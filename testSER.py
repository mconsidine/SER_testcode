#Python test code for reading SER files
#  Matt Considine with the help of the internet, 2021-07-30
import datetime as dt
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/44169233/fastest-way-to-read-in-and-slice-binary-data-files-in-python
#https://currentmillis.com/?-62135596799830#julian-date
# date Jan 1, 0001 00:00:00 UTC
# millisec since 1/1/1970  -62135596799830
# millisec since 1/1/1900  -59926607999830
# ticks since 1/1/1601     -504911231998300000
# ticks adjustment to agree with PIPP -143998300000
#                                     

def SER_time_seconds(h):
    timestamp_1970 = int(621355967998300000) - int(1e7*(4*60*60-0.17))
    s=float(h-timestamp_1970)/1e7 # convert to seconds
    return s # number of seconds from 0001 to 1970

SER_hdr_dt = np.dtype([
    ('FileID', 'S14'),
    ('LuID', '<u4'),
    ('ColorID', '<u4'),
    ('little_Endian', '<u4'),
    ('Width', '<u4'),
    ('Height', '<u4'),
    ('PixelDepthPerPlane', '<u4'),
    ('FrameCount', '<u4'),
    ('Observer', 'S40'),
    ('Instrument', 'S40'),
    ('Telescope', 'S40'),
    ('DateTime', '<u8'),
    ('DateTimeUTC', '<u8')
])

#------- start of optional reading of settings file
SER_settingsdata = []
SER_log_filename = "./TestZWO.CameraSettings.txt"
with open(SER_log_filename, 'r') as f:
    SER_settingsdata = f.readlines()

if (len(SER_settingsdata) > 0) == True:	    
    for line in SER_settingsdata[1:]:
        line = line.strip()
        if line.startswith("[") == True:
            SER_SettingsCamera = line
            print("SER settings camera : ",SER_SettingsCamera)
        else:
            avar = line.split('=')[0]
            aval = line.split('=')[1]
            if avar == "Exposure":
                SER_SettingsExposure = float(aval)
                print("SER settings exposure : ",SER_SettingsExposure)
            if avar == "TimeStamp":
                SER_SettingsTimeStamp = dt.datetime.strptime(aval,"%Y-%m-%dT%H:%M:%S.%f7%z")
                print("SER settings timestamp : ",SER_SettingsTimeStamp)
            if avar == "Tilt":
                SER_SettingsTilt = int(aval)
                print("SER settings tilt : ",SER_SettingsTilt)
            if avar == "Pan":
                SER_SettingsPan = int(aval)
                print("SER settings pan : ",SER_SettingsPan);
else:
    print("No settings file found or retrieved.  Continuing")
#------- end of optional reading of settings file

SER_headersize = 178 #bytes
SER_filename = "./2021-07-22-1645_5.ser"

#now open video file
SER_filesize = os.stat(SER_filename).st_size
SER_fileref = open(SER_filename, 'rb')

SER_header = np.frombuffer(SER_fileref.read(SER_headersize),dtype=SER_hdr_dt)

print(SER_header)

FileID=SER_header["FileID"]
LuID=SER_header["LuID"]
ColorID=SER_header["ColorID"]
little_Endian=SER_header["little_Endian"]
Width=int(SER_header["Width"])
Height=int(SER_header["Height"])
PixelDepthPerPlane=int(SER_header["PixelDepthPerPlane"])
FrameCount=int(SER_header["FrameCount"])
Observer=SER_header["Observer"]
Instrument=SER_header["Instrument"]
Telescope=SER_header["Telescope"]
DateTime=SER_header["DateTime"]
DateTimeUTC=SER_header["DateTimeUTC"]

if (ColorID < 100):
  NumberOfPlanes = 1
else:
  NumberOfPlanes = 3

if (PixelDepthPerPlane < 9):
  BytesPerPixel = 1*NumberOfPlanes
  SER_datatypesize = np.uint8
else:
  BytesPerPixel = 2*NumberOfPlanes
  SER_datatypesize = np.uint16

SER_framesize = NumberOfPlanes *  Width * Height
SER_framesizebytes = SER_framesize*BytesPerPixel

SER_traileroffset = SER_headersize + FrameCount * Width * Height * BytesPerPixel
SER_trailersize = SER_filesize-SER_headersize-2*Width*Height*FrameCount
SER_timestamps = SER_trailersize/8 #should equal FrameCount

SER_hastimestamps = False
if SER_timestamps == FrameCount:
  SER_hastimestamps = True

print(dt.datetime.fromtimestamp(SER_time_seconds(DateTime)))
print(dt.datetime.fromtimestamp(SER_time_seconds(DateTimeUTC)))

flag_rotate = False
ih=Height
iw=Width

if (Width > Height) == True:
    flag_rotate = True
    ih = Width
    iw = Height

#structure to hold summed up frame values
my_data = np.zeros((ih*iw),dtype='uint64')
  
for framenum in range(0,FrameCount-1):
    myresult = np.frombuffer(SER_fileref.read(SER_framesizebytes),dtype=SER_datatypesize)
    if len(myresult)>0:
        #myresult = np.reshape(myresult,(ih, iw)).astype('uint64')
        my_data=np.add(myresult,my_data)
        
#don't need to reshape/rotate until after if we want to view average frame
my_data=np.reshape(my_data,(ih, iw))
if flag_rotate:
    my_data = np.rot90(my_data)
    
plt.title("Average frame")
plt.imshow(my_data/int(FrameCount), cmap='gray')
plt.show()

#close file so we can read the frames again
SER_fileref.close()

#structure to hold one column of each frame to form disc
disc = np.zeros((ih,FrameCount), dtype='uint16')

#read frames; if use "with" it will automatically close file, eg
#     with open(SER_filename,'rb') as SER_fileref:
SER_fileref = open(SER_filename, 'rb')
#reread header
SER_header = np.frombuffer(SER_fileref.read(SER_headersize),dtype=SER_hdr_dt)

for framenum in range(0,FrameCount):
    myresult = np.frombuffer(SER_fileref.read(SER_framesizebytes),dtype=SER_datatypesize)
    #print(framenum+1,SER_filesize-(framenum+1)*SER_framesizebytes-SER_headersize, SER_trailersize)
    if len(myresult)>0:
        myresult = np.reshape(myresult,(Height, Width)).astype('uint16')
        disc[:,framenum]=myresult[:,Width//2].astype('uint16') #placeholder for IntensiteRaie calc
            
#don't need to rotate until after the fact        
if flag_rotate:
    disc = np.rot90(disc)
    
##show reconstructed disk
plt.title("reconstructed disk")
plt.imshow(disc, cmap='gray')
plt.show()

##check last frame if needed
#plt.imshow(myresult, cmap='hot')
#plt.show()

#now read timestamps; these could be added to frames or the final frame for example
#info could also be used to estimate exposure intervals and exposure times, perhaps
#improving building of final image
print("whats left reading trailer:",SER_filesize-SER_headersize-FrameCount*SER_framesize*BytesPerPixel-SER_trailersize)

SER_timestampdata = np.frombuffer(SER_fileref.read(SER_trailersize),dtype='<u8')

print("Count of timestamps ",len(SER_timestampdata))

for timestamps in range(0,FrameCount):
    if (timestamps+1) %  100 == 0:
        print(timestamps+1,dt.datetime.fromtimestamp(SER_time_seconds(SER_timestampdata[timestamps])))

SER_elapsed_time = SER_time_seconds(SER_timestampdata[FrameCount-1])-SER_time_seconds(SER_timestampdata[0])
print("Elapsed time : ",SER_elapsed_time)
print("Avg frames per second : ",(FrameCount-1)/SER_elapsed_time) #not sure about -1 here??
print("Avg interval between frames : ", SER_elapsed_time/(FrameCount-1))

#now close file
SER_fileref.close()
'''''''''''''''''''''''''''''
Info from PIPP

Filesize: 226767082 bytes.

Header Details:
 * FileID: LUCAM-RECORDER
 * LuID: 0x1234
 * ColorID: 0 (MONO)
 * LittleEndian: 0
 * ImageWidth: 88
 * ImageHeight: 608
 * PixelDepth: 16
 * FrameCount: 2119
 * Observer: Observer                                
 * Instrument: ZWO ASI174MM(29234266)                  
 * Telescope: telescope                               
 * DateTime: 22/07/2021 12:44:22.707232 (0x08d94d0e6e6bc141)
 * DateTime_UTC: 22/07/2021 16:44:22.707232 (0x08d94d2ff57d6141)

Timestamps:
 * Timestamps are all in order
 * Min timestamp: 22/07/2021 16:44:22.663438 (0x08d94d2ff576b290)
 * Max timestamp: 22/07/2021 16:46:46.789968 (0x08d94d304b5ea926)
 * Min to Max timestamp difference: 2 mins 24.126530 s
 * Average frames per second: 14.695421

Timestamps List:
 * 0001: 22/07/2021 16:44:22.663438 (0x08d94d2ff576b290)
 * 0002: 22/07/2021 16:44:22.731232 (0x08d94d2ff5810ac0)
 * 0003: 22/07/2021 16:44:22.799343 (0x08d94d2ff58b6f5b)
 * 0004: 22/07/2021 16:44:22.867883 (0x08d94d2ff595e4b0)
'''
