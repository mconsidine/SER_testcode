#Python test code for reading AVI files
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

def AVI_time_seconds(h):
    timestamp_1970 = int(621355967998300000) - int(1e7*(4*60*60-0.17))
    s=float(h-timestamp_1970)/1e7 # convert to seconds
    return s # number of seconds from 0001 to 1970

AVI_hdr_dt = np.dtype([
    ('starthdr', 'S32'),
    ('DT', '<u4'),
    ('mischdr', 'S12'),
    ('FrameCount', '<u4'),
    ('mischdr2', 'S12'),
    ('Width', '<u4'),
    ('Height', '<u4')
])


#------- start of optional reading of settings file
AVI_settingsdata = []
AVI_log_filename = "./TestZWO.CameraSettings.txt"
with open(AVI_log_filename, 'r') as f:
    AVI_settingsdata = f.readlines()

if (len(AVI_settingsdata) > 0) == True:	    
    for line in AVI_settingsdata[1:]:
        line = line.strip()
        if line.startswith("[") == True:
            AVI_SettingsCamera = line
            print("AVI settings camera : ",AVI_SettingsCamera)
        else:
            avar = line.split('=')[0]
            aval = line.split('=')[1]
            if avar == "Exposure":
                AVI_SettingsExposure = float(aval)
                print("AVI settings exposure : ",AVI_SettingsExposure)
            if avar == "TimeStamp":
                AVI_SettingsTimeStamp = dt.datetime.strptime(aval,"%Y-%m-%dT%H:%M:%S.%f7%z")
                print("AVI settings timestamp : ",AVI_SettingsTimeStamp)
            if avar == "Tilt":
                AVI_SettingsTilt = int(aval)
                print("AVI settings tilt : ",AVI_SettingsTilt)
            if avar == "Pan":
                AVI_SettingsPan = int(aval)
                print("AVI settings pan : ",AVI_SettingsPan);
else:
    print("No settings file found or retrieved.  Continuing")
#------- end of optional reading of settings file

AVI_headersize = 72 #bytes
AVI_filename = './pipp_20210727_211356/2021-07-22-1645_5_pipp.avi'

#now open video file
AVI_filesize = os.stat(AVI_filename).st_size
AVI_fileref = open(AVI_filename, 'rb')

AVI_header = np.frombuffer(AVI_fileref.read(AVI_headersize),dtype=AVI_hdr_dt)
AVI_fileref.close()

print(AVI_header)

ColorID = 1
#ColorID=AVI_header["ColorID"]
Width=int(AVI_header["Width"])
Height=int(AVI_header["Height"])
PixelDepthPerPlane = 8
#PixelDepthPerPlane=int(AVI_header["PixelDepthPerPlane"])
FrameCount=int(AVI_header["FrameCount"])
#DateTime=AVI_header["DateTime"]
#DateTimeUTC=AVI_header["DateTimeUTC"]

if (ColorID < 100):
  NumberOfPlanes = 1
else:
  NumberOfPlanes = 3

if (PixelDepthPerPlane < 9):
  BytesPerPixel = 1*NumberOfPlanes
  AVI_datatypesize = np.uint8
else:
  BytesPerPixel = 2*NumberOfPlanes
  AVI_datatypesize = np.uint16

AVI_framesize = NumberOfPlanes *  Width * Height
AVI_framesizebytes = AVI_framesize*BytesPerPixel

AVI_traileroffset = AVI_headersize + FrameCount * Width * Height * BytesPerPixel
AVI_trailersize = AVI_filesize-AVI_headersize-BytesPerPixel*Width*Height*FrameCount
AVI_timestamps = AVI_trailersize/8 #should equal FrameCount

AVI_hastimestamps = False
if AVI_timestamps == FrameCount:
  AVI_hastimestamps = True

flag_rotate = False
ih=Height
iw=Width

if (Width > Height) == True:
    flag_rotate = True
    ih = Width
    iw = Height

#structure to hold summed up frame values
my_data = np.zeros((ih,iw),dtype='uint64')

AVI_fileref = cv2.VideoCapture(AVI_filename)  #MattC new line; we access the file 2 ways
  
for framenum in range(0,FrameCount-1):
    #myresult = np.frombuffer(AVI_fileref.read(AVI_framesizebytes),dtype=AVI_datatypesize)
    (ret, myresult) = AVI_fileref.read() #MattC
    if len(myresult)>0:
        #myresult = np.reshape(myresult,(ih, iw)).astype('uint64')
        myresult=cv2.cvtColor(myresult, cv2.COLOR_BGR2GRAY)
        my_data=np.add(myresult,my_data)

#don't need to reshape/rotate until after if we want to view average frame
my_data=np.reshape(my_data,(ih, iw))
if flag_rotate:
    my_data = np.rot90(my_data)
    
plt.title("Average frame")
plt.imshow(my_data/int(FrameCount), cmap='gray')
plt.show()

#close file so we can read the frames again
AVI_fileref.release()

#------------------------------------------------------
#structure to hold one column of each frame to form disc
disc = np.zeros((ih,FrameCount), dtype='uint16')

#read frames; if use "with" it will automatically close file, eg
#     with open(AVI_filename,'rb') as AVI_fileref:
AVI_fileref = cv2.VideoCapture(AVI_filename)  #MattC new line

#reread header
#  not needed with opencv
        
for framenum in range(0,FrameCount):
    #myresult = np.frombuffer(AVI_fileref.read(AVI_framesizebytes),dtype=AVI_datatypesize)
    (ret, myresult) = AVI_fileref.read() #MattC
    #print(framenum+1,AVI_filesize-(framenum+1)*AVI_framesizebytes-AVI_headersize, AVI_trailersize)
    if len(myresult)>0:
        myresult=cv2.cvtColor(myresult, cv2.COLOR_BGR2GRAY)
        myresult = np.reshape(myresult,(Height, Width))#.astype('uint16')
        disc[:,framenum]=myresult[:,Width//2]#.astype('uint16') #placeholder for IntensiteRaie calc
         
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

#there aren't timestamps for frames, as far as I know, for avi files  So this should show 0
print("whats left reading trailer:",AVI_filesize-AVI_headersize-FrameCount*AVI_framesize*BytesPerPixel-AVI_trailersize)
'''
AVI_timestampdata = np.frombuffer(AVI_fileref.read(AVI_trailersize),dtype='<u8')

print("Count of timestamps ",len(AVI_timestampdata))

for timestamps in range(0,FrameCount):
    if (timestamps+1) %  100 == 0:
        print(timestamps+1,dt.datetime.fromtimestamp(AVI_time_seconds(AVI_timestampdata[timestamps])))

AVI_elapsed_time = AVI_time_seconds(AVI_timestampdata[FrameCount-1])-AVI_time_seconds(AVI_timestampdata[0])
print("Elapsed time : ",AVI_elapsed_time)
print("Avg frames per second : ",(FrameCount-1)/AVI_elapsed_time) #not sure about -1 here??
print("Avg interval between frames : ", AVI_elapsed_time/(FrameCount-1))
'''
#now close file
AVI_fileref.release()

