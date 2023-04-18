from picamera import PiCamera
import time

#start the camera
cam = PiCamera()

#config of the camera
cam.resolution = (640, 480)
cam.vflip = True
cam.hflip = False

#Warm up is always needed for the PiCamera
cam.start_preview()
time.sleep(2)       #wait 2 sec min to avoid error...

#singel picture
#the photo will save as "test.jgp" in this location of the .py file
cam.capture("test.jgp")

#recording