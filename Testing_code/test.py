import torch
import torchvision

import time
from picamera2 import Picamera2, Preview

# torch==1.10.2
# torchvision==0.11.3


#capture a photo
picam = Picamera2()
config = picam.create_preview_configuration()
picam.configure(config)
picam.start()

picam.capture_file("test.jpg")
time.sleep(0.5)

picam.stop()

#load  model
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')


im = 'test.jpg'


#Inference
results = model(im)
print(results)
