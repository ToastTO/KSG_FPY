import time
from picamera2 import Picamera2, Preview

picam = Picamera2()

config = picam.create_preview_configuration(main={"size": (240,240)})
picam.configure(config)
picam.start()

picam.capture_file("test.jpg")
print("Captured image 1")
time.sleep(3)

picam.stop()
