import torch
import cv2
import numpy
import time
import os
from picamera2 import Picamera2, Preview

picam = Picamera2()

config = picam.create_preview_configuration(main={"size": (640,640), "format":("RGB888")})
picam.configure(config)

# init model by hub load
model = torch.hub.load('yolov5', 'custom', path='best.onnx', source='local', device='cpu')

# Model config
# model.conf = 0.25  # NMS confidence threshold
# model.iou = 0.45  # NMS IoU threshold
# model.agnostic = False  # NMS class-agnostic
# model.multi_label = True  # NMS multiple labels per box
# model.max_det = 1000  # maximum number of detections per image
# model.amp = False  # Automatic Mixed Precision (AMP) inference

# save model class information
cls_dict = model.names
# print(cls_dict)

# load Image
picam.start()

print("take picture in 3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)

image = picam.capture_array()
print("image captured")
cv2.imshow("detection result", image)

# Inference
results = model(image)
results.print()

# unattened cooking detection logic
log_dict = {
  "fire":    False,
  "hand":   False,
  "human":  False,
  "pot":    False
}

# image = cv2.imread(im_path)

cls_color = {
  "fire":    (14,14,255),
  "hand":   (14,88,207),
  "human":  (207,66,14),
  "pot":    (104,207,14)
}

msgs = []

# visualization result
for tensor in results.xyxy:
    for detect_ret in tensor.numpy():
        point1 = (detect_ret[0].astype(int), detect_ret[1].astype(int))     # top left
        point2 = (detect_ret[2].astype(int), detect_ret[3].astype(int))     # bottom right
        point3 = (detect_ret[0].astype(int)+3, detect_ret[3].astype(int)-10)
        confidance = detect_ret[4]*100              # confidance
        cls = int(detect_ret[5])                    # class index
        class_name = cls_dict[cls]                  # class name of that index
        log_dict[cls] = True                        # chage logic dictionary
        # print(point1, point2, confidance, class_name)

        msgs.append([class_name+"{:.2f}%".format(confidance), point3, cls_color[class_name]])
        

        image = cv2.rectangle(image, point1, point2, (50,50,50), 5)
        image = cv2.rectangle(image, point1, point2, cls_color[class_name], 2)
        # image = cv2.putText(image, print_msg, point3, 0, 0.5, (50,50,50), 2)
        # image = cv2.putText(image, print_msg, point3, 0, 0.5, cls_color[class_name], 2)

for msg in msgs:
    image = cv2.putText(image, msg[0], msg[1], 0, 0.5, (50,50,50), 5)
    image = cv2.putText(image, msg[0], msg[1], 0, 0.5, msg[2], 2)

# print(msgs)x
flag = (not((log_dict["human"] or log_dict["hand"]) and log_dict["pot"])) and log_dict["fire"]
if flag:
    image = cv2.putText(image, "CAUTION", (10,20), 0, 0.6, (50,50,50),5)
    image = cv2.putText(image, "CAUTION", (10,20), 0, 0.6, (0,0,255),2)
else:
    image = cv2.putText(image, "SAFE", (10,20), 0, 0.6, (50,50,50),5)
    image = cv2.putText(image, "SAFE", (10,20), 0, 0.6, (0,255,0),2)

cv2.imwrite('result.jpg', image)

cv2.imshow("detection result", image)
while not(cv2.waitKey(1)==ord('q')):
    pass
cv2.destroyAllWindows()