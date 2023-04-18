import cv2
import time
import torch
import numpy
from picamera2 import Picamera2

# Load model
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
print(cls_dict)

# Camera setup
picam = Picamera2()

# Camera config
picam.preview_configuration.main.size = (640,640)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")

# variable setup
cls_color = {
  "fire":    (14,14,255),
  "hand":   (14,88,207),
  "human":  (207,66,14),
  "pot":    (104,207,14)
}


picam.start()
currTime = time.time()
prevTime = None
while True:
    prevTime = currTime
    currTime = time.time()
    fps = 1.00/(currTime - prevTime)

    im = picam.capture_array()

    # AI inference
    result = model(im)
    # result.print()
    log_dict = {
        "fire":    False,
        "hand":   False,
        "human":  False,
        "pot":    False
    }
    msgs = []
    for tensor in result.xyxy:
        for detect_ret in tensor.numpy():
            point1 = (detect_ret[0].astype(int), detect_ret[1].astype(int))     # top left
            point2 = (detect_ret[2].astype(int), detect_ret[3].astype(int))     # bottom right
            point3 = (detect_ret[0].astype(int)+3, detect_ret[3].astype(int)-10)
            confidance = detect_ret[4]*100              # confidance
            cls = int(detect_ret[5])                    # class index
            class_name = cls_dict[cls]                  # class name of that index
            log_dict[class_name] = True                 # chage logic dictionary

            msgs.append([class_name+"{:.2f}%".format(confidance), point3, cls_color[class_name]])
            

            im = cv2.rectangle(im, point1, point2, (50,50,50), 5)
            im = cv2.rectangle(im, point1, point2, cls_color[class_name], 2)

    for msg in msgs:
        im = cv2.putText(im, msg[0], msg[1], 0, 0.5, (50,50,50), 5)
        im = cv2.putText(im, msg[0], msg[1], 0, 0.5, msg[2], 2)

    print(log_dict)
    flag = (not((log_dict["human"] or log_dict["hand"]) and log_dict["pot"])) and log_dict["fire"]
    if flag:
        im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (50,50,50),5)
        im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (0,0,255),2)
    else:
        im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (50,50,50),5)
        im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (0,255,0),2)
    im = cv2.putText(im, f'FPS: {round(fps,2)}',  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 200), 5, cv2.LINE_AA)
    im = cv2.putText(im, f'FPS: {round(fps,2)}',  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera", im)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()