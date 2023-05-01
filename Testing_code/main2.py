import cv2
import time
import torch
import numpy
from picamera2 import Picamera2

# Load model
model = torch.hub.load('yolov5', 'custom', path='best.onnx', source='local', device='cpu')

# Model configuration
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = True  # NMS multiple labels per box
model.max_det = 100  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

# save model class information
cls_dict = model.names
#check if class dictionary can be written correctly
#print(cls_dict)

# Camera setup
picam = Picamera2()

# Camera config
# main.size: defines the width and height of the camera intake 
# main.format: format of image intake, "RGB": Red-Green-Blue Format ; "888": 8-bit information per pixel
# 
dim = [640, 640]
picam.preview_configuration.main.size = (dim[0], dim[1])
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")

# variable setup
# cls_color: dictionary
# weight_fire: 
# weight_human: 
# weight_pot: 
#
cls_color = {
  "fire":    (14,14,255),
  "hand":   (14,88,207),
  "human":  (165,15,240),
  "pot":    (104,207,14)
}
weight_fire = 0.3
weight_human = -0.5
weight_pot = 0.2
threshold = (weight_fire + weight_human + weight_pot)/2

picam.start()
currTime = time.time()
prevTime = None
flag = 0 # initialize flag for calculation
while True:
    prevTime = currTime
    currTime = time.time()
    fps = 1.00/(currTime - prevTime)
    im = picam.capture_array()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Image preprocessing: convertScaleAbs (image, alpha,beta)
    # alpha(num, [0,2]): contrast
    # beta(num, [-127,127]): brightness
    im = cv2.convertScaleAbs(im, 10, 0.98)
    # AI inference
    result = model(im)
    # result.print()
    log_dict = {
        "fire":   False,
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
        im = cv2.putText(im, msg[0], msg[1], 0, 0.6, (50,50,50), 5)
        image = cv2.putText(im, msg[0], msg[1], 0, 0.6, msg[2], 2)

    print(log_dict)
    pastFlag = flag*0.5 if flag else 0
    flag = pastFlag + (log_dict["human"] or log_dict["hand"])* -0.5 + log_dict["pot"] * 0.2 + log_dict["fire"] * 0.3
    flag = flag*1 if flag > 0.05 else 0
    cv2.putText(im,f'Score: {round(flag,2)}', (10,80), 0, 0.6, (50,50,50),2)
    if round(flag,2) > threshold+0.01:
        im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (50,50,50),5)
        im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (0,0,255),2)
    else:
        im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (50,50,50),5)
        im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (0,255,0),2)

    im = cv2.putText(im, f'FPS: {round(fps,2)}',  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("Camera", im)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
