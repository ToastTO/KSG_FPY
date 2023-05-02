import cv2
import time
import torch
import numpy

# Load model
model = torch.hub.load('yolov5', 'custom', path='best.onnx', source='local', device='cpu')

# Model config
#model.conf = 0.25  # NMS confidence threshold
#model.iou = 0.45  # NMS IoU threshold
#model.agnostic = False  # NMS class-agnostic
#model.multi_label = True  # NMS multiple labels per box
#model.max_det = 100  # maximum number of detections per image
#model.amp = False  # Automatic Mixed Precision (AMP) inference

# save model class information
cls_dict = model.names
#print(cls_dict)
# variable setup
cls_color = {
  "fire":    (14,14,255),
  "hand":   (14,88,207),
  "human":  (165,15,240),
  "pot":    (104,207,14)
}
weight_fire = 1
weight_hand = -0.6
weight_human = -1.2
weight_pot = 0.4
threshold = 5
flag = 0 # initialize flag for calculation

# read video
cap = cv2.VideoCapture("./20230103_102853000_iOS.mp4")
pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
while True:
    ready, frame = cap.read()
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Image preprocessing: convertScaleAbs (image, alpha,beta)
    # alpha(num, [0,2]): contrast
    # beta(num, [-127,127]): brightness
    #print("cap.read() success")
    #print(ready)
    if ready:
        im = cv2.resize(frame,(640,640))
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
        flag = pastFlag + log_dict["human"] * weight_human + log_dict["hand"]* weight_hand + log_dict["pot"] *weight_pot + log_dict["fire"] * weight_fire
        flag = flag if flag > 0 else 0
        cv2.putText(im, str(flag) , (320,20), 0, 0.6, (50,50,50),2)
        if flag > threshold:
            im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (50,50,50),5)
            im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (0,0,255),2)
        else:
            im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (50,50,50),5)
            im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (0,255,0),2)

        #im = cv2.putText(im, f'FPS: {round(fps,2)}',  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Camera", im)
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
        cv2.waitKey(1000)
    
    if cv2.waitKey(1)==ord('q'):
        break
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break
cv2.destroyAllWindows()
