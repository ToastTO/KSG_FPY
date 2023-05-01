import cv2
import time
import torch
import numpy
import pygame
from picamera2 import Picamera2

class KSG_torch:
    def __init__(self):
        # Model init
        self.model = torch.hub.load('yolov5', 'custom', path='model/best.pt', source='local', device='cpu')
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = True  # NMS multiple labels per box
        self.model.max_det = 100  # maximum number of detections per image
        self.model.amp = False  # Automatic Mixed Precision (AMP) inference
        self.cls_dict = self.model.names

        self.cls_color = {
            "fire":    (14,14,255),
            "hand":   (14,88,207),
            "human":  (165,15,240),
            "pot":    (104,207,14)
        }

        #picam init
        self.picam = Picamera2()
        self.picam.preview_configuration.main.size = (640, 640)
        self.picam.preview_configuration.main.format = "RGB888"
        self.picam.preview_configuration.align()
        self.picam.configure("preview")

        # Music output setup
        pygame.mixer.init()
        pygame.mixer.music.load('Testing_code/Z7E8E5U-beep-beep.mp3')
    
    def __str__(self):
        return("model info...")
    
    def __call__(self, mode):     #run inference
        pass

    def inference(self,image):
        # run model interence, input can be (3,640,640) or (N,3,640,640)
        results = self.model(image)
        return results
    
    def draw_bbox(self, im, point1, point2, cls_name, conf):

        # bBox
        im = cv2.rectangle(im, point1, point2, (50,50,50), 3)
        im = cv2.rectangle(im, point1, point2, self.cls_color[cls_name], 1)

        #cls
        msg = cls_name + "{:.2f}%".format(conf)
        x = int(point1[0]) + 3
        y = int(point1[1]) - 3
        im = cv2.putText(im, msg, (x,y), 0, 0.6, (50,50,50), 3)
        im = cv2.putText(im, msg, (x,y), 0, 0.6, self.cls_color[cls_name], 1)
        return im

    def live_stream(self):
        # live AI inference with unattended cooking detection algorithm
        cls_w = {
            "fire":   0.3,
            "hand":   -0.5,
            "human":  -0.5,
            "pot":    -0.2
        }
        threshold = 0.45
        
        flag = 0

        self.picam.start()
        currTime = time.time()
        prevTime = None
        # press "q" to exit loop
        while cv2.waitKey(1) != ord('q'):
            # calulcation FPS
            prevTime = currTime
            currTime = time.time()
            fps = round(1.00/(currTime - prevTime),2)

            # capture frame
            im = self.picam.capture_array()
            im = cv2.convertScaleAbs(im, 10, 0.98)

            # singel inference
            result = self.inference(im).xyxy[0]

            # record what detected in each image
            log_dict = {
                "fire":   False,
                "hand":   False,
                "human":  False,
                "pot":    False
            }

            # draw box for each detected object
            for detect_ret in result.numpy():
                upL = (detect_ret[0].astype(int), detect_ret[1].astype(int))     # top left
                BottomR = (detect_ret[2].astype(int), detect_ret[3].astype(int))     # bottom right
                conf = detect_ret[4]*100              # confidance
                cls = int(detect_ret[5])                    # class index
                class_name = self.cls_dict[cls]  # class name of that index
                log_dict[class_name] = True                 # chage logic dictionary
                
                #draw bBox, conf, cls to im
                im = self.draw_bbox(im, upL, BottomR, class_name, conf)
            
            pastFlag = pastFlag = flag*0.5 if flag else 0
            flag = pastFlag + (log_dict["human"] or log_dict["hand"])*cls_w["hand"] + log_dict["pot"]*cls_w["pot"] + log_dict["fire"]*cls_w["fire"]
            if flag < 0.05:
                flag = 0
            flag = round(flag,2)
            # print algorithm result
            if flag > threshold:
                im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (50,50,50),5)
                im = cv2.putText(im, "CAUTION", (10,20), 0, 0.6, (0,0,255),2)
                pygame.mixer.music.play()
            else:
                im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (50,50,50),5)
                im = cv2.putText(im, "SAFE", (10,20), 0, 0.6, (0,255,0),2)
            # print FPS
            im = cv2.putText(im, f'FPS: {fps}', (10, 50), 0, 0.6, (50, 50, 50), 5)
            im = cv2.putText(im, f'FPS: {fps}', (10, 50), 0, 0.6, (255, 0, 0), 2)
            # print flag
            im = cv2.putText(im,f'Score: {flag}', (10,80), 0, 0.6, (50,50,50),5)
            im = cv2.putText(im,f'Score: {flag}', (10,80), 0, 0.6, (255, 0, 0),2)
            
            cv2.imshow("KSG Live Streaming",im)
                
        cv2.destroyAllWindows()
        return

