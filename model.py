import cv2
import time
import torch
import numpy as np
import pygame
import onnx
import onnxruntime as ort
from nms import non_max_suppression
from picamera2 import Picamera2

class KSG:
    def __init__(self, model):
        self.model_name = model
        if model == "torch":
            # toch Model init
            self.model = torch.hub.load('yolov5', 'custom', path='model/best.pt', source='local', device='cpu', _verbose=False) # load silently
            self.model.conf = 0.25  # NMS confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = True  # NMS multiple labels per box
            self.model.max_det = 100  # maximum number of detections per image
            self.model.amp = False  # Automatic Mixed Precision (AMP) inference
            self.cls_dict = self.model.names
        elif model == "onnx":
            self.model = onnx.load("model/best.onnx")
            onnx.checker.check_model(self.model)
            self.ort_sess = ort.InferenceSession('model/best.onnx')
            self.outname = [i.name for i in self.ort_sess.get_outputs()]
            self.inname = [i.name for i in self.ort_sess.get_inputs()]

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
    
    def __call__(self, mode, path=None):     #run inference
        if mode == "live":
            print("now start KSG live Streaming")
            self.live_stream()
            print("live streaming finish")

        if mode == "file":
            pass

    def preprocess(image):
        print(image.shape)
        # data process
        image = cv2.resize(image, (640,640), cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        print(image.shape)
        image = np.expand_dims(image, axis=0)
        image = image.transpose(0,3,1,2).astype(np.float32)
        print(image.shape)    # onnx input
        return image

    def inference(self,input):
        # run model interence, input can be (3,640,640) or (N,3,640,640)
        if self.model_name == "torch":
            results = self.model(input)

        elif self.model_name == "onnx":
            input = {self.inname[0]: input}
            outputs = self.ort_sess.run(self.outname, input)
            results = non_max_suppression(outputs)
        return results
    
    # input (image, [x, y, x, y, conf, class])
    def draw_bbox(self, im, dectection):
        upL = (dectection[0].astype(int), dectection[1].astype(int))     # top left
        BottomR = (dectection[2].astype(int), dectection[3].astype(int))     # bottom right
        conf = dectection[4]*100              # confidance
        cls = int(dectection[5])                    # class index
        class_name = self.cls_dict[cls]  # class name of that index
        color = self.cls_color[class_name] 

        # bBox
        im = cv2.rectangle(im, upL, BottomR, (50,50,50), 3)
        im = cv2.rectangle(im, upL, BottomR, self.cls_color[color], 1)

        #cls
        msg = class_name + "{:.2f}%".format(conf)
        x = int(upL[0]) + 3
        y = int(BottomR[1]) - 10
        im = cv2.putText(im, msg, (x,y), 0, 0.6, (50,50,50), 3)
        im = cv2.putText(im, msg, (x,y), 0, 0.6, self.cls_color[color], 1)
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
        while cv2.waitKey(1) != ord('q') and cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:
            # calulcation FPS
            prevTime = currTime
            currTime = time.time()
            fps = round(1.00/(currTime - prevTime),2)

            # capture frame
            im = self.picam.capture_array()
            im = cv2.convertScaleAbs(im, 10, 0.98)

            # singel inference
            if self.model_name == "torch":
                result = self.inference(im).xyxy[0]
            elif self.model_name == "onnx":
                im = self.preprocess(im)
                result = self.inference(im)

            # record what detected in each image
            log_dict = {
                "fire":   False,
                "hand":   False,
                "human":  False,
                "pot":    False
            }

            # draw box for each detected object
            for detect_ret in result.numpy():
                
                log_dict[detect_ret[5]] = True
                
                #draw bBox, conf, cls to im
                im = self.draw_bbox(im, detect_ret)
            
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
    
def video_path2arrary(path):
    cap = cv2.VideoCapture(path)
    
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    array = np.zeros((N,H,W,3))
    # print(array.shape)
    for i in range(N):
        ret, frame = cap.read()
        if ret:
            # cv2.imshow('frame',frame)
            # cv2.waitKey(0)
            array[i,...] = frame
            # print(i)
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return array