import onnx
import onnxruntime as ort
import time
import numpy as np
import cv2
import torch
import sys
from nms import non_max_suppression

def draw_bbox(im, dectection):
        cls_dict = {
                0: "fire",
                1: "hand",
                2: "human",
                3: "pot"
        }
        cls_color = {
            "fire":    (14,14,255),
            "hand":   (14,88,207),
            "human":  (165,15,240),
            "pot":    (104,207,14)
        }
        upL = (dectection[0].astype(int), dectection[1].astype(int))     # top left
        BottomR = (dectection[2].astype(int), dectection[3].astype(int))     # bottom right
        conf = dectection[4]*100              # confidance
        cls = int(dectection[5])                    # class index
        class_name = cls_dict[cls]  # class name of that index
        color = cls_color[class_name] 
        print(upL,BottomR,conf, class_name, color)
        # bBox
        im = cv2.rectangle(im, upL, BottomR, (50,50,50), 3)
        im = cv2.rectangle(im, upL, BottomR, color, 2)

        #cls
        msg = class_name + "{:.2f}%".format(conf)
        x = int(upL[0]) +10
        y = int(BottomR[1]) - 10
        im = cv2.putText(im, msg, (x,y), 0, 0.6, (50,50,50), 3)
        im = cv2.putText(im, msg, (x,y), 0, 0.6, color, 2)
        return im

onnx_model = onnx.load("model/best.onnx")
onnx.checker.check_model(onnx_model)

# get image
im_path = "Test_img/test1.jpg"
image = cv2.imread(im_path, cv2.IMREAD_COLOR)
print(image.shape)

# data process
H = image.shape[0]
W = image.shape[1]
x = cv2.resize(image, (640,640), cv2.INTER_LINEAR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = x.astype(np.float32)
x /= 255.0
print(x.shape)
x = np.expand_dims(x, axis=0)
x = x.transpose(0,3,1,2).astype(np.float32)
print(x.shape)    # onnx input


ort_sess = ort.InferenceSession('model/best.onnx')
outname = [i.name for i in ort_sess.get_outputs()]
inname = [i.name for i in ort_sess.get_inputs()]
meta = ort_sess.get_modelmeta().custom_metadata_map

# print(meta, "\n", outname, "\n", inname)
# print(len(outname))
# print(len(inname))

input = {inname[0]: x}
outputs = ort_sess.run(outname, input)[0]
print(outputs.shape)
# output shape
# (num_of image, (3 input reshape 8, 16, 32), num_cls+xyxy+conf)

colors = {
  "fire":    (14,14,255),
  "hand":   (14,88,207),
  "human":  (207,66,14),
  "pot":    (104,207,14)
}

pred = non_max_suppression(outputs)

print(pred[0].shape)

for object in pred[0]:
    print(object)
    xyxy = object[:4]
    xyxy[0] *= W/640
    xyxy[2] *= W/640
    xyxy[1] *= H/640
    xyxy[3] *= H/640
    
    print(xyxy)
    object[:4] = xyxy
    print(object)
    image = draw_bbox(image, object)

cv2.imshow("detection result", image)
while not(cv2.waitKey(1)==ord('q')):
    pass
cv2.destroyAllWindows()