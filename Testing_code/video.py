import cv2
import numpy as np
import torch


def xyxynp2point(detect_ret):
  cls_dict = {
    0: "fire",
    1: "hand",
    2: "human",
    3: "pot"
  }
  upL = (detect_ret[0].astype(int), detect_ret[1].astype(int))     # top left
  BottomR = (detect_ret[2].astype(int), detect_ret[3].astype(int))     # bottom right
  conf = detect_ret[4]*100              # confidance
  cls = int(detect_ret[5])                    # class index
  class_name = cls_dict[cls]  # class name of that index
  return upL, BottomR, class_name, conf

def draw_bbox(im, point1, point2, cls_name, conf):
  cls_color = {
            "fire":    (14,14,255),
            "hand":   (14,88,207),
            "human":  (165,15,240),
            "pot":    (104,207,14)
        }
  # bBox
  im = cv2.rectangle(im, point1, point2, (50,50,50), 3)
  im = cv2.rectangle(im, point1, point2, cls_color[cls_name], 1)
  #cls
  msg = cls_name + "{:.2f}%".format(conf)
  x = int(point1[0]) + 3
  y = int(point1[1]) - 3
  im = cv2.putText(im, msg, (x,y), 0, 0.6, (50,50,50), 3)
  im = cv2.putText(im, msg, (x,y), 0, 0.6, cls_color[cls_name], 1)
  return im

def output_mp4(name, images):
  fps = 15
  size = (images[0].shape[0],images[0].shape[1])
  print(size)
  fourcc = cv2.VideoWriter_fourcc(*'avc1') #‘avc1'
  writer = cv2.VideoWriter('out.avi', fourcc, fps, size, 1)

  for i in range(len(images)):
    writer.write(images[i])
    
  writer.release()
  return

cap = cv2.VideoCapture("Test_img/testing1.mp4")

N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (W,H)

# print((W,H))

frames = []
images = []

for i in range(30):
  ret, frame = cap.read()
  if ret: 
    images.append(frame)
    frames.append(frame.transpose((2,0,1)))
  else:
    break
cap.release()

print(len(frames))

output_mp4("raw", images)

# print(len(frames))
# print(frames[4])

# output = np.vstack(frames)
# # print(output.shape)   #should be 10,W,H,3

# X = output.transpose(0,3,1,2)
# print(X.shape)   #should be 10,3,W,H

# model = torch.hub.load('yolov5', 'custom', path='model/best.pt', source='local', device='cpu')
# results = model(frames)    

# print(results)
# print(len(results.xyxy)) # retuen back a list
# for idx, result in enumerate(results.xyxy):

#   for detect_ret in result.numpy():
#     # print(detect_ret.shape)
#     upL, bottomR, class_name, conf = xyxynp2point(detect_ret)
#     print(upL, bottomR, class_name, conf)
#     images[idx] = draw_bbox(images[idx], upL, bottomR, class_name, conf)
#     pass

# output_mp4("out", images)