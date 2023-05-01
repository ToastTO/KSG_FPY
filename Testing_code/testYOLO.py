import torch

# Model
model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')  # local repo

# Image
im = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(im)

results.pandas().xyxy[0]