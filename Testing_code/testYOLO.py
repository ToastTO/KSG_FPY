import torch

# Model
model = torch.hub.load('yolov5', 'custom', path='model/best.pt', source='local')  # local repo

# Image
im = 'Test_img/test1.jpg'

# Inference
results = model(im)
results.show()

results.pandas().xyxy[0]