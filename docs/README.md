# Kitchen Safety Guide <br>
The Kitchen Safety Guide is an AI-embedded surveillance system for kitchen safety monitoring. By using the RGB MIPI camera, the system can check the existence of unattended fire in real-time. This project is a FYP by HKUST ECE students.

# Features
- **Real-time unattended fire detection:** The system can recognize 4 object which is fire, human, hand, and pot. If there is nobody taking care of the fire, the system can respond immediately, which can remind the user as soon as possible. <br>
- **Lower price:** Compare to the recent kitchen safety equipment, kitchen Safety Guided uses the RGB camera instead of expensive sensor or fire suppression equipment, which is affordable for household usage. <br>
- **user-friendly:** our system will output through a monitor, user can easily turn on and control the system and get the message from the monitor

# How it works
The MIPI RGB camera will first collect the raw data and create an RGB numpy array. After that, each image will be resized to 1x3x640x640 array. The brightness, contrast, and normalization will be adjusted. The adjusted image will undergo a custom-trained YOLOv5 AI model to detect the object in the image. The algorithm will calculate the danger score of the environment depending on the object. At last,the score and the result will be shown on the monitor to remind users via visual warining and beeping sound.

# Component need
- Raspberry Pi 4
- MIPI RGB camera
- Display Monitor (HDMI/ mini-HDMI recommended)
- Keyboard and mouse 

# Library used
OpenCV, NumPy, PiCamera2, PyGame

# Installation 
First, clone repository and download the required library by following command:
```
git clone https://github.com/ToastTO/KSG_FYP
cd KSG_FYP
pip install -r docs/requirements.txt
```
Since our AI using YOLOv5 framework by ultralytics, downloading the ultralytics/yolov5 repository is needed as well. <br>
Details of installing yolov5 can found at its repository: https://github.com/ultralytics/yolov5
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

# Quick start
After installing the required repositories and libraries, connect your MIPI camera and run the following command in terminal:
```
python main.py
```
The detection result and FPS will show up in the terminal. When an unattended cooking event is detected, the device will make a beeping sound to warm the user.

you can also use .onnx AI model for inferencing as well
```
python main.py -o
```

to show inference result in action, run the command followed by `--display` or `-d` 
```
python main.py -d
```
A live streaming window will show up. Press 'q' on the window to end the program and exit the streaming.

type in `python main.py -h` for more information.

# Documentation
All Documents are in "docs" folder, including the FYP Final Report, Poster, Documentation, and a power point for presentation.

# Credit 
Some libraries and online resources are used in this poject:
- <a href="https://github.com/ultralytics/yolov5">YOLOv5</a>
- <a href="https://www.pygame.org/">PyGame</a>
- <a href="https://pytorch.org/get-started/locally">PyTorch</a>
- <a href="https://pytorch.org/vision/stable/index.html">TorchVision</a>
- <a href="https://onnx.ai/">ONNX</a>

# Member
CHUI, Chi To Anson

CHUNG, Wai Lok

CHOY, Yu Hin

Special Thanks: Professor Mow Wai Ho @ HKUST ECE Department
