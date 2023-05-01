# Kitchen Safety Guided <br>
The Kitchen Safety Guided is an AI-embedded surveillance system for kitchen safety monitoring. By using the RGB MIPI camera, the system can check the existence of unattended fire in real-time.

# Features
-Real-time unattended fire detection: The system can recognize 4 object which is fire, human, hand, and pot.If there is nobody taking care of the fire, the system can respond immediately, which can remind the user as soon as possible. <br>
-Lower price: Compare to the recent kitchen safety equipment, kitchen Safety Guided uses the RGB camera instead of expensive sensor or fire suppression equipment, which is affordable for household usage. <br>
-user-friendly: our system will output through a monitor, user can easily turn on and control the system and get the message from the monitor

# How it works
The MIPI RGB camera will first collect the raw data and create an RGB numpy array. After that,each image will be resized to 1x3x640x640, the brightness, contrast, and normalization will be adjusted. The adjusted image will undergo a custom-trained YOLOv5 AI model to detect the object in the image. The algorithm will calculate the danger score of the environment depending on the object. At last,the score and the result will be shown on the monitor to remind users.

# Component need
Raspberry Pi, MIPI RGB camera, Display Monitor, keyboard, mouse 

# Installation 

# How to start
