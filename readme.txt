Project 1: Video Special Effects
Name: Shivang Patel
Date: January 23, 2026
Video Links: None
Time Travel Days Used: 0

About This Project
This project is about applying different filters to live video from webcam.
I used OpenCV for image processing and also added depth estimation using 
the Depth Anything V2 model.

How to Run
1. Go to src folder
2. Run "make vid" to build the video app
3. Run "../bin/vid" to start
4. Press different keys to switch filters

Keyboard Controls
q = quit
s = save screenshot
c = normal color
g = greyscale (opencv)
h = greyscale (my own method)
p = sepia (old photo look)
b = blur
x = sobel x (shows vertical lines)
y = sobel y (shows horizontal lines)
m = edge strength
l = blur + quantize (less colors)
f = face detection
d = depth map
1 = spotlight (face in color, rest grey)
2 = neon edges
3 = cartoon effect
4 = fog effect using depth

Files I Made
- imgDisplay.cpp : shows an image
- vidDisplay.cpp : main video app with all filters
- filters.cpp    : all the filter functions
- filters.h      : header for filters
- faceDetect.cpp : face detection code
- faceDetect.h   : header for face detection

Notes
- Need OpenCV 4 installed
- For depth filter, need ONNX Runtime and the depth model in data folder
- On Mac, might need to allow camera permission on first run