---
title: SmartVision
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# 🧠 SmartVision

SmartVision is an AI-based computer vision project that analyzes an uploaded image or a camera photo and provides:

- Person detection
- Detection of eyeglasses, pens, bags, cups, mobile phones, ties, and bottles
- Dominant colour detection for ties
- Confidence percentages for seven facial emotions
- Estimated age prediction

## Technologies Used

- Python
- Streamlit
- OpenCV
- Ultralytics YOLO-World
- DeepFace
- Pillow
- NumPy

## How to Use

1. Select the **Upload Image** mode and upload an image.
2. Select the **Camera Capture** mode to take a photo using a webcam or mobile camera.
3. The app will display detected objects, tie colour, emotion confidence percentages, and the estimated age.

## Note

The live webcam mode works best on a local computer. In the deployed app, please use **Upload Image** and **Camera Capture**.