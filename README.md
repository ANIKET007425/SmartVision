# 🧠 SmartVision

SmartVision is an AI-based computer vision web app built with Streamlit. Upload an image or take a photo with your camera, and SmartVision will detect everyday objects and analyze the face in the picture.

## Features

- **Object detection** with YOLO-World: person, eyeglasses, pen, bag, cup, mobile phone, tie and bottle
- **Tie colour detection**: tells you the dominant colour of a detected tie
- **Emotion analysis**: confidence percentages for seven facial emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Estimated age range**: a lightweight model predicts one of eight age ranges (0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- **No-face check**: if no real face is found in the image, the app says "Face not detected" instead of guessing an emotion

## How to Use

1. Choose **Upload Image** and select a photo, or choose **Camera Capture** and take a photo with a webcam or mobile camera.
2. The app shows the detected objects with boxes and confidence scores.
3. If a face is found, you also get the dominant emotion, the emotion percentages and the estimated age range.

> **Note:** A live webcam mode works on a local computer only. In the deployed app, please use **Upload Image** or **Camera Capture**.

## Technologies Used

- Python
- Streamlit
- OpenCV (including the OpenCV DNN module)
- Ultralytics YOLO-World
- DeepFace (MTCNN face detector, emotion model)
- Age model by Gil Levi and Tal Hassner (Caffe, trained on the Adience dataset)
- PyTorch, Pillow, NumPy

## Project Structure

```
.
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python packages
├── packages.txt            # System packages needed on Streamlit Cloud
├── yolov8s-worldv2.pt      # YOLO-World model
└── models/
    ├── age_deploy.prototxt # Age model structure
    └── age_net.caffemodel  # Age model weights
```

## Run Locally

1. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the `models` folder contains `age_deploy.prototxt` and `age_net.caffemodel`. If they are missing, the app still works, but the age will show as "N/A".

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Notes and Limitations

- The age is an **estimate** and is shown as a range. Lighting, camera angle and image quality can change the result.
- Emotion percentages are model predictions, not facts about how a person feels.
- Very small objects (such as pens) can be missed, especially in low-quality images.
- The app code does not save uploaded images or camera photos.

## Credits

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO-World
- [DeepFace](https://github.com/serengil/deepface) for emotion analysis
- Gil Levi and Tal Hassner for the age classification model ([project page](https://talhassner.github.io/home/publication/2015_CVPR))