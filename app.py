import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from deepface import DeepFace

# ---------------- Page Setup ----------------

st.set_page_config(
    page_title="SmartVision",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SmartVision")
st.write(
    " Upload an image, Take photo with camera, or Run a live webcam. "
    " Detection of Person, eyeglasses, pen, bag, cup, mobile phone, tie and bottle. "
    " You will also get the emotion percentage and estimated age from the face."
)

# ---------------- Load YOLO-World Model ----------------

@st.cache_resource
def load_yolo_world():
    model = YOLOWorld("yolov8s-worldv2.pt")

    custom_classes = [
        "person",
        "eyeglasses",
        "spectacles",
        "pen",
        "ballpoint pen",
        "bag",
        "handbag",
        "backpack",
        "coffee cup",
        "cup",
        "mobile phone",
        "cell phone",
        "necktie",
        "black necktie",
        "blue necktie",
        "red necktie",
        "white shirt",
        "bottle"
    ]

    model.set_classes(custom_classes)
    return model

yolo_model = load_yolo_world()

# ---------------- Helper: Tie Colour ----------------

def get_dominant_color_name(frame, box):
    x1, y1, x2, y2 = map(int, box)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))

    if v < 50:
        return "Black"
    if s < 40:
        return "White/Grey"

    if h < 10 or h >= 170:
        return "Red"
    elif h < 25:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 100:
        return "Cyan"
    elif h < 130:
        return "Blue"
    elif h < 150:
        return "Purple"
    else:
        return "Pink"

# ---------------- Helper: Object Analysis ----------------

def analyze_objects(frame):
    results = yolo_model.predict(
        frame,
        conf=0.50,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    details = []

    for result in results:
        for box in result.boxes:
            label = yolo_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            box_coords = box.xyxy[0].tolist()

            if label == "tie":
                tie_colour = get_dominant_color_name(frame, box_coords)

                details.append(
                    f"👔 **Tie detected** — Colour: **{tie_colour}** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["eyeglasses", "spectacles"]:
                details.append(
                    f"👓 **Eyeglasses detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["pen", "ballpoint pen"]:
                details.append(
                    f"🖊️ **Pen detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["bag", "handbag", "backpack"]:
                details.append(
                    f"🎒 **Bag detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["coffee cup", "cup"]:
                details.append(
                    f"☕ **Cup detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["mobile phone", "cell phone"]:
                details.append(
                    f"📱 **Mobile phone detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label == "bottle":
                details.append(
                    f"🍶 **Bottle detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label == "person":
                details.append(
                    f"🧍 **Person detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            else:
                details.append(
                    f"- **{label.capitalize()}** — "
                    f"Confidence: **{confidence:.2f}**"
                )

    return annotated_rgb, details

# ---------------- Helper: Face Analysis ----------------

def analyze_face(frame):
    try:
        analysis = DeepFace.analyze(
            frame,
            actions=["emotion", "age"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        if analysis:
            face = analysis[0]

            age = int(face["age"])
            dominant = str(face["dominant_emotion"])
            emotions = face["emotion"]
            dominant_score = float(emotions[dominant])

            return age, dominant, dominant_score, emotions

    except Exception as e:
        st.error(f"Face not detected: {type(e).__name__} — {e}")

    return None, None, None, None

# ---------------- Helper: Show Face Results ----------------

def show_face_results(age, dominant, dominant_score, emotions):
    if age is None:
        st.info("Face not detected.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Estimated Age", age)

    with col2:
        st.metric(
            "Dominant Emotion",
            dominant.capitalize(),
            f"{dominant_score:.1f}%"
        )

    st.subheader("📊 Emotion Confidence")

    for emotion, score in emotions.items():
        score = float(score)
        progress_value = min(score / 100.0, 1.0)

        st.progress(progress_value)
        st.write(f"**{emotion.capitalize()}**: {score:.1f}%")

# ---------------- Mode Selection ----------------

mode = st.radio(
    "Choose mode:",
    ["🖼️ Upload Image", "📸 Camera Capture"]
)

# ---------------- 1. Upload Image ----------------

if mode == "🖼️ Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"The image did not open: {e}")
            st.stop()

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📦 Object Detection")

        with st.spinner("Objects are being detected..."):
            annotated_rgb, details = analyze_objects(frame)

        st.image(
            annotated_rgb,
            caption="Detected Objects",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📋 Detection Details")

        if details:
            for detail in details:
                st.write(detail)
        else:
            st.info("No object detected.")

        st.subheader("🙂 Face Analysis")

        with st.spinner("Face emotion and age are being analyzed..."):
            age, dominant, dominant_score, emotions = analyze_face(frame)

        show_face_results(age, dominant, dominant_score, emotions)

# ---------------- 2. Camera Capture ----------------

elif mode == "📸 Camera Capture":

    camera_photo = st.camera_input("📸 Take photo with camera")

    if camera_photo is not None:
        try:
            image = Image.open(camera_photo).convert("RGB")
        except Exception as e:
            st.error(f"The camera image did not open: {e}")
            st.stop()

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(
            image,
            caption="Captured Photo",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📦 Object Detection")

        with st.spinner("Objects are being detected..."):
            annotated_rgb, details = analyze_objects(frame)

        st.image(
            annotated_rgb,
            caption="Detected Objects",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📋 Detection Details")

        if details:
            for detail in details:
                st.write(detail)
        else:
            st.info("No object detected.")

        st.subheader("🙂 Face Analysis")

        with st.spinner("Face emotion and age are being analyzed..."):
            age, dominant, dominant_score, emotions = analyze_face(frame)

        show_face_results(age, dominant, dominant_score, emotions)

# ---------------- 3. Live Webcam ----------------

elif mode == "🎥 Live Webcam":

    st.warning(
        "Press start, Allow camera permission . "
        "Press Stop to turn off."
    )

    start = st.button("▶️ Start Live Webcam")
    stop = st.button("⏹️ Stop")

    # Placeholder for video
    frame_placeholder = st.empty()

    # Placeholder for detailed output below
    details_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        face_info = None
        latest_details = []

        while cap.isOpened() and not stop:
            ret, frame = cap.read()

            if not ret:
                st.error("Not getting a frame from the camera.")
                break

            frame_count += 1

            # ---------------- YOLO Object Detection ----------------

            results = yolo_model.predict(
                frame,
                conf=0.25,
                imgsz=480,
                verbose=False
            )

            latest_details = []

            for result in results:
                for box in result.boxes:
                    label = yolo_model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    box_coords = box.xyxy[0].tolist()

                    x1, y1, x2, y2 = map(int, box_coords)

                    # Person: green box
                    if label == "person":
                        color = (0, 255, 0)
                        text = f"Person: {confidence:.2f}"

                        latest_details.append(
                            f"🧍 **Person detected** — "
                            f"Confidence: **{confidence:.2f}**"
                        )

                    # Remaining objects: blue box
                    else:
                        color = (255, 100, 0)
                        text = label.capitalize()

                        if label == "tie":
                            tie_colour = get_dominant_color_name(
                                frame,
                                box_coords
                            )

                            latest_details.append(
                                f"👔 **Tie detected** — "
                                f"Colour: **{tie_colour}** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                            text = f"Tie: {tie_colour}"

                        elif label in ["eyeglasses", "spectacles"]:
                            latest_details.append(
                                f"👓 **Eyeglasses detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        elif label in ["pen", "ballpoint pen"]:
                            latest_details.append(
                                f"🖊️ **Pen detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        elif label in ["bag", "handbag", "backpack"]:
                            latest_details.append(
                                f"🎒 **Bag detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        elif label in ["coffee cup", "cup"]:
                            latest_details.append(
                                f"☕ **Cup detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        elif label in ["mobile phone", "cell phone"]:
                            latest_details.append(
                                f"📱 **Mobile phone detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        elif label == "bottle":
                            latest_details.append(
                                f"🍶 **Bottle detected** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                        else:
                            latest_details.append(
                                f"- **{label.capitalize()}** — "
                                f"Confidence: **{confidence:.2f}**"
                            )

                    # Draw a box on the video
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            # ---------------- DeepFace Face Analysis ----------------

            # Analyze the face on every 10th frame
            if frame_count % 10 == 0:
                try:
                    analysis = DeepFace.analyze(
                        frame,
                        actions=["emotion", "age"],
                        enforce_detection=False,
                        detector_backend="opencv",
                        silent=True
                    )

                    if analysis:
                        face_info = analysis[0]

                except Exception:
                    face_info = None

            # ---------------- Draw a box around the face in the video ----------------

            if face_info:
                region = face_info["region"]
                fx, fy = region["x"], region["y"]
                fw, fh = region["w"], region["h"]

                emotions = face_info["emotion"]
                dominant = face_info["dominant_emotion"]
                age = face_info["age"]

                # Green face box
                cv2.rectangle(
                    frame,
                    (fx, fy),
                    (fx + fw, fy + fh),
                    (0, 255, 0),
                    2
                )

                # Emotion on the face
                cv2.putText(
                    frame,
                    f"{dominant}: {emotions[dominant]:.1f}%",
                    (fx, fy - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Age
                cv2.putText(
                    frame,
                    f"Age: {age}",
                    (fx, fy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Top 3 emotions beneath the surface
                top_emotions = sorted(
                    emotions.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:3]

                y_pos = fy + fh + 25

                for emotion, score in top_emotions:
                    cv2.putText(
                        frame,
                        f"{emotion}: {score:.1f}%",
                        (fx, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                    y_pos += 22

            # ---------------- Show live video  ----------------

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                rgb_frame,
                channels="RGB",
                width="stretch"
            )

            # ---------------- Detailed output below ----------------

            with details_placeholder.container():

                st.subheader("📋 Live Detection Details")

                if latest_details:
                    for detail in latest_details:
                        st.write(detail)
                else:
                    st.info("No object has been detected yet.")

                st.subheader("🙂 Live Face Analysis")

                if face_info:
                    age = int(face_info["age"])
                    dominant = str(face_info["dominant_emotion"])
                    emotions = face_info["emotion"]
                    dominant_score = float(emotions[dominant])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Estimated Age", age)

                    with col2:
                        st.metric(
                            "Dominant Emotion",
                            dominant.capitalize(),
                            f"{dominant_score:.1f}%"
                        )

                    st.subheader("📊 Emotion Confidence")

                    for emotion, score in emotions.items():
                        score = float(score)
                        progress_value = min(score / 100.0, 1.0)

                        st.progress(progress_value)
                        st.write(
                            f"**{emotion.capitalize()}**: {score:.1f}%"
                        )

                else:
                    st.info(
                        "Face not detected. Face in front of camera "
                        "Place it in the center and improve the lighting."
                    )

        cap.release()
        cv2.destroyAllWindows()
