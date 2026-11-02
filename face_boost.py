"""Face helpers for SmartVision: find faces, estimate age range and emotion,
and keep the answers steady.

Why this file exists
--------------------
Age and emotion from ONE camera frame are noisy. A small AI model can be right
on one frame and wrong on the next (light, head angle, glasses, blur). Three cheap
tricks make the answers better without a new model:

  1. Many looks at one face (age): the Levi-Hassner age network was trained on
     256x256 faces with random 227x227 crops. We do the same at test time
     (centre + corners + mirror image) and AVERAGE the answers.
  2. Mirror image (emotion): the emotion network sees the face and its mirror
     image, and the two answers are averaged.
  3. Averaging over time (live video): the numbers are smoothed over the last
     few seconds, so one bad frame cannot flip the label.

It also says "Not sure" when the model is not confident, instead of showing a
wrong label with a straight face.

Nothing here needs Streamlit. DeepFace is imported only when an emotion is
really needed. Nothing is written to disk and nothing is sent anywhere.
"""

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# The eight output classes of the Levi-Hassner network, in output order.
AGE_RANGES = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

# DeepFace's emotion names.
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_LABELS = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
    "sad": "Sad", "surprise": "Surprise", "neutral": "Neutral",
}

# Mean pixel values (BGR) the age network was trained with.
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

# Below this top probability the label says "Not sure".
EMOTION_SURE = 0.40
AGE_SURE = 0.50
# A second age range is mentioned when it has at least this probability.
AGE_ALSO = 0.25

_LOAD_LOCK = threading.Lock()
_AGE_LOCK = threading.Lock()        # cv2.dnn networks are not thread-safe
_EMOTION_LOCK = threading.Lock()    # one DeepFace emotion call at a time
_AGE_NET = {"net": None, "tried": False}


# ======================================================================
# Age network (OpenCV DNN, Caffe)
# ======================================================================

def _model_folders():
    here = Path(__file__).resolve().parent
    return [here / "models", Path.cwd() / "models"]


def get_age_net(reload=False):
    """Load the age network once for the whole process. Returns None if the
    two model files are missing or cannot be read (the app then shows N/A)."""
    with _LOAD_LOCK:
        if reload:
            _AGE_NET.update(net=None, tried=False)
        if _AGE_NET["net"] is not None or _AGE_NET["tried"]:
            return _AGE_NET["net"]
        _AGE_NET["tried"] = True
        for folder in _model_folders():
            proto = folder / "age_deploy.prototxt"
            weights = folder / "age_net.caffemodel"
            if proto.exists() and weights.exists():
                try:
                    _AGE_NET["net"] = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
                    break
                except Exception:
                    _AGE_NET["net"] = None
        return _AGE_NET["net"]


def pad_box(box, frame_shape, margin=0.25):
    """Grow a box (x1, y1, x2, y2) by `margin` on every side, kept inside the picture."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    return max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy)


def crop_face(frame_bgr, box, margin=0.25):
    """Cut a face (with some margin) out of a picture. Returns None if empty."""
    x1, y1, x2, y2 = pad_box(box, frame_bgr.shape, margin)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame_bgr[y1:y2, x1:x2]


def age_crops(face_bgr, fast=False):
    """The 227x227 looks at one face.

    fast=True  -> centre + its mirror image         (2 looks, for live video)
    fast=False -> centre + 4 corners, all mirrored   (10 looks, for photos)
    """
    face = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    offsets = [(14, 14)] if fast else [(14, 14), (0, 0), (29, 0), (0, 29), (29, 29)]
    crops = [face[y:y + 227, x:x + 227] for x, y in offsets]
    crops += [cv2.flip(c, 1) for c in crops]
    return crops


def age_probabilities(net, face_bgr, fast=False):
    """Average probability of each of the eight age ranges (numpy array, sums to 1).
    Returns None if the network gives nothing usable."""
    crops = age_crops(face_bgr, fast=fast)
    blob = cv2.dnn.blobFromImages(crops, 1.0, (227, 227), AGE_MEAN, swapRB=False)
    with _AGE_LOCK:
        net.setInput(blob)
        out = net.forward()
    out = np.asarray(out, dtype=np.float64).reshape(len(crops), -1)[:, :len(AGE_RANGES)]
    if out.shape[1] != len(AGE_RANGES) or not np.all(np.isfinite(out)):
        return None
    probs = out.mean(axis=0)
    total = probs.sum()
    return probs / total if total > 0 else None


def describe_age(probs):
    """Turn eight probabilities into a small, honest summary."""
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    confidence = float(probs[top])
    also = AGE_RANGES[second] if float(probs[second]) >= AGE_ALSO else None
    text = AGE_RANGES[top] + (f" or {also}" if also else "")
    return {
        "range": AGE_RANGES[top],
        "also": also,
        "confidence": confidence,
        "sure": confidence >= AGE_SURE,
        "text": text,
    }


# ======================================================================
# Emotion (DeepFace, on a face that is already cut out)
# ======================================================================

def emotion_vector(face_bgr):
    """Seven emotion probabilities (order of EMOTIONS) for one cut-out face.
    The face and its mirror image are both analysed and averaged.
    Raises an error if DeepFace cannot do it (the caller catches it)."""
    from deepface import DeepFace

    vectors = []
    for img in (face_bgr, cv2.flip(face_bgr, 1)):
        with _EMOTION_LOCK:
            result = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                detector_backend="skip",      # the face is already cut out
                enforce_detection=False,
                align=False,
                silent=True,
            )
        item = result[0] if isinstance(result, list) else result
        scores = item["emotion"]
        vectors.append([float(scores.get(name, 0.0)) for name in EMOTIONS])
    vec = np.mean(vectors, axis=0)
    total = vec.sum()
    return vec / total if total > 0 else None


def describe_emotion(vec):
    vec = np.asarray(vec, dtype=np.float64)
    order = np.argsort(vec)[::-1]
    top, second = int(order[0]), int(order[1])
    confidence = float(vec[top])
    name = EMOTION_LABELS[EMOTIONS[top]]
    maybe = EMOTION_LABELS[EMOTIONS[second]]
    sure = confidence >= EMOTION_SURE
    percent = int(round(confidence * 100))
    text = f"{name} {percent}%" if sure else f"Not sure (maybe {name})"
    return {
        "name": name, "second": maybe, "confidence": confidence,
        "sure": sure, "text": text,
    }


# ======================================================================
# Finding faces (OpenCV cascade; no download needed)
# ======================================================================

class FaceFinder:
    """Finds frontal faces with an OpenCV cascade file.

    The cascade files ship inside the OpenCV package (cv2.data), so nothing has
    to be downloaded. It is light (about 10 ms), but it only finds faces that look
    roughly at the camera - that is fine for a webcam.
    """

    CASCADES = ("haarcascade_frontalface_alt2.xml", "haarcascade_frontalface_default.xml")

    def __init__(self, cascade_path=None):
        self.cascade_path = cascade_path
        self._cascade = None
        self._tried = False
        self.error = None

    def _load(self):
        if self._tried:
            return
        self._tried = True
        paths = []
        if self.cascade_path:
            paths.append(str(self.cascade_path))
        else:
            try:
                base = cv2.data.haarcascades
                paths += [base + name for name in self.CASCADES]
            except Exception as e:
                self.error = f"no cascade folder: {e}"
        for path in paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    self._cascade = cascade
                    return
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        if self.error is None:
            self.error = "face cascade file not found"

    def available(self):
        self._load()
        return self._cascade is not None

    def find(self, frame_bgr, max_faces=2, min_size=36):
        """Return up to `max_faces` boxes (x1, y1, x2, y2), biggest first."""
        self._load()
        if self._cascade is None:
            return []
        h, w = frame_bgr.shape[:2]
        scale = 480.0 / max(h, w) if max(h, w) > 480 else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size),
        )
        boxes = []
        for x, y, bw, bh in (rects if len(rects) else []):
            boxes.append((
                int(x / scale), int(y / scale),
                int((x + bw) / scale), int((y + bh) / scale),
            ))
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes[:max_faces]


def inside_any(box, outer_boxes, tolerance=0.1):
    """True if the CENTRE of `box` is inside any of the outer boxes (slightly grown)."""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    for x1, y1, x2, y2 in outer_boxes:
        dx, dy = (x2 - x1) * tolerance, (y2 - y1) * tolerance
        if x1 - dx <= cx <= x2 + dx and y1 - dy <= cy <= y2 + dy:
            return True
    return False


# ======================================================================
# Keeping the answers steady
# ======================================================================

class _Track:
    def __init__(self, track_id, box, now):
        self.id = track_id
        self.set_box(box)
        self.seen = now
        self.emotion = None     # smoothed probabilities
        self.age = None
        self.updates = 0

    def set_box(self, box):
        self.cx = (box[0] + box[2]) / 2.0
        self.cy = (box[1] + box[3]) / 2.0
        self.size = float(max(box[2] - box[0], box[3] - box[1]))


class FaceTracker:
    """Remembers faces between analyses and smooths their answers.

    A new box is matched to the nearest remembered face. The answers are an
    exponential moving average: new = a * latest + (1 - a) * old. With a = 0.35
    one odd frame cannot change the label, but a new expression that stays for
    two or three analyses does.
    """

    def __init__(self, alpha_emotion=0.35, alpha_age=0.3, forget_seconds=4.0):
        self.alpha_emotion = alpha_emotion
        self.alpha_age = alpha_age
        self.forget_seconds = forget_seconds
        self.tracks = []
        self._next_id = 1

    def assign(self, box, now):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        size = float(max(box[2] - box[0], box[3] - box[1]))
        best, best_distance = None, None
        for track in self.tracks:
            distance = math.hypot(track.cx - cx, track.cy - cy)
            if distance <= 0.8 * max(size, track.size) and (best is None or distance < best_distance):
                best, best_distance = track, distance
        if best is None:
            best = _Track(self._next_id, box, now)
            self._next_id += 1
            self.tracks.append(best)
        else:
            best.set_box(box)
        best.seen = now
        return best

    @staticmethod
    def _blend(old, new, alpha):
        if new is None:
            return old
        new = np.asarray(new, dtype=np.float64)
        return new if old is None else alpha * new + (1.0 - alpha) * old

    def update(self, track, emotion_vec, age_vec):
        track.emotion = self._blend(track.emotion, emotion_vec, self.alpha_emotion)
        track.age = self._blend(track.age, age_vec, self.alpha_age)
        track.updates += 1

    def forget(self, now):
        self.tracks = [t for t in self.tracks if now - t.seen <= self.forget_seconds]


# ======================================================================
# Everything together (this is what the live page uses)
# ======================================================================

class FacePipeline:
    """find faces -> emotion + age for each -> smooth over time -> short text.

    finder, age_net_fn and emotion_fn can be replaced (used by the tests).
    """

    def __init__(self, finder=None, age_net_fn=None, emotion_fn=None,
                 want_age=True, want_emotion=True, max_faces=2):
        self.finder = finder or FaceFinder()
        self.age_net_fn = age_net_fn or get_age_net
        self.emotion_fn = emotion_fn or emotion_vector
        self.want_age = want_age
        self.want_emotion = want_emotion
        self.max_faces = max_faces
        self.tracker = FaceTracker()
        self.status = ""        # short note for the page, e.g. "no face finder"
        self.error = None       # last problem (shown only to the owner)

    def reset(self):
        self.tracker = FaceTracker()

    def analyze(self, frame_bgr, person_boxes=None, now=None):
        """Returns a list of dicts: id, box, lines (texts to show), emotion, age."""
        now = time.monotonic() if now is None else now
        self.status = ""

        if not self.finder.available():
            self.status = "Face finder is not available on this server."
            self.error = self.finder.error
            return []

        boxes = self.finder.find(frame_bgr, max_faces=self.max_faces + 2)
        if person_boxes:
            boxes = [b for b in boxes if inside_any(b, person_boxes)]
        boxes = boxes[:self.max_faces]

        net = None
        if self.want_age:
            try:
                net = self.age_net_fn()
            except Exception as e:
                self.error = f"age: {type(e).__name__}: {e}"
        if self.want_age and net is None:
            self.status = "Age model files not found (age is skipped)."

        results = []
        for box in boxes:
            track = self.tracker.assign(box, now)
            emotion_vec = age_vec = None

            if self.want_emotion:
                face = crop_face(frame_bgr, box, margin=0.08)
                if face is not None:
                    try:
                        emotion_vec = self.emotion_fn(face)
                    except Exception as e:
                        self.error = f"emotion: {type(e).__name__}: {e}"
                        self.status = "Emotion could not be read (age still works)."

            if net is not None:
                face = crop_face(frame_bgr, box, margin=0.25)
                if face is not None:
                    try:
                        age_vec = age_probabilities(net, face, fast=True)
                    except Exception as e:
                        self.error = f"age: {type(e).__name__}: {e}"

            self.tracker.update(track, emotion_vec, age_vec)

            lines = []
            emotion = describe_emotion(track.emotion) if track.emotion is not None else None
            age = describe_age(track.age) if track.age is not None else None
            if emotion:
                lines.append(emotion["text"])
            if age:
                lines.append(f"Age {age['text']}")
            results.append({"id": track.id, "box": box, "lines": lines, "emotion": emotion, "age": age})

        self.tracker.forget(now)
        return results