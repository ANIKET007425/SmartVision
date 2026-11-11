"""Photo face reading for SmartVision (Upload Image and Camera Capture).

Why this file exists
--------------------
The two photo modes used to look at a face only ONE time. One look is noisy: a small
change in light, head angle or crop can flip the answer. The Live Webcam
(face_boost.py) already uses better habits. This file gives the same habits to photos:

  * Emotion: the face and its mirror image are both analysed and averaged. If the
    best emotion is below 40 % the page says "Not sure" instead of a confident
    wrong label.
  * Several faces: the BIGGEST real face is used (a tiny face in the background no
    longer decides the answer).
  * The face is cut out from the ORIGINAL picture (not the shrunk copy), so it is sharper.

(The estimated-age feature was removed from SmartVision: age models are guesses that
are often wrong, for example for children and for filtered photos.)

These are better habits, not magic: the emotion stays an estimate and the model can
still be wrong (bad light, glasses, a turned head, a blurry photo).

Nothing here needs Streamlit or DeepFace to be installed (DeepFace is only called
through face_boost.emotion_vector). Nothing is written to disk.
"""

import cv2
import numpy as np

from face_boost import (
    EMOTION_LABELS,
    EMOTIONS,
    crop_face,
    describe_emotion,
    emotion_vector,
)

# A face finder that is less than 50 % sure means "no face".
MIN_FACE_CONFIDENCE = 0.5

EMOTION_MARGIN = 0.08  # emotion likes a tight crop


def _region(face):
    """(x, y, w, h) of a DeepFace result, or None if it is not usable."""
    region = face.get("region") if isinstance(face, dict) else None
    if not isinstance(region, dict):
        return None
    try:
        x, y = float(region["x"]), float(region["y"])
        w, h = float(region["w"]), float(region["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def choose_face(faces):
    """From the DeepFace answers pick the biggest REAL face.

    Returns (face, count): `count` is how many real faces were found.
    (None, 0) means "Face not detected".
    """
    real = []
    for face in faces or []:
        if not isinstance(face, dict):
            continue
        if float(face.get("face_confidence", 1)) < MIN_FACE_CONFIDENCE:
            continue
        region = _region(face)
        if region is None:
            continue
        real.append((region[2] * region[3], face))
    if not real:
        return None, 0
    real.sort(key=lambda item: item[0], reverse=True)
    return real[0][1], len(real)


def region_to_box(region, scale, shape):
    """Turn a face region found on the SHRUNK picture into a box (x1, y1, x2, y2)
    on the ORIGINAL picture. `scale` is shrunk / original (1.0 = not shrunk).
    Returns None if the box is empty."""
    h, w = shape[:2]
    scale = scale if scale and scale > 0 else 1.0
    x1 = int(round(region[0] / scale))
    y1 = int(round(region[1] / scale))
    x2 = int(round((region[0] + region[2]) / scale))
    y2 = int(round((region[1] + region[3]) / scale))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def _vector_from_percentages(scores):
    """DeepFace's own answer ({'happy': 87.1, ...} in percent) as 7 probabilities."""
    try:
        vec = np.array([float(scores.get(name, 0.0)) for name in EMOTIONS], dtype=np.float64)
    except Exception:
        return None
    total = vec.sum()
    if not np.all(np.isfinite(vec)) or total <= 0:
        return None
    return vec / total


def _thumbnail(face_bgr, size=192):
    """The face the model looked at, as a small RGB picture for the page."""
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return None
    h, w = face_bgr.shape[:2]
    k = size / float(max(h, w))
    resized = cv2.resize(face_bgr, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def read_face(original_bgr, face, scale=1.0, emotion_fn=None):
    """Read the emotion of one face.

    original_bgr : the full-size picture (BGR)
    face         : one DeepFace answer (has 'region' and 'emotion')
    scale        : shrunk / original, if the face was found on a shrunk copy
    emotion_fn   : replaceable for tests; default is face_boost.emotion_vector

    Returns a dict (box, face_rgb, emotion, emotion_text, emotion_scores), or None
    when the box is empty.
    """
    region = _region(face)
    box = region_to_box(region, scale, original_bgr.shape) if region else None
    if box is None:
        return None

    # ---- emotion: face + mirror image; DeepFace's first answer is the fallback
    emotion_vec = None
    crop = crop_face(original_bgr, box, margin=EMOTION_MARGIN)
    if crop is not None:
        try:
            emotion_vec = (emotion_fn or emotion_vector)(crop)
        except Exception:
            emotion_vec = None
    if emotion_vec is None:
        emotion_vec = _vector_from_percentages(face.get("emotion") or {})

    emotion = describe_emotion(emotion_vec) if emotion_vec is not None else None
    scores = []
    if emotion_vec is not None:
        scores = [(EMOTION_LABELS[name], round(float(p) * 100.0, 1)) for name, p in zip(EMOTIONS, emotion_vec)]
        scores.sort(key=lambda item: item[1], reverse=True)

    return {
        "box": box,
        "face_rgb": _thumbnail(crop),
        "emotion": emotion,
        "emotion_text": emotion["text"] if emotion else None,
        "emotion_scores": scores,
    }