"""Photo face reading for SmartVision (Upload Image and Camera Capture).

Why this file exists
--------------------
Until now the two photo modes looked at a face only ONE time: one crop for the age
network and one pass for the emotion network. One look is noisy: a small change in
light, head angle or crop can flip the answer. The Live Webcam (face_boost.py)
already uses better habits. This file gives the same habits to photos:

  * Age: the face is cut out from the ORIGINAL picture (not the shrunk copy). The small
    ONNX age model (age_model.py) looks at 14 slightly different crops of it and the
    median age is used. If that model is not available, the old 8-group model looks at the
    face 10 times (centre + 4 corners, each also mirrored) and averages them.
    The ONNX model is weak with young children, so when it says "35 years or less" the old
    model is asked for a second opinion; if the old model says "child", the page shows both
    ("4-6 or 25-32") and says it is not sure.
  * Emotion: the face and its mirror image are both analysed and averaged. If the
    best emotion is below 40 % the page says "Not sure" instead of a confident
    wrong label.
  * Several faces: the BIGGEST real face is used (a tiny face in the background no
    longer decides the answer).

These are better habits, not magic: age and emotion stay estimates and the models
can still be wrong (bad light, glasses, a turned head, a blurry photo).

Nothing here needs Streamlit or DeepFace to be installed (DeepFace is only called
through face_boost.emotion_vector). Nothing is written to disk.
"""

import cv2
import numpy as np

from age_model import describe_years
from face_boost import (
    AGE_RANGES,
    EMOTION_LABELS,
    EMOTIONS,
    age_probabilities,
    crop_face,
    describe_age,
    describe_emotion,
    emotion_vector,
)

# Same rule as before: a face finder that is less than 50 % sure means "no face".
MIN_FACE_CONFIDENCE = 0.5

AGE_MARGIN = 0.25      # extra picture around the face for the age network
EMOTION_MARGIN = 0.08  # emotion likes a tighter crop

# Young children are the weak spot of the small ONNX age model: it often says "about 28"
# for a 4-year-old. The older 8-group model is better with children (its groups start at
# 0-2, 4-6, 8-12), so it is asked for a SECOND OPINION, but only when the ONNX model says
# the person is young (35 years or less). An elderly face is never touched by this.
CHILD_CHECK_MAX_YEARS = 35.0
CHILD_CHECK_MASS = 0.50   # the old model must put at least this much on the 3 child groups
CHILD_GROUPS = 3          # AGE_RANGES[:3] = 0-2, 4-6, 8-12


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


def child_second_opinion(age, original_bgr, box, net):
    """If the ONNX age is young (<= 35) and the OLD model thinks "child", say so.

    Changes `age` in place and returns True when it did. Never raises: a missing or broken
    old model simply means "no second opinion".
    """
    if age is None or net is None or age.get("source") != "onnx":
        return False
    years = age.get("years")
    if years is None or years > CHILD_CHECK_MAX_YEARS:
        return False
    if years_is_child_group(age):
        return False
    try:
        old_net = net() if callable(net) else net
        crop = crop_face(original_bgr, box, margin=AGE_MARGIN) if old_net is not None else None
        probs = age_probabilities(old_net, crop, fast=False) if crop is not None else None
    except Exception:
        return False
    if probs is None or len(probs) < CHILD_GROUPS:
        return False
    child_mass = float(np.sum(probs[:CHILD_GROUPS]))
    if child_mass < CHILD_CHECK_MASS:
        return False
    child_group = AGE_RANGES[int(np.argmax(probs[:CHILD_GROUPS]))]
    main_group = age["text"].split(" or ")[0]
    age["text"] = f"{child_group} or {main_group}"
    age["sure"] = False
    age["child_check"] = round(child_mass, 2)
    age["source"] = "onnx+caffe"
    age["note"] = (
        "The main age model is weak with young children, and the older model thinks this face "
        "may be a child. So this age is not sure."
    )
    return True


def years_is_child_group(age):
    """True if the ONNX answer already starts with one of the 3 child groups."""
    first = (age.get("text") or "").split(" or ")[0]
    return first in AGE_RANGES[:CHILD_GROUPS]


def read_face(original_bgr, face, scale=1.0, net=None, emotion_fn=None, age_fn=None):
    """Read age and emotion of one face.

    original_bgr : the full-size picture (BGR)
    face         : one DeepFace answer (has 'region' and 'emotion')
    scale        : shrunk / original, if the face was found on a shrunk copy
    net          : the OLD age network, or a function that loads it. It is only used
                   (and only loaded) when age_fn gives no answer. None = no backup.
    emotion_fn   : replaceable for tests; default is face_boost.emotion_vector
    age_fn       : the small ONNX age model: age_fn(picture, box) -> {"years", ...} or None

    Returns a dict, or None when the box is empty.
    """
    region = _region(face)
    box = region_to_box(region, scale, original_bgr.shape) if region else None
    if box is None:
        return None

    # ---- age: the small ONNX model first (14 crops, median) ...
    age = None
    seen = None            # the face picture the age model looked at (shown on the page)
    if age_fn is not None:
        try:
            found = age_fn(original_bgr, box)
        except Exception:
            found = None
        if found is not None:
            age = describe_years(found["years"], found.get("low"), found.get("high"))
            seen = found.get("face")
            child_second_opinion(age, original_bgr, box, net)

    # ... and the old 8-group model (10 looks, averaged) as the backup
    if age is None and net is not None:
        try:
            old_net = net() if callable(net) else net
        except Exception:
            old_net = None
        crop = crop_face(original_bgr, box, margin=AGE_MARGIN) if old_net is not None else None
        if crop is not None:
            try:
                probs = age_probabilities(old_net, crop, fast=False)
                if probs is not None:
                    age = describe_age(probs)
                    age["source"] = "caffe"
                    seen = crop
            except Exception:
                age = None

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

    if seen is None:
        seen = crop_face(original_bgr, box, margin=EMOTION_MARGIN)

    return {
        "box": box,
        "face_rgb": _thumbnail(seen),
        "age": age,
        "age_text": age["text"] if age else None,
        "emotion": emotion,
        "emotion_text": emotion["text"] if emotion else None,
        "emotion_scores": scores,
    }