"""Age from a face with a small ONNX model (InsightFace "genderage", 1.3 MB).

Why this file exists
--------------------
SmartVision used the OpenCV "Levi-Hassner" age model (2015). It sorts a face into 8
age groups, it is known to lean towards "25-32", and it is easily fooled. Here we use a
newer small model that gives a NUMBER (an age in years). On a published test with
1000 photos (arXiv 2511.14689) the InsightFace model had a typical error of about
7.5 years, and DeepFace's age model about 11 years. It is still an ESTIMATE: it can be
wrong, mostly with strange light, a turned head or a very blurry face.

How it works (same recipe as the InsightFace library, written with OpenCV + NumPy)
  1. The face box is made 1.5 times bigger, so hair, forehead and chin are included.
  2. That area is cut out and resized to 96 x 96 pixels (this is the picture the model
     really sees; it is also returned, so the page can show it).
  3. The model returns three numbers. The first two are a gender guess: SmartVision does
     NOT use or show them. The third one x 100 is the age in years.
  4. One crop is a bit lucky or unlucky, so 7 slightly different crops of the same face
     (a little bigger, smaller, moved left/right/up/down) are tried, each also as a mirror
     image. The MEDIAN of these 14 ages is the answer. This makes the result steadier.

If onnxruntime or the model file (models/genderage.onnx) is missing, the functions
return None and the app falls back to the old age model. Nothing is saved anywhere.

Licence note: InsightFace's pre-trained models are for non-commercial research use,
which fits a university project. Mention it in the report.
"""

import os
import threading

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "genderage.onnx")
INPUT_SIZE = 96
FACE_SCALE = 1.5          # the face box is made 1.5 x bigger before cutting
# (scale, shift left/right, shift up/down) - the shifts are a share of the face size
CROP_VARIANTS = [
    (1.5, 0.0, 0.0),
    (1.3, 0.0, 0.0), (1.7, 0.0, 0.0),
    (1.5, 0.06, 0.0), (1.5, -0.06, 0.0),
    (1.5, 0.0, 0.06), (1.5, 0.0, -0.06),
]
MIN_AGE, MAX_AGE = 1.0, 100.0

# The same 8 groups the app always showed (the report and slides talk about them).
AGE_GROUPS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
NEAR_YEARS = 4.0          # an age this close to a second group shows both groups
UNSTEADY_YEARS = 12.0     # if the middle half of the 14 tries spans more than this, say "not sure"

_lock = threading.Lock()
_session = None
_tried = False


def get_session():
    """Load the ONNX model once. Returns None if it cannot be loaded."""
    global _session, _tried
    with _lock:
        if _tried:
            return _session
        _tried = True
        try:
            if not os.path.exists(MODEL_PATH):
                return None
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.log_severity_level = 3
            _session = ort.InferenceSession(MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as e:  # never crash the app because of this model
            print("Age model (ONNX) could not be loaded:", type(e).__name__, e)
            _session = None
        return _session


def prepare_face(img_bgr, box, face_scale=None, shift_x=0.0, shift_y=0.0):
    """Cut the face (box made 1.5 x bigger) out of the picture as a 96 x 96 BGR image."""
    face_scale = face_scale or FACE_SCALE
    x1, y1, x2, y2 = [float(v) for v in box]
    side = max(x2 - x1, y2 - y1)
    if side < 8:
        return None
    scale = INPUT_SIZE / (side * face_scale)
    cx = (x1 + x2) / 2.0 + shift_x * side
    cy = (y1 + y2) / 2.0 + shift_y * side
    matrix = np.array(
        [[scale, 0.0, INPUT_SIZE / 2.0 - cx * scale],
         [0.0, scale, INPUT_SIZE / 2.0 - cy * scale]],
        dtype=np.float32,
    )
    return cv2.warpAffine(img_bgr, matrix, (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)


def _run(session, faces_bgr):
    """Ages (years) for a list of 96 x 96 BGR faces."""
    # The model wants RGB, values 0..255, shape (N, 3, 96, 96) (it normalises inside).
    blob = cv2.dnn.blobFromImages(faces_bgr, 1.0, (INPUT_SIZE, INPUT_SIZE), (0.0, 0.0, 0.0), swapRB=True)
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    return [float(row[2]) * 100.0 for row in np.asarray(out).reshape(len(faces_bgr), -1)]


def predict_age(img_bgr, box, session=None):
    """Age in years for the face in `box` = (x1, y1, x2, y2), or None.

    Returns {"years": median age, "low": ..., "high": ..., "face": 96x96 BGR picture
    the model saw (the main crop)}. `low`/`high` are the 25 % / 75 % points of the 14
    tries: a wide gap means the model is unsteady about this face.
    """
    session = session or get_session()
    if session is None:
        return None
    face = prepare_face(img_bgr, box)
    if face is None:
        return None
    crops = []
    for face_scale, shift_x, shift_y in CROP_VARIANTS:
        crop = prepare_face(img_bgr, box, face_scale, shift_x, shift_y)
        if crop is not None:
            crops.append(crop)
            crops.append(cv2.flip(crop, 1))                  # and its mirror image
    try:
        ages = _run(session, crops)
    except Exception as e:
        print("Age model (ONNX) failed:", type(e).__name__, e)
        return None
    ages = np.array([a for a in ages if np.isfinite(a)], dtype=np.float64)
    if ages.size == 0:
        return None
    ages = np.clip(ages, MIN_AGE, MAX_AGE)
    return {
        "years": float(np.median(ages)),
        "low": float(np.percentile(ages, 25)),
        "high": float(np.percentile(ages, 75)),
        "face": face,
    }


def years_to_text(years):
    """Years -> one of the 8 groups; two groups ("25-32 or 38-43") when the age sits
    between them. Returns (text, sure)."""
    def label(group):
        return f"{group[0]}-{group[1]}"

    def distance(group):
        lo, hi = group
        if lo <= years <= hi:
            return 0.0
        return min(abs(years - lo), abs(years - hi))

    ranked = sorted(AGE_GROUPS, key=lambda g: (distance(g), g[0]))
    best = ranked[0]
    second = ranked[1]
    if distance(best) == 0.0 and distance(second) > NEAR_YEARS:
        return label(best), True
    if distance(second) <= NEAR_YEARS:
        pair = sorted([best, second])
        return f"{label(pair[0])} or {label(pair[1])}", False
    return label(best), False


def describe_years(years, low=None, high=None):
    """The dictionary the page uses (same keys as face_boost.describe_age, plus years)."""
    text, sure = years_to_text(years)
    note = None if sure else "This age is close to the border between two age groups."
    # a wide spread between the 14 tries -> the model itself is unsure about this face
    if low is not None and high is not None and (high - low) > UNSTEADY_YEARS:
        sure = False
        note = "The model is not steady about this face (its tries differ a lot)."
    return {"text": text, "sure": sure, "years": years, "low": low, "high": high, "note": note, "source": "onnx"}