"""Automated checks for the photo upgrade of SmartVision (age_model.py, photo_faces.py,
the new analyze_face / show_face_results in app.py, and the dashboard hook).

Run from the project folder (the one that holds app.py, age_model.py, photo_faces.py,
face_boost.py and dashboard_page.py):

    python test_photo_faces.py

No camera, GPU, DeepFace, Streamlit server or internet is needed. A fake age
network, a fake DeepFace and a fake Streamlit are used, so only OUR code is tested
(not the quality of the AI models).
"""
import ast
import os
import sys
import types

import cv2
import numpy as np

SRC = os.getcwd()
sys.path.insert(0, SRC)

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name + (f"  ({detail})" if detail and not condition else ""))


# ------------------------------------------------------------------ fakes
class _State(dict):
    pass


class FakeSt(types.ModuleType):
    """Records what the page would draw."""

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.events = []

    def _log(self, kind, *args):
        self.events.append((kind,) + args)

    def metric(self, label, value, delta=None):
        self._log("metric", label, value, delta)

    def caption(self, text):
        self._log("caption", text)

    def info(self, text):
        self._log("info", text)

    def error(self, text):
        self._log("error", text)

    def warning(self, text):
        self._log("warning", text)

    def subheader(self, text):
        self._log("subheader", text)

    def progress(self, value):
        self._log("progress", value)

    def write(self, text):
        self._log("write", text)

    def image(self, img, width=None, **kw):
        self._log("image", getattr(img, "shape", None), width)

    def expander(self, label, **kw):
        self._log("expander", label)

        class Box:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False
        return Box()

    def columns(self, n):
        class Col:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False
        return [Col() for _ in range(n)]

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]


st = FakeSt()
sys.modules["streamlit"] = st

import age_model as am          # noqa: E402
import face_boost as fb          # noqa: E402
import photo_faces as pf         # noqa: E402
import dashboard_page as dash    # noqa: E402


class FakeNet:
    """Gives the same eight probabilities for every look and records the blob."""

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float64)
        self.blob = None

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return np.tile(self.vec, (self.blob.shape[0], 1))


def age_vec(index, strength, second=None, second_strength=0.0):
    v = np.full(8, 0.0)
    v[index] = strength
    if second is not None:
        v[second] = second_strength
    rest = 1.0 - v.sum()
    others = [i for i in range(8) if v[i] == 0.0]
    for i in others:
        v[i] = rest / len(others)
    return v


def emo_vec(name, strength):
    v = np.full(7, (1 - strength) / 6)
    v[fb.EMOTIONS.index(name)] = strength
    return v


def face_dict(x, y, w, h, conf=0.99, emotion=None):
    d = {"region": {"x": x, "y": y, "w": w, "h": h}, "face_confidence": conf}
    if emotion is not None:
        d["emotion"] = emotion
    return d


PICTURE = np.full((600, 800, 3), 120, np.uint8)

# ================================================================== 1. choose_face
check("choose_face: nothing -> (None, 0)", pf.choose_face([]) == (None, 0) and pf.choose_face(None) == (None, 0))
small, big = face_dict(10, 10, 40, 40), face_dict(100, 100, 120, 130)
face, count = pf.choose_face([small, big])
check("choose_face: the BIGGEST real face wins", face is big and count == 2)
weak = face_dict(0, 0, 500, 500, conf=0.3)
face, count = pf.choose_face([weak, small])
check("choose_face: a face below 50 % is ignored, even if huge", face is small and count == 1)
check("choose_face: only weak faces -> (None, 0)", pf.choose_face([weak]) == (None, 0))
no_conf = {"region": {"x": 1, "y": 1, "w": 50, "h": 50}}
check("choose_face: missing face_confidence counts as sure (as before)", pf.choose_face([no_conf])[0] is no_conf)
bad = [{"region": {"x": 1}}, {"region": None}, "text", {"region": {"x": 0, "y": 0, "w": 0, "h": 9}}]
check("choose_face: broken answers are skipped without a crash", pf.choose_face(bad) == (None, 0))

# ================================================================== 2. region_to_box
check("region_to_box: scale 1.0 keeps the numbers", pf.region_to_box((100, 50, 80, 90), 1.0, PICTURE.shape) == (100, 50, 180, 140))
check("region_to_box: shrunk copy (scale 0.5) maps back to the original (x 2)",
      pf.region_to_box((100, 50, 80, 90), 0.5, PICTURE.shape) == (200, 100, 360, 280))
check("region_to_box: box is clipped to the picture", pf.region_to_box((700, 500, 400, 400), 1.0, PICTURE.shape) == (700, 500, 800, 600))
check("region_to_box: empty box -> None", pf.region_to_box((10, 10, 3, 3), 1.0, PICTURE.shape) is None)
check("region_to_box: scale 0 or None is treated as 1.0", pf.region_to_box((100, 50, 80, 90), 0, PICTURE.shape) == (100, 50, 180, 140))

# ================================================================== 3. read_face
seen_crops = []


def fake_emotion(crop):
    seen_crops.append(crop.shape)
    return emo_vec("happy", 0.80)


net = FakeNet(age_vec(4, 0.80))                 # 25-32
f = face_dict(300, 200, 100, 120, emotion={"happy": 10.0, "sad": 90.0})
res = pf.read_face(PICTURE, f, scale=1.0, net=net, emotion_fn=fake_emotion)
check("read_face: age network gets 10 looks (centre + 4 corners, mirrored)", net.blob is not None and net.blob.shape[0] == 10, str(None if net.blob is None else net.blob.shape))
check("read_face: every look is 227 x 227", net.blob.shape[2:] == (227, 227))
check("read_face: age text is the top range", res["age_text"] == "25-32" and res["age"]["sure"])
check("read_face: emotion comes from emotion_fn, not from DeepFace's first answer", res["emotion"]["name"] == "Happy" and res["emotion"]["sure"])
check("read_face: emotion text is 'Happy 80%'", res["emotion_text"] == "Happy 80%", res["emotion_text"])
scores = res["emotion_scores"]
check("read_face: 7 emotion scores, biggest first, adding up to about 100",
      len(scores) == 7 and scores[0][0] == "Happy" and all(scores[i][1] >= scores[i + 1][1] for i in range(6)) and abs(sum(s for _, s in scores) - 100) < 0.5,
      str(scores))
check("read_face: the face box is returned", res["box"] == (300, 200, 400, 320))
check("read_face: emotion crop is the tight crop (8 % margin) of the face",
      seen_crops and seen_crops[-1] == (int(120 + 2 * int(120 * 0.08)), int(100 + 2 * int(100 * 0.08)), 3), str(seen_crops))

# a face that is found on a shrunk copy is cut from the ORIGINAL picture
seen_crops.clear()
res = pf.read_face(PICTURE, face_dict(150, 100, 50, 60), scale=0.5, net=net, emotion_fn=fake_emotion)
check("read_face: scale 0.5 -> the crop is cut from the original (twice as big)", seen_crops and seen_crops[-1][0] > 100, str(seen_crops))
check("read_face: box is in original coordinates", res["box"] == (300, 200, 400, 320), str(res["box"]))

# second range and "not sure"
res = pf.read_face(PICTURE, f, net=FakeNet(age_vec(4, 0.45, second=5, second_strength=0.35)), emotion_fn=fake_emotion)
check("read_face: two close ages are both shown", res["age_text"] == "25-32 or 38-43" and not res["age"]["sure"], res["age_text"])
res = pf.read_face(PICTURE, f, net=net, emotion_fn=lambda c: emo_vec("neutral", 0.30))
check("read_face: weak emotion -> 'Not sure (maybe Neutral)'", res["emotion_text"].startswith("Not sure (maybe") and not res["emotion"]["sure"], res["emotion_text"])

# missing pieces
res = pf.read_face(PICTURE, f, net=None, emotion_fn=fake_emotion)
check("read_face: no age network -> age is None, emotion still works", res["age"] is None and res["age_text"] is None and res["emotion"] is not None)


def broken_emotion(crop):
    raise RuntimeError("DeepFace is not available")


res = pf.read_face(PICTURE, f, net=net, emotion_fn=broken_emotion)
check("read_face: if the mirror emotion fails, DeepFace's own answer is used",
      res["emotion"] is not None and res["emotion"]["name"] == "Sad", str(res["emotion"]))
res = pf.read_face(PICTURE, face_dict(300, 200, 100, 120), net=net, emotion_fn=broken_emotion)
check("read_face: no emotion anywhere -> emotion None, age still works", res["emotion"] is None and res["emotion_scores"] == [] and res["age_text"] == "25-32")


class ExplodingNet(FakeNet):
    def forward(self):
        raise RuntimeError("net broke")


res = pf.read_face(PICTURE, f, net=ExplodingNet(age_vec(4, 0.8)), emotion_fn=fake_emotion)
check("read_face: a crash inside the age network is caught", res["age"] is None and res["emotion"] is not None)
check("read_face: empty box -> None", pf.read_face(PICTURE, face_dict(5, 5, 2, 2), net=net, emotion_fn=fake_emotion) is None)


# ================================================================== 3b. age_model.py (the small ONNX age model)
check("years_to_text: 79 -> '60-100' and sure", am.years_to_text(79.0) == ("60-100", True))
check("years_to_text: 28 -> '25-32' and sure", am.years_to_text(28.0) == ("25-32", True))
check("years_to_text: 35 sits between two groups -> both", am.years_to_text(35.0) == ("25-32 or 38-43", False))
check("years_to_text: 14 -> '8-12 or 15-20'", am.years_to_text(14.0) == ("8-12 or 15-20", False))
check("years_to_text: 55.2 is outside every group -> nearest, not sure", am.years_to_text(55.2) == ("48-53", False))
check("years_to_text: 1 -> '0-2 or 4-6'", am.years_to_text(1.0) == ("0-2 or 4-6", False))
check("years_to_text: 100 -> '60-100'", am.years_to_text(100.0) == ("60-100", True))
d = am.describe_years(79.6, 78.4, 80.1)
check("describe_years: steady and inside a group -> sure, years kept", d["sure"] and d["text"] == "60-100" and abs(d["years"] - 79.6) < 1e-9 and d["source"] == "onnx")
d = am.describe_years(79.6, 60.0, 80.0)
check("describe_years: unsteady tries (spread > 12 years) -> not sure + note", (not d["sure"]) and "not steady" in d["note"])

black = np.zeros((400, 600, 3), np.uint8)
black[100:110, 200:210] = 255                                   # a white square at the top-left corner of the box
face96 = am.prepare_face(black, (200, 100, 300, 200))
check("prepare_face: a 96 x 96 BGR picture", face96 is not None and face96.shape == (96, 96, 3))
check("prepare_face: the box (100 px) fills 96 / 1.5 = 64 px, so its corner lands at (16, 16)",
      face96[17:21, 17:21].mean() > 200 and face96[4:12, 4:12].mean() < 5 and face96[60:80, 60:80].mean() < 5,
      str((face96[17:21, 17:21].mean(), face96[4:12, 4:12].mean())))
check("prepare_face: a tiny box -> None", am.prepare_face(black, (10, 10, 14, 14)) is None)
check("prepare_face: a box next to the edge still gives a picture (black border)", am.prepare_face(black, (550, 350, 600, 400)).shape == (96, 96, 3))


class FakeSession:
    def __init__(self, age=55.0, fail=False, nan=False):
        self.age, self.fail, self.nan = age, fail, nan
        self.blob = None

    def get_inputs(self):
        return [types.SimpleNamespace(name="data")]

    def run(self, names, feed):
        if self.fail:
            raise RuntimeError("onnx broke")
        self.blob = feed["data"]
        n = self.blob.shape[0]
        out = np.zeros((n, 3), np.float32)
        out[:, 2] = (float("nan") if self.nan else self.age / 100.0)
        return [out]


red = np.zeros((300, 300, 3), np.uint8)
red[:, :, 2] = 255                                              # pure RED in BGR order
fs = FakeSession(55.0)
res = am.predict_age(red, (100, 100, 200, 200), session=fs)
check("predict_age: 14 tries (7 crops x mirror) in one batch", fs.blob.shape == (14, 3, 96, 96), str(fs.blob.shape))
check("predict_age: the model gets RGB, values 0..255 (red picture -> channel 0 is 255)", fs.blob[:, 0].min() > 250 and fs.blob[:, 2].max() < 5)
check("predict_age: median age and the spread", abs(res["years"] - 55.0) < 1e-6 and res["low"] == res["high"] == res["years"] or abs(res["low"] - 55.0) < 1e-3)
check("predict_age: also returns the 96 x 96 face the model saw", res["face"].shape == (96, 96, 3))
check("predict_age: ages are limited to 1..100", am.predict_age(red, (100, 100, 200, 200), session=FakeSession(250.0))["years"] == 100.0)
check("predict_age: a crash inside onnx -> None", am.predict_age(red, (100, 100, 200, 200), session=FakeSession(fail=True)) is None)
check("predict_age: only NaN answers -> None", am.predict_age(red, (100, 100, 200, 200), session=FakeSession(nan=True)) is None)
check("predict_age: tiny box -> None", am.predict_age(red, (10, 10, 12, 12), session=FakeSession()) is None)
saved = (am.MODEL_PATH, am._session, am._tried)
am.MODEL_PATH, am._session, am._tried = "/this/file/does/not/exist.onnx", None, False
check("get_session: model file missing -> None (the app uses the old model)", am.get_session() is None and am.predict_age(red, (100, 100, 200, 200)) is None)
am.MODEL_PATH, am._session, am._tried = saved

# a median is not moved by one odd crop: make the 3rd crop answer 99 and the rest 40
class OddSession(FakeSession):
    def run(self, names, feed):
        out = super().run(names, feed)[0]
        out[2, 2] = 0.99
        return [out]


check("predict_age: one odd crop does not move the median", abs(am.predict_age(red, (100, 100, 200, 200), session=OddSession(40.0))["years"] - 40.0) < 1e-3)

# the REAL model file, if it is next to this script and onnxruntime is installed
try:
    import onnxruntime  # noqa: F401
    real_ok = os.path.exists(am.MODEL_PATH)
except Exception:
    real_ok = False
if real_ok:
    rng_img = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    real = am.predict_age(rng_img, (80, 80, 220, 220))
    check("REAL model: loads and answers with a number between 1 and 100", real is not None and 1.0 <= real["years"] <= 100.0 and np.isfinite(real["years"]))
else:
    print("SKIP  REAL model test (onnxruntime or models/genderage.onnx not found here)")

# read_face with the ONNX age function
face_img_bgr = np.zeros((96, 96, 3), np.uint8)
face_img_bgr[:, :, 0] = 255                                      # BLUE in BGR
calls = {"net": 0}


def lazy_net():
    calls["net"] += 1
    return net


good = lambda img, box: {"years": 79.6, "low": 78.4, "high": 80.1, "face": face_img_bgr}   # noqa: E731
res = pf.read_face(PICTURE, f, net=lazy_net, emotion_fn=fake_emotion, age_fn=good)
check("read_face + ONNX: age text, years and source", res["age_text"] == "60-100" and res["age"]["sure"] and abs(res["age"]["years"] - 79.6) < 1e-9 and res["age"]["source"] == "onnx")
check("read_face + ONNX: the old model is not even loaded", calls["net"] == 0)
check("read_face + ONNX: the face picture is returned as RGB, 192 px", res["face_rgb"].shape == (192, 192, 3) and res["face_rgb"][0, 0].tolist() == [0, 0, 255], str(res["face_rgb"][0, 0]))
res = pf.read_face(PICTURE, f, net=lazy_net, emotion_fn=fake_emotion, age_fn=lambda img, box: None)
check("read_face: ONNX gives no answer -> the old model is the backup", res["age"]["source"] == "caffe" and calls["net"] == 1 and res["age_text"] is not None)
check("read_face: backup shows the face crop the old model used", res["face_rgb"] is not None and max(res["face_rgb"].shape[:2]) == 192)


def crashing_age(img, box):
    raise RuntimeError("boom")


res = pf.read_face(PICTURE, f, net=net, emotion_fn=fake_emotion, age_fn=crashing_age)
check("read_face: a crash in the ONNX age function -> backup model", res["age"]["source"] == "caffe")


def broken_loader():
    raise RuntimeError("cannot load")


res = pf.read_face(PICTURE, f, net=broken_loader, emotion_fn=fake_emotion, age_fn=lambda i, b: None)
check("read_face: no ONNX and the old model cannot load -> age None, emotion still works", res["age"] is None and res["emotion"] is not None)
res = pf.read_face(PICTURE, f, net=None, emotion_fn=fake_emotion, age_fn=lambda i, b: {"years": 30.0, "low": 20.0, "high": 45.0, "face": face_img_bgr})
check("read_face + ONNX: unsteady tries -> not sure", res["age"] is not None and not res["age"]["sure"])

# ================================================================== 4. the glue in app.py
tree = ast.parse(open(os.path.join(SRC, "app.py"), encoding="utf-8").read())
wanted = {"analyze_face", "show_face_results", "count_for_dashboard"}
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
check("app.py: the three helper functions exist", {f.name for f in funcs} == wanted, str([f.name for f in funcs]))


class FakeDeepFace:
    def __init__(self):
        self.answer = []
        self.seen_shape = None
        self.error = None

    def analyze(self, img, actions=None, enforce_detection=True, detector_backend=None, silent=False):
        self.seen_shape = img.shape
        self.kw = dict(actions=actions, enforce_detection=enforce_detection, detector_backend=detector_backend)
        if self.error:
            raise self.error
        return self.answer


deepface = FakeDeepFace()
glue_net = FakeNet(age_vec(5, 0.7))             # 38-43
GLUE_AGE = {"answer": None}                     # the ONNX model gives no answer -> the old model is used
ns = {
    "cv2": cv2, "gc": __import__("gc"), "st": st, "DeepFace": deepface,
    "choose_face": pf.choose_face, "read_face": pf.read_face,
    "get_age_net": lambda: glue_net,
    "predict_age": lambda img, box: GLUE_AGE["answer"],
    "record_photo": dash.record_photo, "photo_key": dash.photo_key,
}
exec(compile(ast.Module(body=funcs, type_ignores=[]), "app.py", "exec"), ns)
analyze_face, show_face_results, count_for_dashboard = ns["analyze_face"], ns["show_face_results"], ns["count_for_dashboard"]

pf_original_emotion = pf.emotion_vector
crop_sizes = []


def recording_emotion(crop):
    crop_sizes.append(crop.shape)
    return emo_vec("surprise", 0.7)


pf.emotion_vector = recording_emotion

BIG = np.full((1200, 1600, 3), 90, np.uint8)
deepface.answer = [face_dict(100, 80, 60, 70, emotion={"happy": 99.0}), face_dict(200, 150, 160, 180, emotion={"happy": 99.0})]
result = analyze_face(BIG)
check("analyze_face: the face finder gets a copy of at most 640 pixels", max(deepface.seen_shape[:2]) == 640, str(deepface.seen_shape))
check("analyze_face: face finder settings are unchanged (emotion, mtcnn, no forced face)",
      deepface.kw == {"actions": ["emotion"], "enforce_detection": False, "detector_backend": "mtcnn"}, str(deepface.kw))
check("analyze_face: two faces -> biggest is used and the count is stored", result is not None and result["faces"] == 2)
check("analyze_face: box is in ORIGINAL picture coordinates (x 2.5)", result["box"] == (500, 375, 900, 825), str(result["box"]))
check("analyze_face: emotion is read from the ORIGINAL picture (big crop)", crop_sizes and crop_sizes[-1][0] > 300, str(crop_sizes))
check("analyze_face: age and emotion are in the answer", result["age_text"] == "38-43" and result["emotion"]["name"] == "Surprise")

small_pic = np.full((300, 400, 3), 90, np.uint8)
deepface.answer = [face_dict(100, 60, 80, 90)]
result = analyze_face(small_pic)
check("analyze_face: a small picture is not shrunk", deepface.seen_shape == small_pic.shape and result["box"] == (100, 60, 180, 150))

deepface.answer = [face_dict(0, 0, 400, 300, conf=0.0)]
check("analyze_face: face_confidence 0 -> None (Face not detected)", analyze_face(small_pic) is None)
deepface.answer = []
check("analyze_face: empty answer -> None", analyze_face(small_pic) is None)
deepface.answer = face_dict(100, 60, 80, 90)        # DeepFace may return a single dict
check("analyze_face: a single dict answer is accepted", analyze_face(small_pic) is not None)

st.events.clear()
deepface.error = ValueError("boom")
check("analyze_face: a crash in DeepFace shows an error and returns None", analyze_face(small_pic) is None and len(st.of("error")) == 1)
deepface.error = None
pf.emotion_vector = pf_original_emotion

GLUE_AGE["answer"] = {"years": 79.6, "low": 78.4, "high": 80.1, "face": face_img_bgr}
deepface.answer = [face_dict(100, 60, 80, 90, emotion={"happy": 99.0})]
result = analyze_face(small_pic)
check("analyze_face: the ONNX age is used when the model answers", result["age"]["source"] == "onnx" and result["age_text"] == "60-100")
GLUE_AGE["answer"] = None

# ---- show_face_results
st.events.clear()
show_face_results(None)
check("show: no face -> 'Face not detected.'", st.events == [("info", "Face not detected.")], str(st.events))

sure = pf.read_face(PICTURE, f, net=FakeNet(age_vec(4, 0.8)), emotion_fn=lambda c: emo_vec("happy", 0.8))
sure["faces"] = 1
st.events.clear()
show_face_results(sure)
metrics = {e[1]: e for e in st.of("metric")}
check("show: age metric", metrics["Estimated Age"][2] == "25-32")
check("show: sure emotion -> name and percent, as before", metrics["Dominant Emotion"][2] == "Happy" and metrics["Dominant Emotion"][3] == "80.0%", str(metrics["Dominant Emotion"]))
check("show: 7 emotion bars", len(st.of("progress")) == 7 and len(st.of("write")) == 7)
check("show: the honest note is always shown", any("estimates" in e[1] for e in st.of("caption")))
check("show: no 'not sure' notes for a sure answer", not any("not very sure" in e[1] or "Best guess" in e[1] for e in st.of("caption")))

unsure = pf.read_face(PICTURE, f, net=FakeNet(age_vec(4, 0.40, second=5, second_strength=0.30)), emotion_fn=lambda c: emo_vec("neutral", 0.30))
unsure["faces"] = 3
st.events.clear()
show_face_results(unsure)
metrics = {e[1]: e for e in st.of("metric")}
caps = " | ".join(e[1] for e in st.of("caption"))
check("show: unsure emotion -> 'Not sure' plus the best guess", metrics["Dominant Emotion"][2] == "Not sure" and "Best guess: Neutral (30%)" in caps, caps)
check("show: unsure age -> note", "not very sure about this age" in caps and metrics["Estimated Age"][2] == "25-32 or 38-43", caps)
check("show: several faces -> note", "3 faces found" in caps)

no_age = dict(sure, age=None, age_text=None)
st.events.clear()
show_face_results(no_age)
check("show: no age model -> 'N/A'", {e[1]: e for e in st.of("metric")}["Estimated Age"][2] == "N/A")
no_emotion = dict(sure, emotion=None, emotion_text=None, emotion_scores=[])
st.events.clear()
show_face_results(no_emotion)
check("show: no emotion -> 'N/A' and no bars", {e[1]: e for e in st.of("metric")}["Dominant Emotion"][2] == "N/A" and not st.of("progress"))

onnx_res = pf.read_face(PICTURE, f, net=None, emotion_fn=lambda c: emo_vec("happy", 0.8), age_fn=good)
onnx_res["faces"] = 1
st.events.clear()
show_face_results(onnx_res)
caps = " | ".join(e[1] for e in st.of("caption"))
check("show: ONNX answer -> no 'old model' warning", not st.of("warning"))
check("show: the exact guess in years is shown with the honest error note", "about 80 years" in caps and "7-8 years" in caps, caps)
check("show: 'the face the model looked at' is in an expander with a picture", any("face the model looked at" in e[1] for e in st.of("expander")) and len(st.of("image")) == 1)
check("show: a steady, sure age has no 'not sure' note", "not steady" not in caps and "border" not in caps and "not very sure" not in caps)

# the old (Caffe) model was used because the ONNX model is not available -> tell the user
caffe_res = pf.read_face(PICTURE, f, net=FakeNet(age_vec(4, 0.8)), emotion_fn=lambda c: emo_vec("happy", 0.8), age_fn=None)
caffe_res["faces"] = 1
st.events.clear()
show_face_results(caffe_res)
warn = " ".join(e[1] for e in st.of("warning"))
check("show: old model used -> warning names get_age_model.py", "get_age_model.py" in warn and "genderage.onnx" in warn, warn)

# a stale photo_faces.py (no age_fn) gives a clear message, not a confusing TypeError
def _stale_read_face(*a, **kw):
    raise TypeError("read_face() got an unexpected keyword argument 'age_fn'")
_saved_read_face = analyze_face.__globals__["read_face"]
analyze_face.__globals__["read_face"] = _stale_read_face
deepface.answer = [face_dict(100, 60, 80, 90, emotion={"happy": 99.0})]
st.events.clear()
_res = analyze_face(small_pic)
errs = " ".join(e[1] for e in st.of("error"))
check("analyze_face: stale photo_faces.py -> clear 'OLD version' message", _res is None and "OLD version" in errs and "age_fn" in errs, errs)
analyze_face.__globals__["read_face"] = _saved_read_face

# an empty / tiny model file is reported in the console and treated as 'no model'
import contextlib, io, tempfile
import age_model as am
_old_path, _old_tried, _old_session = am.MODEL_PATH, am._tried, am._session
with tempfile.TemporaryDirectory() as tmp:
    empty = os.path.join(tmp, "genderage.onnx")
    open(empty, "wb").close()
    am.MODEL_PATH, am._tried, am._session = empty, False, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = am.get_session()
    check("age_model: empty model file -> None plus a hint to run get_age_model.py", got is None and "get_age_model.py" in buf.getvalue(), buf.getvalue())
    am.MODEL_PATH, am._tried, am._session = os.path.join(tmp, "nothing.onnx"), False, None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = am.get_session()
    check("age_model: missing model file -> None plus a hint", got is None and "get_age_model.py" in buf.getvalue(), buf.getvalue())
am.MODEL_PATH, am._tried, am._session = _old_path, _old_tried, _old_session
shaky = pf.read_face(PICTURE, f, net=None, emotion_fn=lambda c: emo_vec("happy", 0.8), age_fn=lambda i, b: {"years": 35.0, "low": 30.0, "high": 50.0, "face": face_img_bgr})
shaky["faces"] = 1
st.events.clear()
show_face_results(shaky)
check("show: an unsteady age shows the note", any("not steady" in e[1] for e in st.of("caption")))
no_pic = dict(onnx_res, face_rgb=None)
st.events.clear()
show_face_results(no_pic)
check("show: no face picture -> no expander", not st.of("expander"))

# ---- the dashboard hook
st.session_state.clear()


class FakeFile:
    def __init__(self, fid):
        self.file_id = fid


details = ["🧍 **Person detected** — Confidence: **0.90**", "☕ **Cup detected** — Confidence: **0.70**"]
count_for_dashboard(details, sure, FakeFile("a"))
count_for_dashboard(details, sure, FakeFile("a"))        # a rerun of the same photo
stats = dash.get_stats()
check("dashboard: the same photo is counted only once", stats["photos"] == 1)
check("dashboard: objects, emotion and age of the photo are counted",
      stats["objects"]["Person"] == 1 and stats["objects"]["Cup"] == 1 and stats["emotions"]["Happy"] == 1 and stats["ages"]["25-32"] == 1 and stats["faces"] == 1)
count_for_dashboard([], None, FakeFile("b"))
check("dashboard: a photo without a face counts as a photo, not as a face", stats["photos"] == 2 and stats["faces"] == 1)
count_for_dashboard([], unsure, FakeFile("c"))
check("dashboard: 'Not sure' emotion is not counted, the age range is",
      stats["emotions"]["Neutral"] == 0 and stats["ages"]["25-32"] == 2 and stats["faces"] == 2)
count_for_dashboard(details, sure, object())            # no id, no getvalue -> key None
check("dashboard: a broken photo object never breaks the page", True)

count_for_dashboard([], onnx_res, FakeFile("d"))
check("dashboard: the new age text ('60-100') is counted", stats["ages"]["60-100"] == 1)
count_for_dashboard([], dict(onnx_res, age_text="48-53 or 60-100"), FakeFile("e"))
check("dashboard: 'a or b' counts the first group", stats["ages"]["48-53"] == 1)

# ================================================================== 5. the app wiring (text checks)
source = open(os.path.join(SRC, "app.py"), encoding="utf-8").read()
check("app.py: the mode list has the Dashboard", '"📊 Dashboard"' in source and 'elif "Dashboard" in mode' in source)
check("app.py: no old single-look age code is left", "estimate_age" not in source and "load_age_model" not in source and "readNetFromCaffe" not in source)
check("app.py: both photo modes record the photo for the dashboard", source.count("count_for_dashboard(details, face_result,") == 2)
check("app.py: the face-finder import order is unchanged (webrtc first)",
      source.index("import streamlit_webrtc") < source.index("import cv2") < source.index("from deepface import DeepFace") < source.index("from photo_faces import"))


check("app.py: phone photos are turned upright (EXIF) in both photo modes", source.count("ImageOps.exif_transpose(") == 2 and "from PIL import Image, ImageOps" in source)
check("app.py: the ONNX age function is used and the old model is loaded lazily",
      "from age_model import predict_age" in source and "net=get_age_net, age_fn=predict_age" in source)
req = os.path.join(SRC, "requirements.txt")
if os.path.exists(req):
    check("requirements.txt: onnxruntime is listed", "onnxruntime" in open(req, encoding="utf-8").read())
else:
    print("SKIP  requirements.txt not found here")

passed = sum(1 for r in RESULTS if r[1])
print("\n" + "=" * 60)
print(f"TOTAL        {passed}/{len(RESULTS)} passed")
sys.exit(0 if passed == len(RESULTS) else 1)