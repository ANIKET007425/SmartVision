"""Live webcam mode for SmartVision.

How it works (simple version):
    Browser camera  ->  WebRTC  ->  this server  ->  boxes drawn  ->  back to the browser

Why it is built this way:
  * cv2.VideoCapture(0) opens a camera on the SERVER, and a cloud server has no
    camera. streamlit-webrtc sends the visitor's browser camera to the server.
  * YOLO is slow on a free CPU server. So the video callback only draws the
    LAST known boxes (fast, smooth video), while a separate worker thread runs
    YOLO on the newest frame every `interval` seconds.
  * Every visitor gets their OWN LiveDetector (kept in st.session_state), so one
    visitor's detections are never shown to another visitor.

V2 changes (smoother video + emotion):
  * Video quality presets (Smooth / Balanced / Sharp). A smaller picture and
    fewer frames per second travel much better over a phone network.
  * The video callback does almost nothing when there is nothing to draw, and
    the AI worker thread uses at most two processor cores.
  * The emotion is found for each face (face_boost.py) and is
    averaged over a few seconds, so one bad frame cannot flip the label.
    (The estimated-age feature was removed: age models are often wrong.)
  * A speed meter shows how many frames per second the server really handles.

The class LiveDetector does not need streamlit or av, so it can be tested alone.
"""

import os
import subprocess
import sys
import threading
import time

import cv2

try:
    import face_boost                   # emotion helpers (optional file)
except Exception:                       # the live page still works without it
    face_boost = None

LIVE_VERSION = "V3 (smooth video + emotion)"


def _trace(message):
    """Short progress note (with memory use) in the Streamlit Cloud log."""
    try:
        import psutil
        mb = psutil.Process().memory_info().rss / (1024 * 1024)
        loaded = ", ".join(n for n in ("torch", "tensorflow", "cv2", "av") if n in sys.modules)
        print(f"[live] {message} (memory {mb:.0f} MB; loaded: {loaded})", file=sys.stderr, flush=True)
    except Exception:
        print(f"[live] {message}", file=sys.stderr, flush=True)


# Only one YOLO call at a time, across all visitors (the model is shared).
_MODEL_LOCK = threading.Lock()

# The worker thread stops itself if no frames arrive for this long.
IDLE_SECONDS = 20

FRIENDLY_NAMES = {
    "person": "Person",
    "eyeglasses": "Eyeglasses",
    "spectacles": "Eyeglasses",
    "pen": "Pen",
    "ballpoint pen": "Pen",
    "bag": "Bag",
    "handbag": "Bag",
    "backpack": "Bag",
    "coffee cup": "Cup",
    "cup": "Cup",
    "mobile phone": "Mobile phone",
    "cell phone": "Mobile phone",
    "tie": "Tie",
    "bottle": "Bottle",
    "book": "Book",
}

ICONS = {
    "Person": "🧍",
    "Eyeglasses": "👓",
    "Pen": "🖊️",
    "Bag": "🎒",
    "Cup": "☕",
    "Mobile phone": "📱",
    "Tie": "👔",
    "Bottle": "🍶",
    "Book": "📖",
}

GREEN = (0, 255, 0)        # BGR: person
ORANGE = (255, 100, 0)     # BGR: everything else (looks blue in BGR order)
FACE_COLOUR = (255, 0, 200)  # BGR: face box with the emotion (pink-violet)

# Camera size and speed that the browser is asked for. The server sends the
# picture BACK to the phone, so a smaller picture = less data = smoother video.
QUALITY_PRESETS = {
    "Smooth (best for phone)": {"id": "smooth", "width": 320, "height": 240, "fps": 10},
    "Balanced": {"id": "balanced", "width": 480, "height": 360, "fps": 12},
    "Sharp (good network)": {"id": "sharp", "width": 640, "height": 480, "fps": 15},
}


class LiveDetector:
    """Runs YOLO (and, if switched on, face analysis) in a background thread and
    draws the latest results on frames."""

    def __init__(self, model, colour_fn=None, face_pipeline=None):
        self.model = model
        self.colour_fn = colour_fn      # function(frame_bgr, [x1, y1, x2, y2]) -> "Red"

        # Settings (the Streamlit page updates these on every rerun)
        self.conf = 0.45
        self.imgsz = 320
        self.interval = 0.5             # minimum seconds between two detections
        self.faces_on = False           # emotion (needs face_boost.py)
        self.face_interval = 1.0        # minimum seconds between two face analyses

        self._lock = threading.Lock()
        self._latest_frame = None
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._busy = False
        self._last_done = 0.0
        self._last_frame_time = time.monotonic()

        self._boxes = []                # (x1, y1, x2, y2, text, is_person)
        self._frame_size = None         # (width, height) the boxes belong to
        self._counts = {}
        self._infer_seconds = None
        self._detections_done = 0
        self._error = None

        # Faces (emotion)
        self._pipeline = face_pipeline  # created on first use if None
        self._faces = []                # dicts from FacePipeline.analyze()
        self._last_face_time = 0.0
        self._face_status = ""
        self._face_error = None

        # Video speed meter (filled by the video callback)
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self._fps = None
        self._callback_ms = None

    # ---------- called from the video callback thread ----------

    def tick(self):
        """Called for EVERY frame. Cheap. Returns True when this frame should be
        handed to the detector (a detection is due and the worker is free)."""
        now = time.monotonic()
        self._last_frame_time = now
        self._ensure_running()
        return not (self._busy or (now - self._last_done) < self.interval)

    def offer(self, frame_bgr):
        """Give the detector a frame (only call it when tick() returned True)."""
        self._busy = True
        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._new_frame.set()

    def submit(self, frame_bgr):
        """Offer a frame for detection. Never blocks; most frames are skipped."""
        if self.tick():
            self.offer(frame_bgr)

    def has_overlay(self):
        """True if there is anything to draw on the video."""
        with self._lock:
            return bool(self._boxes or self._faces)

    def note_frame(self, milliseconds):
        """Speed meter: how many frames per second the server really handles."""
        now = time.monotonic()
        self._callback_ms = (
            milliseconds if self._callback_ms is None
            else 0.9 * self._callback_ms + 0.1 * milliseconds
        )
        self._fps_count += 1
        elapsed = now - self._fps_t0
        if elapsed >= 2.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0 = now

    def draw(self, frame_bgr):
        """Draw the last known boxes on the frame (in place) and return it."""
        with self._lock:
            boxes = list(self._boxes)
            faces = list(self._faces)
            size = self._frame_size

        if (not boxes and not faces) or size is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        sx = w / float(size[0])
        sy = h / float(size[1])

        for x1, y1, x2, y2, text, is_person in boxes:
            colour = GREEN if is_person else ORANGE
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)
            _draw_label(frame_bgr, text, x1, y1, colour)

        for face in faces:
            fx1, fy1, fx2, fy2 = face["box"]
            fx1, fx2 = int(fx1 * sx), int(fx2 * sx)
            fy1, fy2 = int(fy1 * sy), int(fy2 * sy)
            cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), FACE_COLOUR, 2)
            # labels go BELOW the face box so they do not hide the object labels
            y = fy2
            for line in face["lines"]:
                y = _draw_label(frame_bgr, line, fx1, y, FACE_COLOUR, below=True)

        return frame_bgr

    # ---------- called from the Streamlit page ----------

    def summary(self):
        """Small snapshot for the details panel."""
        with self._lock:
            return {
                "counts": dict(self._counts),
                "infer_seconds": self._infer_seconds,
                "detections_done": self._detections_done,
                "error": self._error,
                "faces": [dict(f) for f in self._faces],
                "face_status": self._face_status,
                "face_error": self._face_error,
                "fps": self._fps,
                "callback_ms": self._callback_ms,
            }

    def clear(self):
        """Forget old boxes (used when the stream stops)."""
        with self._lock:
            self._boxes = []
            self._counts = {}
            self._frame_size = None
            self._latest_frame = None
            self._faces = []
            self._face_status = ""
            self._fps = None
            self._callback_ms = None
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        if self._pipeline is not None:
            try:
                self._pipeline.reset()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self._new_frame.set()

    # ---------- worker thread ----------

    def _ensure_running(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._busy = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        _limit_threads()
        try:
            while not self._stop.is_set():
                got_frame = self._new_frame.wait(timeout=1.0)

                if not got_frame:
                    if time.monotonic() - self._last_frame_time > IDLE_SECONDS:
                        break          # nobody is sending frames: stop quietly
                    continue

                self._new_frame.clear()

                with self._lock:
                    frame = self._latest_frame
                    self._latest_frame = None

                if frame is None:
                    self._busy = False
                    continue

                try:
                    self._detect(frame)
                except Exception as e:        # never let the thread die silently
                    with self._lock:
                        self._error = f"{type(e).__name__}: {e}"
                finally:
                    self._last_done = time.monotonic()
                    self._busy = False
        finally:
            self._busy = False

    def _detect(self, frame):
        h, w = frame.shape[:2]

        started = time.monotonic()
        with _MODEL_LOCK:
            results = self.model.predict(
                frame,
                conf=self.conf,
                imgsz=self.imgsz,
                agnostic_nms=True,
                verbose=False,
            )
        took = time.monotonic() - started

        boxes = []
        counts = {}

        for result in results:
            for box in result.boxes:
                raw_name = self.model.names[int(box.cls[0])]
                score = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                name = FRIENDLY_NAMES.get(raw_name, str(raw_name).capitalize())
                text = f"{name} {score:.2f}"

                if name == "Tie" and self.colour_fn is not None:
                    try:
                        text = f"Tie ({self.colour_fn(frame, [x1, y1, x2, y2])}) {score:.2f}"
                    except Exception:
                        pass

                boxes.append((x1, y1, x2, y2, text, name == "Person"))
                counts[name] = counts.get(name, 0) + 1

        # Faces: only when somebody (a "person") is in the picture, and not more
        # often than face_interval seconds. None = "leave the old faces as they are".
        faces = None
        face_status = None
        face_error = None
        if self.faces_on:
            person_boxes = [(b[0], b[1], b[2], b[3]) for b in boxes if b[5]]
            if not person_boxes:
                faces, face_status = [], ""
                if self._pipeline is not None:
                    self._pipeline.reset()
            elif time.monotonic() - self._last_face_time >= self.face_interval:
                faces, face_status, face_error = self._analyze_faces(frame, person_boxes)
                self._last_face_time = time.monotonic()

        with self._lock:
            self._boxes = boxes
            self._frame_size = (w, h)
            self._counts = counts
            self._infer_seconds = took
            self._detections_done += 1
            self._error = None
            if faces is not None:
                self._faces = faces
                self._face_status = face_status or ""
                self._face_error = face_error
            elif not self.faces_on:
                self._faces = []

    def _analyze_faces(self, frame, person_boxes):
        """Returns (faces, status, error). Never raises."""
        try:
            if self._pipeline is None:
                if face_boost is None:
                    return [], "The emotion needs the file face_boost.py.", "face_boost.py not found"
                self._pipeline = face_boost.FacePipeline()
            faces = self._pipeline.analyze(frame, person_boxes)
            return faces, self._pipeline.status, self._pipeline.error
        except Exception as e:
            return [], "Face analysis stopped because of a problem.", f"{type(e).__name__}: {e}"


def _limit_threads():
    """Keep the AI model from using EVERY processor core, so the video thread
    always gets some CPU time (this is what keeps the live picture smooth on a
    small shared server). Called inside the worker thread only."""
    try:
        import torch
        torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2) // 2)))
    except Exception:
        pass


def _draw_label(frame_bgr, text, x, y, colour, below=False):
    """Draw a text label with a coloured background.

    below=False: the label sits ABOVE y (used for object boxes).
    below=True : the label sits BELOW y; returns the next free y (for stacking).
    """
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    height = th + base + 4
    if below:
        top = min(max(0, frame_bgr.shape[0] - height), y)
    else:
        top = max(0, y - height)
    cv2.rectangle(frame_bgr, (x, top), (x + tw + 6, top + height), colour, -1)
    cv2.putText(
        frame_bgr, text, (x + 3, top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
    )
    return top + height


# ======================================================================
# Streamlit part
# ======================================================================

def _read_secret(name):
    """Environment first, then st.secrets. Never raises."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _rtc_configuration():
    """Returns (rtc_configuration, turn_is_configured).

    With Cloudflare TURN keys in Secrets, streamlit-webrtc fetches the relay
    credentials by itself, so we leave rtc_configuration as None.
    Without them we fall back to a public STUN server (works locally and on
    many home networks, but is often blocked on Streamlit Community Cloud).
    """
    key_id = _read_secret("CLOUDFLARE_TURN_KEY_ID")
    token = _read_secret("CLOUDFLARE_TURN_KEY_API_TOKEN")

    if key_id and token:
        os.environ["CLOUDFLARE_TURN_KEY_ID"] = key_id
        os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = token
        return None, True

    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, False


# ----------------------------------------------------------------------
# Safety check: import the video libraries in a SEPARATE process first.
# If a library crashes (segmentation fault), only that small process dies and
# the app shows a message instead of going down for every visitor.
# ----------------------------------------------------------------------

_PREFLIGHT_CACHE = {}

PREFLIGHT_CODE = "import cv2, av, aiortc, streamlit_webrtc"

SELF_TESTS = [
    ("av alone", "import av"),
    ("aiortc alone", "import aiortc"),
    ("streamlit_webrtc alone", "import streamlit_webrtc"),
    ("cv2, then av", "import cv2, av"),
    ("av, then cv2", "import av, cv2"),
    ("cv2, then aiortc", "import cv2, aiortc"),
    ("cv2, then streamlit_webrtc", "import cv2, streamlit_webrtc"),
]

# "X first, then pylibsrtp": pylibsrtp is the library that crashed in the app
# (Fatal Python error: Segmentation fault ... pylibsrtp/__init__.py). These tests
# find out WHICH other library it clashes with. Each one runs in its own process.
CLASH_TESTS = [
    (f"{name}, then pylibsrtp", f"import {name}; import pylibsrtp")
    for name in [
        "numpy", "PIL", "yaml", "google.protobuf", "websockets",
        "pyarrow", "pandas", "h5py", "onnxruntime", "cryptography", "psutil",
    ]
]

def _in_thread(code, then=""):
    """Python source that runs `code` inside a NON-main thread (like Streamlit's
    script thread does), and afterwards runs `then` in the main thread.
    A crash kills the process (-11); a normal import error gives exit code 1."""
    body = "\n".join("        " + line for line in code.splitlines())
    return (
        "import sys, threading\n"
        "errors = []\n"
        "def _run():\n"
        "    try:\n"
        f"{body}\n"
        "    except BaseException as e:\n"
        "        errors.append(e)\n"
        "        print(repr(e), file=sys.stderr)\n"
        "t = threading.Thread(target=_run)\n"
        "t.start()\n"
        "t.join()\n"
        "if errors:\n"
        "    sys.exit(1)\n"
        f"{then}\n"
    )


# The imports that app.py does at the top (before the live page is opened)
APP_IMPORTS = "import cv2\nfrom ultralytics import YOLOWorld\nfrom deepface import DeepFace"

# Known from the last run: "tensorflow, then pylibsrtp" CRASHES, torch is fine.
HEAVY_TESTS = [
    ("tensorflow, then pylibsrtp", "import tensorflow\nimport pylibsrtp"),
    ("pylibsrtp FIRST, then tensorflow", "import pylibsrtp\nimport tensorflow"),
    ("streamlit_webrtc FIRST, then tensorflow", "import streamlit_webrtc\nimport tensorflow"),
    ("app imports, then streamlit_webrtc", APP_IMPORTS + "\nimport streamlit_webrtc"),
    ("streamlit_webrtc FIRST, then app imports", "import streamlit_webrtc\n" + APP_IMPORTS),
    ("pylibsrtp alone, in a non-main thread", _in_thread("import pylibsrtp")),
    ("app imports, then streamlit_webrtc, all in a thread",
     _in_thread(APP_IMPORTS + "\nimport streamlit_webrtc")),
    ("streamlit_webrtc FIRST in a thread, then app imports",
     _in_thread("import streamlit_webrtc", then=APP_IMPORTS)),
]

WATCHED_MODULES = [
    "torch", "tensorflow", "keras", "cv2", "ultralytics", "deepface",
    "av", "aiortc", "pylibsrtp", "streamlit_webrtc",
]


def _loaded_report():
    """Which heavy libraries are already loaded in THIS process."""
    return ", ".join(
        f"{name}={'yes' if name in sys.modules else 'no'}" for name in WATCHED_MODULES
    )

# The crash is fixed (app.py loads streamlit_webrtc before deepface), so normal
# visitors see no diagnostics and no extra button. For the OWNER there is a
# debug mode: add `&debug=1` at the end of the web address. In debug mode the
# diagnostics are shown and the video libraries load only after a button click,
# so the tools stay usable even if a future update brings the crash back.
# Set this to True to ask every visitor to click first.
REQUIRE_CLICK_BEFORE_LOADING = False


def _debug_mode(st):
    """True when the web address ends with `debug=1` (owner-only tools)."""
    try:
        return str(st.query_params.get("debug", "")) == "1"
    except Exception:
        return False

REPORT_PACKAGES = [
    "streamlit", "streamlit-webrtc", "av", "aiortc", "pylibsrtp", "cryptography",
    "opencv-python", "opencv-contrib-python-headless", "numpy", "torch", "tensorflow",
]


def _run_snippet(code, timeout=90):
    """Run `code` in a fresh Python process. Returns (return_code, stderr, seconds)."""
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"      # a crash prints which line of Python it was on
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return proc.returncode, (proc.stderr or "").strip(), time.monotonic() - started
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout} s", time.monotonic() - started
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", time.monotonic() - started


def _describe(return_code):
    if return_code == 0:
        return "OK"
    if return_code is None:
        return "ERROR"
    if return_code < 0:
        name = " = segfault" if return_code == -11 else ""
        return f"CRASH (signal {-return_code}{name})"
    return f"FAILED (exit code {return_code})"


def _tail(text, lines=12):
    return "\n".join(text.splitlines()[-lines:])


def _preflight():
    """(True, "") if the live libraries import fine, else (False, details)."""
    if "result" not in _PREFLIGHT_CACHE:
        code, err, _ = _run_snippet(PREFLIGHT_CODE)
        if code == 0:
            _PREFLIGHT_CACHE["result"] = (True, "")
        else:
            _PREFLIGHT_CACHE["result"] = (False, f"{_describe(code)}\n{_tail(err)}")
    return _PREFLIGHT_CACHE["result"]


def _package_versions():
    from importlib.metadata import PackageNotFoundError, version

    lines = []
    for name in REPORT_PACKAGES:
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            lines.append(f"{name}: not installed")
    return lines


def run_self_test(heavy=False):
    """Text report: which import combinations work in a fresh process."""
    lines = [
        "Python " + sys.version.split()[0],
        *_package_versions(),
        "Already loaded in the app: " + _loaded_report(),
        "",
    ]
    tests = SELF_TESTS + CLASH_TESTS + (HEAVY_TESTS if heavy else [])

    for name, code in tests:
        rc, err, secs = _run_snippet(code)
        lines.append(f"{_describe(rc):<24} {name:<36} {secs:5.1f}s")
        if rc != 0 and err:
            for line in _tail(err, 4).splitlines():
                lines.append("      " + line[:160])

    return "\n".join(lines)


def _diagnostics_ui(st):
    """Buttons that run the import tests in separate processes (cannot crash the app)."""
    with st.expander("Diagnostics (to find the live webcam crash)"):
        st.caption(
            "These tests run in separate processes, so they cannot crash the app. "
            "Copy the result text and send it."
        )
        st.caption("Already loaded in the app: " + _loaded_report())

        if st.button("Run light self-test", key="diag_light"):
            with st.spinner("Testing library combinations (about a minute)..."):
                st.session_state["selftest_report"] = run_self_test()

        if st.button("Run heavy self-test (slow, uses a lot of memory)", key="diag_heavy"):
            with st.spinner("Testing with torch and tensorflow (a few minutes)..."):
                st.session_state["selftest_report"] = run_self_test(heavy=True)

        report = st.session_state.get("selftest_report")
        if report:
            st.code(report)


def run_live(yolo_model, conf_value, imgsz_value, colour_fn=None):
    """Draws the whole Live Webcam page. Call it from app.py."""
    import streamlit as st

    try:
        import faulthandler
        faulthandler.enable(all_threads=True)   # a crash prints a Python traceback in the logs
    except Exception:
        pass

    debug = _debug_mode(st)

    _trace("run_live started")
    if "streamlit_webrtc" in sys.modules and not debug:
        # app.py already loaded it at start-up (and survived), so it is safe
        ok, details = True, ""
    else:
        _trace("running the safety check in a separate process")
        ok, details = _preflight()
        _trace(f"safety check finished: {'OK' if ok else 'FAILED'}")
    if not ok:
        if "No module named" in details:
            st.error(
                "Live webcam needs the package `streamlit-webrtc`. "
                "Add `streamlit-webrtc==0.78.1` to requirements.txt."
            )
        else:
            st.error(
                "Live webcam cannot start on this server because a video library "
                "crashed when it was loaded. The rest of the app still works."
            )
        if debug:
            st.code(details)
            _diagnostics_ui(st)
        return

    if debug:
        _diagnostics_ui(st)

    if (REQUIRE_CLICK_BEFORE_LOADING or debug) and not st.session_state.get("live_libs_requested"):
        st.info(
            "The safety check passed. Press the button to load the live video "
            "libraries in the app. (If this crashes the app, open the page again "
            "and use the diagnostics above.)"
        )
        if st.button("Load live webcam", key="load_live"):
            st.session_state["live_libs_requested"] = True
            st.rerun()
        return

    try:
        _trace("importing av and streamlit_webrtc in the app process")
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        _trace("imports done")
    except ImportError:
        st.error(
            "Live webcam needs the package `streamlit-webrtc`. "
            "Add `streamlit-webrtc==0.78.1` to requirements.txt."
        )
        return

    st.info(
        "Press **START**, then allow camera permission in your browser. "
        "Your browser camera is sent to the server only to draw the boxes. "
        "The app code does not save the video."
    )

    quality_name = st.radio(
        "Video quality (choose before pressing START)",
        list(QUALITY_PRESETS),
        index=0,
        horizontal=True,
        key="live_quality",
        help=(
            "The video goes to the server and comes back with boxes. A smaller "
            "picture is much smoother on a phone or a slow network. "
            "Changing this restarts the camera."
        ),
    )
    quality = QUALITY_PRESETS[quality_name]

    interval = st.sidebar.slider(
        "Live: detect every (seconds)",
        min_value=0.2, max_value=2.0, value=0.5, step=0.1,
        help="Smaller = more updates but a slower server. Bigger = smoother video.",
    )
    faces_wanted = st.sidebar.checkbox(
        "Live: show emotion",
        value=True,
        help=(
            "Finds faces and shows the emotion. "
            "Uses more of the server. Switch it off if the video is not smooth."
        ),
    )

    # One detector per visitor session (never shared between visitors)
    if "live_detector" not in st.session_state:
        st.session_state.live_detector = LiveDetector(yolo_model, colour_fn)

    detector = st.session_state.live_detector
    detector.conf = conf_value
    detector.imgsz = imgsz_value
    detector.interval = interval
    detector.faces_on = bool(faces_wanted) and face_boost is not None

    def video_frame_callback(frame):
        started = time.monotonic()
        due = detector.tick()                       # cheap, runs for every frame
        if due or detector.has_overlay():
            # Only now do we pay for the conversion to a numpy picture
            img = frame.to_ndarray(format="bgr24")
            if due:
                detector.offer(img)
            img = detector.draw(img)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        detector.note_frame((time.monotonic() - started) * 1000.0)
        return frame

    rtc_configuration, turn_ok = _rtc_configuration()

    _trace("calling webrtc_streamer")
    ctx = webrtc_streamer(
        key=f"smartvision-live-{quality['id']}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": quality["width"]},
                "height": {"ideal": quality["height"]},
                "frameRate": {"ideal": quality["fps"], "max": quality["fps"]},
            },
            "audio": False,
        },
        async_processing=True,
    )
    _trace("webrtc_streamer returned")

    if debug:
        if turn_ok:
            st.caption("Connection: TURN relay is configured.")
        else:
            st.caption(
                "Connection: no TURN relay set up. This is fine on your own "
                "computer. The online app may need one if the video does not start."
            )
    else:
        st.caption(
            "Video not starting? Some office or school networks block live video. "
            "Try another network or a mobile hotspot."
        )
    st.caption(f"Live Webcam {LIVE_VERSION}")

    if faces_wanted and face_boost is None:
        st.warning(
            "The emotion is switched off because the file face_boost.py "
            "is missing from the app folder."
        )

    panel = st.empty()

    if not ctx.state.playing:
        detector.clear()
        return

    # Keep this script run alive while the video plays, and refresh the panel.
    while ctx.state.playing:
        info = detector.summary()

        with panel.container():
            st.subheader("📋 Live Detection Details")

            if info["error"]:
                st.warning(f"Detection problem: {info['error']}")

            if info["counts"]:
                for name, count in sorted(info["counts"].items()):
                    st.write(f"{ICONS.get(name, '•')} **{name}** × {count}")
            elif info["detections_done"] == 0:
                st.info("Getting ready... the first detection can take a few seconds.")
            else:
                st.info("No object detected right now.")

            if detector.faces_on:
                st.markdown("**🙂 Faces (estimates)**")
                if info["faces"]:
                    for number, face in enumerate(info["faces"], start=1):
                        st.write(f"Face {number}: " + " · ".join(face["lines"]))
                elif info["face_status"]:
                    st.caption(info["face_status"])
                elif info["counts"].get("Person"):
                    st.caption("Looking for a face... look at the camera in good light.")
                else:
                    st.caption("No person in the picture yet.")
                if debug and info["face_error"]:
                    st.caption(f"Face problem (owner only): {info['face_error']}")
                st.caption(
                    "The emotion is a rough estimate from a small AI model. "
                    "Light, angle and glasses can make it wrong, so it is "
                    "averaged over a few seconds to stay steady."
                )

            if info["infer_seconds"] is not None:
                st.caption(
                    f"One detection takes about {info['infer_seconds']:.2f} s "
                    f"on this server (updates every {interval:.1f} s or slower)."
                )

            if info["fps"] is not None:
                st.caption(
                    f"Video speed on the server: about {info['fps']:.0f} frames/s "
                    f"(asked for {quality['fps']})."
                )
                if info["fps"] < 0.6 * quality["fps"]:
                    st.warning(
                        "The video is arriving slowly. Choose **Smooth** quality, "
                        "switch off the emotion, or try a better network."
                    )

        time.sleep(0.5)

    detector.clear()