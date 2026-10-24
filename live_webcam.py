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

The class LiveDetector does not need streamlit or av, so it can be tested alone.
"""

import os
import threading
import time

import cv2

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


class LiveDetector:
    """Runs YOLO in a background thread and draws the latest boxes on frames."""

    def __init__(self, model, colour_fn=None):
        self.model = model
        self.colour_fn = colour_fn      # function(frame_bgr, [x1, y1, x2, y2]) -> "Red"

        # Settings (the Streamlit page updates these on every rerun)
        self.conf = 0.45
        self.imgsz = 320
        self.interval = 0.5             # minimum seconds between two detections

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

    # ---------- called from the video callback thread ----------

    def submit(self, frame_bgr):
        """Offer a frame for detection. Never blocks; most frames are skipped."""
        now = time.monotonic()
        self._last_frame_time = now
        self._ensure_running()

        if self._busy or (now - self._last_done) < self.interval:
            return

        self._busy = True
        with self._lock:
            self._latest_frame = frame_bgr.copy()
        self._new_frame.set()

    def draw(self, frame_bgr):
        """Draw the last known boxes on the frame (in place) and return it."""
        with self._lock:
            boxes = list(self._boxes)
            size = self._frame_size

        if not boxes or size is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        sx = w / float(size[0])
        sy = h / float(size[1])

        for x1, y1, x2, y2, text, is_person in boxes:
            colour = GREEN if is_person else ORANGE
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, 2)

            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - th - base - 4)
            cv2.rectangle(frame_bgr, (x1, top), (x1 + tw + 6, top + th + base + 4), colour, -1)
            cv2.putText(
                frame_bgr, text, (x1 + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
            )

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
            }

    def clear(self):
        """Forget old boxes (used when the stream stops)."""
        with self._lock:
            self._boxes = []
            self._counts = {}
            self._frame_size = None
            self._latest_frame = None

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

        with self._lock:
            self._boxes = boxes
            self._frame_size = (w, h)
            self._counts = counts
            self._infer_seconds = took
            self._detections_done += 1
            self._error = None


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


def run_live(yolo_model, conf_value, imgsz_value, colour_fn=None):
    """Draws the whole Live Webcam page. Call it from app.py."""
    import streamlit as st

    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
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

    interval = st.sidebar.slider(
        "Live: detect every (seconds)",
        min_value=0.2, max_value=2.0, value=0.5, step=0.1,
        help="Smaller = more updates but a slower server. Bigger = smoother video.",
    )

    # One detector per visitor session (never shared between visitors)
    if "live_detector" not in st.session_state:
        st.session_state.live_detector = LiveDetector(yolo_model, colour_fn)

    detector = st.session_state.live_detector
    detector.conf = conf_value
    detector.imgsz = imgsz_value
    detector.interval = interval

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        detector.submit(img)
        img = detector.draw(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    rtc_configuration, turn_ok = _rtc_configuration()

    ctx = webrtc_streamer(
        key="smartvision-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if turn_ok:
        st.caption("Connection: TURN relay is configured.")
    else:
        st.caption(
            "Connection: no TURN relay set up. This is fine on your own "
            "computer. The online app may need one if the video does not start."
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

            if info["infer_seconds"] is not None:
                st.caption(
                    f"One detection takes about {info['infer_seconds']:.2f} s "
                    f"on this server (updates every {interval:.1f} s or slower)."
                )

        time.sleep(0.5)

    detector.clear()