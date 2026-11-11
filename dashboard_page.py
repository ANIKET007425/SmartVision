"""3D dashboard page for SmartVision.

How it is used (three small steps, see the bottom of this file for the exact lines):

  1. After every photo is analysed, app.py tells the dashboard what was found:

         from dashboard_page import record_photo, photo_key
         record_photo(details, emotion, key=photo_key(uploaded_file), face=True)

  2. A new page in the sidebar calls:

         from dashboard_page import show_dashboard
         show_dashboard()

  3. That is all. The numbers live only in this visit (st.session_state).
     Nothing is saved on disk, no photo is stored.

What the page shows
-------------------
  * four glass cards with big numbers (photos, objects, faces, most seen object);
  * two 3D bar charts that sway slowly: objects and emotions;
  * four small cards that describe the models and tools of the app.

It is only HTML + CSS (no JavaScript, no extra package), the same way the
welcome page works. The CSS 3D comes from `perspective` and `transform-style:
preserve-3d`: every bar is a box made of five flat faces.

The HTML is built by plain functions (build_dashboard_html) that do not need
Streamlit, so the tests can run them anywhere.
"""

import hashlib
import html
import re

# ---------------------------------------------------------------------------
# What the dashboard counts (the names are the ones the app already uses)
# ---------------------------------------------------------------------------

OBJECT_CLASSES = ["Person", "Glasses", "Pen", "Bag", "Cup", "Phone", "Tie", "Bottle", "Book"]
EMOTION_CLASSES = ["Happy", "Neutral", "Surprise", "Sad", "Angry", "Fear", "Disgust"]

_OBJECT_ALIASES = {
    "person": "Person",
    "eyeglasses": "Glasses", "spectacles": "Glasses", "glasses": "Glasses",
    "pen": "Pen", "ballpoint pen": "Pen",
    "bag": "Bag", "handbag": "Bag", "backpack": "Bag",
    "cup": "Cup", "coffee cup": "Cup", "mug": "Cup",
    "mobile phone": "Phone", "cell phone": "Phone", "phone": "Phone",
    "tie": "Tie",
    "bottle": "Bottle",
    "book": "Book",
}

MAX_BAR_HEIGHT = 170   # px, the tallest bar

SAMPLE_STATS = {
    "photos": 12,
    "faces": 9,
    "objects": {"Person": 9, "Glasses": 4, "Pen": 3, "Bag": 2, "Cup": 5,
                "Phone": 6, "Tie": 1, "Bottle": 3, "Book": 2},
    "emotions": {"Happy": 5, "Neutral": 4, "Surprise": 1, "Sad": 1,
                 "Angry": 0, "Fear": 0, "Disgust": 0},
}


# ---------------------------------------------------------------------------
# The numbers (kept in st.session_state, so every visitor has their own)
# ---------------------------------------------------------------------------

def new_stats():
    return {
        "photos": 0,
        "faces": 0,
        "objects": {name: 0 for name in OBJECT_CLASSES},
        "emotions": {name: 0 for name in EMOTION_CLASSES},
        "keys": set(),          # photos already counted (a rerun must not count twice)
    }


def get_stats():
    import streamlit as st
    if "sv_stats" not in st.session_state:
        st.session_state["sv_stats"] = new_stats()
    return st.session_state["sv_stats"]


def reset_stats():
    import streamlit as st
    st.session_state["sv_stats"] = new_stats()


def photo_key(uploaded):
    """A short id for an uploaded or camera photo, so it is counted only once.
    Streamlit runs the whole script again after every click, and the photo is
    still there each time."""
    for attr in ("file_id",):
        value = getattr(uploaded, attr, None)
        if value:
            return str(value)
    try:
        data = uploaded.getvalue()
    except Exception:
        return None
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Turning the app's own words into the dashboard's names
# ---------------------------------------------------------------------------

def normalize_object(name):
    return _OBJECT_ALIASES.get(str(name).strip().lower())


def labels_from_details(details):
    """The detection lines of app.py look like '👔 **Tie detected** — Colour: ...'
    or '- **Book** — Confidence: ...'. Returns the dashboard names, e.g. ['Tie']."""
    labels = []
    for line in details or []:
        text = str(line)
        match = re.search(r"\*\*\s*([A-Za-z ]+?)\s+detected\s*\*\*", text, re.I)
        if not match:
            match = re.search(r"^\W*\*\*\s*([A-Za-z ]+?)\s*\*\*\s*[—-]", text)
        if match:
            name = normalize_object(match.group(1))
            if name:
                labels.append(name)
    return labels


def normalize_emotion(value):
    if value is None:
        return None
    match = re.match(r"\s*([A-Za-z]+)", str(value))
    if not match:
        return None
    name = match.group(1).capitalize()
    return name if name in EMOTION_CLASSES else None


def record_photo(details=None, emotion=None, labels=None, key=None, face=False):
    """Count one analysed photo. Returns True if it was counted, False if this
    photo (same key) was counted before.

    details : the detection lines of app.py (list of text) - or give labels
    labels  : list of object names, for example ["person", "cup"]
    emotion : 'happy', 'Happy 87%', ... (anything that starts with the emotion)
    key     : photo_key(file), so a rerun does not count the same photo again
    face    : True if a face was found (counted even when the emotion is "Not sure")
    """
    stats = get_stats()
    if key is not None:
        if key in stats["keys"]:
            return False
        stats["keys"].add(key)

    found = list(labels_from_details(details))
    for label in labels or []:
        name = normalize_object(label)
        if name:
            found.append(name)
    for name in found:
        stats["objects"][name] += 1

    emotion_name = normalize_emotion(emotion)
    if emotion_name:
        stats["emotions"][emotion_name] += 1
    if emotion_name or face:
        stats["faces"] += 1

    stats["photos"] += 1
    return True


# ---------------------------------------------------------------------------
# The look
# ---------------------------------------------------------------------------
# Rules for the HTML (Streamlit reads it as Markdown first): no empty lines
# inside one block and no lines that start with spaces. _compact() takes care
# of that, so the text below can be written in a readable way.

STYLE = """
<style>
.sv3 { max-width: 72rem; margin: 0 auto; color: #efe9ff; }
.sv3 * { box-sizing: border-box; }
.sv3 p { margin: 0; }
/* ---- header with the turning cube ---- */
.sv3-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: 0 0 1.2rem; }
.sv3-kicker { color: #5ee7ef; font-weight: 600; font-size: clamp(.68rem, 2.6vw, .82rem); letter-spacing: .3em; margin: 0 0 .5rem; }
.sv3-title { font-size: clamp(1.8rem, 7vw, 3.1rem); font-weight: 800; letter-spacing: .1em; line-height: 1.1; color: #fff; text-shadow: 0 0 22px rgba(123,77,255,.55); }
.sv3-title span { color: #ff7ac6; text-shadow: 0 0 26px rgba(255,62,165,.9), 0 0 64px rgba(255,62,165,.45); }
.sv3-sub { margin: .7rem 0 0; max-width: 36rem; color: #d9d0ee; font-size: clamp(.88rem, 3.3vw, 1rem); line-height: 1.55; }
.sv3-cubewrap { flex: none; width: 84px; height: 84px; perspective: 520px; }
.sv3-cube { position: relative; width: 100%; height: 100%; transform-style: preserve-3d; animation: sv3-spin 14s linear infinite; }
.sv3-cube i { position: absolute; display: block; inset: 0; border: 1.5px solid rgba(94,231,239,.85); background: rgba(123,77,255,.16); box-shadow: inset 0 0 18px rgba(255,62,165,.4), 0 0 14px rgba(94,231,239,.35); }
.sv3-cube i:nth-child(1) { transform: translateZ(42px); }
.sv3-cube i:nth-child(2) { transform: rotateY(180deg) translateZ(42px); }
.sv3-cube i:nth-child(3) { transform: rotateY(90deg) translateZ(42px); }
.sv3-cube i:nth-child(4) { transform: rotateY(-90deg) translateZ(42px); }
.sv3-cube i:nth-child(5) { transform: rotateX(90deg) translateZ(42px); }
.sv3-cube i:nth-child(6) { transform: rotateX(-90deg) translateZ(42px); }
@keyframes sv3-spin { from { transform: rotateX(-24deg) rotateY(0deg); } to { transform: rotateX(-24deg) rotateY(360deg); } }
/* ---- glass cards that tilt in 3D when the mouse is over them ---- */
.sv3-card { position: relative; overflow: hidden; border-radius: 1.1rem; padding: 1rem 1.1rem; background: linear-gradient(145deg, rgba(255,255,255,.10), rgba(255,255,255,.03)); border: 1px solid rgba(255,255,255,.15); box-shadow: 0 18px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.14); transform: perspective(800px) rotateX(0deg) rotateY(0deg); transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease; }
.sv3-card:hover { transform: perspective(800px) rotateX(6deg) rotateY(-8deg) translateY(-5px); border-color: rgba(94,231,239,.55); box-shadow: 18px 26px 50px rgba(0,0,0,.45), 0 0 30px rgba(123,77,255,.38); }
.sv3-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: 0 0 1rem; }
.sv3-kpi .n { font-size: clamp(1.7rem, 6vw, 2.6rem); font-weight: 800; line-height: 1.1; color: #fff; text-shadow: 0 0 18px rgba(255,62,165,.55); overflow-wrap: anywhere; }
.sv3-kpi .l { margin-top: .4rem; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #bfb4dd; }
.sv3-kpi:nth-child(2) .n { text-shadow: 0 0 18px rgba(123,77,255,.75); }
.sv3-kpi:nth-child(3) .n { text-shadow: 0 0 18px rgba(94,231,239,.65); }
.sv3-banner { display: inline-block; margin: 0 0 .9rem; padding: .35rem .9rem; border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .12em; color: #5ee7ef; background: rgba(94,231,239,.1); border: 1px solid rgba(94,231,239,.4); }
/* ---- the 3D bar charts ---- */
.sv3-charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem; margin: 0 0 1rem; }
.sv3-charts .wide { grid-column: 1 / -1; }
.sv3-ct { font-weight: 700; letter-spacing: .08em; color: #fff; }
.sv3-ch { margin: .15rem 0 0; color: #bfb4dd; font-size: .82rem; }
.sv3-scene { perspective: 1500px; perspective-origin: 50% 20%; height: 345px; margin-top: .4rem; overflow: hidden; }
.sv3-stage { position: relative; height: 100%; transform-style: preserve-3d; transform: rotateX(-16deg) rotateY(-12deg); animation: sv3-sway 16s ease-in-out infinite alternate; }
@keyframes sv3-sway { from { transform: rotateX(-16deg) rotateY(-19deg); } to { transform: rotateX(-16deg) rotateY(-5deg); } }
.sv3-row { position: absolute; left: 5%; right: 5%; bottom: 104px; height: 215px; display: flex; align-items: flex-end; transform-style: preserve-3d; }
.sv3-col { position: relative; flex: 1 1 0; min-width: 0; height: 100%; transform-style: preserve-3d; }
.sv3-bar { --w: 30px; --d: 30px; position: absolute; left: 50%; bottom: 0; width: var(--w); margin-left: calc(var(--w) / -2); transform-style: preserve-3d; transform-origin: 50% 100%; animation: sv3-grow .95s cubic-bezier(.2,.9,.3,1) both; }
.sv3-bar i { position: absolute; display: block; }
.sv3-bar .f { left: 0; top: 0; width: var(--w); height: 100%; background: var(--f); transform: translateZ(calc(var(--d) / 2)); box-shadow: inset 0 0 14px rgba(255,255,255,.12), 0 0 18px var(--g); }
.sv3-bar .b { left: 0; top: 0; width: var(--w); height: 100%; background: var(--s); transform: rotateY(180deg) translateZ(calc(var(--d) / 2)); }
.sv3-bar .l { left: calc((var(--w) - var(--d)) / 2); top: 0; width: var(--d); height: 100%; background: var(--s); transform: rotateY(-90deg) translateZ(calc(var(--w) / 2)); }
.sv3-bar .r { left: calc((var(--w) - var(--d)) / 2); top: 0; width: var(--d); height: 100%; background: var(--s); transform: rotateY(90deg) translateZ(calc(var(--w) / 2)); }
.sv3-bar .t { left: 0; top: calc(var(--d) / -2); width: var(--w); height: var(--d); background: var(--t); transform: rotateX(90deg); }
.sv3-bar.k0 { --f: linear-gradient(180deg, #ff6bbd, #d4157f); --s: #a30c60; --t: #ffa3d6; --g: rgba(255,62,165,.55); }
.sv3-bar.k1 { --f: linear-gradient(180deg, #9a78ff, #5a33d6); --s: #3f22a3; --t: #c2adff; --g: rgba(123,77,255,.6); }
.sv3-bar.k2 { --f: linear-gradient(180deg, #4ff0f3, #12a9b4); --s: #0b7580; --t: #a8fbfd; --g: rgba(94,231,239,.5); }
.sv3-bar.zero { opacity: .55; }
@keyframes sv3-grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
.sv3-val { position: absolute; left: -10px; right: -10px; text-align: center; font-weight: 700; font-size: .85rem; color: #fff; text-shadow: 0 0 10px rgba(255,62,165,.8); animation: sv3-pop .6s ease .6s both; }
@keyframes sv3-pop { from { opacity: 0; } to { opacity: 1; } }
.sv3-name { position: absolute; left: -12px; right: -12px; top: calc(100% + 12px); text-align: center; font-size: .78rem; font-weight: 600; color: #ece6ff; white-space: nowrap; transform: translateZ(45px); }
.sv3-floor { position: absolute; left: -4%; right: -4%; top: 100%; height: 130px; transform-origin: 50% 0; transform: translateZ(-70px) rotateX(90deg); border-radius: 14px; border: 1px solid rgba(94,231,239,.4); background: linear-gradient(rgba(94,231,239,.18) 1px, transparent 1px) 0 0 / 100% 26px, linear-gradient(90deg, rgba(94,231,239,.18) 1px, transparent 1px) 0 0 / 26px 100%, radial-gradient(ellipse at 50% 45%, rgba(123,77,255,.38), rgba(123,77,255,.05) 72%); box-shadow: 0 0 40px rgba(123,77,255,.38); }
/* ---- the four model cards ---- */
.sv3-sys { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .9rem; margin: 0 0 1rem; }
.sv3-sys .k { color: #5ee7ef; font-size: .7rem; font-weight: 700; letter-spacing: .22em; }
.sv3-sys .v { margin: .35rem 0 .45rem; font-size: clamp(1.25rem, 4.5vw, 1.7rem); font-weight: 800; color: #fff; text-shadow: 0 0 16px rgba(123,77,255,.7); }
.sv3-sys .d { color: #cfc6e6; font-size: .84rem; line-height: 1.5; }
.sv3-note { margin: .4rem 0 0; color: #a79ecb; font-size: .8rem; line-height: 1.55; text-align: center; }
/* ---- phones ---- */
@media (max-width: 760px) {
  .sv3-kpis, .sv3-sys { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .sv3-charts { grid-template-columns: minmax(0, 1fr); }
  .sv3-cubewrap { width: 62px; height: 62px; }
  .sv3-cube i:nth-child(1) { transform: translateZ(31px); }
  .sv3-cube i:nth-child(2) { transform: rotateY(180deg) translateZ(31px); }
  .sv3-cube i:nth-child(3) { transform: rotateY(90deg) translateZ(31px); }
  .sv3-cube i:nth-child(4) { transform: rotateY(-90deg) translateZ(31px); }
  .sv3-cube i:nth-child(5) { transform: rotateX(90deg) translateZ(31px); }
  .sv3-cube i:nth-child(6) { transform: rotateX(-90deg) translateZ(31px); }
  .sv3-bar { --w: 20px; --d: 20px; }
  .sv3-name { font-size: .54rem; left: -6px; right: -6px; }
  .sv3-card.wide .sv3-name { font-size: .49rem; letter-spacing: -.01em; }
  .sv3-val { font-size: .72rem; }
  .sv3-card { padding: .85rem .9rem; }
}
@media (prefers-reduced-motion: reduce) {
  .sv3-cube, .sv3-stage, .sv3-bar, .sv3-val { animation: none; }
}
</style>
"""


def _compact(text):
    """Remove empty lines and the spaces at the start of lines (see the note above)."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _kpi(number, label):
    return f'<div class="sv3-card sv3-kpi"><p class="n">{html.escape(str(number))}</p><p class="l">{html.escape(label)}</p></div>'


def _bar(index, name, value, top):
    if value <= 0:
        height, extra = 6, " zero"
    else:
        height, extra = max(16, round(MAX_BAR_HEIGHT * value / top)), ""
    delay = index * 70
    return (
        '<div class="sv3-col">'
        f'<span class="sv3-val" style="bottom:{height + 26}px">{value}</span>'
        f'<div class="sv3-bar k{index % 3}{extra}" style="height:{height}px;animation-delay:{delay}ms">'
        '<i class="f"></i><i class="b"></i><i class="l"></i><i class="r"></i><i class="t"></i></div>'
        f'<span class="sv3-name">{html.escape(name)}</span>'
        '</div>'
    )


def build_chart_html(title, hint, names, values, wide=False):
    top = max(values) if values else 0
    bars = "".join(_bar(i, n, int(v), top) for i, (n, v) in enumerate(zip(names, values)))
    css = "sv3-card wide" if wide else "sv3-card"
    return (
        f'<div class="{css}">'
        f'<p class="sv3-ct">{html.escape(title)}</p>'
        f'<p class="sv3-ch">{html.escape(hint)}</p>'
        '<div class="sv3-scene"><div class="sv3-stage"><div class="sv3-row">'
        f'{bars}<div class="sv3-floor"></div>'
        '</div></div></div></div>'
    )


def _top_name(counts):
    if not counts or max(counts.values()) <= 0:
        return "-"
    return max(counts, key=lambda name: counts[name])


def build_dashboard_html(stats, sample=False):
    """The whole dashboard as one HTML text. `stats` looks like new_stats()."""
    objects = [stats["objects"].get(n, 0) for n in OBJECT_CLASSES]
    emotions = [stats["emotions"].get(n, 0) for n in EMOTION_CLASSES]

    parts = ['<div class="sv3">']
    parts.append(
        '<div class="sv3-head"><div>'
        '<p class="sv3-kicker">LIVE STATS · THIS VISIT</p>'
        '<div class="sv3-title">SMART <span>DASHBOARD</span></div>'
        '<p class="sv3-sub">Everything SmartVision found in the photos of this visit: '
        'objects and emotions, drawn as 3D bars.</p>'
        '</div>'
        '<div class="sv3-cubewrap"><div class="sv3-cube"><i></i><i></i><i></i><i></i><i></i><i></i></div></div>'
        '</div>'
    )
    if sample:
        parts.append('<div class="sv3-banner">SAMPLE DATA · NOT FROM REAL PHOTOS</div>')
    parts.append(
        '<div class="sv3-kpis">'
        + _kpi(stats.get("photos", 0), "Photos analysed")
        + _kpi(sum(objects), "Objects found")
        + _kpi(stats.get("faces", 0), "Faces read")
        + _kpi(_top_name(stats["objects"]), "Most seen object")
        + '</div>'
    )
    parts.append(
        '<div class="sv3-charts">'
        + build_chart_html("OBJECTS", "How many times each object was found", OBJECT_CLASSES, objects, wide=True)
        + build_chart_html("EMOTIONS", "Dominant emotion of each face", EMOTION_CLASSES, emotions, wide=True)
        + '</div>'
    )
    parts.append(
        '<div class="sv3-sys">'
        '<div class="sv3-card"><p class="k">OBJECTS</p><p class="v">9 kinds</p>'
        '<p class="d">YOLO-World finds a person, glasses, pen, bag, cup, phone, tie, bottle and book.</p></div>'
        '<div class="sv3-card"><p class="k">EMOTION</p><p class="v">7 emotions</p>'
        '<p class="d">DeepFace reads the face. The answer is an estimate, not a fact.</p></div>'
        '<div class="sv3-card"><p class="k">SECURITY LAB</p><p class="v">4 tools</p>'
        '<p class="d">Hidden message, QR reader, file fingerprint and image facts. These are calculations, not guesses.</p></div>'
        '<div class="sv3-card"><p class="k">LIVE WEBCAM</p><p class="v">~0.06 s</p>'
        '<p class="d">One detection on the cloud server (a single test at image size 320).</p></div>'
        '</div>'
    )
    parts.append(
        '<p class="sv3-note">The numbers live only in this visit. The app code does not store photos or video. '
        'Emotion is an estimate and can be wrong.</p>'
    )
    parts.append('</div>')
    return _compact("\n".join(parts))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def show_dashboard():
    """Draw the dashboard page (call it from the sidebar menu of app.py)."""
    import streamlit as st

    stats = get_stats()
    st.markdown(_compact(STYLE), unsafe_allow_html=True)

    sample = False
    if stats["photos"] == 0:
        st.info("No photo has been analysed in this visit yet. Analyse a photo in "
                "Upload Image or Camera Capture and come back, or look at sample data first.")
        sample = st.checkbox("Show sample data (for a demo)", value=False, key="sv_dash_sample")
    show = SAMPLE_STATS if sample else stats
    st.markdown(build_dashboard_html(show, sample=sample), unsafe_allow_html=True)

    if stats["photos"] > 0 and st.button("Reset dashboard numbers", key="sv_dash_reset"):
        reset_stats()
        st.rerun()


# ---------------------------------------------------------------------------
# The three lines for app.py (the exact place is shown in the chat)
# ---------------------------------------------------------------------------
#
#   from dashboard_page import show_dashboard, record_photo, photo_key
#
#   # after the photo was analysed (Upload Image and Camera Capture):
#   record_photo(details, dominant, key=photo_key(uploaded_file), face=True)
#
#   # in the sidebar menu: a new page "📊 Dashboard" that calls
#   show_dashboard()