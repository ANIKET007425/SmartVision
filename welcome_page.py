"""The welcome (landing) page of SmartVision.

app.py uses it like this, right after the neon background code:

    from welcome_page import show_welcome, tighten_layout
    show_welcome()
    tighten_layout()

show_welcome() draws the page. If the visitor has not pressed "Start" yet, it
STOPS the script, so the AI models below it in app.py are not loaded yet.
When Start is pressed it returns and the normal app continues.

tighten_layout() is for all the other pages (detection, camera, live, how it
works). Streamlit leaves a big empty strip above and below the content, which
makes a page scroll even when there is nothing to see below. This makes the
strip smaller. It only runs after Start, because show_welcome() stops the
script before it on the welcome page.

Every new visit (a fresh link, a refresh, a bookmark) opens on THIS page. An old
address that ends with ?page=app does not skip it any more.

What this layout does on purpose:
  * the page is exactly one screen tall, so it does not scroll on a normal
    phone or laptop. Two things used to push it taller than the screen:
    Streamlit's empty space at the bottom, and the extra top/bottom margin
    that the neon-background CSS in app.py puts on the same box. Both are
    switched off here, and on a normal-height screen scrolling is locked;
  * everything is centred in the middle of the screen, both ways;
  * the intro animation only fades in (it no longer slides), so nothing jumps;
  * text sizes use clamp(), so they shrink on a phone and grow on a laptop;
  * the four small labels sit in a tidy 2 x 2 grid on a phone and in one row
    on a laptop; the button has one fixed width.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Rules for the HTML below (Streamlit reads it as Markdown first):
#   * no empty lines inside one HTML block (an empty line ends the block)
#   * no lines that start with spaces (4 spaces would turn into a code block)
# ---------------------------------------------------------------------------

STYLE = """
<style>
/* 1) The stage: as tall as the visible screen, content in the middle, no extra space below */
[data-testid="stMainBlockContainer"], .block-container {
  background: transparent; border: none; box-shadow: none; backdrop-filter: none;
  min-height: 100vh; min-height: 100dvh;
  padding: 3.75rem 1rem !important;
  margin-top: 0 !important; margin-bottom: 0 !important;
  display: flex; flex-direction: column; justify-content: center;
}
/* 1b) Normal-height screens: the page is exactly one screen, so lock the scrolling.
   (Very short screens, like a phone held sideways, are left alone so nothing gets cut off.) */
@media (min-height: 601px) {
  [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main { overflow: hidden !important; }
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 1.3rem; }
/* 2) The text block */
.sv-hero { width: 100%; max-width: 46rem; margin: 0 auto; padding-bottom: 1.5rem; text-align: center; animation: sv-fade 1.2s ease both; }
.sv-hero [data-testid="stHeaderActionElements"], .sv-hero h1 a { display: none !important; }
@keyframes sv-fade { from { opacity: 0; } to { opacity: 1; } }
.sv-hero p { margin: 0 !important; text-align: center; }
.sv-hero .sv-kicker { color: #5ee7ef; font-weight: 600; font-size: clamp(.7rem, 2.7vw, .85rem); letter-spacing: .3em; padding-left: .3em; line-height: 1.9; margin-bottom: .9rem !important; }
.sv-kicker span { display: inline-block; }
.sv-kicker .sv-dot { margin: 0 .55em; }
h1.sv-welcome { font-size: clamp(1.9rem, 8vw, 4.2rem); font-weight: 800; letter-spacing: .1em; padding: 0 0 0 .1em; margin: 0 0 1.1rem 0; line-height: 1.15; color: #fff; text-shadow: 0 0 22px rgba(123,77,255,.55); }
h1.sv-welcome span { color: #ff7ac6; text-shadow: 0 0 26px rgba(255,62,165,.9), 0 0 64px rgba(255,62,165,.45); animation: sv-flicker 7s infinite; }
@keyframes sv-flicker { 0%,18%,22%,24%,54%,57%,100% { opacity: 1; } 20%,23%,55.5% { opacity: .55; } }
.sv-hero .sv-sub { max-width: 34rem; margin: 0 auto 1.4rem !important; color: #d9d0ee; font-size: clamp(.95rem, 3.6vw, 1.1rem); line-height: 1.6; text-wrap: balance; }
/* 3) The four small labels: 2 x 2 on a phone, one row on a laptop */
.sv-chips { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; max-width: 26rem; margin: 0 auto; }
.sv-chips span { padding: .55rem .5rem; border-radius: 999px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.14); font-size: clamp(.8rem, 3.1vw, .95rem); text-align: center; line-height: 1.25; }
@media (min-width: 700px) {
  .sv-chips { grid-template-columns: repeat(4, max-content); justify-content: center; max-width: none; }
  .sv-chips span { padding: .5rem 1.1rem; white-space: nowrap; }
}
/* 4) The Start button: one fixed width, centred */
div.stButton { display: flex; justify-content: center; }
[data-testid="stElementContainer"]:has(.stButton), .element-container:has(.stButton) { width: 100% !important; display: flex; justify-content: center; }
div.stButton > button { width: min(20rem, 100%); background: linear-gradient(90deg, #ff3ea5, #7b4dff); border: 0; padding: .8rem 1.5rem; border-radius: 999px; box-shadow: 0 0 22px rgba(255,62,165,.55), 0 0 60px rgba(123,77,255,.35); transition: transform .2s ease, box-shadow .2s ease; }
div.stButton > button, div.stButton > button p { color: #fff !important; font-size: 1.1rem; font-weight: 700; letter-spacing: .08em; }
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 0 30px rgba(255,62,165,.8), 0 0 80px rgba(123,77,255,.5); }
/* 5) Very short screens (a phone held sideways): tighten everything, the page may scroll */
@media (max-height: 600px) {
  [data-testid="stMainBlockContainer"], .block-container { padding-top: 3.25rem !important; padding-bottom: 1rem !important; }
  h1.sv-welcome { font-size: clamp(1.5rem, 5vw, 2.4rem); margin-bottom: .6rem; }
  .sv-hero .sv-kicker { margin-bottom: .4rem !important; }
  .sv-hero .sv-sub { margin-bottom: .8rem !important; }
}
@media (max-height: 460px) { .sv-chips { display: none; } }
@media (max-width: 520px) {
  .sv-kicker span { display: block; }
  .sv-kicker .sv-dot { display: none; }
}
@media (prefers-reduced-motion: reduce) { .sv-hero, h1.sv-welcome span { animation: none; } }
</style>
"""

HERO = """
<div class="sv-hero">
<p class="sv-kicker"><span>AI VISION · OBJECTS</span><span class="sv-dot">·</span><span>EMOTION · AGE</span></p>
<h1 class="sv-welcome">WELCOME TO<br><span>SMART VISION</span></h1>
<p class="sv-sub">Upload a photo, use your camera or go live. SmartVision finds everyday objects, reads the emotion on a face and estimates an age range.</p>
<div class="sv-chips"><span>🎯 Object detection</span><span>🙂 Emotion analysis</span><span>🎂 Age range</span><span>🎥 Live webcam</span></div>
</div>
"""


TIGHT = """
<style>
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 4.5rem !important;
  padding-bottom: 2.5rem !important;
  margin-top: .5rem !important;
  margin-bottom: 1rem !important;
}
</style>
"""


def tighten_layout():
    """Make the empty strip above and below the content smaller (all pages after Start)."""
    st.markdown(TIGHT, unsafe_allow_html=True)


def show_welcome():
    """Draw the welcome page, or return at once if the visitor already started."""

    # A new visit always begins on the welcome page. (The Home button in the
    # sidebar sets this back to False, so it shows the welcome page again.)
    if "started" not in st.session_state:
        st.session_state.started = False

    # An empty box that we can clear instantly when Start is pressed
    landing = st.empty()

    if st.session_state.started:
        return

    with landing.container():
        st.markdown(STYLE, unsafe_allow_html=True)
        st.markdown(HERO, unsafe_allow_html=True)
        start_clicked = st.button("🚀 Start SmartVision")

    if start_clicked:
        st.session_state.started = True
        landing.empty()  # remove the welcome page right now
    else:
        # Stop here: the models are NOT loaded until the visitor presses Start
        st.stop()