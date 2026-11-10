"""The "How it works" page of SmartVision.

app.py shows this page when the visitor picks "ℹ️ How it works" in the mode list:

    elif mode == "ℹ️ How it works":
        from how_it_works import show_how_it_works
        show_how_it_works()

It is only text and styling. No AI model is loaded here, so the page opens
instantly. The facts below (classes, emotions, age groups, thresholds) are the
same ones app.py and live_webcam.py use - if you change them there, change
them here too.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Rules for the HTML below (Streamlit reads it as Markdown first):
#   * no empty lines inside one HTML block (an empty line ends the block)
#   * no lines that start with spaces (4 spaces would turn into a code block)
# ---------------------------------------------------------------------------

STYLE = """
<style>
.hiw { color: #d9d0ee; line-height: 1.6; }
.hiw h2.hiw-title { font-size: clamp(1.6rem, 4vw, 2.3rem); font-weight: 800; margin: 0 0 .4rem 0; padding: 0; color: #fff; text-shadow: 0 0 22px rgba(255,62,165,.45), 0 0 50px rgba(123,77,255,.35); }
.hiw p.hiw-lead { font-size: 1.05rem; max-width: 760px; margin: 0 0 1.4rem 0; }
.hiw h3.hiw-h { font-size: 1.25rem; font-weight: 700; margin: 2rem 0 .8rem 0; padding: 0; color: #5ee7ef; letter-spacing: .02em; }
.hiw .hiw-flow { display: flex; align-items: stretch; gap: .6rem; flex-wrap: wrap; }
.hiw .hiw-step, .hiw .hiw-card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); border-radius: 16px; padding: 1rem 1.1rem; backdrop-filter: blur(6px); transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; }
.hiw .hiw-step:hover, .hiw .hiw-card:hover { transform: translateY(-3px); border-color: rgba(255,122,198,.6); box-shadow: 0 0 22px rgba(255,62,165,.28); }
.hiw .hiw-step { flex: 1 1 200px; }
.hiw .hiw-step h4, .hiw .hiw-card h4 { margin: 0 0 .35rem 0; padding: 0; font-size: 1.08rem; color: #fff; }
.hiw .hiw-step p, .hiw .hiw-card p { margin: 0 0 .5rem 0; font-size: .96rem; }
.hiw .hiw-arrow { align-self: center; font-size: 1.5rem; color: #ff7ac6; text-shadow: 0 0 14px rgba(255,62,165,.8); }
.hiw .hiw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: .9rem; }
.hiw .hiw-card .hiw-tag { display: inline-block; font-size: .78rem; letter-spacing: .06em; color: #5ee7ef; border: 1px solid rgba(94,231,239,.45); border-radius: 999px; padding: .1rem .65rem; margin-bottom: .6rem; }
.hiw .hiw-card ul { margin: .2rem 0 0 0; padding-left: 1.15rem; }
.hiw .hiw-card li { margin-bottom: .3rem; font-size: .94rem; }
.hiw .hiw-live { counter-reset: hiw; list-style: none; margin: 0; padding: 0; display: grid; gap: .6rem; }
.hiw .hiw-live li { counter-increment: hiw; position: relative; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); border-radius: 14px; padding: .8rem 1rem .8rem 3.2rem; }
.hiw .hiw-live li::before { content: counter(hiw); position: absolute; left: .9rem; top: .75rem; width: 1.7rem; height: 1.7rem; border-radius: 50%; background: linear-gradient(135deg, #ff3ea5, #7b4dff); color: #fff; font-weight: 700; text-align: center; line-height: 1.7rem; box-shadow: 0 0 14px rgba(255,62,165,.6); }
.hiw .hiw-note { margin-top: .9rem; padding: .7rem 1rem; border-left: 3px solid #5ee7ef; background: rgba(94,231,239,.08); border-radius: 0 12px 12px 0; font-size: .95rem; }
.hiw .hiw-foot { margin-top: 2rem; text-align: center; font-size: .88rem; color: #a99fc4; }
@media (max-width: 700px) { .hiw .hiw-arrow { display: none; } }
@media (prefers-reduced-motion: reduce) { .hiw .hiw-step, .hiw .hiw-card { transition: none; } .hiw .hiw-step:hover, .hiw .hiw-card:hover { transform: none; } }
</style>
"""

INTRO = """
<div class="hiw">
<h2 class="hiw-title">ℹ️ How SmartVision works</h2>
<p class="hiw-lead">SmartVision chains a few small AI models together. Here is what happens between giving it a picture and seeing the results.</p>
<div class="hiw-flow">
<div class="hiw-step"><h4>📥 1. You give a picture</h4><p>Upload a photo, take one with your camera, or switch on the live webcam.</p></div>
<div class="hiw-arrow">➜</div>
<div class="hiw-step"><h4>🧠 2. The AI looks at it</h4><p>Up to three models run, and each one has a single job (see below).</p></div>
<div class="hiw-arrow">➜</div>
<div class="hiw-step"><h4>📋 3. You get the results</h4><p>Boxes with names and colours, an emotion score for every emotion, and an age range.</p></div>
</div>
</div>
"""

MODELS = """
<div class="hiw">
<h3 class="hiw-h">Three models, three jobs</h3>
<div class="hiw-grid">
<div class="hiw-card"><span class="hiw-tag">YOLO-World</span><h4>🎯 Finds objects</h4>
<p>We give it a list of words and it looks for them in the picture.</p>
<ul>
<li>Looks for: person, eyeglasses, pen, bag, cup, mobile phone, tie, bottle and book.</li>
<li>Draws a box around each thing it finds and gives the box a score from 0 to 1.</li>
<li>Boxes with a score below the <b>Confidence</b> slider are hidden.</li>
<li>For a tie it also names the colour, by averaging the colours inside the box.</li>
</ul></div>
<div class="hiw-card"><span class="hiw-tag">DeepFace</span><h4>🙂 Reads the emotion</h4>
<p>First it must find a face, then it scores the expression.</p>
<ul>
<li>A face finder (MTCNN) looks for a face. If it is less than 50% sure, you get "Face not detected" instead of a made-up answer. If there are several faces, the biggest one is used.</li>
<li>Then a neural network scores 7 emotions: angry, disgust, fear, happy, sad, surprise and neutral. It scores the face and its mirror image and averages the two, so one odd look cannot decide.</li>
<li>The scores add up to about 100%. The biggest one is the "Dominant Emotion". If the biggest score is below 40%, the app says "Not sure" and only shows its best guess.</li>
</ul></div>
<div class="hiw-card"><span class="hiw-tag">OpenCV age model</span><h4>🎂 Guesses an age range</h4>
<p>A small model that only looks at the face.</p>
<ul>
<li>The face is cut out of the original picture with extra room around it (hair, forehead and chin) and resized to 96 × 96 pixels. A small model (InsightFace "genderage", 1.3 MB) looks at 7 slightly different crops of it, each also as a mirror image, and the median of the 14 answers is the age in years. The app does not use or show its gender guess.</li>
<li>The age is then put into one of 8 groups: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53 and 60-100. If the age is close to the border of two groups, you see both, for example 25-32 or 38-43. The exact guess in years is shown too.</li>
<li>It is a guess, not an exact age. On a published test the average error of this kind of model was about 7-8 years, and some faces are off by much more. If the 14 answers differ a lot, the app tells you it is not very sure. (If the model file is missing, the app uses the old model instead: 8 groups, 10 looks at 227 × 227 pixels, "not sure" when its best score is below 50%.)</li>
</ul></div>
</div>
</div>
"""

LIVE = """
<div class="hiw">
<h3 class="hiw-h">🎥 What is different in Live Webcam?</h3>
<ol class="hiw-live">
<li><b>Your browser</b> opens the camera, after you press Allow.</li>
<li>The video travels to the SmartVision server using <b>WebRTC</b>, the same technology as video calls.</li>
<li>The server runs the object detector on the <b>newest frame</b> only, about every 0.5 seconds. Older frames are skipped, so the video never freezes while the AI is thinking.</li>
<li>If a person is in the picture, about once a second the server also looks at the <b>face</b> and estimates the emotion and an age range. The answers are averaged over a few seconds so one bad frame cannot change them.</li>
<li>The boxes are drawn on <b>every frame</b> and the video comes back to your screen, with the names listed below the video.</li>
</ol>
<p class="hiw-note">The <b>Video quality</b> choice sets the size of the camera picture. A small picture (Smooth) is the best choice on a phone or a slow network. The age and emotion in live mode are rough estimates and can be switched off in the sidebar if the video is not smooth. The video is processed in memory and thrown away. Nothing is recorded or saved, and pressing STOP ends it.</p>
</div>
"""

SLIDERS = """
- **Confidence:** every box gets a score from 0 to 1. Boxes below the slider are hidden. A higher value shows fewer but surer boxes. A lower value finds more things, but also makes more mistakes.
- **Image size (320 or 480):** the picture is shrunk to this size before the detector looks at it. 480 sees small things (like a pen) better. 320 is faster.
- **Live: detect every (seconds):** how often Live Webcam runs the detector. A smaller number updates the boxes more often, but the server works harder.
- **Live: show age and emotion:** switches the face estimates in Live Webcam on or off. Off = lighter for the server and smoother video.
"""

HOW_WORDS = """
Older detectors can only find the things they were trained on. YOLO-World also understands **words**.
It is paired with a text-and-image model (CLIP) that links words with what they look like, so
we can simply hand it a list such as "pen", "eyeglasses" or "book" and it looks for those, without
training a new model for each one. That is why adding a new object to SmartVision only needs a new word in the list.
"""

LIMITS = """
- **These are AI guesses, not facts.** The age is a range and the emotion is only the most likely expression.
  Light, camera angle, glasses, make-up and photo quality can all change the answer.
- **Small or partly hidden objects can be missed.** Try Image size 480 and move closer to the camera.
- **Your pictures are not saved.** A photo you upload or take is sent to the SmartVision server, analysed in memory
  and not stored by the app. Big photos are shrunk to at most 640 pixels when the face is searched (to save memory); the age and emotion are then read from the original picture.
- **The first start is slower.** The models are loaded once, and after that they are reused.
"""

FOOT = """
<div class="hiw"><p class="hiw-foot">Built with Streamlit · Ultralytics YOLO-World · DeepFace · OpenCV · streamlit-webrtc</p></div>
"""


def show_how_it_works():
    """Draw the whole page."""
    st.markdown(STYLE + INTRO, unsafe_allow_html=True)
    st.markdown(MODELS, unsafe_allow_html=True)
    st.markdown(LIVE, unsafe_allow_html=True)

    st.markdown('<div class="hiw"><h3 class="hiw-h">Good to know</h3></div>', unsafe_allow_html=True)
    with st.expander("What do the sliders in the sidebar do?"):
        st.markdown(SLIDERS)
    with st.expander("How can it find a pen if nobody trained it on pens?"):
        st.markdown(HOW_WORDS)
    with st.expander("Accuracy, limits and privacy"):
        st.markdown(LIMITS)

    st.markdown(FOOT, unsafe_allow_html=True)