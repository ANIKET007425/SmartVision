import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  

import gc
import streamlit as st
# Live webcam: this video library MUST be loaded before cv2 / ultralytics / deepface
# (deepface loads TensorFlow). In the other order Streamlit Cloud crashes (segfault).
try:
    import streamlit_webrtc  # noqa: F401
except Exception:
    pass
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from deepface import DeepFace

# ---------------- Page Setup ----------------

st.set_page_config(
    page_title="SmartVision",
    page_icon="🧠",
    layout="wide"
)
# ---------------- Live Background (original neon-city look) ----------------

st.markdown(
    """
<style>
:root { --bg-dark: #0b0716; }
/* ---- base sky (static) ---- */
.stApp {
  background:
    radial-gradient(ellipse 90% 50% at 50% 105%, rgba(255,120,40,.78), rgba(255,60,120,.28) 45%, transparent 72%),
    linear-gradient(170deg, #0b0716 0%, #1a0b2e 45%, #2b0f2e 75%, #0b0716 100%);
  background-attachment: fixed;
}
/* ---- layer 1: drifting neon haze (pink / violet / ember / teal) ---- */
.stApp::before {
  content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(40% 35% at 20% 25%, rgba(255,62,165,.40), transparent 70%),
    radial-gradient(38% 40% at 80% 30%, rgba(123,77,255,.44), transparent 70%),
    radial-gradient(35% 30% at 55% 72%, rgba(255,90,60,.30), transparent 70%),
    radial-gradient(30% 28% at 28% 80%, rgba(25,211,218,.24), transparent 70%);
  animation: sv-drift 38s ease-in-out infinite alternate;
  will-change: transform;
}
@keyframes sv-drift {
  0%   { transform: translate3d(-3%, -2%, 0) rotate(0deg) scale(1); }
  100% { transform: translate3d(4%, 3%, 0) rotate(6deg) scale(1.08); }
}
/* ---- layer 2: falling rain ---- */
.stApp::after {
  content: ""; position: fixed; inset: -30%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 1px 28px at 112px 52px, rgba(190,215,255,0.4), transparent),
    radial-gradient(ellipse 1px 24px at 42px 162px, rgba(190,215,255,0.5), transparent),
    radial-gradient(ellipse 1px 26px at 190px 63px, rgba(190,215,255,0.5), transparent),
    radial-gradient(ellipse 1px 23px at 139px 60px, rgba(190,215,255,0.5), transparent),
    radial-gradient(ellipse 1px 18px at 108px 173px, rgba(190,215,255,0.3), transparent),
    radial-gradient(ellipse 1px 28px at 152px 155px, rgba(190,215,255,0.3), transparent),
    radial-gradient(ellipse 1px 28px at 43px 129px, rgba(190,215,255,0.5), transparent),
    radial-gradient(ellipse 1px 19px at 22px 137px, rgba(190,215,255,0.3), transparent),
    radial-gradient(ellipse 1px 16px at 103px 44px, rgba(190,215,255,0.5), transparent),
    radial-gradient(ellipse 1px 23px at 160px 88px, rgba(190,215,255,0.4), transparent),
    radial-gradient(ellipse 1px 22px at 38px 68px, rgba(190,215,255,0.3), transparent);
  background-size: 240px 240px;
  transform: rotate(12deg);
  animation: sv-rain 1.6s linear infinite;
  opacity: .55;
  will-change: transform;
}
@keyframes sv-rain {
  from { transform: rotate(12deg) translateY(0); }
  to   { transform: rotate(12deg) translateY(240px); }
}
/* ---- layer 3: city skyline with glowing windows ---- */
[data-testid="stAppViewContainer"]::before {
  content: ""; position: fixed; left: 0; right: 0; bottom: 0; height: 42vh;
  z-index: 0; pointer-events: none;
  background: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%201600%20420'%20preserveAspectRatio='xMidYMax%20slice'%3E%3Crect%20x='-20'%20y='322'%20width='91'%20height='98'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='71'%20y='223'%20width='59'%20height='197'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='100'%20y='201'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='134'%20y='231'%20width='57'%20height='189'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='162'%20y='209'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='191'%20y='253'%20width='105'%20height='167'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='243'%20y='231'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='296'%20y='252'%20width='120'%20height='168'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='356'%20y='230'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='420'%20y='303'%20width='65'%20height='117'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='489'%20y='259'%20width='57'%20height='161'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='517'%20y='237'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='547'%20y='326'%20width='55'%20height='94'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='574'%20y='304'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='603'%20y='330'%20width='119'%20height='90'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='726'%20y='334'%20width='73'%20height='86'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='804'%20y='265'%20width='74'%20height='155'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='841'%20y='243'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='883'%20y='345'%20width='58'%20height='75'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='944'%20y='251'%20width='118'%20height='169'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1065'%20y='268'%20width='108'%20height='152'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1119'%20y='246'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1179'%20y='298'%20width='73'%20height='122'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1215'%20y='276'%20width='2'%20height='22'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1254'%20y='234'%20width='117'%20height='186'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1376'%20y='287'%20width='107'%20height='133'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1483'%20y='229'%20width='65'%20height='191'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='1554'%20y='322'%20width='93'%20height='98'%20fill='%233a1550'%20opacity='0.45'/%3E%3Crect%20x='-20'%20y='291'%20width='49'%20height='129'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='12'%20y='329'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='3'%20y='357'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='32'%20y='118'%20width='79'%20height='302'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='91'%20y='128'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='55'%20y='198'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='82'%20y='226'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='91'%20y='310'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='37'%20y='380'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='116'%20y='96'%20width='89'%20height='324'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='175'%20y='120'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='139'%20y='134'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='130'%20y='176'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='139'%20y='260'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='175'%20y='260'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='193'%20y='274'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='121'%20y='302'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='148'%20y='330'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='157'%20y='330'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='175'%20y='330'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='166'%20y='358'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='209'%20y='62'%20width='61'%20height='358'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='232'%20y='72'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='241'%20y='86'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='232'%20y='100'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='223'%20y='114'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='223'%20y='128'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='241'%20y='128'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='241'%20y='156'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='214'%20y='184'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='241'%20y='240'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='214'%20y='282'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='250'%20y='310'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='223'%20y='324'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='232'%20y='338'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='241'%20y='338'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='232'%20y='366'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='232'%20y='380'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='232'%20y='394'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='272'%20y='178'%20width='80'%20height='242'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='312'%20y='156'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='277'%20y='230'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='304'%20y='272'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='286'%20y='300'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='304'%20y='314'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='277'%20y='328'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='313'%20y='342'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='277'%20y='370'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='340'%20y='398'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='353'%20y='100'%20width='94'%20height='320'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='430'%20y='110'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='376'%20y='166'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='394'%20y='180'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='430'%20y='180'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='358'%20y='208'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='385'%20y='222'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='358'%20y='236'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='376'%20y='292'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='394'%20y='292'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='421'%20y='320'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='385'%20y='334'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='403'%20y='334'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='421'%20y='348'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='367'%20y='362'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='403'%20y='376'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='453'%20y='176'%20width='69'%20height='244'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='487'%20y='154'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='485'%20y='186'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='485'%20y='214'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='476'%20y='270'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='494'%20y='284'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='485'%20y='298'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='476'%20y='312'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='458'%20y='326'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='458'%20y='354'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='503'%20y='396'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='528'%20y='46'%20width='101'%20height='374'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='614'%20y='84'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='98'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='560'%20y='112'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='587'%20y='112'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='542'%20y='126'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='614'%20y='126'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='560'%20y='140'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='569'%20y='140'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='569'%20y='154'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='560'%20y='182'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='533'%20y='224'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='533'%20y='238'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='578'%20y='238'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='533'%20y='252'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='266'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='280'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='569'%20y='280'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='294'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='308'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='578'%20y='308'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='551'%20y='350'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='569'%20y='364'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='634'%20y='246'%20width='57'%20height='174'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='648'%20y='284'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='657'%20y='298'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='691'%20y='52'%20width='76'%20height='368'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='729'%20y='30'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='705'%20y='76'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='741'%20y='90'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='696'%20y='104'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='723'%20y='174'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='714'%20y='216'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='723'%20y='272'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='705'%20y='314'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='714'%20y='328'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='714'%20y='384'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='741'%20y='384'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='696'%20y='398'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='772'%20y='222'%20width='54'%20height='198'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='799'%20y='200'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='804'%20y='246'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='804'%20y='274'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='777'%20y='302'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='795'%20y='330'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='813'%20y='386'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='777'%20y='400'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='828'%20y='81'%20width='65'%20height='339'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='878'%20y='133'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='869'%20y='161'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='869'%20y='217'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='833'%20y='287'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='860'%20y='301'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='851'%20y='343'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='851'%20y='357'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='851'%20y='399'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='878'%20y='399'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='898'%20y='302'%20width='105'%20height='118'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='950'%20y='280'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='912'%20y='312'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='948'%20y='326'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='966'%20y='340'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='948'%20y='354'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='912'%20y='382'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='1007'%20y='104'%20width='61'%20height='316'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1037'%20y='82'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1039'%20y='170'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1021'%20y='184'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1021'%20y='198'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1030'%20y='212'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1039'%20y='226'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1021'%20y='254'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1030'%20y='268'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1039'%20y='282'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1048'%20y='352'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1039'%20y='366'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1048'%20y='394'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1069'%20y='203'%20width='89'%20height='217'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1083'%20y='213'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1101'%20y='213'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1092'%20y='227'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1101'%20y='227'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1110'%20y='297'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1101'%20y='311'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1119'%20y='311'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1137'%20y='311'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1119'%20y='381'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1160'%20y='112'%20width='70'%20height='308'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1174'%20y='122'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1174'%20y='248'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1183'%20y='248'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1210'%20y='248'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1183'%20y='262'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1201'%20y='304'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1201'%20y='318'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1174'%20y='332'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='1201'%20y='332'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1201'%20y='346'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1232'%20y='235'%20width='92'%20height='185'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1278'%20y='213'%20width='2'%20height='22'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1300'%20y='259'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='1291'%20y='315'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1309'%20y='315'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1282'%20y='329'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1309'%20y='357'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1237'%20y='385'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1330'%20y='309'%20width='80'%20height='111'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1371'%20y='319'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1398'%20y='347'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='1380'%20y='361'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1371'%20y='375'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1380'%20y='389'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1415'%20y='262'%20width='82'%20height='158'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1420'%20y='300'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1438'%20y='314'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1474'%20y='328'%20width='4'%20height='6'%20fill='%23ffb347'%20opacity='0.9'/%3E%3Crect%20x='1474'%20y='342'%20width='4'%20height='6'%20fill='%23ffd9a0'%20opacity='0.9'/%3E%3Crect%20x='1438'%20y='384'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1483'%20y='384'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3Crect%20x='1500'%20y='285'%20width='95'%20height='135'%20fill='%2307030f'%20opacity='1.0'/%3E%3Crect%20x='1541'%20y='295'%20width='4'%20height='6'%20fill='%23ff5a5a'%20opacity='0.9'/%3E%3Crect%20x='1559'%20y='337'%20width='4'%20height='6'%20fill='%235ee7ef'%20opacity='0.9'/%3E%3Crect%20x='1568'%20y='351'%20width='4'%20height='6'%20fill='%23ff6ec7'%20opacity='0.9'/%3E%3C/svg%3E") bottom center / cover no-repeat;
  -webkit-mask-image: linear-gradient(to top, #000 65%, transparent);
          mask-image: linear-gradient(to top, #000 65%, transparent);
  animation: sv-breathe 7s ease-in-out infinite alternate;
}
@keyframes sv-breathe { from { opacity: .80; } to { opacity: 1; } }
/* ---- layer 4: scanning line (SmartVision "vision" touch) ---- */
[data-testid="stAppViewContainer"]::after {
  content: ""; position: fixed; left: 0; right: 0; top: 0; height: 170px;
  z-index: 0; pointer-events: none;
  background: linear-gradient(to bottom,
    transparent 0%, rgba(25,211,218,.10) 55%, rgba(25,211,218,.22) 90%,
    rgba(150,255,255,.75) 99%, transparent 100%);
  animation: sv-scan 10s linear infinite;
  will-change: transform;
}
@keyframes sv-scan {
  0%   { transform: translateY(-180px); }
  70%  { transform: translateY(100vh); }
  100% { transform: translateY(100vh); }
}
/* ---- make the real app readable on top ---- */
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMain"], section.main { position: relative; z-index: 1; }
.block-container, [data-testid="stMainBlockContainer"] {
  background: rgba(10,7,22,.60);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 18px;
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 40px rgba(0,0,0,.35);
  margin-top: 1rem; margin-bottom: 2rem;
}
[data-testid="stSidebar"] {
  background: rgba(12,8,24,.74);
  backdrop-filter: blur(8px);
  border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebarContent"] { background: transparent; }
h1 { text-shadow: 0 0 18px rgba(255,62,165,.55); }
/* ---- respect users who turn animations off ---- */
@media (prefers-reduced-motion: reduce) {
  .stApp::before, .stApp::after,
  [data-testid="stAppViewContainer"]::before,
  [data-testid="stAppViewContainer"]::after { animation: none; }
}
</style>
""",
    unsafe_allow_html=True
)
# ---------------- Welcome (landing) page ----------------

from welcome_page import show_welcome, tighten_layout
show_welcome()
tighten_layout()

st.title("🧠 SmartVision")
st.write(
    " Upload an image, take a photo, or use the live webcam. "
    " Detection of Person, eyeglasses, pen, bag, cup, mobile phone, tie, bottle and book. "
    " You will also get the emotion percentage and an estimated age range from the face."
)
# ---------------- Load YOLO-World Model ----------------

@st.cache_resource
def load_yolo_world():
    model = YOLOWorld("yolov8s-worldv2.pt")

    custom_classes = [
        "person",
        "eyeglasses",
        "spectacles",
        "pen",
        "ballpoint pen",
        "bag",
        "cup",
        "mobile phone",
        "cell phone",
        "tie",
        "bottle",
        "book",
    ]

    model.set_classes(custom_classes)
    return model

yolo_model = load_yolo_world()

# ---------------- Age Model (small, OpenCV) ----------------

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")

AGE_RANGES = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

@st.cache_resource
def load_age_model():
    # If the model files are missing, the app still runs (age is just skipped)
    if not (os.path.exists(AGE_PROTO) and os.path.exists(AGE_MODEL)):
        return None
    try:
        return cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    except Exception as e:
        # Do not crash the whole app if this OpenCV version cannot load the model
        print("Age model could not be loaded:", type(e).__name__, e)
        return None

age_net = load_age_model()

def estimate_age(frame, region):
    if age_net is None:
        return None

    h, w = frame.shape[:2]
    x, y = int(region["x"]), int(region["y"])
    fw, fh = int(region["w"]), int(region["h"])

    # Take a little extra area around the face
    pad = int(0.2 * max(fw, fh))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)

    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return None

    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227), AGE_MEAN, swapRB=False
    )
    age_net.setInput(blob)
    predictions = age_net.forward()[0]

    return AGE_RANGES[int(predictions.argmax())]

# ---------------- Helper: Tie Colour ----------------

def get_dominant_color_name(frame, box):
    x1, y1, x2, y2 = map(int, box)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h = float(np.mean(hsv[:, :, 0]))
    s = float(np.mean(hsv[:, :, 1]))
    v = float(np.mean(hsv[:, :, 2]))

    if v < 50:
        return "Black"
    if s < 40:
        return "Grey"

    if h < 10 or h >= 170:
        return "Red"
    elif h < 25:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 100:
        return "Cyan"
    elif h < 130:
        return "Blue"
    elif h < 150:
        return "Purple"
    else:
        return "Pink"

# ---------------- Sidebar: Detection Settings ----------------
if st.sidebar.button("🏠 Home"):
       st.session_state.started = False
       st.query_params.clear()
       st.rerun()

# Click the title to open or close the settings (they stay closed until clicked)
with st.sidebar.expander("⚙️ Detection Settings", expanded=False):

    conf_value = st.slider(
        "Confidence (higher = fewer but surer detections)",
        min_value=0.10,
        max_value=0.80,
        value=0.40,
        step=0.05
    )

    imgsz_value = st.selectbox(
        "Image size (bigger = better for small objects, but slower)",
        [320, 480],
        index=0
    )

# ---------------- Helper: Object Analysis ----------------

def analyze_objects(frame):
    results = yolo_model.predict(
        frame,
        conf=conf_value,
        imgsz=imgsz_value,
        agnostic_nms=True,
        verbose=False
    )

    annotated_frame = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    details = []

    for result in results:
        for box in result.boxes:
            label = yolo_model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            box_coords = box.xyxy[0].tolist()

            if label == "tie":
                tie_colour = get_dominant_color_name(frame, box_coords)

                details.append(
                    f"👔 **Tie detected** — Colour: **{tie_colour}** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["eyeglasses", "spectacles"]:
                details.append(
                    f"👓 **Eyeglasses detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["pen", "ballpoint pen"]:
                details.append(
                    f"🖊️ **Pen detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["bag", "handbag", "backpack"]:
                details.append(
                    f"🎒 **Bag detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["coffee cup", "cup"]:
                details.append(
                    f"☕ **Cup detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label in ["mobile phone", "cell phone"]:
                details.append(
                    f"📱 **Mobile phone detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label == "bottle":
                details.append(
                    f"🍶 **Bottle detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            elif label == "person":
                details.append(
                    f"🧍 **Person detected** — "
                    f"Confidence: **{confidence:.2f}**"
                )

            else:
                details.append(
                    f"- **{label.capitalize()}** — "
                    f"Confidence: **{confidence:.2f}**"
                )

    return annotated_rgb, details

# ---------------- Helper: Face Analysis ----------------

def analyze_face(frame):
    try:
        # Resize large images to reduce memory usage
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        analysis = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="mtcnn",
            silent=True
        )

        if analysis:
            face = analysis[0]

            # No real face found -> confidence is 0
            if face.get("face_confidence", 1) < 0.5:
                return None, None, None, None

            dominant = str(face["dominant_emotion"])
            emotions = face["emotion"]
            dominant_score = float(emotions[dominant])
            age_range = estimate_age(frame, face["region"])

            return age_range, dominant, dominant_score, emotions

    except Exception as e:
        st.error(f"Face not detected: {type(e).__name__} — {e}")
    finally:
        gc.collect()

    return None, None, None, None

# ---------------- Helper: Show Face Results ----------------

def show_face_results(age_range, dominant, dominant_score, emotions):
    if dominant is None:
        st.info("Face not detected.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Estimated Age", age_range if age_range else "N/A")

    with col2:
        st.metric(
            "Dominant Emotion",
            dominant.capitalize(),
            f"{dominant_score:.1f}%"
        )

    st.subheader("📊 Emotion Confidence")

    for emotion, score in emotions.items():
        score = float(score)
        progress_value = min(score / 100.0, 1.0)

        st.progress(progress_value)
        st.write(f"**{emotion.capitalize()}**: {score:.1f}%")

# ---------------- Mode Selection ----------------

mode = st.radio(
    "Choose mode:",
    ["🖼️ Upload Image", "📸 Camera Capture","🎥 Live Webcam","ℹ️How it works"]
)

# ---------------- 1. Upload Image ----------------

if mode == "🖼️ Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"The image did not open: {e}")
            st.stop()

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📦 Object Detection")

        with st.spinner("Objects are being detected..."):
            annotated_rgb, details = analyze_objects(frame)

        st.image(
            annotated_rgb,
            caption="Detected Objects",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📋 Detection Details")

        if details:
            for detail in details:
                st.write(detail)
        else:
            st.info("No object detected.")

        st.subheader("🙂 Face Analysis")

        with st.spinner("Face emotion and age are being analyzed..."):
            age_range, dominant, dominant_score, emotions = analyze_face(frame)

        show_face_results(age_range, dominant, dominant_score, emotions)

# ---------------- 2. Camera Capture ----------------

elif mode == "📸 Camera Capture":

    camera_photo = st.camera_input("📸 Take photo with camera")

    if camera_photo is not None:
        try:
            image = Image.open(camera_photo).convert("RGB")
        except Exception as e:
            st.error(f"The camera image did not open: {e}")
            st.stop()

        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(
            image,
            caption="Captured Photo",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📦 Object Detection")

        with st.spinner("Objects are being detected..."):
            annotated_rgb, details = analyze_objects(frame)

        st.image(
            annotated_rgb,
            caption="Detected Objects",
            width="stretch",
            output_format="PNG"
        )

        st.subheader("📋 Detection Details")

        if details:
            for detail in details:
                st.write(detail)
        else:
            st.info("No object detected.")

        st.subheader("🙂 Face Analysis")

        with st.spinner("Face emotion and age are being analyzed..."):
            age_range, dominant, dominant_score, emotions = analyze_face(frame)

        show_face_results(age_range, dominant, dominant_score, emotions)

# ---------------- 3. Live Webcam ----------------

elif mode == "🎥 Live Webcam":
    from live_webcam import run_live
    run_live(yolo_model, conf_value, imgsz_value, get_dominant_color_name)
# ---------------- 4. How it works ----------------

elif "How it works" in mode:
    from how_it_works import show_how_it_works
    show_how_it_works()