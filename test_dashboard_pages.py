"""Tests for dashboard_page.py (they need no Streamlit: a tiny fake one is used)."""
import sys, types, re

class _State(dict):
    pass

calls = {"markdown": [], "info": 0, "checkbox": None, "button": None, "rerun": 0}
fake = types.ModuleType("streamlit")
fake.session_state = _State()
fake.markdown = lambda text, unsafe_allow_html=False: calls["markdown"].append((text, unsafe_allow_html))
def _info(text): calls["info"] += 1
fake.info = _info
fake.checkbox = lambda label, value=False, key=None: calls["checkbox"]
fake.button = lambda label, key=None: calls["button"]
def _rerun(): calls["rerun"] += 1
fake.rerun = _rerun
sys.modules["streamlit"] = fake

sys.path.insert(0, ".")
import dashboard_page as d

passed = 0
def check(name, cond):
    global passed
    assert cond, "FAILED: " + name
    passed += 1

# 1. the detection lines of app.py -> dashboard names
lines = [
    "👔 **Tie detected** — Colour: **Blue** — Confidence: **0.80**",
    "👓 **Eyeglasses detected** — Confidence: **0.74**",
    "📱 **Mobile phone detected** — Confidence: **0.61**",
    "🧍 **Person detected** — Confidence: **0.91**",
    "☕ **Cup detected** — Confidence: **0.75**",
    "🎒 **Bag detected** — Confidence: **0.50**",
    "🍶 **Bottle detected** — Confidence: **0.52**",
    "🖊️ **Pen detected** — Confidence: **0.55**",
    "- **Book** — Confidence: **0.48**",
    "- **Something else** — Confidence: **0.99**",
]
got = d.labels_from_details(lines)
check("all 9 kinds found", got == ["Tie","Glasses","Phone","Person","Cup","Bag","Bottle","Pen","Book"])
check("empty details", d.labels_from_details([]) == [] and d.labels_from_details(None) == [])

# 2. emotion
check("emotion plain", d.normalize_emotion("happy") == "Happy")
check("emotion with percent", d.normalize_emotion("Happy 87%") == "Happy")
check("emotion not sure", d.normalize_emotion("Not sure (maybe Happy)") is None)
check("emotion none", d.normalize_emotion(None) is None)
check("emotion face not detected", d.normalize_emotion("Face not detected") is None)

# 3. age
check("age range", d.normalize_age("25-32") == "25-32")
check("age two ranges", d.normalize_age("25-32 or 38-43") == "25-32")
check("age text", d.normalize_age("Age 8-12") == "8-12")
check("age number", d.normalize_age(27) == "25-32")
check("age 0", d.normalize_age(0) == "0-2")
check("age 70", d.normalize_age(70) == "60-100")
check("age N/A", d.normalize_age("N/A") is None and d.normalize_age(None) is None)
check("age bool ignored", d.normalize_age(True) is None)

# 4. counting, and a rerun must not count the same photo twice
fake.session_state.clear()
check("first photo counted", d.record_photo(lines[:3], "Happy 80%", "25-32", key="a") is True)
check("same photo again is ignored", d.record_photo(lines[:3], "Happy 80%", "25-32", key="a") is False)
check("second photo counted", d.record_photo(labels=["person", "pen"], emotion="sad", age=40, key="b") is True)
st = d.get_stats()
check("photos = 2", st["photos"] == 2)
check("faces = 2", st["faces"] == 2)
check("objects counted", st["objects"]["Tie"] == 1 and st["objects"]["Person"] == 1 and st["objects"]["Pen"] == 1 and st["objects"]["Glasses"] == 1)
check("emotions counted", st["emotions"]["Happy"] == 1 and st["emotions"]["Sad"] == 1)
check("ages counted", st["ages"]["25-32"] == 1 and st["ages"]["38-43"] == 1)
d.record_photo(details=["- **Book** — Confidence: **0.5**"], emotion="Face not detected", age=None, key="c")
check("photo without a face adds no face", d.get_stats()["faces"] == 2 and d.get_stats()["photos"] == 3)
d.record_photo(labels=["person"])   # no key: always counted
d.record_photo(labels=["person"])
check("no key = counted each time", d.get_stats()["photos"] == 5)

class _Upload:     # looks like a Streamlit UploadedFile
    file_id = "abc123"
class _Camera:     # no file_id, only bytes
    def getvalue(self): return b"hello"
check("photo_key uses file_id", d.photo_key(_Upload()) == "abc123")
check("photo_key hashes bytes", len(d.photo_key(_Camera())) == 32)

d.reset_stats()
check("reset", d.get_stats()["photos"] == 0 and d.get_stats()["objects"]["Person"] == 0)

# 5. the HTML
body = d.build_dashboard_html(d.SAMPLE_STATS, sample=True)
check("24 bars (9 + 7 + 8)", body.count('class="sv3-bar') == 24)
check("sample banner shown", "SAMPLE DATA" in body)
check("no banner for real data", "SAMPLE DATA" not in d.build_dashboard_html(d.new_stats(), sample=False))
check("no empty lines", all(line.strip() for line in body.splitlines()))
check("no line starts with spaces", not any(re.match(r"^ ", line) for line in body.splitlines()))
check("style also compact", all(line.strip() and not line.startswith(" ") for line in d._compact(d.STYLE).splitlines()))
check("most seen = Person", ">Person</p><p class=\"l\">Most seen object" in body)
check("zero data shows dash", ">-</p>" in d.build_dashboard_html(d.new_stats()))
tall = [m for m in re.findall(r"sv3-bar k\d[^\"]*\" style=\"height:(\d+)px", body)]
check("tallest bar is the limit", max(int(x) for x in tall) == d.MAX_BAR_HEIGHT)
check("escapes names", "<script>" not in d.build_chart_html("<script>", "x", ["<b>"], [1]))

# 6. the page itself, with a fake Streamlit
fake.session_state.clear(); calls.update(markdown=[], info=0, checkbox=False, button=False, rerun=0)
d.show_dashboard()
check("empty page shows info + sample box", calls["info"] == 1 and len(calls["markdown"]) == 2)
check("html allowed", all(flag for _, flag in calls["markdown"]))
calls.update(markdown=[], info=0, checkbox=True)
d.show_dashboard()
check("sample mode shows banner", "SAMPLE DATA" in calls["markdown"][1][0])
d.record_photo(labels=["cup"], emotion="happy", age="15-20", key="z")
calls.update(markdown=[], info=0, checkbox=False, button=True, rerun=0)
d.show_dashboard()
check("page with data: no info box", calls["info"] == 0)
check("reset button resets and reruns", calls["rerun"] == 1 and d.get_stats()["photos"] == 0)

print(f"all {passed} checks passed")