"""
BukoType scanner demo (Streamlit).

Run (same folder as train_bukotype.py, the two .joblib models, and optionally yolo_best.pt):
    pip install streamlit ultralytics
    python -m streamlit run app.py

Two scan modes:
  Auto-detect (YOLO)   needs yolo_best.pt (from train_yolo.py). YOLO finds each coconut and names it;
                       the scikit-learn model gives a second opinion on the crop and can be fused
                       with a tap recording.
  Manual center crop   works without YOLO. Frame ONE coconut in the middle and set the zoom.
"""

import io
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw

import train_bukotype as tb  # reuses the exact feature code from training

STAGE_TO_NAME = {stage: name for name, stage in tb.IMAGE_TO_STAGE.items()}  # immature -> malauhog
CONFIDENCE_MIN = 0.50
YOLO_WEIGHTS = Path("yolo_best.pt")

st.set_page_config(page_title="BukoType Scanner", page_icon="🥥")
st.title("🥥 BukoType Scanner")
st.caption("Scan a coconut photo to estimate its stage. Demo built on public datasets.")


@st.cache_resource
def load_models():
    img = joblib.load("image_model.joblib")["model"]
    try:
        aud = joblib.load("audio_model.joblib")["model"]
    except FileNotFoundError:
        aud = None
    return img, aud


@st.cache_resource
def load_yolo():
    if not YOLO_WEIGHTS.exists():
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    return YOLO(str(YOLO_WEIGHTS))


try:
    image_model, audio_model = load_models()
except FileNotFoundError:
    st.error("image_model.joblib not found. Run train_bukotype.py first, in this same folder.")
    st.stop()

yolo = load_yolo()
if yolo is None:
    st.info("YOLO not available (needs yolo_best.pt and `pip install ultralytics`). "
            "Using manual center crop.")
    mode = "Manual center crop"
else:
    mode = st.radio("Scan mode", ["Auto-detect (YOLO)", "Manual center crop"], horizontal=True)

source = st.radio("Image source", ["Camera", "Upload photo"], horizontal=True)
if source == "Camera":
    shot = st.camera_input("Show the coconut(s) to the camera")
else:
    shot = st.file_uploader("Coconut photo", type=["jpg", "jpeg", "png"])

audio_src = st.radio("Tap audio (optional)", ["None", "Record with mic", "Upload WAV"], horizontal=True)
wav = None
if audio_src == "Record with mic":
    st.caption("Hold the phone near the coconut, press record, tap it once or twice, then stop.")
    wav = st.audio_input("Tap recording")
elif audio_src == "Upload WAV":
    wav = st.file_uploader("Tap recording (WAV)", type=["wav"])
w_image = st.slider("Weight on image model (when audio is given)", 0.0, 1.0, 0.75, step=0.05) \
    if wav is not None else 1.0

if shot is None:
    st.info("Take or upload a photo to begin.")
    st.stop()

img = Image.open(shot).convert("RGB")
W, H = img.size
yolo_label = None

# ---------------------------------------------------------------- find the coconut
if mode == "Auto-detect (YOLO)":
    det_conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, step=0.05)
    res = yolo.predict(img, conf=det_conf, verbose=False)[0]
    if len(res.boxes) == 0:
        st.warning("No coconut detected. Try better lighting, a closer shot, or a lower "
                   "detection confidence.")
        st.stop()
    st.image(Image.fromarray(res.plot()[..., ::-1]), caption="YOLO detections",
             use_container_width=True)

    boxes = res.boxes.xyxy.tolist()
    classes = [int(c) for c in res.boxes.cls.tolist()]
    confs = res.boxes.conf.tolist()
    order = sorted(range(len(boxes)),
                   key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    labels = [f"#{n + 1}: {res.names[classes[i]]} ({confs[i]:.0%})" for n, i in enumerate(order)]
    pick = order[labels.index(st.selectbox("Coconut to analyze", labels))]

    x1, y1, x2, y2 = boxes[pick]
    pad = 0.05 * max(x2 - x1, y2 - y1)
    crop = img.crop((max(int(x1 - pad), 0), max(int(y1 - pad), 0),
                     min(int(x2 + pad), W), min(int(y2 + pad), H)))
    yolo_label, yolo_conf = res.names[classes[pick]], confs[pick]
    st.image(crop, caption="Selected coconut", width=240)

    if yolo_label.lower() == "niyog":
        st.subheader(f"niyog  —  {yolo_conf:.0%}")
        st.info("Niyog is the fully mature brown coconut, not one of the three buko stages, "
                "so no stage estimate is made.")
        st.stop()
else:
    zoom = st.slider("Crop zoom (smaller = tighter around the coconut)", 30, 100, 80, step=5)
    side = int(min(W, H) * zoom / 100)
    left, top = (W - side) // 2, (H - side) // 2
    box = (left, top, left + side, top + side)
    crop = img.crop(box)
    preview = img.copy()
    ImageDraw.Draw(preview).rectangle(box, outline="red", width=max(3, W // 200))
    c1, c2 = st.columns(2)
    c1.image(preview, caption="Scan area (red box)", use_container_width=True)
    c2.image(crop, caption="What the model sees", use_container_width=True)

# ---------------------------------------------------------------- scikit-learn + optional audio
buf = io.BytesIO()
crop.save(buf, format="PNG")
buf.seek(0)
p_img = image_model.predict_proba(tb.image_features(buf).reshape(1, -1))[0]
p = p_img

if wav is not None:
    if audio_model is None:
        st.warning("audio_model.joblib not found, so only the image is used.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav.getvalue())
        p_aud = audio_model.predict_proba(tb.audio_features(tmp.name).reshape(1, -1))[0]
        p = w_image * p_img + (1 - w_image) * p_aud
        st.caption("Image + audio fused.")

best = int(p.argmax())
stage = tb.STAGES[best]
st.subheader(f"{STAGE_TO_NAME[stage]} ({stage})  —  {p[best]:.0%}")
if yolo_label is not None:
    st.caption(f"YOLO's own label for this coconut: {yolo_label} ({yolo_conf:.0%})")
if p[best] < CONFIDENCE_MIN:
    st.warning("Low confidence. Re-frame the coconut, adjust the zoom, or improve the lighting.")

fig, ax = plt.subplots(figsize=(5, 2.6))
ax.barh([f"{STAGE_TO_NAME[s]} ({s})" for s in tb.STAGES], p, color="#59a14f")
ax.set_xlim(0, 1)
ax.set_xlabel("Probability")
fig.tight_layout()
st.pyplot(fig)
