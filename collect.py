"""
Collect your OWN paired data (photo + tap audio + label for the same coconut).
Works on a phone browser or laptop:

    python -m streamlit run collect.py

Saves to own_data/images, own_data/audio and own_data/manifest.csv
"""

import csv
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

DATA = Path("own_data")
(DATA / "images").mkdir(parents=True, exist_ok=True)
(DATA / "audio").mkdir(parents=True, exist_ok=True)
MANIFEST = DATA / "manifest.csv"
FIELDS = ["buko_id", "take", "image_path", "audio_path", "label", "saved_at"]
LABELS = ["(label later)", "malauhog", "malakanin", "malakatad"]


def read_rows():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


st.title("🥥 BukoType data collector")
st.caption("One ID per coconut. Photo once, tap 1-3 times, label after you cut it open.")

buko_id = re.sub(r"[^A-Za-z0-9_-]", "", st.text_input("Buko ID (e.g. buko_001)"))
label = st.selectbox("Label (your criteria, after cutting it open)", LABELS)
photo = st.camera_input("Photo: fill the frame with the coconut")
takes = [st.audio_input(f"Tap recording {i}", key=f"take{i}") for i in (1, 2, 3)]

if st.button("Save this coconut", type="primary"):
    recs = [(i, t) for i, t in enumerate(takes, 1) if t is not None]
    if not buko_id or photo is None or not recs:
        st.error("Need an ID, a photo, and at least one tap recording.")
    else:
        img_path = DATA / "images" / f"{buko_id}.jpg"
        Image.open(photo).convert("RGB").save(img_path, quality=95)
        rows = [r for r in read_rows() if r["buko_id"] != buko_id]  # re-saving replaces old rows
        for i, t in recs:
            wav_path = DATA / "audio" / f"{buko_id}_tap{i}.wav"
            wav_path.write_bytes(t.getvalue())
            rows.append({"buko_id": buko_id, "take": i, "image_path": str(img_path),
                         "audio_path": str(wav_path), "label": "" if label == LABELS[0] else label,
                         "saved_at": datetime.now().isoformat(timespec="seconds")})
        write_rows(rows)
        st.success(f"Saved {buko_id} with {len(recs)} tap(s).")

st.divider()
st.subheader("Add or fix a label")
fix_id = st.text_input("Buko ID to label")
fix_label = st.selectbox("New label", LABELS[1:], key="fixlabel")
if st.button("Update label"):
    rows = read_rows()
    hit = [r for r in rows if r["buko_id"] == fix_id]
    for r in hit:
        r["label"] = fix_label
    write_rows(rows)
    st.success(f"Updated {len(hit)} row(s).") if hit else st.error("ID not found.")

rows = read_rows()
if rows:
    coconuts = {r["buko_id"]: r["label"] or "unlabeled" for r in rows}
    counts = {}
    for lab in coconuts.values():
        counts[lab] = counts.get(lab, 0) + 1
    st.subheader(f"Collected so far: {len(coconuts)} coconuts")
    st.write(counts)
