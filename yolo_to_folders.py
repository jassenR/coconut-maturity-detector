"""
Convert a Roboflow YOLO export (object detection) into a classification
folder structure by cropping each bounding box.

Input  (unzipped YOLO export):
    yolo_export/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/*.jpg   valid/labels/*.txt
        test/images/*.jpg    test/labels/*.txt

Output:
    data/images/train|valid|test/<class_name>/*.jpg

pip install pillow pyyaml
"""

from pathlib import Path

import yaml
from PIL import Image

SRC = Path("yolo_export")
DST = Path("data/images")
MIN_SIZE = 16
EXCLUDE = {"niyog"}

names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
print("Classes:", names)

counts = {}
for split in ["train", "valid", "test"]:
    img_dir, lbl_dir = SRC / split / "images", SRC / split / "labels"
    if not img_dir.exists():
        continue
    for img_path in img_dir.glob("*"):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not lbl_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        for i, line in enumerate(lbl_path.read_text().splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if names[cls].lower() in EXCLUDE:
                continue
            vals = [float(v) for v in parts[1:]]
            if len(vals) == 4:                       # detection: cx cy w h
                cx, cy, w, h = vals
                xs, ys = [cx - w / 2, cx + w / 2], [cy - h / 2, cy + h / 2]
            else:                                    # segmentation polygon: x1 y1 x2 y2 ...
                xs, ys = vals[0::2], vals[1::2]
            left, right = int(min(xs) * W), int(max(xs) * W)
            top, bottom = int(min(ys) * H), int(max(ys) * H)
            left, top = max(left, 0), max(top, 0)
            right, bottom = min(right, W), min(bottom, H)
            if right - left < MIN_SIZE or bottom - top < MIN_SIZE:
                continue
            out_dir = DST / split / names[cls]
            out_dir.mkdir(parents=True, exist_ok=True)
            img.crop((left, top, right, bottom)).save(out_dir / f"{img_path.stem}_{i}.jpg")
            counts[(split, names[cls])] = counts.get((split, names[cls]), 0) + 1

for k in sorted(counts):
    print(k, counts[k])
