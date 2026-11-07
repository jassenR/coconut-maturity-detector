"""
Train YOLOv8 (segmentation) on your Roboflow export so the scanner can FIND coconuts
and name each one (malauhog / malakanin / malakatad / niyog).

Setup:
    pip install ultralytics
    folder layout:  yolo_export/data.yaml, yolo_export/train|valid|test/images + labels

Run:
    python train_yolo.py          (train, test, and draw the training-curve chart)
    python train_yolo.py plot     (only redraw the chart from an earlier run)

Training on a CPU is slow (roughly an hour or more). Google Colab with a free GPU
(Runtime > Change runtime type > T4 GPU) takes about 10-20 minutes with the same script.

Output: yolo_best.pt  (copy this next to app.py), plus charts in runs/bukotype/
"""

import csv
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

EXPORT_DIR = Path("yolo_export")
MODEL = "yolov8n-seg.pt"  # n = fastest. yolov8s-seg.pt is more accurate but ~3x slower
EPOCHS = 50
IMGSZ = 640
RUN_NAME = "bukotype"


def fix_yaml(export_dir):
    """Roboflow's data.yaml often has relative paths that break. Write a corrected copy."""
    cfg = yaml.safe_load((export_dir / "data.yaml").read_text())
    cfg["path"] = export_dir.resolve().as_posix()
    for key, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        if (export_dir / folder / "images").exists():
            cfg[key] = f"{folder}/images"
        else:
            cfg.pop(key, None)
    out = export_dir / "data_fixed.yaml"
    out.write_text(yaml.safe_dump(cfg))
    print("Classes:", cfg["names"])
    return out


def plot_training_curves(run_dir=Path("runs") / RUN_NAME):
    """Accuracy over time: detection quality and error at every training epoch."""
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        print("No results.csv found in", run_dir)
        return
    with open(csv_path, newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]

    def col(name):
        return [float(r[name]) if r.get(name) not in (None, "") else float("nan") for r in rows]

    epochs = [int(float(r["epoch"])) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in [("metrics/mAP50(B)", "Box detection quality (mAP50)"),
                       ("metrics/mAP50(M)", "Outline/mask quality (mAP50)"),
                       ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")]:
        if key in rows[0]:
            ax[0].plot(epochs, col(key), label=label)
    ax[0].set_title("Accuracy over training time"); ax[0].set_xlabel("Training epoch (1 epoch = one pass over the data)")
    ax[0].set_ylabel("Score (0 to 1, higher is better)"); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
    for key, label in [("train/box_loss", "Training error (box)"), ("val/box_loss", "Validation error (box)")]:
        if key in rows[0]:
            ax[1].plot(epochs, col(key), label=label)
    ax[1].set_title("Error over training time"); ax[1].set_xlabel("Training epoch")
    ax[1].set_ylabel("Loss (lower is better)"); ax[1].legend(fontsize=8)
    text = ("LEFT: how well YOLO finds and outlines coconuts after each epoch (mAP50 = share of coconuts "
            "found with a well-placed box). RIGHT: its error. If validation error rises while training "
            "error keeps falling, the model is overfitting.")
    fig.text(0.5, 0.01, textwrap.fill(text, width=int(fig.get_figwidth() * 9)),
             ha="center", va="bottom", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = Path(run_dir) / "training_curves.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)
    plt.show()


def main():
    import torch
    from ultralytics import YOLO

    data_yaml = fix_yaml(EXPORT_DIR)
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Training on", "GPU" if device == 0 else "CPU (slow; consider Google Colab)")

    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ,
        batch=16 if device == 0 else 8, device=device,
        project="runs", name=RUN_NAME, exist_ok=True,
        patience=15,   # stop early if validation stops improving
        workers=0,     # avoids multiprocessing problems on Windows
    )

    best = Path(model.trainer.best)
    shutil.copy(best, "yolo_best.pt")
    print("Saved yolo_best.pt")

    # Final check on the held-out TEST split
    metrics = YOLO(best).val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                             device=device, workers=0)
    names = YOLO(best).names
    print("\nTEST RESULTS")
    print(f"  box mAP50 {metrics.box.map50:.3f}   box mAP50-95 {metrics.box.map:.3f}")
    if hasattr(metrics, "seg"):
        print(f"  mask mAP50 {metrics.seg.map50:.3f}  mask mAP50-95 {metrics.seg.map:.3f}")
    for idx, m in zip(metrics.box.ap_class_index, metrics.box.maps):
        print(f"  {names[int(idx)]:<10} mAP50-95 {m:.3f}")
    print("\nCharts (confusion matrix, PR curves, sample predictions): runs/", RUN_NAME)
    plot_training_curves()


if __name__ == "__main__":  # required on Windows
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_training_curves()
    else:
        main()
