"""
BukoType training script (scikit-learn): audio model + image model + class-paired late fusion.

Run from the folder that contains `data/`:
    data/audio_wav/<immature|mature|overmature>/RidgeA_c10_m.wav   (from xlsx_to_wav.py)
    data/images/<train|valid|test>/<malauhog|malakanin|malakatad>/*.jpg   (from yolo_to_folders.py)

pip install numpy scipy scikit-learn librosa scikit-image pillow joblib matplotlib

Outputs: audio_model.joblib, image_model.joblib (used by the demo app),
         figures/*.png (charts for your slides).

LABEL MAPPING IS AN ASSUMPTION (confirm it against the dataset descriptions):
    malauhog -> immature, malakanin -> mature, malakatad -> overmature
"""

import textwrap
import warnings
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)  # hides the SVC(probability=True) notice

ROOT = Path(".")
FIG_DIR = ROOT / "figures"
AUDIO_DIR = ROOT / "data" / "audio_wav"
IMG_DIR = ROOT / "data" / "images"
IMG_SIZE = (128, 128)
SEED = 42

STAGES = ["immature", "mature", "overmature"]  # fixed order for every model
IMAGE_TO_STAGE = {"malauhog": "immature", "malakanin": "mature", "malakatad": "overmature"}
S2I = {s: i for i, s in enumerate(STAGES)}
BEST = {}  # which model type won for each modality (filled during training)


# ------------------------------------------------------------------ features
def audio_features(path):
    y, sr = librosa.load(path, sr=44100)  # dataset is 44.1 kHz; phone audio gets resampled
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    parts = [
        mfcc,
        librosa.feature.delta(mfcc),
        librosa.feature.spectral_contrast(y=y, sr=sr),
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y),
        librosa.feature.rms(y=y),
    ]
    return np.concatenate([np.hstack([p.mean(axis=1), p.std(axis=1)]) for p in parts])


def image_features(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img) / 255.0
    hsv = rgb2hsv(arr)
    color = np.concatenate([
        np.histogram(hsv[..., i], bins=16, range=(0, 1), density=True)[0] for i in range(3)
    ])
    gray = np.asarray(img.convert("L")) / 255.0
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.concatenate([color, shape])


# ------------------------------------------------------------------ data
def load_audio():
    cache = ROOT / "audio_features.joblib"
    if cache.exists():
        return joblib.load(cache)
    X, y, groups = [], [], []
    for stage in STAGES:
        for f in sorted((AUDIO_DIR / stage).glob("*.wav")):
            X.append(audio_features(f))
            y.append(S2I[stage])
            groups.append(f.stem.split("_")[1])  # coconut id, e.g. "c10"; same for all 3 ridges
    out = (np.array(X), np.array(y), np.array(groups))
    joblib.dump(out, cache)
    return out


def load_images(split):
    cache = ROOT / f"image_features_{split}.joblib"
    if cache.exists():
        return joblib.load(cache)
    X, y = [], []
    for cls, stage in IMAGE_TO_STAGE.items():
        for f in sorted((IMG_DIR / split / cls).glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                X.append(image_features(f))
                y.append(S2I[stage])
    out = (np.array(X), np.array(y))
    joblib.dump(out, cache)
    return out


def make_model(kind):
    if kind == "svm":
        clf = SVC(C=10, gamma="scale", probability=True, class_weight="balanced",
                  random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def report(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"  accuracy          {np.mean(y_true == y_pred):.3f}")
    print(f"  balanced accuracy {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"  macro F1          {f1_score(y_true, y_pred, average='macro'):.3f}")
    print("  confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=range(len(STAGES))))


# ------------------------------------------------------------------ audio
def run_audio():
    X, y, groups = load_audio()
    print(f"AUDIO: {len(X)} clips, {len(set(groups))} coconuts, "
          f"clips per stage: { {s: int((y == S2I[s]).sum()) for s in STAGES} }")

    # Coconut-level CV: all 3 ridges of one coconut stay in the same fold.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_kind, best_bal, best_proba = None, -1, None
    for kind in ["svm", "rf"]:
        proba = cross_val_predict(make_model(kind), X, y, groups=groups, cv=cv,
                                  method="predict_proba")
        bal = balanced_accuracy_score(y, proba.argmax(axis=1))
        print(f"  {kind}: out-of-fold balanced accuracy {bal:.3f}")
        if bal > best_bal:
            best_kind, best_bal, best_proba = kind, bal, proba

    report(f"AUDIO ({best_kind}, coconut-level 5-fold CV)", y, best_proba.argmax(axis=1))
    print(classification_report(y, best_proba.argmax(axis=1), target_names=STAGES))

    BEST["audio"] = best_kind
    final = make_model(best_kind).fit(X, y)
    joblib.dump({"model": final, "stages": STAGES}, ROOT / "audio_model.joblib")
    return y, best_proba


# ------------------------------------------------------------------ image
def run_image():
    Xtr, ytr = load_images("train")
    Xva, yva = load_images("valid")
    Xte, yte = load_images("test")
    print(f"\nIMAGE: train {len(Xtr)}, valid {len(Xva)}, test {len(Xte)}")

    best_kind, best_bal = None, -1
    for kind in ["svm", "rf"]:
        m = make_model(kind).fit(Xtr, ytr)
        bal = balanced_accuracy_score(yva, m.predict(Xva))
        print(f"  {kind}: validation balanced accuracy {bal:.3f}")
        if bal > best_bal:
            best_kind, best_bal = kind, bal

    BEST["image"] = best_kind
    model = make_model(best_kind).fit(Xtr, ytr)
    report(f"IMAGE ({best_kind}) on TEST set", yte, model.predict(Xte))
    test_proba = model.predict_proba(Xte)

    # Refit on train + valid for the saved/demo model
    final = make_model(best_kind).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    joblib.dump({"model": final, "stages": STAGES}, ROOT / "image_model.joblib")
    return yte, test_proba


# ------------------------------------------------------------------ fusion
def fusion_demo(y_img, p_img, y_aud, p_aud, repeats=20):
    """
    PROOF OF CONCEPT ONLY. Images and audio come from different coconuts, so each test
    image is paired with a random audio clip of the same stage (out-of-fold probabilities).
    This shows the fusion mechanics, not real same-coconut performance.
    """
    rng = np.random.default_rng(SEED)
    by_stage = {k: np.where(y_aud == k)[0] for k in range(len(STAGES))}
    print("\nFUSION DEMO (class-paired, NOT same-coconut pairs)")
    print("  weight on image -> mean balanced accuracy over", repeats, "random pairings")
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    means = []
    for w in weights:
        scores = []
        for _ in range(repeats):
            idx = np.array([rng.choice(by_stage[c]) for c in y_img])
            fused = w * p_img + (1 - w) * p_aud[idx]
            scores.append(balanced_accuracy_score(y_img, fused.argmax(axis=1)))
        tag = " (audio only)" if w == 0 else " (image only)" if w == 1 else ""
        print(f"  w={w:.2f}: {np.mean(scores):.3f}{tag}")
        means.append(float(np.mean(scores)))
    return weights, means


def predict_pair(image_path, audio_path, w_image=0.5):
    """Used by the demo app: fused stage prediction for one image + one WAV."""
    img = joblib.load(ROOT / "image_model.joblib")["model"]
    aud = joblib.load(ROOT / "audio_model.joblib")["model"]
    p_i = img.predict_proba(image_features(image_path).reshape(1, -1))[0]
    p_a = aud.predict_proba(audio_features(audio_path).reshape(1, -1))[0]
    p = w_image * p_i + (1 - w_image) * p_a
    return STAGES[int(p.argmax())], dict(zip(STAGES, p.round(3)))


# ------------------------------------------------------------------ figures
def caption(fig, text):
    """Plain-language explanation printed under a figure."""
    wrapped = textwrap.fill(text, width=int(fig.get_figwidth() * 9))
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])


def learning_curves():
    """Accuracy as the amount of training data grows (the closest thing to 'accuracy over time')."""
    rng = np.random.default_rng(SEED)
    fracs = np.linspace(0.1, 1.0, 8)

    # IMAGE: train on a growing slice of the training set; always score on the fixed VALID set
    Xtr, ytr = load_images("train")
    Xva, yva = load_images("valid")
    sizes, tr_sc, va_sc = [], [], []
    for fr in fracs:
        n = max(int(len(Xtr) * fr), 12)
        t_s, v_s = [], []
        for _ in range(3):
            idx = rng.choice(len(Xtr), n, replace=False)
            if len(set(ytr[idx])) < len(STAGES):
                continue
            m = make_model(BEST.get("image", "svm")).fit(Xtr[idx], ytr[idx])
            t_s.append(balanced_accuracy_score(ytr[idx], m.predict(Xtr[idx])))
            v_s.append(balanced_accuracy_score(yva, m.predict(Xva)))
        if t_s:
            sizes.append(n); tr_sc.append(np.mean(t_s)); va_sc.append(np.mean(v_s))
    image_curve = (sizes, tr_sc, va_sc)

    # AUDIO: coconut-level cross-validation on a growing number of clips
    audio_curve = None
    try:
        Xa, ya, ga = load_audio()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        sz, tr, va = learning_curve(make_model(BEST.get("audio", "svm")), Xa, ya, groups=ga, cv=cv,
                                    train_sizes=np.linspace(0.3, 1.0, 6),
                                    scoring="balanced_accuracy", n_jobs=1)
        audio_curve = (sz, tr.mean(axis=1), va.mean(axis=1))
    except Exception as e:  # tiny classes can break the smallest subsets
        print("Audio learning curve skipped:", e)
    return image_curve, audio_curve


def make_figures(y_aud, p_aud, y_img, p_img, fusion):
    """Saves labeled PNGs to figures/ and shows them. Close each window to see the next."""
    FIG_DIR.mkdir(exist_ok=True)
    img_names = list(IMAGE_TO_STAGE)

    # 0) Scores at a glance
    scores = {}
    for name, yt, pr in [("Audio\n(coconut-level CV)", y_aud, p_aud.argmax(axis=1)),
                         ("Image\n(test set)", y_img, p_img.argmax(axis=1))]:
        scores[name] = (np.mean(yt == pr), balanced_accuracy_score(yt, pr),
                        f1_score(yt, pr, average="macro"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = np.arange(len(scores)); wd = 0.25
    for k, (metric, color) in enumerate(zip(["Accuracy", "Balanced accuracy", "Macro F1"],
                                            ["#4c78a8", "#f58518", "#59a14f"])):
        vals = [v[k] for v in scores.values()]
        bars = ax.bar(xs + (k - 1) * wd, vals, wd, label=metric, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.axhline(0.75, ls="--", c="gray", label="75% target")
    ax.set_xticks(xs); ax.set_xticklabels(list(scores)); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (0 to 1, higher is better)"); ax.set_xlabel("Model")
    ax.set_title("How well does each model perform?"); ax.legend(loc="lower right", fontsize=8)
    caption(fig, "Accuracy = share of correct answers. Balanced accuracy averages each class's hit rate, "
                 "so a model can't look good by only guessing the biggest class. Macro F1 balances precision "
                 "and recall per class.")
    fig.savefig(FIG_DIR / "0_scores_summary.png", dpi=150)

    # 1) Class distribution
    n_img = [sum(len(list((IMG_DIR / sp / c).glob("*"))) for sp in ["train", "valid", "test"])
             for c in IMAGE_TO_STAGE]
    n_aud = [int((y_aud == i).sum()) for i in range(len(STAGES))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    ax[0].bar(STAGES, n_aud, color="#4c78a8")
    ax[0].set_title("Audio recordings per stage"); ax[0].set_xlabel("Maturity stage")
    ax[0].set_ylabel("Number of recordings")
    ax[1].bar(img_names, n_img, color="#59a14f")
    ax[1].set_title("Cropped photos per class"); ax[1].set_xlabel("Coconut class")
    ax[1].set_ylabel("Number of photos")
    for a, vals in zip(ax, [n_aud, n_img]):
        for i, v in enumerate(vals):
            a.text(i, v, str(v), ha="center", va="bottom")
    caption(fig, "How much data each class has. A model learns tall-bar classes best, so uneven bars "
                 "(like audio) make the small classes harder to recognize.")
    fig.savefig(FIG_DIR / "1_class_distribution.png", dpi=150)

    # 2) Confusion matrices (row-normalized)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for a, (title, yt, yp, labels) in zip(ax, [
        ("Audio model (coconut-level CV)", y_aud, p_aud.argmax(axis=1), STAGES),
        ("Image model (test set)", y_img, p_img.argmax(axis=1), img_names),
    ]):
        ConfusionMatrixDisplay.from_predictions(
            yt, yp, labels=range(len(STAGES)), display_labels=labels, normalize="true",
            values_format=".2f", cmap="Blues", ax=a, colorbar=False)
        a.set_title(title); a.set_xlabel("Predicted by the model"); a.set_ylabel("Actual (true) class")
    caption(fig, "Each ROW is the true class; the numbers show what share the model assigned to each "
                 "column. A dark diagonal means correct answers. Numbers off the diagonal are mix-ups "
                 "(e.g. 0.12 = 12% of that class was mistaken for the column's class).")
    fig.savefig(FIG_DIR / "2_confusion_matrices.png", dpi=150)

    # 3) Fusion weight curve
    weights, means = fusion
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(weights, means, marker="o")
    ax.axhline(0.75, ls="--", c="gray", label="75% target")
    ax.set_xlabel("Weight on the image model (0 = audio only, 1 = image only)")
    ax.set_ylabel("Balanced accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Does combining image + audio help? (demo)"); ax.legend()
    caption(fig, "Each point blends the two models' probabilities at a different image weight. "
                 "Demo only: image and audio come from DIFFERENT coconuts, paired by class, so this "
                 "shows the mechanism, not real same-coconut performance.")
    fig.savefig(FIG_DIR / "3_fusion_curve.png", dpi=150)

    # 4) Example tap: waveform + spectrogram per stage
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, s in enumerate(STAGES):
        f = sorted((AUDIO_DIR / s).glob("*.wav"))[0]
        y, sr = librosa.load(f, sr=None)
        ax[0, j].plot(np.arange(len(y)) / sr, y, lw=0.5)
        ax[0, j].set_title(f"{s}: waveform"); ax[0, j].set_xlabel("Time (seconds)")
        ax[0, j].set_ylabel("Loudness (amplitude)")
        ax[1, j].specgram(y, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
        ax[1, j].set_ylim(0, 8000); ax[1, j].set_title(f"{s}: spectrogram")
        ax[1, j].set_xlabel("Time (seconds)"); ax[1, j].set_ylabel("Pitch / frequency (Hz)")
    caption(fig, "TOP: the recorded tap as loudness over time (the spike is the knock). BOTTOM: a "
                 "spectrogram, where brighter color = more energy at that pitch and moment. The audio "
                 "model looks for differences in these patterns between stages.")
    fig.savefig(FIG_DIR / "4_audio_examples.png", dpi=150)

    # 5) Sample image crops per class
    fig, ax = plt.subplots(3, 4, figsize=(10, 7.5))
    for i, (cls, stage) in enumerate(IMAGE_TO_STAGE.items()):
        files = sorted((IMG_DIR / "train" / cls).glob("*"))[:4]
        for j in range(4):
            ax[i, j].axis("off")
            if j < len(files):
                ax[i, j].imshow(Image.open(files[j]))
        ax[i, 0].set_title(f"{cls} (treated as {stage})", loc="left", fontsize=10)
    caption(fig, "Example cropped coconut photos the image model learns from, one row per class. "
                 "The model uses color and texture/shape patterns in these crops.")
    fig.savefig(FIG_DIR / "5_image_samples.png", dpi=150)

    # 6) Accuracy as training data grows
    image_curve, audio_curve = learning_curves()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, title, curve, xl in [(ax[0], "Image model", image_curve, "Number of training photos"),
                                (ax[1], "Audio model", audio_curve, "Number of training recordings")]:
        if curve is not None:
            a.plot(curve[0], curve[1], marker="o", label="On the data it trained on")
            a.plot(curve[0], curve[2], marker="o",
                   label="On unseen data (validation)" if title == "Image model"
                   else "On unseen coconuts (cross-validation)")
        a.set_title(title + ": accuracy vs amount of training data"); a.set_xlabel(xl)
        a.set_ylabel("Balanced accuracy"); a.set_ylim(0, 1.05); a.legend(loc="lower right", fontsize=8)
    caption(fig, "Accuracy as the model sees more examples. If the unseen-data line is still climbing at "
                 "the right edge, more data would help. A large gap between the two lines means "
                 "overfitting (memorizing instead of learning).")
    fig.savefig(FIG_DIR / "6_accuracy_vs_training_data.png", dpi=150)

    print(f"\nSaved charts to {FIG_DIR.resolve()}")
    plt.show()


if __name__ == "__main__":
    y_aud, p_aud = run_audio()
    y_img, p_img = run_image()
    fusion = fusion_demo(y_img, p_img, y_aud, p_aud)
    make_figures(y_aud, p_aud, y_img, p_img, fusion)
