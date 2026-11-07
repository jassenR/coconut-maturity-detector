"""
Audio classification with scikit-learn.

Expected dataset layout (folder name = class label):
    data/
        class_a/  *.wav
        class_b/  *.wav
        ...

Install:
    pip install librosa scikit-learn numpy joblib
"""

import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path("data")
SR = 22050          # sample rate
DURATION = 3.0      # seconds; clips are padded/trimmed to this length
N_MFCC = 20


def extract_features(path: Path) -> np.ndarray:
    """Return a fixed-length feature vector for one audio file."""
    y, sr = librosa.load(path, sr=SR, duration=DURATION)
    y = librosa.util.fix_length(y, size=int(SR * DURATION))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    # Summarize each time-series feature with mean and std
    feats = [mfcc, delta, chroma, contrast, centroid, rolloff, zcr, rms]
    return np.concatenate([np.hstack([f.mean(axis=1), f.std(axis=1)]) for f in feats])


def load_dataset(data_dir: Path):
    X, labels = [], []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for f in class_dir.glob("*"):
            if f.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
                continue
            try:
                X.append(extract_features(f))
                labels.append(class_dir.name)
            except Exception as e:
                print(f"Skipping {f}: {e}", file=sys.stderr)
    return np.array(X), np.array(labels)


def main():
    X, labels = load_dataset(DATA_DIR)
    print(f"Loaded {len(X)} clips, {X.shape[1]} features, "
          f"{len(set(labels))} classes")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
    }

    best_name, best_score = None, -1
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"{name}: CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    best = models[best_name].fit(X_train, y_train)
    y_pred = best.predict(X_test)
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    try:
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=le.classes_, xticks_rotation=45
        )
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
    except ImportError:
        pass

    joblib.dump({"model": best, "label_encoder": le}, "audio_model.joblib")
    print("Saved audio_model.joblib")


def predict(file_path: str, model_path: str = "audio_model.joblib") -> str:
    """Classify a single audio file with the saved model."""
    bundle = joblib.load(model_path)
    feats = extract_features(Path(file_path)).reshape(1, -1)
    pred = bundle["model"].predict(feats)[0]
    return bundle["label_encoder"].inverse_transform([pred])[0]


if __name__ == "__main__":
    main()
