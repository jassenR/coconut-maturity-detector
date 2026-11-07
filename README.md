# 🥥 BukoType: Coconut Maturity Detector

A scanner that finds a coconut in a photo and estimates its stage (**malauhog**, **malakanin**, **malakatad**) with probabilities. It can also take a tap recording from a microphone. Built for a course project on multimodal (image + audio) classification of young coconut (buko) quality.

- **Image branch:** YOLOv8 segmentation (finds and names coconuts) + a scikit-learn classifier on color and shape features.
- **Audio branch:** MFCC / spectral features of the tap sound + a scikit-learn classifier. **Prototype only** (see [Limitations](#limitations)).
- **App:** Streamlit dashboard with camera, photo upload, and mic recording.

## Quick start (run the app)

The large model files are not stored in this repository. Download them first:

**📦 Download: `https://drive.google.com/drive/folders/12-iaRUQipyGbFnGSZwwcpCzPScowXtc_?usp=drive_link`**

| File | Needed to run the app? |
|------|------------------------|
| `yolo_best.pt` | Yes, for auto-detect mode (without it the app uses manual center crop) |
| `image_model.joblib` | **Yes** (the app stops without it) |
| `audio_model.joblib` | Only for the optional tap-audio fusion |
| `audio_features.joblib`, `image_features_*.joblib` | No (caches that only speed up retraining) |

Then, in PowerShell (Windows):

```powershell
git clone https://github.com/jassenR/coconut-maturity-detector.git
cd coconut-maturity-detector

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy the downloaded files into the project folder (next to `app.py`), then start the app:

```powershell
python -m streamlit run app.py
```

It opens at `http://localhost:8501`. Always start it with `streamlit run`, not `python app.py`.

On macOS/Linux, activate the environment with `source venv/bin/activate` instead.

## Using the scanner

1. Choose **Auto-detect (YOLO)** or **Manual center crop**.
2. Take a photo with the camera or upload one. Use even light and a plain background.
3. In auto-detect mode, pick the coconut to analyze. The app shows YOLO's label and a scikit-learn estimate with a probability bar chart.
4. Optional: record or upload a tap sound to fuse the audio with the image.

A brown **niyog** is recognized as niyog and gets no buko-stage estimate. Low-confidence results (under 50%) show a warning.

> Phone browsers only allow camera and mic on HTTPS or `localhost`. For a phone demo, use a tunnel such as ngrok, or run the app on the laptop.

## Project structure

```
app.py               Streamlit scanner app
train_bukotype.py    Trains the image + audio models, prints scores, saves charts to figures/
train_yolo.py        Trains YOLOv8-seg, tests it, saves yolo_best.pt and training curves
collect.py           Streamlit tool to record your own photo + tap + label per coconut
yolo_to_folders.py   Converts the Roboflow YOLO export into class folders (crops)
xlsx_to_wav.py       Converts the acoustic dataset spreadsheet into labeled WAV files
requirements.txt     Python dependencies
```

## Retrain from scratch

1. **Datasets** (not included; download them yourself):
   - Images: Roboflow Universe, "Coconut Maturity for PLP" (sandhya-tjeor). Export as **YOLOv8** and unzip to `yolo_export/`.
   - Audio: Mendeley Data, "Acoustic Signal Dataset: Tall Coconut Fruit Species" (<https://data.mendeley.com/datasets/hxh8kd3snj/1>). Save as `coconut_acoustic signals.xlsx` (or edit the path in `xlsx_to_wav.py`).
2. **Prepare data**
   ```powershell
   python yolo_to_folders.py     # crops -> data/images/<split>/<class>/
   python xlsx_to_wav.py         # WAVs  -> data/audio_wav/<stage>/
   ```
3. **Train the scikit-learn models and make charts**
   ```powershell
   python train_bukotype.py      # saves image_model.joblib, audio_model.joblib, figures/
   ```
4. **Train YOLO** (use a GPU, e.g. Google Colab T4; CPU training is very slow)
   ```powershell
   python train_yolo.py          # saves yolo_best.pt, runs/bukotype/
   ```
5. **Collect your own data** (recommended): `python -m streamlit run collect.py`

## Results

| Model | Evaluation | Result |
|-------|-----------|--------|
| Image (scikit-learn) | Roboflow test split | 84.9% accuracy, 84.9% balanced accuracy |
| Audio (scikit-learn) | Coconut-level 5-fold cross-validation | 75.3% accuracy, **57.0% balanced accuracy** |
| YOLOv8-seg | Roboflow test split | `<ADD mAP50 and mAP50-95 from train_yolo.py output>` |

Plain accuracy for audio looks high mainly because 65% of the recordings are one class (overmature), so balanced accuracy is the fairer number.

## Limitations

- **Label mapping is an assumption.** The image classes (malauhog, malakanin, malakatad) are treated as matching the audio dataset's stages (immature, mature, overmature). Verify this against both dataset descriptions.
- **The datasets are unpaired.** Images and audio come from different coconuts, so image + audio fusion is only demonstrated with class-matched pairs. It shows the mechanism, not real same-coconut performance.
- **Audio is a prototype.** The public recordings are from a different coconut type and recording setup. Phone-microphone taps will likely differ, so the audio model needs your own recordings (`collect.py`) before it can be trusted.
- **Image accuracy is from public test photos.** Real-world lighting, angles, and backgrounds can lower it.
- The model only knows three buko stages plus niyog. Other objects give meaningless results.

## Credits

Datasets belong to their original authors (Roboflow Universe user sandhya-tjeor; the Mendeley Data acoustic dataset authors). Check each dataset's license before reuse. Built with scikit-learn, librosa, Ultralytics YOLOv8, and Streamlit.
