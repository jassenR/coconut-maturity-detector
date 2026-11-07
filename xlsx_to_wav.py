"""
Convert coconut_acoustic_signals.xlsx into WAV files.

Each sheet (Ridge A/B/C) holds one raw waveform PER COLUMN:
  132,300 rows = 3 s at 44,100 Hz (sample rate inferred from the row count;
  the file itself does not store it).
The label is in the column name suffix: _im (immature), _m (mature), _om (overmature).

Output:
  data/audio_wav/<label>/<ridge>_<column>.wav

pip install openpyxl numpy scipy
"""

import re
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.io import wavfile

XLSX = Path("coconut_acoustic signals.xlsx")
OUT = Path("data/audio_wav")
SR = 44100
PEAK_NORMALIZE = False  # True = scale each clip to full range (hides loudness differences)

LABELS = {"im": "immature", "m": "mature", "om": "overmature"}

wb = load_workbook(XLSX, read_only=True)
count = 0
for ws in wb.worksheets:
    ridge = ws.title.replace(" ", "")          # RidgeA, RidgeB, RidgeC
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    data = np.array([r for r in rows], dtype=np.float32)   # (132300, n_cols)
    print(ridge, data.shape)

    for j, col in enumerate(header):
        label = LABELS[re.sub(r"^c?\d+_", "", str(col))]
        y = data[:, j]
        y = y[~np.isnan(y)]
        if PEAK_NORMALIZE and np.abs(y).max() > 0:
            y = y / np.abs(y).max() * 0.95
        out_dir = OUT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(out_dir / f"{ridge}_{col}.wav", SR,
                      (np.clip(y, -1, 1) * 32767).astype(np.int16))
        count += 1

print(f"Wrote {count} WAV files to {OUT}")
