"""Per-frame audio features for the STAViS datasets (research/log.md section 11D). CPU only.

    python -u scripts/prep_stavis_audio.py [--threads 16]
Input:  D:/Datasets/<DB>/extracted/video_audio/<name>/<video>/<video>.wav (the STAViS release),
        fps from the STAViS fold lists, frame count from extracted/video_frames.
Model:  AST (MIT/ast-finetuned-audioset-10-10-0.4593, BSD-3-Clause), frozen, on consecutive 10.24 s chunks of 16 kHz
        mono audio; patch tokens averaged over the 12 frequency rows -> a 768-d feature per 10 ms-hop column.
Output: D:/Datasets/<DB>/surge_audio/<video>.npy, float16 (n_frames, 768): feature of frame k (1-based) = mean of the
        columns whose centre lies within +-0.5 s of (k - 0.5) / fps. Videos without a WAV get no file (zeros at load).
        D:/Datasets/<DB>/surge_audio/index.csv (video, frames, fps, seconds, has_audio).
"""
from __future__ import annotations

import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from scipy.signal import resample_poly

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surge.data import STAVIS, STAVIS_FOLDS  # noqa: E402

AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
CHUNK = 163840  # 10.24 s
HALF_WIN = 0.5


def fps_table() -> dict[tuple[str, str], float]:
    out = {}
    for f in Path(STAVIS_FOLDS).glob("*_fps.txt"):
        for line in open(f):
            t = line.split()
            if len(t) >= 3:
                out[(f.name.split("_list_")[0], t[0])] = float(t[2])
    return out


def load_wav(path: Path) -> np.ndarray:
    sr, x = wavfile.read(str(path))
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float32) - 128) / 128
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


@torch.no_grad()
def columns(x: np.ndarray, fe, model) -> tuple[np.ndarray, np.ndarray]:
    """-> (centre times in s (n_cols,), features (n_cols, 768)) over the whole waveform."""
    times, feats = [], []
    for c0 in range(0, max(len(x), 1), CHUNK):
        seg = x[c0:c0 + CHUNK]
        inp = fe(seg, sampling_rate=SR, return_tensors="pt")
        h = model(**inp).last_hidden_state[0]  # 2 + 12*101 tokens
        td = 101
        assert h.shape[0] == 2 + 12 * td, h.shape
        p = h[2:].reshape(12, td, h.shape[-1])  # frequency-major (HF ASTEmbeddings: conv output (768, 12, 101) flattened)
        col = p.mean(0).numpy()
        t = c0 / SR + (10 * np.arange(td) + 8) * 0.01
        keep = t <= c0 / SR + len(seg) / SR  # drop columns that only see padding
        times.append(t[keep]); feats.append(col[keep])
    return np.concatenate(times), np.concatenate(feats)


def per_frame(times, feats, n, fps):
    out = np.zeros((n, feats.shape[1]), np.float32)
    for k in range(n):
        t = (k + 0.5) / fps
        m = np.abs(times - t) <= HALF_WIN
        out[k] = feats[m].mean(0) if m.any() else feats[np.argmin(np.abs(times - t))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default=None, help="one STAViS key, e.g. coutrot2")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(AST)
    model = ASTModel.from_pretrained(AST).eval()
    fps = fps_table()
    for key, (folder, name, _) in STAVIS.items():
        if a.only and key != a.only:
            continue
        base = Path("D:/Datasets") / folder
        out = base / "surge_audio"
        out.mkdir(exist_ok=True)
        rows = []
        vids = sorted(p.name for p in (base / "extracted" / "video_frames" / name).iterdir() if p.is_dir())
        for v in vids:
            n = sum(1 for p in (base / "extracted" / "video_frames" / name / v).iterdir() if p.suffix.lower() == ".jpg")
            f = fps.get((name, v))
            wav = base / "extracted" / "video_audio" / name / v / f"{v}.wav"
            if f is None or not wav.exists():
                rows.append((v, n, f or "", "", 0))
                continue
            dst = out / f"{v}.npy"
            x = load_wav(wav)
            if not dst.exists():
                t, c = columns(x, fe, model)
                np.save(dst, per_frame(t, c, n, f).astype(np.float16))
            rows.append((v, n, f, round(len(x) / SR, 2), 1))
            print(key, v, n, f, round(len(x) / SR, 1), "s", flush=True)
        with open(out / "index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "frames", "fps", "seconds", "has_audio"])
            w.writerows(rows)
        print(key, "done:", sum(r[4] for r in rows), "of", len(rows), "with audio", flush=True)


if __name__ == "__main__":
    main()
