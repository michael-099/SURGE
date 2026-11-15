"""Light-ASD speaking scores on MVVA's own face tracks (research/log.md section 15N, part 1 input). CPU.

For every MVVA video: each face's runs of consecutive frames with a finite box (MVVA face tracking, x1, y1, x2, y2 in
native pixels), if at least 1 s long, are scored by Light-ASD (Liao et al., CVPR 2023; AVA-ActiveSpeaker weights
`weight/pretrain_AVA_CVPR.model`, MIT) from the native video frames and the soundtrack (`all_audio`). Crops, MFCCs,
resampling to 25 fps, window averaging and the +-2 frame smoothing are those of scripts/asd_speaker.py (14B); there is
no face detection and no tracking, because MVVA provides the tracks. Light-ASD's model code is imported from its
repository, appended to the END of sys.path.

Output per video: <out>/<id>.npz with arrays frame (N,), face (N,), score (N,) (speaking logit; > 0 = speaking) and
fps, n_frames.

    python scripts/inv_mvva_asd.py --out runs/inv/mvva_asd [--videos 001 002] [--workers 6]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SRC = Path("D:/Datasets/MVVA/extracted")
INDEX = Path("D:/Datasets/MVVA/surge/index.csv")
ASD_REPO = Path("D:/Research/baselines/Light-ASD")


def load_asd(device):
    import torch
    import torch.nn as nn
    if str(ASD_REPO) not in sys.path:
        sys.path.append(str(ASD_REPO))
    from model.Model import ASD_Model
    from loss import lossAV

    class _ASD(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ASD_Model()
            self.lossAV = lossAV()

    asd = _ASD()
    sd = torch.load(ASD_REPO / "weight" / "pretrain_AVA_CVPR.model", map_location="cpu", weights_only=True)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k.startswith(("model.", "lossAV."))}
    asd.load_state_dict(sd, strict=True)
    return asd.to(device).eval()


def runs(ok: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """(first, last) frame of each run of True in ok that is at least min_len long."""
    out, i, n = [], 0, len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def find(folder: Path, vid: str, *exts: str) -> Path:
    for ext in exts:
        for name in (vid, vid.lstrip("0") or "0", f"{int(vid):03d}"):
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"{folder}/{vid}{exts}")


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    cap.release()
    return frames


def one_video(args):
    vid, fps, out = args
    import python_speech_features
    import torch
    from scipy import signal
    from scipy.io import wavfile

    from scripts.asd_speaker import crop_track, score_track
    torch.set_num_threads(3)
    dst = Path(out) / f"{vid}.npz"
    if dst.exists():
        return vid, -1, 0.0
    t0 = time.time()
    device = torch.device("cpu")
    asd = load_asd(device)
    t = np.load(find(SRC / "all_face_tracking_release", vid, ".npy"), allow_pickle=False)
    frames = read_frames(find(SRC / "mvva_database_v1" / "our_database_video", vid, ".mp4", ".avi"))
    sr, wav = wavfile.read(str(find(SRC / "all_audio", vid, ".wav")))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = signal.resample_poly(wav.astype(np.float64), 16000, sr) if sr != 16000 else wav.astype(np.float64)
    n = min(len(frames), t.shape[0]) if t.ndim == 3 else 0
    rec_f, rec_j, rec_s = [], [], []
    for j in range(t.shape[1] if t.ndim == 3 else 0):
        b = t[:n, j].astype(np.float64)
        ok = np.all(np.isfinite(b), 1)
        b = np.where(ok[:, None], b, 0.0)
        b = np.stack([np.minimum(b[:, 0], b[:, 2]), np.minimum(b[:, 1], b[:, 3]),
                      np.maximum(b[:, 0], b[:, 2]), np.maximum(b[:, 1], b[:, 3])], 1)
        ok &= (b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)
        for f0, f1 in runs(ok, int(np.ceil(fps))):
            track = {"frame": np.arange(f0, f1 + 1), "bbox": b[f0:f1 + 1]}
            crops = crop_track(frames, track)
            k = np.arange(int((f1 - f0 + 1) / fps * 25))
            if len(k) < 25:
                continue
            idx = np.clip(np.round(k * fps / 25.0).astype(int), 0, len(crops) - 1)
            seg = wav[int(f0 / fps * 16000):int((f1 + 1) / fps * 16000)]
            if len(seg) < 16000:  # soundtrack ends before the track: under 1 s of audio, not scored
                continue
            mfcc = python_speech_features.mfcc(seg, 16000, numcep=13, winlen=0.025, winstep=0.010)
            s25 = score_track(asd, crops[idx], mfcc, device)
            if len(s25) == 0:
                continue
            for f in track["frame"]:
                kk = min(int(round((f - f0) * 25.0 / fps)), len(s25) - 1)
                rec_f.append(int(f))
                rec_j.append(j)
                rec_s.append(float(s25[max(kk - 2, 0):min(kk + 3, len(s25))].mean()))
    np.savez_compressed(dst, frame=np.array(rec_f, np.int32), face=np.array(rec_j, np.int32),
                        score=np.array(rec_s, np.float32), fps=fps, n_frames=n)
    return vid, len(rec_s), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(INDEX)))
    fps = {r["id"]: float(r["fps"]) for r in rows}
    vids = a.videos or [r["id"] for r in rows]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (v, nrec, sec) in enumerate(ex.map(one_video, [(v, fps[v], a.out) for v in vids])):
            print(f"[{i + 1}/{len(vids)}] {v}: " + ("exists" if nrec < 0 else f"{nrec} face-frames, {sec:.0f} s"),
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
