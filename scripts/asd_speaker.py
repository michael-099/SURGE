"""Speaker maps for the speaker-aware branch (research/log.md section 14B).

For every video of a STAViS dataset: S3FD face detection on each frame, IoU face tracking, face crops and
Light-ASD (Liao et al., CVPR 2023; AVA-ActiveSpeaker weights `weight/pretrain_AVA_CVPR.model`, MIT) speaking scores
per face and frame from the frames and the soundtrack. The steps follow Light-ASD's own inference
(Columbia_test.py): tracks of more than 10 frames with gaps of at most 10 frames and IoU > 0.5, box size and centre
median-filtered (kernel 13), crop scale 0.40, 224x224 crops reduced to their grey 112x112 centre, 13 MFCCs at 100 Hz,
scores averaged over windows of 1 to 6 s. Two deliberate differences: no scene detection (tracks end where IoU
breaks), and the crops are resampled to the 25 fps that the model expects using each video's frame rate from the
STAViS fold lists (Light-ASD assumes 25 fps input). Light-ASD's model code is imported from its repository, which is
appended to the END of sys.path; its demo script (which calls shell commands) is not used.

Output per video: <dataset folder>/surge_speaker/<video>.npz with arrays
    frame (N,)  frame index (0-based) of each face detection kept in a track
    box   (N,4) x1, y1, x2, y2 in pixels of the STAViS frames
    score (N,)  Light-ASD 'speaking' logit, smoothed over +-2 frames as in Light-ASD's visualisation
    track (N,)  track id
plus fps, n_frames, height, width.

    python -I scripts/asd_speaker.py --dataset coutrot2 [--videos clip1] [--max-frames 200] [--device cpu]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import python_speech_features  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from surge.data import STAVIS, STAVIS_FOLDS  # noqa: E402

ASD_REPO = Path("D:/Research/baselines/Light-ASD")
NUM_FAILED_DET, MIN_TRACK, IOU_THRES, CROP_SCALE = 10, 10, 0.5, 0.40
DURATIONS = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]


def load_models(device):
    """S3FD and Light-ASD from the Light-ASD repository. The repository resolves the S3FD weight path relative to the
    current directory, so the working directory is the repository while the models are built."""
    sys.path.append(str(ASD_REPO))
    if not hasattr(np, "int"):  # S3FD's box_utils.nms_ uses the alias np.int, removed in NumPy 1.24
        np.int = int
    cwd = os.getcwd()
    os.chdir(ASD_REPO)
    try:
        from model.faceDetector.s3fd import S3FD
        from model.Model import ASD_Model
        from loss import lossAV
        det = S3FD(device=str(device))
    finally:
        os.chdir(cwd)

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
    return det, asd.to(device).eval()


def fps_table(key: str) -> dict[str, float]:
    _, name, nf = STAVIS[key]
    out = {}
    for f in Path(STAVIS_FOLDS).glob(f"{name}_list_*_fps.txt"):
        for line in open(f):
            t = line.split()
            if len(t) >= 3:
                out[t[0]] = float(t[2])
    return out


def iou(a, b):
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def track_faces(faces):
    """faces: list over frames of lists of dicts {frame, bbox}. Greedy IoU tracking as in Light-ASD's track_shot."""
    faces = [list(f) for f in faces]
    tracks = []
    while True:
        track = []
        for frame_faces in faces:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if iou(face["bbox"], track[-1]["bbox"]) > IOU_THRES:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > MIN_TRACK:
            fr = np.array([f["frame"] for f in track])
            bb = np.array([f["bbox"] for f in track])
            fi = np.arange(fr[0], fr[-1] + 1)
            bi = np.stack([interp1d(fr, bb[:, j])(fi) for j in range(4)], axis=1)
            tracks.append({"frame": fi, "bbox": bi})
    return tracks


def crop_track(frames, track):
    """Grey 112x112 face crops of a track, following Light-ASD's crop_video and evaluate_network."""
    s = signal.medfilt(np.maximum(track["bbox"][:, 3] - track["bbox"][:, 1], track["bbox"][:, 2] - track["bbox"][:, 0]) / 2, 13)
    y = signal.medfilt((track["bbox"][:, 1] + track["bbox"][:, 3]) / 2, 13)
    x = signal.medfilt((track["bbox"][:, 0] + track["bbox"][:, 2]) / 2, 13)
    out = []
    for i, f in enumerate(track["frame"]):
        bs = s[i]
        bsi = int(bs * (1 + 2 * CROP_SCALE))
        img = np.pad(frames[f], ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = y[i] + bsi, x[i] + bsi
        face = img[int(my - bs):int(my + bs * (1 + 2 * CROP_SCALE)), int(mx - bs * (1 + CROP_SCALE)):int(mx + bs * (1 + CROP_SCALE))]
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[56:168, 56:168]
        out.append(face)
    return np.stack(out)


@torch.no_grad()
def score_track(asd, faces25, mfcc, device):
    """Light-ASD scores at 25 fps, averaged over windows of 1-6 s (Light-ASD's evaluate_network)."""
    length = min(mfcc.shape[0] / 100.0, faces25.shape[0] / 25.0)
    a = mfcc[:int(round(length * 100))]
    v = faces25[:int(round(length * 25))]
    all_scores = []
    for d in DURATIONS:
        sc = []
        for i in range(int(math.ceil(length / d))):
            ia = torch.FloatTensor(a[i * d * 100:(i + 1) * d * 100]).unsqueeze(0).to(device)
            iv = torch.FloatTensor(v[i * d * 25:(i + 1) * d * 25]).unsqueeze(0).to(device)
            if ia.shape[1] < 4 or iv.shape[1] < 1:
                continue
            out = asd.model.forward_audio_visual_backend(asd.model.forward_audio_frontend(ia), asd.model.forward_visual_frontend(iv))
            sc.extend(asd.lossAV.forward(out, labels=None))
        all_scores.append(np.array(sc[:v.shape[0]], dtype=np.float32))
    n = min(len(s) for s in all_scores)
    return np.mean(np.stack([s[:n] for s in all_scores]), axis=0)


def process_video(key, video, fps, det, asd, device, max_frames=None, scale=1.0, conf=0.9):
    folder, name, _ = STAVIS[key]
    base = Path("D:/Datasets") / folder / "extracted"
    fpaths = sorted((base / "video_frames" / name / video).glob("*.jpg"))[:max_frames]
    frames = [cv2.imread(str(p)) for p in fpaths]
    n = len(frames)
    H, W = frames[0].shape[:2]
    faces = []
    for i, im in enumerate(frames):
        bb = det.detect_faces(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), conf_th=conf, scales=[scale])
        faces.append([{"frame": i, "bbox": b[:4].tolist()} for b in bb])
    tracks = track_faces(faces)
    sr, wav = wavfile.read(str(base / "video_audio" / name / video / f"{video}.wav"))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = signal.resample_poly(wav.astype(np.float64), 16000, sr) if sr != 16000 else wav.astype(np.float64)
    rec = {"frame": [], "box": [], "score": [], "track": []}
    for tid, tr in enumerate(tracks):
        crops = crop_track(frames, tr)
        f0, f1 = int(tr["frame"][0]), int(tr["frame"][-1])
        dur = (f1 - f0 + 1) / fps
        k = np.arange(int(dur * 25))
        if len(k) < 25:  # under one second at 25 fps: too short to score
            continue
        idx = np.clip(np.round(k * fps / 25.0).astype(int), 0, len(crops) - 1)
        seg = wav[int(f0 / fps * 16000):int((f1 + 1) / fps * 16000)]
        mfcc = python_speech_features.mfcc(seg, 16000, numcep=13, winlen=0.025, winstep=0.010)
        s25 = score_track(asd, crops[idx], mfcc, device)
        for j, f in enumerate(tr["frame"]):
            kk = min(int(round((f - f0) * 25.0 / fps)), len(s25) - 1)
            sm = s25[max(kk - 2, 0):min(kk + 3, len(s25))].mean()  # +-2 frames, as Light-ASD's visualisation
            rec["frame"].append(int(f))
            rec["box"].append(tr["bbox"][j])
            rec["score"].append(float(sm))
            rec["track"].append(tid)
    return rec, n, H, W, len(tracks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(STAVIS))
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="output folder (default <dataset folder>/surge_speaker)")
    a = ap.parse_args()
    device = torch.device(a.device)
    det, asd = load_models(device)
    fps = fps_table(a.dataset)
    folder, name, _ = STAVIS[a.dataset]
    out = Path(a.out) if a.out else Path("D:/Datasets") / folder / "surge_speaker"
    out.mkdir(parents=True, exist_ok=True)
    videos = a.videos or sorted(fps)
    for vi, v in enumerate(videos):
        dst = out / f"{v}.npz"
        if dst.exists() and a.max_frames is None:
            print(f"[{vi + 1}/{len(videos)}] {v} exists", flush=True)
            continue
        t0 = time.time()
        rec, n, H, W, nt = process_video(a.dataset, v, fps[v], det, asd, device, a.max_frames)
        np.savez_compressed(dst, frame=np.array(rec["frame"], np.int32), box=np.array(rec["box"], np.float32).reshape(-1, 4),
                            score=np.array(rec["score"], np.float32), track=np.array(rec["track"], np.int32),
                            fps=fps[v], n_frames=n, height=H, width=W)
        sc = np.array(rec["score"])
        print(f"[{vi + 1}/{len(videos)}] {v}: {n} frames, {nt} tracks, {len(sc)} face-frames, "
              f"speaking (score>0) {float((sc > 0).mean()) if len(sc) else 0:.2f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
