"""Per-frame sound-source localisation maps for the STAViS datasets (research/log.md section 14F). CPU is enough.

    python -I -u scripts/ssl_maps.py --ckpt D:/Research/baselines/ezvsl_ckpt/vggsound_full_best.pth --rate-check
    python -I -u scripts/ssl_maps.py --ckpt D:/Research/baselines/ezvsl_ckpt/vggsound_full_best.pth --rate R [--only coutrot2]

Model:  EZ-VSL (Mo & Morgado, ECCV 2022; code Apache-2.0), the released VGG-Sound-full checkpoint. The two ResNet-18s
        are re-implemented here (no code from the EZ-VSL repository is run); the checkpoint is read with tensor-only
        unpickling. Inputs as in the EZ-VSL test code: the frame resized to 224x224 (bicubic, ImageNet normalisation);
        3 s of mono audio -> scipy spectrogram (nperseg 512, noverlap 274) -> log(. + 1e-7) / 12.
Audio:  the STAViS release's WAV of each video. The 3 s window is centred on the frame (time (k - 0.5) / fps, k 1-based,
        fps from the fold lists), shifted to stay inside the soundtrack at its ends, and tiled if the soundtrack is
        shorter than 3 s. Audio embeddings are computed every --audio-step s and each frame uses the nearest one.
Output: D:/Datasets/<DB>/surge_ssl/<video>.npy     float16 (n_frames, 7, 7): cosine similarity between the frame's
        visual feature at each of the 7x7 locations and the audio embedding (EZ-VSL's S_AVL times its tau).
        D:/Datasets/<DB>/surge_ssl_rot/<video>.npy  test videos only, the control of 14F: the same with the soundtrack
        of the next video with audio in the video's test-fold list (cyclic), the pairing of `evaluate.py --audio-rotate`.
        D:/Datasets/<DB>/surge_ssl/index.csv (video, frames, fps, has_audio, rot_partner) and meta.json.
--rate-check: label-free choice of the sample rate the checkpoint expects (the paper gives 257x300 spectrograms of
        3 s but no rate). For the middle frame of every video with audio, S[i, j] = max over locations of the cosine
        similarity between frame i and soundtrack j; per rate, the mean over i of the fraction of other videos j with
        S[i, i] > S[i, j] (AUC) and the top-1 retrieval rate. No gaze data is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from math import ceil, gcd
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import signal
from scipy.signal import resample_poly
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surge.data import STAVIS, STAVIS_FOLDS, stavis_list  # noqa: E402

DUR = 3.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class EZVSL(nn.Module):
    """The two encoders of EZ-VSL: visual ResNet-18 -> 1x1 conv (512, 7, 7); audio ResNet-18 (1-channel input,
    global max pool) -> linear (512); both L2-normalised."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.imgnet = resnet18(weights=None)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=1)
        self.audnet = resnet18(weights=None)
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)

    @staticmethod
    def _trunk(net, x):
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        return net.layer4(net.layer3(net.layer2(net.layer1(x))))

    def image(self, x):  # (B, 3, 224, 224) -> (B, 512, 7, 7)
        return F.normalize(self.img_proj(self._trunk(self.imgnet, x)), dim=1)

    def audio(self, s):  # (B, 1, 257, T) -> (B, 512)
        return F.normalize(self.aud_proj(self.audnet.avgpool(self._trunk(self.audnet, s)).flatten(1)), dim=1)


def load_ezvsl(path: str) -> EZVSL:
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    f64 = type(np.dtype("float64"))
    safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, (f64, "numpy.dtypes." + f64.__name__)]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    m = EZVSL()
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in ck["model"].items()})
    return m.eval()


def fps_table() -> dict[tuple[str, str], float]:
    out = {}
    for f in Path(STAVIS_FOLDS).glob("*_fps.txt"):
        for line in open(f):
            t = line.split()
            if len(t) >= 3:
                out[(f.name.split("_list_")[0], t[0])] = float(t[2])
    return out


def load_wav(path: Path, rate: int) -> np.ndarray:
    sr, x = wavfile.read(str(path))
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float32) - 128) / 128
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    if x.ndim == 2:
        x = x.mean(1)
    if sr != rate:
        g = gcd(rate, sr)
        x = resample_poly(x, rate // g, sr // g).astype(np.float32)
    return x


def window(x: np.ndarray, rate: int, t: float) -> np.ndarray:
    """3 s of x centred on time t (shifted inside x at its ends; x tiled if shorter than 3 s)."""
    L = int(round(DUR * rate))
    if len(x) < L:
        x = np.tile(x, L // max(len(x), 1) + 1)
    s = int(round(t * rate - L / 2))
    s = min(max(s, 0), len(x) - L)
    return x[s:s + L]


def spectrogram(w: np.ndarray, rate: int) -> np.ndarray:
    _, _, s = signal.spectrogram(w, rate, nperseg=512, noverlap=274)
    return (np.log(s + 1e-7) / 12.0).astype(np.float32)


def load_frame(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    return ((np.asarray(im, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def frame_files(base: Path, name: str, v: str) -> list[str]:
    d = base / "extracted" / "video_frames" / name / v
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() == ".jpg")


@torch.no_grad()
def audio_embed(model, x: np.ndarray, rate: int, times, batch: int = 32) -> torch.Tensor:
    out = []
    for b0 in range(0, len(times), batch):
        s = np.stack([spectrogram(window(x, rate, t), rate) for t in times[b0:b0 + batch]])
        out.append(model.audio(torch.from_numpy(s)[:, None]))
    return torch.cat(out)


@torch.no_grad()
def rate_check(model, rates):
    fps = fps_table()
    frames, wavs = [], []
    for key, (folder, name, _) in STAVIS.items():
        base = Path("D:/Datasets") / folder
        for v in sorted(p.name for p in (base / "extracted" / "video_frames" / name).iterdir() if p.is_dir()):
            wav = base / "extracted" / "video_audio" / name / v / f"{v}.wav"
            f = fps.get((name, v))
            if f is None or not wav.exists():
                continue
            fr = frame_files(base, name, v)
            i = len(fr) // 2
            frames.append((fr[i], (i + 0.5) / f))
            wavs.append(wav)
    print(f"{len(frames)} videos with audio", flush=True)
    img = torch.cat([model.image(torch.from_numpy(np.stack([load_frame(p) for p, _ in frames[b:b + 32]])))
                     for b in range(0, len(frames), 32)])  # N,512,7,7
    res = {}
    for r in rates:
        aud = torch.cat([audio_embed(model, load_wav(w, r), r, [t]) for w, (_, t) in zip(wavs, frames)])  # N,512
        S = torch.einsum("ichw,jc->ijhw", img, aud).flatten(2).amax(2)  # S[i, j]: frame i, soundtrack j
        d = S.diag()
        N = len(d)
        auc = ((d[:, None] > S).float().sum(1) + 0.5 * ((d[:, None] == S).float().sum(1) - 1)) / (N - 1)
        res[r] = {"auc": float(auc.mean()), "top1_img2aud": float((S.argmax(1) == torch.arange(N)).float().mean()),
                  "top1_aud2img": float((S.argmax(0) == torch.arange(N)).float().mean()),
                  "mean_matched": float(d.mean()), "mean_other": float((S.sum() - d.sum()) / (N * N - N))}
        print(r, json.dumps(res[r]), flush=True)
    return res


def rot_partners(key: str) -> dict[str, str | None]:
    """Test video -> the next video with audio in its test-fold list (cyclic); None if the video has no audio."""
    folder, name, nf = STAVIS[key]
    has = {v for v in (p.name for p in (Path("D:/Datasets") / folder / "extracted" / "video_audio" / name).iterdir())
           if (Path("D:/Datasets") / folder / "extracted" / "video_audio" / name / v / f"{v}.wav").exists()}
    out = {}
    for fold in ([1] if nf == 1 else [1, 2, 3]):
        lst = [v for v, _ in stavis_list(key, fold, "test")]
        for i, v in enumerate(lst):
            assert v not in out, (key, v)
            if v not in has:
                out[v] = None
                continue
            j = (i + 1) % len(lst)
            while lst[j] not in has:
                j = (j + 1) % len(lst)
            out[v] = lst[j] if lst[j] != v else None
    return out


@torch.no_grad()
def extract(model, rate: int, step: float, only: str | None, batch: int):
    fps = fps_table()
    for key, (folder, name, _) in STAVIS.items():
        if only and key != only:
            continue
        base = Path("D:/Datasets") / folder
        out, outr = base / "surge_ssl", base / "surge_ssl_rot"
        out.mkdir(exist_ok=True)
        outr.mkdir(exist_ok=True)
        partners = rot_partners(key)
        rows = []
        for v in sorted(p.name for p in (base / "extracted" / "video_frames" / name).iterdir() if p.is_dir()):
            fr = frame_files(base, name, v)
            n = len(fr)
            f = fps.get((name, v))
            wav = base / "extracted" / "video_audio" / name / v / f"{v}.wav"
            pv = partners.get(v)
            if f is None or not wav.exists():
                rows.append((v, n, f or "", 0, ""))
                continue
            rows.append((v, n, f, 1, pv or ""))
            dst, dstr = out / f"{v}.npy", outr / f"{v}.npy"
            if dst.exists() and (pv is None or dstr.exists()):
                continue
            x = load_wav(wav, rate)
            dur = n / f
            K = int(ceil(dur / step))
            grid = [(k + 0.5) * step for k in range(K)]
            A = audio_embed(model, x, rate, grid)  # K,512
            Ar = None
            if pv is not None:
                xp = load_wav(base / "extracted" / "video_audio" / name / pv / f"{pv}.wav", rate)
                xp = np.tile(xp, int(ceil(max(len(x), 1) / max(len(xp), 1))) + 1)
                Ar = audio_embed(model, xp, rate, grid)
            kidx = np.minimum(((np.arange(n) + 0.5) / f / step).astype(int), K - 1)
            M = np.zeros((n, 7, 7), np.float16)
            Mr = np.zeros((n, 7, 7), np.float16) if Ar is not None else None
            for b0 in range(0, n, batch):
                ims = torch.from_numpy(np.stack([load_frame(p) for p in fr[b0:b0 + batch]]))
                img = model.image(ims)  # B,512,7,7
                k = torch.from_numpy(kidx[b0:b0 + batch])
                M[b0:b0 + len(ims)] = torch.einsum("bchw,bc->bhw", img, A[k]).numpy().astype(np.float16)
                if Ar is not None:
                    Mr[b0:b0 + len(ims)] = torch.einsum("bchw,bc->bhw", img, Ar[k]).numpy().astype(np.float16)
            np.save(dst, M)
            if Mr is not None:
                np.save(dstr, Mr)
            print(key, v, n, "frames", round(dur, 1), "s", "rot<-" + (pv or "-"), flush=True)
        with open(out / "index.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "frames", "fps", "has_audio", "rot_partner"])
            w.writerows(rows)
        print(key, "done:", sum(r[3] for r in rows), "of", len(rows), "with audio", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rate-check", action="store_true")
    ap.add_argument("--rates", default="16000,22050,24000,32000,44100,48000")
    ap.add_argument("--rate", type=int, default=None)
    ap.add_argument("--audio-step", type=float, default=0.1)
    ap.add_argument("--only", default=None, help="one STAViS key, e.g. coutrot2")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None, help="rate check: JSON output")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load_ezvsl(a.ckpt)
    if a.rate_check:
        res = rate_check(model, [int(r) for r in a.rates.split(",")])
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1), newline="\n")
        return
    if a.rate is None:
        raise SystemExit("--rate is required (run --rate-check first)")
    extract(model, a.rate, a.audio_step, a.only, a.batch)
    for key, (folder, _, _) in STAVIS.items():
        if a.only and key != a.only:
            continue
        meta = {"model": "EZ-VSL VGG-Sound full", "ckpt": str(a.ckpt), "rate": a.rate, "audio_step": a.audio_step,
                "window_s": DUR, "map": "cosine similarity, 7x7"}
        (Path("D:/Datasets") / folder / "surge_ssl" / "meta.json").write_text(json.dumps(meta, indent=1), newline="\n")


if __name__ == "__main__":
    main()
