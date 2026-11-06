"""Datasets for SURGE-D, read directly from the original layouts on disk (no GRAFT code or GRAFT layouts).

DHF1K      D:/DHF1K/frames/NNNN/NNNN.jpg, D:/DHF1K/annotations/annotation/NNNN/{maps,fixation}/NNNN.png
UCF Sports D:/Datasets/UCF_Sports/extracted/{training,testing}/<video>/{images,maps,fixation}/*.png
SALICON    D:/Datasets/SALICON/extracted/{images,maps,fixations}/{train,val}/...
STAViS     D:/Datasets/<DB>/extracted/{video_frames,annotations}/<name>/<video>/... (DIEM, Coutrot1/2, ETMD, AVAD, SumMe),
           folds from D:/Datasets/STAViS_folds/fold_lists_x/fold_lists/<name>_list_{train,test}[_k]_fps.txt
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------------------------- videos
@dataclass
class Video:
    name: str
    frames: list[str]
    maps: list[str] = field(default_factory=list)
    fixations: list[str] = field(default_factory=list)
    skip_empty: bool = False  # score only frames with at least one fixation (STAViS datasets; research/log.md section 7)

    def __len__(self) -> int:
        n = len(self.frames)
        if self.maps:
            n = min(n, len(self.maps), len(self.fixations))
        return n


def _sorted_files(d: Path, exts=(".png", ".jpg")) -> list[str]:
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in exts)


def dhf1k_videos(root: str, ids, with_labels: bool = True) -> list[Video]:
    root = Path(root)
    out = []
    for i in ids:
        name = f"{int(i):04d}"
        frames = _sorted_files(root / "frames" / name, (".jpg",))
        if with_labels:
            a = root / "annotations" / "annotation" / name
            maps = _sorted_files(a / "maps", (".png",))
            fix = _sorted_files(a / "fixation", (".png",))
            out.append(Video(name, frames, maps, fix))
        else:
            out.append(Video(name, frames))
    return out


def ucf_videos(root: str, split: str) -> list[Video]:
    base = Path(root) / "extracted" / ("training" if split == "train" else "testing")
    out = []
    for v in sorted(os.listdir(base)):
        d = base / v
        out.append(Video(v, _sorted_files(d / "images", (".png",)), _sorted_files(d / "maps", (".png",)),
                         _sorted_files(d / "fixation", (".png",))))
    return out


# key -> (folder under D:/Datasets, STAViS name, number of folds)
STAVIS = {
    "diem": ("DIEM", "DIEM", 1), "coutrot1": ("Coutrot1", "Coutrot_db1", 3), "coutrot2": ("Coutrot2", "Coutrot_db2", 3),
    "etmd": ("ETMD", "ETMD_av", 3), "avad": ("AVAD", "AVAD", 3), "summe": ("SumMe", "SumMe", 3),
}
STAVIS_FOLDS = "D:/Datasets/STAViS_folds/fold_lists_x/fold_lists"


def stavis_list(key: str, fold: int, part: str) -> list[tuple[str, int]]:
    """(video, listed frame count) from the STAViS fold list; part is 'train' or 'test'."""
    _, name, nf = STAVIS[key]
    f = f"{name}_list_{part}_fps.txt" if nf == 1 else f"{name}_list_{part}_{fold}_fps.txt"
    out = []
    for line in open(Path(STAVIS_FOLDS) / f):
        t = line.split()
        if t:
            out.append((t[0], int(t[1])))
    return out


def stavis_videos(key: str, fold: int, part: str, root: str = "D:/Datasets") -> list[Video]:
    """Videos of one STAViS fold. Frames 1..min(listed count, frames that have image, map and fixation file)."""
    folder, name, _ = STAVIS[key]
    base = Path(root) / folder / "extracted"
    out = []
    for v, listed in stavis_list(key, fold, part):
        fr = _sorted_files(base / "video_frames" / name / v, (".jpg",))
        a = base / "annotations" / name / v
        mp = _sorted_files(a / "maps", (".jpg",))
        fx = sorted(str(p) for p in a.glob("fixMap_*.mat"))
        n = min(listed, len(fr), len(mp), len(fx))
        out.append(Video(v, fr[:n], mp[:n], fx[:n], skip_empty=True))
    return out


def parse_stavis(dataset: str):
    """'avad_f2' -> ('avad', 2); None if not a STAViS dataset name."""
    import re
    m = re.fullmatch(r"(%s)_f(\d)" % "|".join(STAVIS), dataset)
    return (m.group(1), int(m.group(2))) if m else None


def stavis_holdout_file(key: str, fold: int, splits_dir) -> Path:
    return Path(splits_dir) / f"stavis_{key}_f{fold}_holdout.txt"


def stavis_split_videos(dataset: str, split: str, splits_dir) -> list[Video]:
    """split: 'val' (= the fold's test list), 'train' or 'holdout' (the fold's train list minus / only the holdout)."""
    key, fold = parse_stavis(dataset)
    if split in ("val", "test"):
        return stavis_videos(key, fold, "test")
    hold = set(open(stavis_holdout_file(key, fold, splits_dir)).read().split())
    vids = stavis_videos(key, fold, "train")
    return [v for v in vids if (v.name in hold) == (split == "holdout")]


def read_split_file(path: str) -> list[int]:
    with open(path) as f:
        return [int(x) for x in f.read().split()]


# ----------------------------------------------------------------------------------------------- loading
def load_rgb(path: str, size: tuple[int, int]) -> np.ndarray:
    """Read an image and resize (squash) it to size=(H, W); returns float32 normalized HWC."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return (im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def load_gray(path: str) -> np.ndarray:
    if path.endswith(".mat"):  # STAViS fixMap_*.mat: binary 'eyeMap' (0/1)
        import scipy.io as sio
        m = sio.loadmat(path)["eyeMap"]
        return ((m > 0) * 255).astype(np.uint8)
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(path)
    return im


def density_small(m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Area-resize a ground-truth map to `size` and normalize it to a probability distribution."""
    s = cv2.resize(m.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_AREA)
    t = s.sum()
    return s / t if t > 0 else np.full(size, 1.0 / (size[0] * size[1]), np.float32)


def fixations_small(f: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Map fixated pixels of a binary map onto a coarse grid (count per cell)."""
    ys, xs = np.nonzero(f > 127 if f.dtype == np.uint8 else f > 0)
    out = np.zeros(size, np.float32)
    if len(ys):
        gy = np.minimum((ys * size[0]) // f.shape[0], size[0] - 1)
        gx = np.minimum((xs * size[1]) // f.shape[1], size[1] - 1)
        np.add.at(out, (gy, gx), 1.0)
    return out


def window_indices(n: int, start: int, T: int, stride: int) -> list[int]:
    """Indices of a T-frame window starting at `start`; out-of-range positions are clamped (edge replicate)."""
    return [min(max(start + k * stride, 0), n - 1) for k in range(T)]


# ----------------------------------------------------------------------------------------------- training
class ClipDataset(Dataset):
    """Random training windows.

    Each item: frames (T,3,H,W); targets for the central T/2 sampled frames: density (T/2, g, g),
    fixation counts (T/2, g, g). `clips_per_video` windows are drawn per video per epoch.
    """

    def __init__(self, videos: list[Video], T: int = 16, stride: int = 2, size=(256, 256), out_size=(64, 64),
                 clips_per_video: int = 8, hflip: bool = True, seed: int = 0):
        self.videos = [v for v in videos if len(v) >= 1]
        self.T, self.stride, self.size, self.out_size = T, stride, size, out_size
        self.cpv, self.hflip = clips_per_video, hflip
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.videos) * self.cpv

    def __getitem__(self, i):
        v = self.videos[i // self.cpv]
        rng = random.Random((self.seed, self.epoch, i).__hash__())
        n = len(v)
        span = (self.T - 1) * self.stride
        lo, hi = -span // 4, max(n - 1 - span + span // 4, -span // 4)  # allow partial edge padding
        start = rng.randint(lo, hi)
        idx = window_indices(n, start, self.T, self.stride)
        flip = self.hflip and rng.random() < 0.5
        frames = np.stack([load_rgb(v.frames[j], self.size) for j in idx])
        c0 = self.T // 4
        tgt = idx[c0:c0 + self.T // 2]
        dens = np.stack([density_small(load_gray(v.maps[j]), self.out_size) for j in tgt])
        fixs = np.stack([fixations_small(load_gray(v.fixations[j]), self.out_size) for j in tgt])
        if flip:
            frames = frames[:, :, ::-1]
            dens = dens[:, :, ::-1]
            fixs = fixs[:, :, ::-1]
        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2),
            "density": torch.from_numpy(np.ascontiguousarray(dens)),
            "fixations": torch.from_numpy(np.ascontiguousarray(fixs)),
        }


class SaliconDataset(Dataset):
    """Static images (repeated over T frames) with density/fixation targets, for static co-training."""

    def __init__(self, root: str, split: str = "train", T: int = 16, size=(256, 256), out_size=(64, 64),
                 hflip: bool = True, limit: int | None = None):
        r = Path(root) / "extracted"
        ims = sorted((r / "images" / split).glob("*.jpg"))
        if limit:
            ims = ims[:limit]
        self.items = [(str(p), str(r / "maps" / split / (p.stem + ".png")), str(r / "fixations" / split / (p.stem + ".mat")))
                      for p in ims]
        self.T, self.size, self.out_size, self.hflip = T, size, out_size, hflip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import scipy.io as sio
        ip, mp, fp = self.items[i]
        img = load_rgb(ip, self.size)
        m = load_gray(mp)
        dens = density_small(m, self.out_size)
        g = sio.loadmat(fp)["gaze"]
        H, W = m.shape
        fixs = np.zeros(self.out_size, np.float32)
        for s in range(g.shape[0]):
            f = g[s][0]["fixations"]
            if f.size == 0:
                continue
            x = np.clip(f[:, 0].astype(int) - 1, 0, W - 1)
            y = np.clip(f[:, 1].astype(int) - 1, 0, H - 1)
            np.add.at(fixs, ((y * self.out_size[0]) // H, (x * self.out_size[1]) // W), 1.0)
        if self.hflip and random.random() < 0.5:
            img, dens, fixs = img[:, ::-1], dens[:, ::-1], fixs[:, ::-1]
        frames = np.repeat(img[None], self.T, axis=0)
        half = self.T // 2
        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2),
            "density": torch.from_numpy(np.ascontiguousarray(np.repeat(dens[None], half, 0))),
            "fixations": torch.from_numpy(np.ascontiguousarray(np.repeat(fixs[None], half, 0))),
        }


# ----------------------------------------------------------------------------------------------- inference
def inference_windows(n: int, T: int = 16, stride: int = 2):
    """Cover every frame of an n-frame video with windows whose central T/2 sampled frames span hop = T/2*stride
    original frames. Yields (window indices, list of (orig_frame, sampled_pos_float)) assignments.

    The central sampled positions are k = T/4 .. 3T/4-1, i.e. original offsets start + k*stride. Original frame
    f between two sampled frames gets a fractional sampled position used for linear interpolation in time.
    """
    hop = (T // 2) * stride
    c0 = (T // 4) * stride  # offset of the first central sampled frame from the window start
    start = -c0
    while start + c0 < n:
        idx = [start + k * stride for k in range(T)]
        assign = []
        for f in range(start + c0, min(start + c0 + hop, n)):
            assign.append((f, (f - start) / stride))
        yield [min(max(j, 0), n - 1) for j in idx], assign
        start += hop
