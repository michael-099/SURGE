"""YAML configs with inheritance (`base:` key) and command-line overrides (`a.b=value`)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "run_dir": "runs/debug",
    "data": {
        "dataset": "dhf1k", "root": "D:/DHF1K",
        "train_split": "configs/splits/dhf1k_train540.txt", "holdout_split": "configs/splits/dhf1k_holdout60.txt",
        "val_split": "configs/splits/dhf1k_val100.txt",
        "T": 16, "stride": 2, "size": [256, 256], "hflip": True,
        "salicon": False, "salicon_root": "D:/Datasets/SALICON", "salicon_limit": None,
    },
    "model": {"backbone": "vjepa21_l", "scm": False, "prior": True, "aux": True, "channels": 256, "expert": None},
    "train": {
        "stage": 1, "seed": 0, "epochs": 10, "clips_per_video": 8, "batch": 8, "workers": 12,
        "lr": 3e-4, "lr_prior": 1e-3, "lr_lora": 1e-4, "wd": 0.01, "warmup_epochs": 0.5,
        "w_cc": 1.0, "w_nss": 0.1, "w_aux": 0.25, "lora_r": 16, "lora_alpha": 16.0,
        "init_from": None, "grad_ckpt": False, "log_every": 50, "ckpt_every": 500,
    },
    "eval": {"sigma": 0.0, "frame_step": 5, "workers": 12},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, overrides=()) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    y = yaml.safe_load(p.read_text()) or {}
    base = DEFAULTS
    if "base" in y:
        base = load_config(p.parent / y.pop("base"))
    cfg = _merge(base, y)
    for o in overrides:
        k, v = o.split("=", 1)
        cur = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur[kk]
        cur[keys[-1]] = yaml.safe_load(v)
    return cfg
