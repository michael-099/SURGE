"""Figures for the thesis, built only from stored outputs in the SURGE repository (no model is run).

    python Paper/msc-thesis/figures/src/make_plots.py          (run from the repository root)

Writes into Paper/msc-thesis/figures/:
  g1_curves.pdf        holdout and training CC per epoch of the three G1 backbone runs (runs/g1_*/log.jsonl)
  val_delta.pdf        per-video CC difference to SalFoM on DHF1K val (runs/vitg/val_ens3.json,
                       runs/metric_check/salfom_val_all.json)
  sigma.pdf            holdout CC against the output blur sigma (runs/final/sigma_*.json, runs/vitg/sigma_*.json)
  qualitative.pdf      DHF1K-val examples (runs/thesis/qualitative.npz, from scripts/thesis_qualitative.py)
  learning_curve.pdf   holdout CC against the number of training videos (runs/review/learning_curve.json, from
                       scripts/learning_curve.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Paper/msc-thesis/figures"
TEXTW = 6.1  # inches; text width of the thesis class (A4, 3 cm + 2.5 cm margins)



def latin_modern() -> list[str]:
    """Register Latin Modern Roman, the font of the thesis text, from the TeX distribution (found with kpsewhich).

    Returns the family name, or an empty list if no TeX distribution is found; the figures then fall back to the
    Computer Modern font bundled with matplotlib, from which Latin Modern is derived."""
    import shutil
    import subprocess

    from matplotlib import font_manager

    kpse = shutil.which("kpsewhich")
    if kpse is None:
        return []
    found = False
    for style in ("regular", "italic", "bold", "bolditalic"):
        path = subprocess.run([kpse, f"lmroman10-{style}.otf"], capture_output=True, text=True).stdout.strip()
        if path:
            font_manager.fontManager.addfont(path)
            found = True
    return ["Latin Modern Roman"] if found else []


LM = latin_modern()
plt.rcParams.update({
    "font.family": "serif", "font.serif": LM + ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": not LM,
    "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
C = {"vjepa2": "#1f5f7a", "vjepa21": "#c0711f", "umt": "#7a7a7a", "salfom": "#8c2d2d"}


def events(run: str, kind: str):
    rows = [json.loads(ln) for ln in open(ROOT / "runs" / run / "log.jsonl") if ln.strip()]
    rows = [r for r in rows if r.get("event") == kind]
    out = {}
    for r in rows:  # keep the last record per epoch (resumed runs may log an epoch twice)
        out[r["epoch"]] = r
    return [out[k] for k in sorted(out)]


def g1_curves():
    fig, ax = plt.subplots(1, 2, figsize=(TEXTW, 2.3))
    runs = [("g1_vjepa2_l", "V-JEPA 2 ViT-L", C["vjepa2"]), ("g1_vjepa21_l", "V-JEPA 2.1 ViT-L", C["vjepa21"]),
            ("g1_umt_l", "UMT-L", C["umt"])]
    for run, label, col in runs:
        h = events(run, "holdout")
        t = events(run, "train_epoch")
        ax[0].plot([r["epoch"] + 1 for r in h], [r["cc"] for r in h], "-o", ms=3, color=col, label=label)
        ax[1].plot([r["epoch"] + 1 for r in t], [r["cc"] for r in t], "--o", ms=3, color=col, label=label)
        b = max(h, key=lambda r: r["cc"])
        ax[0].plot([b["epoch"] + 1], [b["cc"]], "o", ms=7, mfc="none", color=col)
    ax[0].set_title("(a) Internal holdout (60 videos)")
    ax[1].set_title("(b) Training clips (running mean per epoch)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("CC")
        a.set_xticks(range(1, 11))
        a.grid(alpha=0.25, lw=0.5)
    ax[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "g1_curves.pdf")
    plt.close(fig)


def val_delta():
    ours = json.load(open(ROOT / "runs/vitg/val_ens3.json"))["per_video"]
    sal = json.load(open(ROOT / "runs/metric_check/salfom_val_all.json"))["per_video"]
    vids = sorted(set(ours) & set(sal))
    d = np.array(sorted(ours[v]["cc"] - sal[v]["cc"] for v in vids))
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(TEXTW, 2.1))
    cols = np.where(d > 0, C["vjepa2"], C["salfom"])
    ax.bar(np.arange(len(d)), d, width=0.85, color=cols)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(d.mean(), color="black", lw=0.8, ls="--")
    ax.text(len(d) * 0.02, d.mean() + 0.006, f"mean {d.mean():+.3f}  (95% CI [{lo:+.3f}, {hi:+.3f}])", fontsize=8)
    ax.set_xlim(-1, len(d))
    ax.set_xticks([])
    ax.set_xlabel(f"DHF1K validation videos, sorted ({int((d > 0).sum())} of {len(d)} above zero)")
    ax.set_ylabel(r"$\Delta$CC (SURGE-D $-$ SalFoM)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "val_delta.pdf")
    plt.close(fig)


def sigma():
    files = [("final/sigma_single.json", "ViT-L, single", C["vjepa21"], "--"),
             ("final/sigma_ens3.json", "ViT-L, 3-seed ensemble", C["vjepa21"], "-"),
             ("vitg/sigma_single.json", "ViT-g, single", C["vjepa2"], "--"),
             ("vitg/sigma_ens3.json", "ViT-g, 3-seed ensemble", C["vjepa2"], "-")]
    fig, ax = plt.subplots(figsize=(TEXTW * 0.55, 2.2))
    for f, label, col, ls in files:
        s = json.load(open(ROOT / "runs" / f))
        xs = sorted(float(k) for k in s["scores"])
        ax.plot(xs, [s["scores"][str(x)]["cc"] for x in xs], ls, marker="o", ms=3, color=col, label=label)
    ax.set_xlabel(r"Gaussian blur $\sigma$ (pixels at 640$\times$360)")
    ax.set_ylabel("Holdout CC")
    ax.set_xticks([0, 4, 8, 12, 16, 24])
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "sigma.pdf")
    plt.close(fig)


def qualitative():
    p = ROOT / "runs/thesis/qualitative.npz"
    meta = ROOT / "runs/thesis/qualitative.json"
    if not p.exists():
        print("qualitative.npz missing; skipped")
        return
    z = np.load(p)
    info = json.load(open(meta))
    order = {"p10": 0, "p50": 1, "p90": 2, "best_vs_salfom": 3, "worst_vs_salfom": 4}
    names = sorted(info, key=lambda k: order.get(info[k]["tag"], 9))
    tag_txt = {"p10": "10th pct.", "p50": "median", "p90": "90th pct.",
               "best_vs_salfom": "max. gain", "worst_vs_salfom": "max. loss"}
    cmap = plt.get_cmap("inferno")
    # 4 panels of 16:9 frames across the text width (~1.5 x 0.85 in each), plus room for labels
    fig, axes = plt.subplots(len(names), 4, figsize=(TEXTW, 1.07 * len(names) + 0.3))
    if len(names) == 1:
        axes = axes[None]
    for i, n in enumerate(names):
        img, gt, fix = z[f"{n}__img"], z[f"{n}__gt"], z[f"{n}__fix"]
        maps = [None, gt, z[f"{n}__salfom"], z[f"{n}__ours"]]
        for j in range(4):
            a = axes[i, j]
            a.imshow(img)
            if j == 0:
                ys, xs = np.nonzero(fix)
                a.scatter(xs, ys, s=4, c="#ffd400", edgecolors="black", linewidths=0.3)
            else:
                m = maps[j].astype(np.float64)
                m = (m - m.min()) / (m.max() - m.min() + 1e-12)
                rgba = cmap(m)
                rgba[..., 3] = 0.9 * np.clip(m, 0, 1) ** 0.6  # opacity follows saliency; low values stay clear
                a.imshow(rgba)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        r = info[n]
        axes[i, 0].set_ylabel(f"{n}\n{tag_txt.get(r['tag'], r['tag'])}", fontsize=7)
        axes[i, 2].set_xlabel(f"CC {r['cc_salfom_frame']:.2f}", fontsize=7, labelpad=1)
        axes[i, 3].set_xlabel(f"CC {r['cc_ours_frame']:.2f}", fontsize=7, labelpad=1)
    for j, t in enumerate(["Frame and fixations", "Ground truth", "SalFoM", "SURGE-D (ViT-g, ensemble)"]):
        axes[0, j].set_title(t, fontsize=8)
    fig.subplots_adjust(wspace=0.03, hspace=0.32)
    fig.savefig(OUT / "qualitative.pdf", dpi=200)
    plt.close(fig)


def learning_curve(path=None):
    p = Path(path) if path else ROOT / "runs/review/learning_curve.json"
    if not p.exists():
        print("learning_curve.json missing; skipped")
        return
    arms = json.load(open(p))["arms"]
    ns = sorted(int(k) for k in arms)
    seeds = np.array([arms[str(n)]["cc"]["per_seed"] for n in ns])  # arms x seeds
    mean = seeds.mean(1)
    col = C["vjepa2"]
    fig, ax = plt.subplots(figsize=(TEXTW * 0.62, 2.4))
    for s in range(seeds.shape[1]):  # the subsets are nested within a seed, so each seed is one curve
        ax.plot(ns, seeds[:, s], "-o", lw=0.7, ms=3, color=col, alpha=0.45, mfc="white", zorder=2)
    ax.plot(ns, mean, "-o", lw=1.8, ms=5.5, color=col, zorder=3)
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]  # keeps the labels clear of the seed curves
    for n, m in zip(ns, mean):
        ax.annotate(f"{m:.3f}", (n, m), textcoords="offset points", xytext=(-6, 6), ha="right", va="bottom",
                    fontsize=8, color="black", path_effects=halo, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlim(ns[0] / 1.35, ns[-1] * 1.2)
    lo, hi = seeds.min(), seeds.max()
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.set_xlabel("Training videos (logarithmic scale)")
    ax.set_ylabel("Holdout CC")
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(OUT / (p.stem + ".pdf") if path is None else Path(path).with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    g1_curves()
    val_delta()
    sigma()
    qualitative()
    learning_curve()
    print("figures written to", OUT)
