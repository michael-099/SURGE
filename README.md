# SURGE

Video saliency prediction research: the SURGE-D proposal, its fact-check, and an independent implementation with results.

- **`SURGE_final_proposal.md`**: the fact-checked proposal (SURGE-D), written before the implementation.
- **`research/log.md`**: the research record. Every decision rule was written there *before* the runs it governs;
  results are added with the run folders they come from.
- **`surge/`, `scripts/`**: the implementation (independent of any other project's code).

## Results

Main model: **frozen V-JEPA 2 ViT-g + SURGE-D decoder** (SimpleFPN-3D, (2+1)D top-down decoder with auxiliary heads,
learned center prior; 7.4M trainable parameters), 16 frames at stride 2, 256×256, 16 output frames per pass.
The first version used ViT-L (6.99M trainable); its numbers are kept for comparison. Every test set below was scored
once, with all settings fixed beforehand on held-out training data (`research/log.md`).

**DHF1K val** (the official 100 validation videos, used as a test set; test labels are held out by the authors):

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM (released weights; maps scored by the same evaluator) | 0.5639 | 3.326 | 0.4130 | 1.297 | 0.9243 | 0.7406 |
| SURGE-D ViT-L, single / 3-seed ensemble | 0.5836 / 0.5910 | 3.349 / 3.418 | 0.4492 / 0.4486 | 1.222 / 1.187 | 0.9283 / 0.9302 | 0.7636 / 0.7614 |
| **SURGE-D ViT-g, single** | **0.6003** | **3.457** | **0.4531** | **1.170** | **0.9308** | **0.7709** |
| **SURGE-D ViT-g, 3-seed ensemble** | **0.6029** | **3.486** | **0.4574** | **1.160** | **0.9314** | 0.7680 |

Paired bootstrap over videos, ΔCC [95 % CI]: ViT-g vs SalFoM +0.036 [+0.026, +0.048] (single), +0.039 [+0.029, +0.050]
(ensemble); ViT-g vs ViT-L +0.017 (single), +0.012 (ensemble). Better than SalFoM on all six metrics.

**UCF Sports test** (fine-tuned): ViT-L 0.725, ViT-g 0.731 (not a significant gain), ViT-L 3-seed ensemble **0.733** CC.
Above SalFoM (0.631) and TMFI-Net (0.707), tied with ViNet-A (0.734), below ViNet-E (0.744): not state of the art on UCF.

**STAViS audio-visual datasets** (fine-tuned per fold, fold-mean CC; SURGE-D uses no audio here):

| | DIEM | Coutrot1 | Coutrot2 | ETMD | AVAD | SumMe |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L, single | 0.706 | 0.632 | 0.816 | 0.662 | 0.718 | 0.581 |
| SURGE-D ViT-L, 3-seed ensemble | **0.714** | **0.641** | 0.807 | **0.670** | 0.729 | **0.593** |
| best published visual-only | 0.701 | 0.623 | 0.862 | 0.653 | 0.742 | 0.561 |
| best published audio-visual | 0.685 | 0.661 | 0.845 | 0.667 | 0.746 | 0.573 |

The ensemble is clearly above the best visual-only model on DIEM, Coutrot1, ETMD and SumMe, and below it on Coutrot2
and AVAD (where gaze follows the speaker).

**Hollywood-2 with the Vig, Dorr & Cox gaze data** (884 test clips; *not comparable to published Hollywood-2 results*,
which use different viewers and shots): zero-shot 0.632, fine-tuned 0.680, 3-seed ensemble 0.684 CC.

What the experiments showed about the proposal (details and tables in `research/log.md`):

| Proposal component | Outcome |
|---|---|
| V-JEPA 2.1 ViT-L backbone | Replaced: V-JEPA 2 ViT-L is better on the holdout (0.578 vs 0.569 CC); UMT-L 0.546; V-JEPA 2 ViT-g better still (0.585 vs 0.578, 3-seed means) |
| Predictor "surprise" (SCM) | Gate G2 failed: surprise is anti-correlated with fixations; kept off |
| LoRA fine-tuning | Worse than the frozen backbone in 3/3 seeds (−0.0025 CC); not used |
| Learned center prior, auxiliary heads, SALICON co-training | No measurable effect (single-seed ablations) |
| Short schedule | All decoders overfit after 1–2 epochs; 3-epoch schedule used |
| Multi-frame output, frozen foundation features, light decoder | Kept; together they beat SalFoM |

## Reproduce

```
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install transformers safetensors av numpy pillow scipy opencv-python-headless pyyaml pytest timm einops
.venv\Scripts\python -m pytest -q tests                                  # 30 CPU tests
powershell -File scripts\safety_monitor.ps1                               # start first (separate window)
bash scripts/chain.sh configs/main_b1.yaml                                # B1 seed 0 (3 epochs: add --set overrides)
MAIN=main_b1 bash scripts/final_eval.sh                                   # sigma tuning, DHF1K val, UCF
bash scripts/queue_improve.sh                                             # ViT-g (gate, seeds, val, UCF); seed ensembles
```

Data paths (`surge/config.py`, `surge/backbones.py`): DHF1K `D:/DHF1K`; UCF Sports, SALICON under `D:/Datasets`;
V-JEPA 2.1 weights and code in `D:/models/vjepa2_1`; UMT-L in `D:/models/umt`. Checkpoints (`*.pt`) are not in git.

`bench/` holds the earlier CPU replication of the V-JEPA 2 surprise diagnostics. The original design and
reality-check documents are in git history (commit `199de7a`).
