# SURGE

Video saliency prediction research: the SURGE-D proposal, its fact-check, and an independent implementation with results.

- **`SURGE_final_proposal.md`**: the fact-checked proposal (SURGE-D), written before the implementation.
- **`research/log.md`**: the research record. Every decision rule was written there *before* the runs it governs;
  results are added with the run folders they come from.
- **`surge/`, `scripts/`**: the implementation (independent of any other project's code).

## Results (DHF1K val = the official 100 validation videos, used as a test set)

Main model: **frozen V-JEPA 2 ViT-L + SURGE-D decoder** (SimpleFPN-3D, (2+1)D top-down decoder with auxiliary heads,
learned center prior; 6.99M trainable parameters), 16 frames at stride 2, 256×256, 16 output frames per pass.

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM (released weights; maps scored by the same evaluator) | 0.5639 | 3.326 | 0.4130 | 1.297 | 0.9243 | 0.7406 |
| **SURGE-D, single model** | **0.5836** | **3.349** | **0.4492** | **1.222** | **0.9283** | **0.7636** |
| **SURGE-D, 3-seed ensemble** | **0.5910** | **3.418** | **0.4486** | **1.187** | **0.9302** | **0.7614** |

Paired bootstrap over videos vs SalFoM: ΔCC +0.020 [+0.010, +0.031] (single), +0.027 [+0.017, +0.038] (ensemble).
The single model ties SalFoM on NSS; the ensemble is better on all six metrics. DHF1K **test** labels are held out by
the dataset authors, so no test-server number exists yet.

**UCF Sports test** (fine-tuned): CC 0.725, NSS 3.92. Above SalFoM (0.631) and TMFI-Net (0.707), below ViNet-E
(0.744) and ViNet-A (0.734): not state of the art on UCF.

What the experiments showed about the proposal (details and tables in `research/log.md`):

| Proposal component | Outcome |
|---|---|
| V-JEPA 2.1 ViT-L backbone | Replaced: V-JEPA 2 ViT-L is better on the holdout (0.578 vs 0.569 CC); UMT-L 0.546 |
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
.venv\Scripts\python -m pytest -q tests                                  # 27 CPU tests
powershell -File scripts\safety_monitor.ps1                               # start first (separate window)
bash scripts/chain.sh configs/main_b1.yaml                                # B1 seed 0 (3 epochs: add --set overrides)
MAIN=main_b1 bash scripts/final_eval.sh                                   # sigma tuning, DHF1K val, UCF
```

Data paths (`surge/config.py`, `surge/backbones.py`): DHF1K `D:/DHF1K`; UCF Sports, SALICON under `D:/Datasets`;
V-JEPA 2.1 weights and code in `D:/models/vjepa2_1`; UMT-L in `D:/models/umt`. Checkpoints (`*.pt`) are not in git.

`bench/` holds the earlier CPU replication of the V-JEPA 2 surprise diagnostics. The original design and
reality-check documents are in git history (commit `199de7a`).
