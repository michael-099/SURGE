# SURGE

Video saliency prediction research: the SURGE-D proposal, its fact-check, and an independent implementation with results.

- **`SURGE_final_proposal.md`**: the fact-checked proposal (SURGE-D), written before the implementation.
- **`research/log.md`**: the research record. Every decision rule was written there *before* the runs it governs;
  results are added with the run folders they come from.
- **`surge/`, `scripts/`**: the implementation (independent of any other project's code).
- **`Paper/msc-thesis/`**: the MSc thesis written from these results (`main.pdf`), with an area-chair-style review and
  the record of what was done about it in `review/area-chair-review.md`.

## Results

Main model: **frozen V-JEPA 2 ViT-g + SURGE-D decoder** (SimpleFPN-3D, (2+1)D top-down decoder with auxiliary heads,
learned center prior; 7.4M trainable parameters), 16 frames at stride 2, 256×256, 16 output frames per pass.
The first version used ViT-L (6.99M trainable); its numbers are kept for comparison. Every test set below was scored
once, with all settings fixed beforehand on held-out training data (`research/log.md`).

**DHF1K val** (the official 100 validation videos, used as a test set; test labels are held out by the authors):

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM (released weights; maps scored by the same evaluator) | 0.5639 | 3.326 | 0.4130 | 1.297 | 0.9243 | 0.7406 |
| SalFoM with the blur that is best for it on this set (σ 16; an upper bound for the baseline) | 0.5741 | 3.265 | 0.4009 | 1.280 | 0.9277 | 0.7557 |
| SURGE-D ViT-L, single / 3-seed ensemble | 0.5836 / 0.5910 | 3.349 / 3.418 | 0.4492 / 0.4486 | 1.222 / 1.187 | 0.9283 / 0.9302 | 0.7636 / 0.7614 |
| **SURGE-D ViT-g, single** | **0.6003** | **3.457** | **0.4531** | **1.170** | **0.9308** | **0.7709** |
| **SURGE-D ViT-g, 3-seed ensemble** | **0.6029** | **3.486** | **0.4574** | **1.160** | **0.9314** | 0.7680 |

Paired bootstrap over videos, ΔCC [95 % CI]: ViT-g vs SalFoM +0.036 [+0.026, +0.048] (single), +0.039 [+0.029, +0.050]
(ensemble); ViT-g vs ViT-L +0.017 (single), +0.012 (ensemble). Better than SalFoM on all six metrics.

Two controls, each under a rule written before it was run (`research/log.md` section 12):

- **Baseline blur.** Against SalFoM's maps with their best blur, the ViT-g ensemble stays ahead by +0.029
  [+0.019, +0.039] and every ViT-g seed stays ahead; with ViT-L only the ensemble does (+0.017 [+0.007, +0.027]).
- **Identical protocol.** SalFoM sees only past frames; offline SURGE-D also sees later ones. Retrained to predict the
  last frame of its window and scored on every tenth frame, the causal ViT-g ensemble reaches 0.602 CC, +0.029
  [+0.019, +0.040] over SalFoM at its best blur on the same frames, and the causal ViT-L ensemble 0.590, +0.017
  [+0.007, +0.028]. Both are as accurate as their offline versions, so the advantage does not come from future
  frames. A causal model needs one forward pass per output frame, which gives up the offline saving.

**UCF Sports test** (fine-tuned): ViT-L 0.725, ViT-L 3-seed ensemble 0.733, ViT-g 0.731, ViT-g 3-seed ensemble
**0.735** CC (ViT-g vs ViT-L ensemble +0.002, not significant). Above SalFoM (0.631) and TMFI-Net (0.707), tied with
ViNet-A (0.734), below ViNet-E (0.744): not state of the art on UCF.

**STAViS audio-visual datasets** (fine-tuned per fold, fold-mean CC; SURGE-D uses no audio here):

| | DIEM | Coutrot1 | Coutrot2 | ETMD | AVAD | SumMe |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L, single / 3-seed ensemble | 0.706 / 0.714 | 0.632 / 0.641 | 0.816 / 0.807 | 0.662 / 0.670 | 0.718 / 0.729 | 0.581 / **0.593** |
| SURGE-D ViT-g, single | 0.714 | 0.648 | 0.820 | 0.672 | 0.731 | 0.586 |
| SURGE-D ViT-g, 3-seed ensemble | **0.729** | 0.657 | 0.819 | **0.676** | 0.740 | 0.590 |
| best published visual-only | 0.701 | 0.623 | **0.862** | 0.653 | 0.742 | 0.561 |
| best published audio-visual | 0.685 | **0.661** | 0.845 | 0.667 | **0.746** | 0.573 |

The ViT-g ensemble is clearly above the best visual-only model on DIEM, Coutrot1, ETMD and SumMe, ties it on AVAD,
and is below it on Coutrot2 (gaze follows the speaker). Versus the best audio-visual model: above on DIEM and SumMe,
within 0.01 on Coutrot1, ETMD and AVAD, below on Coutrot2. An audio branch (frozen AST features, FiLM + audio-visual
correlation fusion) did not improve any dataset (`research/log.md` 11D).

**Hollywood-2 with the Vig, Dorr & Cox gaze data** (884 test clips; *not comparable to published Hollywood-2 results*,
which use different viewers and shots): zero-shot 0.632; fine-tuned ViT-L 0.680 / 3-seed ensemble 0.684; ViT-g 0.684 / 3-seed ensemble **0.688** CC
(ViT-g better than ViT-L on all six metrics as a single model).

What the experiments showed about the proposal (details and tables in `research/log.md`):

| Proposal component | Outcome |
|---|---|
| V-JEPA 2.1 ViT-L backbone | Replaced: V-JEPA 2 ViT-L is better on the holdout (0.578 vs 0.569 CC); UMT-L 0.546; V-JEPA 2 ViT-g better still (0.585 vs 0.578, 3-seed means) |
| Predictor "surprise" (SCM) | Gate G2 failed: surprise is anti-correlated with fixations; kept off. Re-scored with the predictor's trained mask token (the HF port reads an untrained one by default) and with a masked-tube readout that follows pre-training (section 13B/13C): still NO-GO. Combined with the center prior, frame difference and atypicality, the error adds +0.069 [+0.035, +0.107] (raw) and +0.062 [+0.032, +0.093] (masked tube) bits per fixation at ~4 fps, but with a **negative** weight: people look away from high-error regions |
| LoRA fine-tuning | Worse than the frozen backbone in 3/3 seeds (−0.0025 CC); not used |
| Learned center prior, auxiliary heads | Three seeds: removing either is equivalent to keeping it (−0.0010 CC each, 90 % interval within ±0.003) |
| SALICON co-training, frame stride 4 | Three seeds: inconclusive (+0.0009 and +0.0029 CC; stride 4 is positive in every seed at half the cost per output frame) |
| Short schedule | All decoders overfit after 1–2 epochs; 3-epoch schedule used |
| Multi-frame output, frozen foundation features, light decoder | Kept; together they beat SalFoM. The gain comes from the V-JEPA 2 encoder, not from freezing as such: frozen UMT-L with the same decoder scores 0.545 on val, below SalFoM |
| Later checks (section 11) | Training on all 600 DHF1K videos: no gain; 32-frame clips: missed the pre-set bar by 0.0002; audio: no gain |
| Inference cost against SalFoM (section 13D) | Released SalFoM and SURGE-D timed on the same RTX A5000 (bf16; three runs, medians within 5.1 %). Per output frame: SalFoM 202–229 ms; offline SURGE-D ViT-g 7.7–10.4 ms and ViT-L 3.2–5.5 ms, so SalFoM needs 22–26× and 41–64× their time. Causal SURGE-D (one pass per output frame): ViT-L 0.25–0.39× and ViT-g 0.61–0.73× SalFoM's time. Encoder TFLOPs per output frame: SalFoM 2.87, SURGE-D ViT-L 0.11 and ViT-g 0.33 offline, 1.82 and 5.26 causal |
| Is the model limited by the amount of training data? (section 13A) | Open. Holdout CC with 135 / 270 / 540 training videos (ViT-L, three seeds): 0.5645 / 0.5729 / 0.5783. The first doubling is an effect (+0.0084 [+0.0004, +0.0164]); the second is inconclusive (+0.0054 [−0.0011, +0.0112]) |

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
