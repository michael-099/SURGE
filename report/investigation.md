# Systematic investigation of SURGE-D: behaviour, limitations and research gaps

**Status:** in progress (last updated 2026-11-15). Results arrive from the queues of research log section 15.
**Object:** SURGE-D ViT-g, the final model of the thesis: a frozen V-JEPA 2 ViT-g/16 encoder (1,012 M parameters)
and a 7.39 M-parameter decoder, unchanged. Reference runs `runs/vitg_s{0..5}`.
**Rules:** every experiment was specified, with its decision rule, in `research/log.md` section 15 and committed before
it ran (rules `5e6fb30`; the scene-cut hypothesis 15L `134513c`; one amendment before any affected run).

**Labels used throughout.**
- **[Completed]**: run and analysed in this investigation; numbers come from the files cited.
- **[Thesis]**: completed earlier for the thesis (research log sections 1–14); cited, not re-run.
- **[Pending]**: pre-registered and queued; no result exists yet. Nothing in this report depends on its outcome.
- **[Proposed]**: designed here, not queued, no result.
- **[Hypothesis]**: an explanation that the cited evidence suggests but does not establish.

---

## Contents
1. Baseline reproduction and evaluation
2. Experimental methodology
3. Component-wise ablation studies
4. Sensitivity and interaction analysis
5. Computational efficiency analysis
6. Robustness and generalisation evaluation
7. Failure cases and unexpected observations
8. Comparison with state-of-the-art methods
9. Experimentally identified architectural limitations
10. Literature-validated research gaps
11. Ranked potential research contributions
12. Recommended research direction
13. Proposed experiments for validating the improvement

---

## 1. Baseline reproduction and evaluation

| Check | Result | Status, evidence |
|---|---|---|
| A1: seed-0 checkpoint re-scored on DHF1K val (all 60,273 frames, six metrics, σ 8) | **Identical** to the stored result on every video and metric (max difference 0.0). CC 0.6003, NSS 3.457, SIM 0.453, KLD 1.170, AUC-J 0.931, sAUC 0.771 | [Completed] `runs/review/inv/a1_val_s0.json` |
| New test-time evaluator vs the thesis evaluator (holdout, 60 videos) | Identical per video (max difference 0.0), checked before any other result of it was used | [Completed] `runs/review/inv/t_orig_s0.json` |
| Seed variability (6 seeds, holdout, best epoch) | CC 0.5861 ± 0.0017 (SD; range 0.5829–0.5881) | [Thesis] `runs/vitg_s{0..5}/log.jsonl` |
| A2: seed 0 retrained with the identical configuration (GPU non-determinism) | — | [Pending] `runs/inv_repro_s0` |
| A3: parameters, FLOPs, time and memory at several input sizes and clip lengths | — | [Pending]; thesis values in section 5 |

**Configuration.** One configuration file (`configs/main_b1.yaml`) with command-line overrides; every run directory
stores its resolved `config.json`. Seeds fix clip sampling, order, augmentation and initialisation; cuDNN kernels are
not forced to be deterministic, which A2 quantifies.

**Convergence [Thesis logs, re-read].** All six ViT-g seeds reach their best holdout CC at the second of three epochs
and lose 0.001–0.004 in the third, while training CC keeps rising (0.640 → 0.676). The 10-epoch ViT-L run of gate G1
peaked at epoch 2 (0.578) and fell monotonically to 0.522 at epoch 9. The decoder starts to overfit the 540 training
videos after about two epochs; the train–holdout gap at the last epoch is ≈0.09 CC. A six-epoch run with a schedule
fitted to six epochs is part of 15D [Pending].

**Learning curve [Thesis, ViT-L, 3 seeds].** 135 → 270 → 540 training videos: 0.5645 → 0.5729 → 0.5783 holdout CC;
the second doubling gives 0.64 of the first doubling's gain. DHF1K has no further annotated training videos.

**Discrepancies investigated.**
- The encoder of the Hugging Face V-JEPA 2 port derives token positions from the `crop_size` in its configuration.
  Feeding another input size without rebuilding the encoder would silently use wrong positions. The code now rebuilds
  the encoder for other sizes and raises an error on a mismatch. All earlier runs used 256² and were not affected
  (implementation hazard, not a bug in any reported result).
- GroupNorm in the decoder pools statistics over the whole clip, so the decoder mixes time steps even where it has no
  temporal convolution. This is a property of the design, found by a CPU test; it changed the design of ablation V3
  (section 3), not any earlier result.

---

## 2. Experimental methodology

**Pre-registration.** Each experiment's design and decision rule were written in `research/log.md` section 15 and
committed before the first run.

**Data use.**
- *Decisions and diagnostics:* the DHF1K holdout (60 of DHF1K's 600 training videos, never trained on), every fifth
  frame (6,609 frames), no blur (σ 0), CC, NSS, SIM and KLD. This is the protocol of every ablation in the thesis.
- *DHF1K val (the thesis test set):* used only to re-score the existing checkpoint (A1) and, descriptively, through
  per-video scores that already existed. No model, setting or threshold in this report was selected on it.
- *Other datasets* (UCF Sports, Hollywood-2 with the Vig gaze, the six STAViS datasets, AViMoS, MVVA) are scored
  zero-shot with a blur fixed in advance (σ = 8 × width/640), or with the existing fine-tuned fold models.

**Statistics (rule of 12E).** Paired differences per video, averaged over seeds. Two-level bootstrap: seeds, then
videos; 10,000 resamples; seed 0.
- **Effect:** the 95 % interval excludes 0 and |Δ| ≥ 0.003 CC.
- **Equivalent:** the 90 % interval lies within ±0.003.
- **Inconclusive:** anything else.

Test-time conditions have one seed, so they use the video bootstrap only. They measure the behaviour of one trained
model, not of the training recipe.

**Tools written for this investigation.** All are off by default, and the default paths are tested to be unchanged
(45 tests).
- `scripts/probe_eval.py`: test-time probes. It covers temporal order, stride, input size, tilings, perturbations,
  per-position scores, flip consistency and shot clamping. With its defaults it reproduces the thesis evaluator
  exactly.
- `scripts/train_multi.py`: trains several decoders on one frozen-encoder stream, with the same clips, order and
  augmentation as the standard trainer. Ablations are therefore paired by construction, and the encoder runs once
  for all of them.
- Decoder options `tap_mode`, `temporal` (per-frame norms), `topdown`, and `LinearReadout`. The default decoder is
  tested to be identical, tensor for tensor, to the committed one.
- `scripts/inv_summary.py`: applies the 12E rule. `scripts/inv_cuts.py`: the scene-cut analysis.

**Experiment register.**

| ID | Question | Status |
|---|---|---|
| A1, evaluator check | Is the reported baseline reproducible from the stored checkpoint? | Completed |
| A2 | How large is run-to-run non-determinism at a fixed seed? | Pending |
| A3 | Cost: parameters, FLOPs, time, memory | Pending (thesis values available) |
| B | What does each encoder depth carry (41 linear probes)? | Pending |
| C | Decoder ablations V1–V6, 3 seeds | Pending |
| D | Learning rate, loss terms, schedule length | Pending |
| E1, E3 | Temporal order and stride at test time | Completed |
| E2 | Video model vs a static-trained model (primary temporal test) | Pending |
| F1, F2 | Resolution ceilings; test-time input size | Completed |
| F3 | Training at 384² | Pending |
| G | Output position and tiling | Completed |
| H | Perturbations; flip consistency | Running (flip completed) |
| I | Zero-shot on five datasets | Pending |
| J | Failure analysis | Completed |
| K | Multi-face scenes (MVVA) | Pending (needs I) |
| 15L | Scene cuts: cross-shot context or post-cut gaze? (hypothesis test) | Pending |

---

## 3. Component-wise ablation studies

### Earlier ablations [Thesis; ViT-L unless stated, holdout, rule of 12E]

| Variant | ΔCC [95 %] | Outcome |
|---|---|---|
| Without the learned centre prior | −0.0010 [−0.0026, +0.0005] | equivalent (dispensable) |
| Without the auxiliary heads | −0.0010 [−0.0023, +0.0003] | equivalent (dispensable) |
| SALICON co-training | +0.0009 [−0.0039, +0.0058] | inconclusive |
| Stride 4 instead of 2 (training and test) | +0.0029 [−0.0009, +0.0068] | inconclusive; positive in 3/3 seeds |
| LoRA-adapted instead of frozen encoder | −0.0029 / −0.0010 / −0.0037 (seeds) | rejected (lost 3 of 3) |
| ViT-g instead of ViT-L (seed 0) | +0.0083 | adopted |
| 32 instead of 16 frames (ViT-g, seed 0) | +0.0036 | gate missed by 0.0002 |
| Second frozen expert, UMT-L (ViT-g, 6 seeds) | +0.0038, 97.5 % [−0.0009, +0.0093] | not established |
| Predictability-gated decoder (seed 0) | learned to ignore the error maps | not adopted |

Two components of the thesis design are therefore already known to be dispensable within ±0.003 CC.

### Decoder ablations of this investigation (15C) [Pending]

Each variant is trained next to the unchanged decoder R on the same encoder stream (seeds 0–2), and compared with R
of the same run:
- V1 linear readout of the four taps;
- V2 the last tap only;
- V3 no temporal mixing in the decoder (per-frame convolutions and norms);
- V4 no top-down path;
- V5 64 instead of 256 channels;
- V6 neither prior nor auxiliary heads (an interaction test of the two dispensable parts).

### What each encoder depth carries (15B) [Pending]

Forty linear probes, one per ViT-g block, each with a learned prior, and a prior-only readout, all trained on one
encoder stream.

---

## 4. Sensitivity and interaction analysis

### 4.1 Where in its 16-frame window a frame is predicted (15G) [Completed, seed 0, holdout]

Every output position of every default window was scored against the frame it shows (≈2,030 frames per position).

| Output position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CC | 0.520 | 0.552 | 0.577 | 0.587 | 0.586 | 0.588 | 0.588 | 0.596 | 0.597 | 0.597 | 0.598 | 0.598 | 0.599 | 0.597 | 0.594 | 0.595 |

Windows hop by 8 positions, so every frame is predicted twice: at position k of one window and k+8 of the next. On
the same frames (video-level bootstrap, 60 videos), moving a frame 8 positions later gives it 8 more sampled frames
(≈0.53 s) of past. Its CC changes by:

| From → to | 0→8 | 1→9 | 2→10 | 3→11 | 4→12 | 5→13 | 6→14 | 7→15 |
|---|---|---|---|---|---|---|---|---|
| ΔCC [95 %] | +0.077 [+0.065, +0.090] | +0.046 [+0.036, +0.056] | +0.023 [+0.016, +0.030] | +0.012 [+0.007, +0.017] | +0.007 [+0.002, +0.012] | +0.003 [−0.002, +0.008] | −0.000 [−0.005, +0.005] | −0.004 [−0.009, +0.001] |

**Finding.** SURGE-D uses temporal context, and only the past.
- A frame with no preceding frames in its clip loses 0.077 CC.
- The loss roughly halves with each extra past frame, and it is gone once ≈5–6 sampled frames (≈0.35–0.4 s) of
  history are available.
- Extra future context never helps (7→15: −0.004, n.s.).

This is the within-clip counterpart of the thesis result that causal and offline inference do not differ measurably.

### 4.2 Temporal order (15E1) [Completed, seed 0, holdout; video bootstrap]

| Condition (model trained on forward video) | CC | ΔCC [95 %] | ΔNSS | Outcome |
|---|---|---|---|---|
| Original order | 0.5866 | — | — | — |
| Each window's frames reversed in time | 0.5228 | −0.064 [−0.079, −0.050] | −0.38 | effect |
| Frames shuffled within each window | 0.5549 | −0.032 [−0.039, −0.024] | −0.22 | effect |
| Static clip (each scored frame repeated 16 times) | 0.5243 | −0.062 [−0.081, −0.046] | −0.39 | effect |

The trained model depends on temporal structure. Removing all motion costs 11 % of its CC, and reversing time costs
as much: twice the cost of shuffling. This fits the past-context result (reversal presents the future as the past).

These conditions are distribution shifts for a decoder trained on real video, so they show dependence, not benefit.
The matched control, a model trained on static clips (E2, three seeds), is [Pending].

**Where the video–static gap is largest [Completed, descriptive, per scored frame].**
- It grows with motion energy: +0.045 in the lowest quintile, +0.096 in the highest.
- It is larger when gaze is off-centre (+0.103 vs +0.025) and when gaze stays in place over five frames (+0.083 vs
  +0.045 when it relocates).

### 4.3 Temporal sampling (15E3) [Completed, seed 0]

| Test-time stride (trained: 2) | Window span at 30 fps | CC | ΔCC [95 %] | Outcome |
|---|---|---|---|---|
| 1 | 0.53 s | 0.5743 | −0.012 [−0.017, −0.008] | effect |
| 2 (as trained) | 1.07 s | 0.5866 | — | — |
| 4 | 2.13 s | 0.5681 | −0.019 [−0.026, −0.011] | effect |

The trained model is tuned to its training frame rate. In both directions the change costs more than the 0.003
margin. This matters for datasets recorded at other frame rates: SumMe and MVVA include 15 fps videos, where stride 2
spans 2.1 s, like stride 4 at 30 fps.

### 4.4 Input size at test time (15F2) [Completed, seed 0; encoder rebuilt for each size]

| Input (trained: 256²) | Tokens per frame | CC | ΔCC [95 %] | ΔNSS | Outcome |
|---|---|---|---|---|---|
| 192² | 12×12 | 0.5395 | −0.047 [−0.059, −0.037] | −0.39 | effect |
| 224² | 14×14 | 0.5726 | −0.014 [−0.021, −0.008] | −0.15 | effect |
| 256² | 16×16 | 0.5866 | — | — | — |
| 320² | 20×20 | 0.5685 | −0.018 [−0.026, −0.011] | **+0.01** | effect (CC) |
| 384² | 24×24 | 0.5379 | −0.049 [−0.062, −0.037] | −0.09 | effect |

Accuracy peaks at the training size. Above it, CC falls but NSS does not at first: at 320² NSS is unchanged
(+0.009) while CC drops by 0.018. Fixation locations are still ranked as well as before; only the shape of the
density changes. [Hypothesis] The maps become sharper or more fragmented at a larger token grid, because the
decoder's receptive fields and centre prior were learned on a 16×16 grid. Whether training at 384² gains accuracy is
the question of F3 [Pending].

### 4.5 Training configuration and interactions (15D, 15C-V6) [Pending]

Learning rate ×1/3 and ×3; no CC term; no NSS term; six epochs; prior × auxiliary-head interaction.

---

## 5. Computational efficiency analysis

### 5.1 Cost of the model [Thesis, section 5 "Computational Cost"; A3 will extend it]

| | ViT-L | ViT-g |
|---|---|---|
| Encoder parameters (frozen) | 303.9 M | 1,012.2 M |
| Decoder parameters (trained) | 6.99 M | 7.39 M |
| Encoder TFLOPs per 16×256² clip | 1.656 | 5.088 |
| Decoder TFLOPs per clip | 0.166 | 0.168 |
| TFLOPs per output frame (offline) | 0.114 | 0.328 |
| Time per output frame, batch 1 / 8 (RTX A5000, bf16) | 5.5 / 3.2 ms | 10.3 / 7.7 ms |
| Peak memory, batch 8 | 3.9 GB | 8.2 GB |

With ViT-g the encoder accounts for 97 % of the operations (ViT-L: 91 %). Offline, one pass serves 16 native
frames. A strictly causal model needs one pass per output frame.

### 5.2 Accuracy against encoder passes: tilings (15G) [Completed, seed 0, holdout]

| Tiling of the 16-position windows | Encoder passes per frame | CC | ΔCC [95 %] | Outcome |
|---|---|---|---|---|
| All 16 positions kept (hop 16 positions) | 0.032 | 0.5803 | −0.0063 [−0.0090, −0.0038] | effect |
| **Default** (hop 8, central 8 kept) | 0.063 | 0.5866 | — | — |
| Central 4 (hop 4) | 0.126 | 0.5880 | +0.0014 [+0.0004, +0.0024] | equivalent |
| Overlap averaging (hop 4, central 8 averaged) | 0.126 | 0.5892 | +0.0025 [+0.0019, +0.0032] | inconclusive (below margin) |
| Dense (one window per scored frame, frame at position 8) | 0.200 | 0.5879 | +0.0013 [+0.0002, +0.0025] | equivalent |

- The default tiling is at the knee of the curve. Doubling or tripling the encoder passes gains at most 0.0025 CC,
  which is below the 0.003 margin.
- Halving the passes by keeping all 16 positions costs 0.0063 CC. That cost comes from the positions without past
  context (section 4.1).
- So the default design discards half of the encoder's output positions because those positions lack history.
  [Hypothesis] A decoder that carried state across windows could keep all 16 positions at the accuracy of the
  default, which would halve the cost. Section 9 discusses this.

### 5.3 Pending
A3: FLOPs and measured time at 192/256/384 pixels and 16/32 frames; training step time and peak memory.

---

## 6. Robustness and generalisation evaluation

### 6.1 Perturbations (15H) [Completed, seed 0, holdout; video bootstrap, rule of 12E]

Ground truth comes from unperturbed viewing, so these results measure the stability of the prediction, not accuracy
under the perturbation. Gaze itself changes under image transformations (Che et al., IEEE TIP 2020). Every
perturbation is applied at the native 640×360 resolution, before resizing to 256².

| Perturbation | CC | ΔCC [95 %] | ΔNSS | Outcome |
|---|---|---|---|---|
| None (reference) | 0.5866 | — | — | — |
| No-op through the perturbation code path (control) | 0.5865 | −0.0002 [−0.0004, +0.0000] | −0.001 | equivalent |
| Gaussian noise σ 0.04 (pixel range 0–1) | 0.5855 | −0.0011 [−0.0026, +0.0003] | −0.009 | equivalent |
| Gaussian noise σ 0.16 | 0.5845 | −0.0021 [−0.0059, +0.0015] | −0.013 | inconclusive |
| Gaussian blur σ 2 native px | 0.5840 | −0.0027 [−0.0070, +0.0008] | −0.029 | inconclusive |
| Gaussian blur σ 8 native px | 0.5346 | **−0.0520 [−0.0689, −0.0368]** | −0.369 | effect |
| JPEG quality 10 | 0.5836 | −0.0031 [−0.0066, +0.0003] | −0.025 | inconclusive |
| Contrast ×0.5 | 0.5868 | +0.0002 [−0.0023, +0.0026] | −0.004 | equivalent |
| Brightness ×0.5 | 0.5828 | −0.0038 [−0.0073, −0.0004] | −0.061 | effect (just above the margin) |
| Greyscale | 0.5833 | −0.0033 [−0.0074, +0.0004] | −0.029 | inconclusive |
| Duplicated frames (both frames of every tubelet identical) | 0.5817 | −0.0050 [−0.0088, −0.0010] | −0.044 | effect |

**Findings.**
- **Photometric changes and compression barely matter.** Strong noise, JPEG quality 10, halved contrast and
  greyscale each change CC by at most 0.0033. Only halved brightness clears the 0.003 margin, by 0.0008.
- **Loss of fine spatial detail is the one severe perturbation.** Blur σ 8 native pixels (≈3 px across and ≈6 px
  down at the 256² input, where a patch is 16 px) costs 0.052 CC and 0.37 NSS. That is close to the cost of reversing time (−0.064) or removing all
  motion (−0.062, section 4.2). Mild blur (σ 2) stays within 0.003.
- [Hypothesis] Two different ways of removing detail cost about the same: blur σ 8 at the trained 16×16 token grid
  (−0.052), and a 192² input with a 12×12 grid (−0.047, section 4.4). If so, much of the 192² loss comes from lost
  detail rather than from the smaller grid. Neither run separates the two factors, so this is not established.
- **Most of the temporal signal comes from change across tubelets, not within them.** V-JEPA 2 embeds each pair of
  sampled frames as one tubelet. Duplicating the first frame of every pair removes all motion inside tubelets but
  keeps the 1.07 s span and the tubelet spacing, and it costs 0.005 CC. Removing all motion costs 0.062, and
  changing the stride costs 0.012–0.019 (section 4.3). One confound: at odd output positions (half of the scored frames)
  the model sees a frame 2 native frames (67 ms) older than the frame being scored, so part of the 0.005 may be this
  lag.

**Flip consistency [Completed].** Maps of the mirrored video, mirrored back, correlate 0.971 with the maps of the
original. The model was trained with flip augmentation, yet the correlation is not 1.0. Accuracy is the same
(CC 0.5863 vs 0.5866).

### 6.2 Cross-dataset generalisation

**Zero-shot and fine-tuned results [Thesis] (CC; the zero-shot model was trained on DHF1K only).**

| Dataset | ViT-L zero-shot | ViT-L fine-tuned | Gain from fine-tuning (ViT-L) | ViT-g fine-tuned |
|---|---|---|---|---|
| UCF Sports (task-driven gaze) | 0.608 | 0.725 | +0.118 | 0.731 |
| DIEM | 0.675 | 0.706 | +0.031 | 0.714 |
| Coutrot1 | 0.600 | 0.632 | +0.032 | 0.648 |
| Coutrot2 (meetings, speech) | 0.645 | 0.816 | +0.171 | 0.820 |
| ETMD | 0.588 | 0.662 | +0.074 | 0.672 |
| AVAD | 0.692 | 0.718 | +0.026 | 0.731 |
| SumMe | 0.551 | 0.581 | +0.030 | 0.586 |
| Hollywood-2 (Vig gaze) | 0.632 | 0.680 | +0.048 | 0.684 |

Transfer is weakest where gaze is driven by a task (UCF Sports) or by who is speaking (Coutrot2). Those are the two
largest fine-tuning gains, and both are factors that a free-viewing visual model cannot infer from DHF1K.

**Zero-shot ViT-g on five datasets, including AViMoS and MVVA (15I)** [Pending].

---

## 7. Failure cases and unexpected observations

### 7.1 Resolution ceilings (F1) [Completed, holdout]

| Ground truth stored at | CC ceiling | NSS ceiling | SIM | KLD |
|---|---|---|---|---|
| 64×64 (SURGE-D's output grid) | 0.9989 | 7.58 | 0.971 | 0.005 |
| 32×32 | 0.9867 | 6.96 | 0.903 | 0.039 |
| 16×16 (the encoder's token grid) | 0.9105 | 5.61 | 0.743 | 0.225 |
| SURGE-D ViT-g (seed 0) | 0.5866 | 3.36 | 0.449 | 1.194 |

The output grid costs nothing, and even the 16×16 token grid would allow CC 0.91. Spatial resolution does not limit
average CC at the current accuracy. Fine localisation matters more for NSS (ceiling 5.61 at 16², 7.58 at 64²).

### 7.2 Scene cuts (15J) [Completed, holdout, seed 0]

17 of the 60 holdout videos contain TransNet V2 cuts. Within those videos, frames in the 15 frames after a cut score
CC 0.562, against 0.605 for their other frames: **−0.043 [−0.084, −0.003]** (video bootstrap). Profile by frames
since the cut (≈60 scored frames per bin, so noisy):

| Frames since cut | 0–4 | 5–9 | 10–14 | 15–29 | 30–59 | ≥60 | before the first cut |
|---|---|---|---|---|---|---|---|
| Model CC | 0.581 | 0.540 | 0.574 | 0.619 | 0.589 | 0.594 | 0.612 |
| GT centre bias (CC with a central Gaussian) | 0.360 | 0.378 | 0.388 | 0.342 | 0.337 | 0.319 | 0.368 |
| GT change over 5 frames | 0.401 | 0.382 | 0.520 | 0.377 | 0.381 | 0.365 | — |

- The dip is deepest 5–9 frames after a cut, when most of the clip's ≈1 s of "past" still belongs to the previous
  shot.
- After cuts the ground truth is more central, and it relocates 10–14 frames after them. This is consistent with
  Tseng et al. (JOV 2009) and Carmi & Itti (Vision Research 2006).
- DIEM, which El-Jiz et al. (WACV 2026) found to be hurt by temporal fusion, has 20.7 cuts per video; its fold-1 test
  has 231. There too, gaze is more central after cuts (centre CC 0.44–0.47 in the first 10 frames, 0.39 after 60
  frames) and less dispersed (entropy 8.57–8.80 vs 8.99).

By centrality and entropy, these frames should be easier for a model with a centre bias, yet the model scores lower
on them. This analysis cannot tell whether cross-shot context causes the dip or post-cut gaze behaviour does; the
pre-registered test 15L does (section 13).

### 7.3 What predicts per-video accuracy [Completed, descriptive]

Spearman ρ with per-video CC (val, 100 videos, existing scores, 95 % bootstrap intervals):

| Attribute | ρ [95 %] |
|---|---|
| GT entropy | −0.81 [−0.88, −0.73] |
| GT change over five frames | −0.84 [−0.90, −0.75] |
| GT frame-to-frame consistency (reliability proxy) | +0.85 [+0.77, +0.91] |
| GT centre bias | +0.36 [+0.17, +0.53] |
| Motion energy | −0.26 [−0.46, −0.06] |
| Scene cuts | −0.03 [−0.17, +0.21] |

- Reliability and entropy of the ground truth explain **74 %** of the between-video variance in CC on val (linear
  model), and 70 % on the holdout.
- After removing them, motion, cuts and gaze change have no association with the residual. Only off-centre gaze
  remains (ρ = +0.22 [+0.01, +0.42] between centre bias and residual).
- Most per-video "failures" are videos whose ground truth is dispersed and unstable, not model errors.
- SURGE-D's advantage over SalFoM does not depend on any of these attributes (all |ρ| ≤ 0.19, every interval
  includes 0).

### 7.4 Under-performing videos (holdout, qualitative) [Completed]

Three videos lie furthest below the prediction of that model: 0492 (−0.19), 0336 (−0.14) and 0346 (−0.14). In all
three, gaze is on small, action-relevant, off-centre targets:
- 0492: the contact point of a power drill;
- 0336: the distant player and ball in a wide tennis shot, where the fixated player covers about one token of the
  16×16 grid;
- 0346: the fingertips folding a piece of cardboard.

All three are cut-free and low-motion. The common factor is target size and position, not dynamics.

### 7.5 Unexpected observations [Completed]
- **Shuffling hurts half as much as reversal** (−0.032 against −0.064). The model is more sensitive to the direction
  of time than to its order.
- **Larger test inputs keep NSS but lose CC** (320²: NSS +0.009, CC −0.018). The ranking of fixated locations is
  unaffected; the density shape is.
- **The default tiling is near the accuracy-efficiency knee**, although it was chosen without a sweep (thesis).
- **Gaze after cuts is more central and more concentrated**, yet the model scores lower there.

---

## 8. Comparison with state-of-the-art methods [Thesis; wording follows `report/claims-ledger.md`]

**DHF1K validation (the thesis test set; official test labels are withheld).**

| Model | CC | NSS | SIM | KLD | AUC-J | sAUC | Evaluator |
|---|---|---|---|---|---|---|---|
| SalFoM (released weights) | 0.5639 | 3.326 | 0.413 | 1.297 | 0.924 | 0.741 | ours |
| SalFoM, its best blur (chosen on val) | 0.5741 | 3.265 | 0.401 | 1.280 | 0.928 | 0.756 | ours |
| SURGE-D ViT-g, single | 0.6003 | 3.457 | 0.453 | 1.170 | 0.931 | 0.771 | ours |
| SURGE-D ViT-g, 3-seed ensemble | 0.6029 | 3.486 | 0.457 | 1.160 | 0.931 | 0.768 | ours |
| TMFI-Net | 0.554 | 3.201 | 0.428 | — | 0.924 | — | published |
| THTD-Net | 0.553 | 3.188 | 0.425 | — | 0.924 | — | published |
| ViNet-E | 0.549 | 3.134 | 0.409 | — | 0.922 | — | published |

- Against SalFoM on identical frames: ViT-g single +0.036 CC [+0.026, +0.048]. Every seed exceeds SalFoM, and so does
  the causal three-seed ensemble (+0.029 [+0.019, +0.040] against SalFoM's best blur).
- Against published validation numbers, computed with other code, SURGE-D is 0.046–0.049 CC higher. Published
  numbers for the same model differ by up to 0.03 between papers.
- **UCF Sports:** competitive, not ahead. The ensemble's 0.735 is level with ViNet-A (0.734) and below ViNet-E
  (0.744).
- **STAViS datasets, without audio:** above the best verified peer-reviewed results on DIEM, Coutrot1, ETMD and
  SumMe, level on AVAD, and lower on Coutrot2. The margins are within the spread of cross-paper evaluations, so the
  accurate wording is "competitive with or above".
- **Cost:** offline, 22–26 times less time per output frame than SalFoM on the same GPU.

**What this comparison does not show.** There is no official DHF1K test result. The test submission is the user's
own step and has not been made. The competitors are mostly single models, and recent models whose tables could not
be accessed are not included.

---

## 9. Experimentally identified architectural limitations

*To be completed once the pending experiments of sections 3–6 have run. Candidates from the completed experiments,
with their type:*

| Candidate | Evidence (completed) | Type (provisional) |
|---|---|---|
| Positions without past context are poor, so half of each window's outputs are discarded | 4.1, 5.2 | design (clip-based processing without state); efficiency cost |
| Post-cut frames score lower | 7.2 | open: architectural (cross-shot context) or data/gaze behaviour (15L decides) |
| Tuned to the training frame rate and input size | 4.3, 4.4 | design/training (no scale or rate augmentation) |
| Small, off-centre, action-relevant targets | 7.3, 7.4 | open: representation or resolution (15B, F3 inform) |
| Decoder overfits after ~2 epochs; data-limited | 1 | optimisation/data |
| Most per-video variance is ground-truth reliability | 7.3 | evaluation limitation, not the model |

## 10. Literature-validated research gaps

*Drafted after the pending experiments; the literature checked so far is listed below.*

**Temporal information in video saliency.**
- Tangemann, Kümmerer, Wallis & Bethge (ECCV 2020, doi:10.1007/978-3-030-58604-1_40): on LEDOV, at least 75 % of
  the explainable information is captured by static features. A static model (DeepGaze MR) beat the video models of
  the time.
- El-Jiz, Kümmerer, Tangemann, Bethge, Bartels & Bannert (WACV 2026) compared a spatio-temporal model with an
  otherwise identical spatial model. Temporal fusion gained 0.20 bits per fixation on LEDOV and lost 0.07 bits on
  DIEM, which they attribute to hard cuts and rapid zooms that "disrupt current temporal fusion approaches". They do
  not test a remedy.
- Together, these papers suggest that temporal integration helps on coherent footage and hurts on edited footage.
  Neither studies a frozen video world-model encoder, positional effects inside a clip, or a remedy for cuts.

**Gaze around cuts (vision science).**
- Centre bias is strongest right after cuts (Tseng, Carmi, Cameron, Munoz & Itti, JOV 2009).
- After MTV-style jump cuts, the first saccades are driven by new salient content (Carmi & Itti, Vision Research
  2006).
- Attentional synchrony is high in edited film (Mital et al., Cognitive Computation 2011; Smith & Mital, JOV 2013).
- No saliency model found so far treats shot boundaries explicitly. Sliding-window models (TASED-Net, ICCV 2019;
  ViNet, IROS 2021; SalFoM, ICPR 2024; SURGE-D) feed cross-shot frames as context.

**Frozen backbones and readouts.**
- DeepGaze II (ICCV 2017) and DeepGaze IIE (ICCV 2021) read out frozen image backbones.
- Video Depth Anything (CVPR 2025) pairs a frozen encoder with a light temporal head.
- Layer-wise analyses of ViTs find mid-to-late layers best for localised tasks (Walmer et al., CVPR 2023).
- No peer-reviewed probe of a frozen video world-model encoder for saliency was found. The closest, PredJSal (NTIRE
  2026), is an arXiv preprint.

**Data scale and mouse tracking.**
- SALICON (Jiang et al., CVPR 2015) established mouse-tracking pre-training for image saliency.
- Mouse data agree only partly with eye tracking and are less reliable for evaluation, but acceptable for training
  (Tavakoli et al., CVPR 2017, doi:10.1109/CVPR.2017.673).
- AViMoS (AIM 2024, ECCV Workshops, doi:10.1007/978-3-031-91856-8_11) offers 1,500 videos of crowdsourced mouse
  tracking, validated against eye tracking with a 300 ms lag correction.
- No test of video mouse-tracking pre-training evaluated on eye-tracking benchmarks was found. The closest work
  (arXiv 1907.00480) goes the other way: eye-tracking priors for a mouse model.

**Faces and speakers.**
- Gaze follows the speaker in conversations (Coutrot & Guyader, JOV 2014).
- MVVA (Liu et al., ECCV 2020) showed that attention in multi-face videos differs with sound when turn-taking occurs.
- AVM-Net (Qiao et al., IJCV 2023, doi:10.1007/s11263-023-01950-3) learns saliency and sound-source localisation
  jointly on MVVA.
- Active-speaker detectors exist: TalkNet (ACM MM 2021, doi:10.1145/3474085.3475587) and Light-ASD (CVPR 2023).
- Explicit speaker modelling is therefore established for multi-face video. What it adds on top of a strong frozen
  visual encoder has not been measured.

**Prediction error as attention.**
- Bayesian surprise (Itti & Baldi, Vision Research 2009) is a change of belief, not a prediction error.
- Prediction-error curiosity is captured by stochastic "noisy TV" content (Burda et al., ICLR 2019); ensemble
  disagreement avoids this (Pathak, Gandhi & Gupta, ICML 2019).
- A V-JEPA prediction error detects physically impossible events (Garrido et al., 2025, arXiv preprint).
- The thesis's negative result (error anti-correlated with fixations) is consistent with the noisy-TV account. A
  disagreement-based (epistemic) readout has not been tested against gaze.

**Robustness and evaluation.**
- Common corruptions: Hendrycks & Dietterich (ICLR 2019).
- Gaze itself changes under image transformations (Che et al., IEEE TIP 2020, doi:10.1109/TIP.2019.2945857).
- Metric behaviour: Bylinskii et al. (TPAMI 2019); Kümmerer et al. (ECCV 2018; PNAS 2015).

## 11. Ranked potential research contributions
*After section 10.*

## 12. Recommended research direction
*After section 11.*

## 13. Proposed experiments for validating the improvement

### 13.1 Pre-registered and queued: scene cuts (15L) [Pending]

Three explanations compete for the post-cut dip:
- **(i) Contamination:** the window's past comes from the previous shot and misleads the model.
- **(ii) Missing context:** after a cut, no past helps, whatever fills the window.
- **(iii) Gaze behaviour:** post-cut gaze is harder for this model, whatever it sees.

**Test.** Dense windows, one per scored frame with the frame at position 8, scored with and without clamping the
window to the frame's own shot. Clamped positions repeat the shot's first or last frame. The models and frames are
the same in both conditions.
- **Primary:** DIEM fold-1 test (17 videos, 231 cuts), fine-tuned ViT-g fold models (3 seeds), frames 0–14 after a
  cut, decided by the 12E rule.
- **Reading:** an effect with ΔCC > 0 supports (i), so shot-aware windows would be a remedy. Equivalence, or a
  negative effect, refutes (i) in favour of (ii) or (iii).

*Further hypotheses are formulated after the pending diagnostics.*
