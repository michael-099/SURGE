# Systematic investigation of SURGE-D: behaviour, limitations and research gaps

**Status:** complete (2026-11-17). Every experiment registered in research log section 15 has run. Sections 11–13
propose further work, none of which has run.
**Object:** SURGE-D ViT-g, the final model of the thesis: a frozen V-JEPA 2 ViT-g/16 encoder (1,012 M parameters)
and a 7.39 M-parameter decoder, unchanged. Reference runs `runs/vitg_s{0..5}`.
**Rules:** every experiment was specified, with its decision rule, in `research/log.md` section 15 and committed before
it ran (rules `5e6fb30`; the scene-cut hypothesis 15L `134513c`; one amendment before any affected run).

**Labels used throughout.**
- **[Completed]**: run and analysed in this investigation; numbers come from the files cited.
- **[Thesis]**: completed earlier for the thesis (research log sections 1–14); cited, not re-run.
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
| A2: seed 0 retrained with the identical configuration (GPU non-determinism) | Holdout CC per epoch 0.5655 / 0.5870 / 0.5851 against 0.5658 / 0.5866 / 0.5844 (differences −0.0003, +0.0004, +0.0007). The best-epoch difference (+0.0004) is about a quarter of the SD over six seeds (0.0017) | [Completed] `runs/inv_repro_s0/log.jsonl` |
| A3: parameters, FLOPs, time and memory at several input sizes and clip lengths | Section 5.3 | [Completed] `runs/review/inv/a3_cost.json` |

**Configuration.** One configuration file (`configs/main_b1.yaml`) with command-line overrides; every run directory
stores its resolved `config.json`. Seeds fix clip sampling, order, augmentation and initialisation; cuDNN kernels are
not forced to be deterministic. A2 puts the resulting run-to-run difference at 0.0003–0.0007 CC per epoch.

**Convergence [Thesis logs, re-read].** All six ViT-g seeds reach their best holdout CC at the second of three epochs
and lose 0.001–0.004 in the third, while training CC keeps rising (0.640 → 0.676). The 10-epoch ViT-L run of gate G1
peaked at epoch 2 (0.578) and fell monotonically to 0.522 at epoch 9. The decoder starts to overfit the 540 training
videos after about two epochs; the train–holdout gap at the last epoch is ≈0.09 CC. A six-epoch run with a schedule
fitted to six epochs (15D, seed 0) also peaks at its second epoch and falls to 0.5694 by the sixth (section 4.5). In
15C/15D, R peaks at the second epoch in all three seeds, and so do most variants (section 4.5).

**Learning curve [Thesis, ViT-L, 3 seeds].** 135 → 270 → 540 training videos: 0.5645 → 0.5729 → 0.5783 holdout CC;
the second doubling gives 0.64 of the first doubling's gain. DHF1K has no further annotated training videos.

**Discrepancies investigated.**
- The encoder of the Hugging Face V-JEPA 2 port derives token positions from the `crop_size` in its configuration.
  Feeding another input size without rebuilding the encoder would silently use wrong positions. The code now rebuilds
  the encoder for other sizes and raises an error on a mismatch. All earlier runs used 256² and were not affected
  (implementation hazard, not a bug in any reported result).
- The preparation of MVVA assumed that every source video is 1280×720. The zero-shot run stopped on the first video
  that is not, before any MVVA score was used. The eight videos of other sizes are now laid out by per-axis scaling.
  A ground-truth-only check then showed that in the five 1920×1080 videos the released maps and fixations disagree
  (NSS of the map at its own fixations ≤ 0.28, against ≥ 4.37 elsewhere), so those five are left out. Video 343 has
  no ground truth. Two portrait AViMoS videos had their fixations on the wrong grid and were corrected the same way
  (research log 15I). Implementation bugs in data preparation; no result had been computed from the affected videos.
- A command-line override written in exponent notation without a decimal point (`train.lr=1e-3`) was read by
  PyYAML as a string. The smoke test of the training queue caught this before any run used such a value. Overrides
  now convert these values to numbers (`surge/config.py`, with a test). Every stored run configuration was checked,
  and no earlier run used such an override (implementation bug, no result affected).
- GroupNorm in the decoder pools statistics over the whole clip, so the decoder mixes time steps even where it has no
  temporal convolution. This is a property of the design, found by a CPU test; it changed the design of ablation V3
  (section 3), not any earlier result. 15C2 later found that this coupling adds nothing measurable to accuracy.

---

## 2. Experimental methodology

**Pre-registration.** Each experiment's design and decision rule were written in `research/log.md` section 15 and
committed before the first run. Analyses added after a related result had been seen are labelled post hoc, both in
the log and here, and are descriptive.

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
(55 tests).
- `scripts/probe_eval.py`: test-time probes. It covers temporal order, stride, input size, tilings, perturbations,
  per-position scores, flip consistency and shot clamping. With its defaults it reproduces the thesis evaluator
  exactly.
- `scripts/train_multi.py`: trains several decoders on one frozen-encoder stream, with the same clips, order and
  augmentation as the standard trainer. Ablations are therefore paired by construction, and the encoder runs once
  for all of them.
- Decoder options `tap_mode`, `temporal` (with per-frame norms), `norm_per_frame` (15C2), `topdown`, and
  `LinearReadout`. The default decoder is tested to be identical, tensor for tensor, to the committed one.
- `scripts/inv_summary.py`: applies the 12E rule. `scripts/inv_cuts.py`: the scene-cut analysis.

**Experiment register.**

| ID | Question | Status |
|---|---|---|
| A1, evaluator check | Is the reported baseline reproducible from the stored checkpoint? | Completed |
| A2 | How large is run-to-run non-determinism at a fixed seed? | Completed |
| A3 | Cost: parameters, FLOPs, time, memory | Completed |
| B | What does each encoder depth carry (41 linear probes)? | Completed |
| C | Decoder ablations V1–V6, 3 seeds | Completed |
| D | Learning rate, loss terms, schedule length | Completed (schedule length: seed 0 only, as planned) |
| 15C2 | Does V3's effect come from the temporal convolutions or from the clip-pooled norm statistics? | Completed |
| E1, E3 | Temporal order and stride at test time | Completed |
| E2 | Video model vs a static-trained model (primary temporal test) | Completed |
| F1, F2 | Resolution ceilings; test-time input size | Completed |
| F3 | Training at 384² | Completed |
| G | Output position and tiling | Completed |
| H | Perturbations; flip consistency | Completed |
| I | Zero-shot on five datasets | Completed |
| J | Failure analysis | Completed |
| K | Multi-face scenes (MVVA) | Completed |
| 15L | Scene cuts: cross-shot context or post-cut gaze? (hypothesis test) | Completed |
| 15M | How much of the zero-shot gap is dataset bias (blur, exponent, centre)? | Completed |
| 15N | Can an off-the-shelf active-speaker detector supply the speaker (MVVA)? | Completed |

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

### Decoder ablations of this investigation (15C) [Completed, seeds 0–2, holdout, rule of 12E]

Each variant is trained next to the unchanged decoder R on the same encoder stream, and compared with R of the same
run at each variant's best epoch (seeds 0–2; `runs/inv_c_s{0,1,2}`, `runs/review/inv/holdout/c_s*_*.json`,
`runs/review/inv/summary_ablation.json`). CC is the mean over the three seeds.

| Variant | Change | Decoder parameters | CC | ΔCC vs R [95 %] | ΔCC per seed | Outcome | ΔNSS |
|---|---|---|---|---|---|---|---|
| R | none (trained by the multi-decoder trainer) | 7.39 M | 0.5867 | — | — | — | — |
| V1 | linear readout of the four taps (LayerNorm, 1×1×1 convolution, 16×16 logits upsampled, prior) | 0.006 M | 0.5147 | −0.0720 [−0.0873, −0.0582] | −0.072, −0.075, −0.069 | effect | −0.45 |
| V2 | tap 40 in all four branches | 7.39 M | 0.5852 | −0.0015 [−0.0048, +0.0016] | −0.001, −0.003, −0.000 | inconclusive | −0.00 |
| V3 | no temporal mixing in the decoder (1×1×1 instead of 3×1×1 convolutions; GroupNorm per time step) | 6.60 M | 0.5630 | −0.0237 [−0.0291, −0.0185] | −0.025, −0.025, −0.021 | effect | −0.17 |
| V4 | no top-down path (1/16 stage to the head, ×4 upsampling) | 2.73 M | 0.5826 | −0.0041 [−0.0066, −0.0016] | −0.003, −0.005, −0.005 | effect | −0.04 |
| V5 | 64 instead of 256 channels | 0.73 M | 0.5827 | −0.0040 [−0.0068, −0.0013] | −0.005, −0.005, −0.003 | effect | −0.04 |
| V6 | neither centre prior nor auxiliary heads | 7.39 M | 0.5858 | −0.0009 [−0.0040, +0.0026] | −0.001, −0.004, +0.002 | inconclusive | −0.01 |

- **The decoder's value lies in non-linear processing on the token grid.**
  - The four-tap linear readout V1 loses 0.072 CC in every seed, the largest of the decoder ablations. It is barely
    above the best single-block probe (0.510; section 3, trained with another learning rate).
  - V1 peaks at its last epoch in every seed, the only variant to do so, but gains only 0.0008–0.0015 in that epoch.
    A longer schedule would not close a gap of this size.
  - V4 keeps the non-linear decoder but drops the top-down path. It then works only on the 16×16 token grid, from
    taps 30 and 40, so the branches of taps 10 and 20 go too. It loses 0.004.
  - Feeding the full decoder only the last tap (V2) loses 0.0015 (inconclusive; at most 0.0048 at 95 %).
  - So neither the finer output nor the combination of depths explains much of the decoder's 0.07 over a linear
    readout. Most of the gain comes from the non-linear processing itself.
- **Temporal mixing inside the decoder matters.** V3 loses 0.024 CC in every seed, although the encoder already
  attends across time.
  - This is not a capacity effect. V3 has 0.79 M fewer parameters and loses 0.024, while V5 has 6.65 M fewer and
    loses 0.004.
  - V3 changes two things at once, the temporal convolutions and the GroupNorm statistics (per time step instead of
    per clip). 15C2 changes each one alone and finds that the temporal convolutions carry the effect (below).
- **Width and the top-down path have small, established effects.** V5 needs a tenth of the parameters and loses
  0.004. Because the decoder costs 0.168 of 5.26 TFLOPs per clip (section 5), neither saves meaningful compute.
- **The thesis's dispensable pair is not shown dispensable at ViT-g.** Removing the centre prior and the auxiliary
  heads together (V6) gives −0.0009, but the 90 % interval [−0.0036, +0.0019] reaches past the −0.003 margin, so the
  result is inconclusive rather than equivalent. Each was equivalent alone at ViT-L (thesis).
- **The trainer.** R of the multi-decoder trainer against the standard runs `runs/vitg_s{0,1,2}`:
  - +0.0013 [−0.0003, +0.0036], inconclusive. Per seed: +0.0003, +0.0029, +0.0006.
  - The clips, order and augmentation are the same, but the decoder's initialisation is not: the multi-decoder
    trainer seeds each decoder separately (seed × 1000 + its index). A2 puts GPU non-determinism at a fixed seed at
    0.0003–0.0007 CC per epoch (section 1). [Hypothesis] The seed-1 difference (+0.0029) therefore comes mostly from
    the different initialisation.
  - Every comparison above is within one run and is not affected.

### Which part of V3 carries its effect (15C2) [Completed, seeds 0–2, rule of 12E]

V3a and V3b each change one of V3's two components. They were trained next to R and V3 on one encoder stream, as in
15C (`runs/inv_c2_s{0,1,2}`, `runs/review/inv/holdout/c2_s*_*.json`, `runs/review/inv/summary_v3split.json`). CC is
the mean over the three seeds.

| Variant | Temporal convolutions | GroupNorm statistics | CC | ΔCC vs R [95 %] | ΔCC per seed | Outcome |
|---|---|---|---|---|---|---|
| R | 3×1×1 | over the clip | 0.5864 | — | — | — |
| V3a | 1×1×1 | over the clip | 0.5635 | −0.0229 [−0.0287, −0.0173] | −0.023, −0.026, −0.020 | effect |
| V3b | 3×1×1 | per time step | 0.5862 | −0.0002 [−0.0019, +0.0014] | +0.000, +0.000, −0.001 | equivalent |
| V3 | 1×1×1 | per time step | 0.5617 | −0.0247 [−0.0304, −0.0189] | −0.027, −0.026, −0.021 | effect |

- **The temporal convolutions carry the effect**, by the reading fixed in advance: V3a is an effect, and V3b is
  equivalent (90 % [−0.0016, +0.0011]).
- **Clip-pooled statistics add nothing measurable** (2×2 factorial, descriptive).
  - The convolutions add +0.023 with clip-pooled statistics and +0.024 with per-step statistics.
  - Clip-pooled statistics add +0.0002 [−0.0014, +0.0019] with the convolutions and +0.0018 [−0.0011, +0.0045]
    without them. The interaction is −0.0016 [−0.0049, +0.0018].
  - The norms do couple time steps (section 1), but this does not matter for accuracy.
- **Where the decoder mixes time.**
  - The decoder's six 3×1×1 convolutions carry its 0.024 CC of temporal mixing. There are two at each of the 1/16,
    1/8 and 1/4 levels, and they act on the eight time steps of a window, with zero padding at its edges.
  - Per-time-step normalisation costs nothing measurable. A decoder that carries features from one window into the
    next can therefore use it, so that carried and new features are normalised alike (section 10.1).
- **Other metrics (descriptive).**
  - V3a: NSS −0.170, SIM −0.020, KLD +0.055.
  - V3b: NSS −0.001 [−0.011, +0.008], SIM +0.0004, KLD −0.0004 [−0.005, +0.004].
- **Run-to-run check (descriptive).**
  - R of 15C2 against R of 15C: −0.0002 [−0.0009, +0.0004]. Seeds, clips, order and initialisation are the same.
  - V3 of 15C2 against V3 of 15C: −0.0013 [−0.0036, +0.0015]. V3's initialisation differs between the two runs: it is
    decoder 3 in 15C and decoder 1 in 15C2.

### What each encoder depth carries (15B) [Completed, seed 0, holdout, descriptive]

Forty linear probes, one per ViT-g block, each with its own learned centre prior, and a prior-only readout. All were
trained on one encoder stream (3 epochs, lr 1e-3) and scored after the last epoch with the protocol of
`runs/review/holdout/vitg_s0.json` (σ 0, every fifth frame; `runs/inv_b_s0`, `summary_probes.json`).

| Readout | Holdout CC | NSS | KLD |
|---|---|---|---|
| Centre prior alone | 0.298 | 1.50 | 2.11 |
| Block 1 | 0.339 | 1.76 | 1.99 |
| Block 10 (decoder tap) | 0.399 | 2.15 | 1.78 |
| Block 20 (decoder tap) | 0.466 | 2.58 | 1.55 |
| Block 27 (first within 1 % of the best) | 0.505 | 2.85 | 1.44 |
| Block 30 (decoder tap) | 0.509 | 2.89 | 1.43 |
| Block 34 (best) | 0.510 | 2.89 | 1.43 |
| Block 40 (decoder tap) | 0.503 | 2.85 | 1.46 |
| Full decoder R on taps 10/20/30/40 (`runs/vitg_s0`) | 0.587 | 3.36 | 1.19 |

- Linearly decodable saliency rises steadily with depth and plateaus over blocks 27–40. The last six blocks lose
  0.006, one seed, so this is not established.
- Block 1 already beats the prior alone by 0.041 CC, so low-level token features carry some saliency.
- The full decoder exceeds the best single block by 0.077 CC. Two things could explain the gap: combining depths, and
  the non-linear decoder with its finer output. The four-tap linear readout of 15C (V1) reaches 0.515 (three seeds),
  so combining depths linearly explains little of the gap and the non-linear decoder most of it.
- Two of the four taps (30, 40) lie on the plateau. The full decoder fed only tap 40 (V2) loses 0.0015 CC (three
  seeds, inconclusive), and the decoder on taps 30 and 40 alone, without the top-down path (V4), loses 0.004. The
  shallower taps 10 and 20 therefore add at most a few thousandths.
- This matches layer-wise ViT analyses in which mid-to-late layers serve localised tasks best (section 10).

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

### 4.2 Temporal order and the value of motion (15E1, 15E2) [Completed]

**Test-time conditions (15E1) [seed 0, holdout; video bootstrap].**

| Condition (model trained on forward video) | CC | ΔCC [95 %] | ΔNSS | Outcome |
|---|---|---|---|---|
| Original order | 0.5866 | — | — | — |
| Each window's frames reversed in time | 0.5228 | −0.064 [−0.079, −0.050] | −0.38 | effect |
| Frames shuffled within each window | 0.5549 | −0.032 [−0.039, −0.024] | −0.22 | effect |
| Static clip (each scored frame repeated 16 times) | 0.5243 | −0.062 [−0.081, −0.046] | −0.39 | effect |

The trained model depends on temporal structure. Removing all motion costs 11 % of its CC, and reversing time costs
as much: twice the cost of shuffling. This fits the past-context result (reversal presents the future as the past).

These conditions are distribution shifts for a decoder trained on real video, so they show dependence, not benefit.

**The matched control: a model trained on static clips (15E2, primary temporal test) [seeds 0–2, holdout, rule of
12E].** The main recipe, with each training clip replaced by one of its central frames, drawn at random and repeated
16 times. That frame's map is the target on all central positions, so the number of clips and the encoder compute are
unchanged. The static model is scored on static clips; the video models are `runs/vitg_s{0,1,2}`
(`runs/inv_static_s{0,1,2}`, `runs/review/inv/summary_static.json`).

| Model (input at test time) | CC | NSS | SIM | KLD |
|---|---|---|---|---|
| Video model (video) | 0.5854 | 3.361 | 0.451 | 1.201 |
| Static model (static clips) | 0.5280 | 2.995 | 0.404 | 1.375 |
| Static − video [95 %] | **−0.0574 [−0.0697, −0.0452]** | −0.37 [−0.45, −0.29] | −0.047 | +0.174 |

- **Outcome: effect, in every seed** (−0.058, −0.059, −0.055). By the reading fixed in advance, SURGE-D uses temporal
  information on DHF1K, worth 0.057 CC, or 9.8 % of its CC.
- **The test-time result was mostly not a distribution shift.** Training on static clips recovers only 0.004 of the
  0.062 that static input costs the video model (seed 0: 0.5286 against 0.5243).
- **Where the temporal benefit enters (descriptive, across experiments).**
  - The video model scores 0.585, the decoder without temporal mixing (V3, section 3) 0.563, and the static model
    0.528.
  - About 0.035 of the benefit therefore reaches a decoder that sees one tubelet at a time, through the encoder
    alone. Another 0.024 needs temporal mixing in the decoder. 15C2 places it in the decoder's temporal
    convolutions, not in its clip-pooled norms (section 3).
- **Relation to the literature.** On LEDOV, static features capture at least 75 % of the explainable information
  (Tangemann et al., ECCV 2020). Here the static model reaches 90 % of the video model's CC. CC is not an
  information measure, so the agreement is qualitative: most of what is predictable is static, and the temporal part
  is clearly measurable.

**Where the video–static gap is largest [Completed, descriptive, per scored frame; `e2_per_frame.json`].** The
video − static difference in CC, by quintile of a per-frame attribute. E2 is averaged over seeds 0–2; E1 is the
test-time static condition of seed 0 (6,549 frames each).

| Frames grouped by | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Motion energy, low → high: E2 | +0.032 | +0.033 | +0.038 | +0.074 | +0.092 |
| Motion energy: E1 | +0.045 | +0.033 | +0.036 | +0.078 | +0.096 |
| Ground-truth change over five frames, gaze stays → relocates: E2 | +0.076 | +0.054 | +0.045 | +0.048 | +0.045 |
| Ground-truth change over five frames: E1 | +0.083 | +0.061 | +0.051 | +0.048 | +0.045 |
| Ground-truth centre bias, off-centre → central: E2 | +0.063 | +0.064 | +0.060 | +0.045 | +0.037 |
| Ground-truth centre bias: E1 | +0.103 | +0.072 | +0.051 | +0.036 | +0.025 |

- The temporal benefit roughly triples from the stillest to the most moving frames.
- It is largest where gaze stays in place over five frames (+0.076) and smaller, though still +0.045, where gaze
  relocates. [Hypothesis] Much of the benefit comes from keeping track of attended content, not from anticipating
  new gaze shifts.
- The matched control cuts the off-centre dependence that E1 showed to a third (Q1 − Q5: +0.026 against +0.078).
  Part of the E1 pattern was therefore the distribution shift.

### 4.3 Temporal sampling (15E3) [Completed, seed 0]

| Test-time stride (trained: 2) | Window span at 30 fps | CC | ΔCC [95 %] | Outcome |
|---|---|---|---|---|
| 1 | 0.53 s | 0.5743 | −0.012 [−0.017, −0.008] | effect |
| 2 (as trained) | 1.07 s | 0.5866 | — | — |
| 4 | 2.13 s | 0.5681 | −0.019 [−0.026, −0.011] | effect |

The trained model is tuned to its training frame rate. In both directions the change costs more than the 0.003
margin. This matters for datasets recorded at other frame rates: SumMe and MVVA include 15 fps videos, where stride 2
spans 2.1 s, like stride 4 at 30 fps.

### 4.4 Input size at test time (15F2) and in training (15F3) [Completed]

**At test time (15F2) [seed 0; encoder rebuilt for each size].**

| Input (trained: 256²) | Tokens per frame | CC | ΔCC [95 %] | ΔNSS | Outcome |
|---|---|---|---|---|---|
| 192² | 12×12 | 0.5395 | −0.047 [−0.059, −0.037] | −0.39 | effect |
| 224² | 14×14 | 0.5726 | −0.014 [−0.021, −0.008] | −0.15 | effect |
| 256² | 16×16 | 0.5866 | — | — | — |
| 320² | 20×20 | 0.5685 | −0.018 [−0.026, −0.011] | **+0.01** | effect (CC) |
| 384² | 24×24 | 0.5379 | −0.049 [−0.062, −0.037] | −0.09 | effect |

Accuracy peaks at the training size. Above it, CC falls but NSS does not at first. At 320², NSS is unchanged
(+0.009) and SIM nearly so (0.4481 vs 0.4494), while CC drops by 0.018 and KLD rises from 1.194 to 1.244. The map
is as high at fixated pixels as before, relative to its own mean and spread, but it agrees less with the
ground-truth density as a whole.

The metrics are known to come apart in this way. The NSS-optimal map is the fixation density itself, while the CC-
and KLD-optimal maps are that density blurred with the kernel used to build the ground truth (Kümmerer, Wallis &
Bethge, ECCV 2018, doi:10.1007/978-3-030-01270-0_47). So a map that becomes sharper keeps NSS but loses CC and KLD.

- [Hypothesis] At a larger token grid the decoder's output becomes sharper or more fragmented, because its receptive
  fields and centre prior were learned on a 16×16 grid.
- [Proposed] A diagnostic: re-choose the output blur on val for the 320² condition. If CC recovers to the 256²
  level, the loss is a calibration mismatch, not a loss of information.
- F3 (below) asks whether training at 384² gains accuracy. A model trained at 384² learns its own output sharpness
  under the CC and KLD loss terms, so this mismatch does not carry over to it.

**Training at 384² (15F3) [Completed, seeds 0–2, holdout, rule of 12E].** The main recipe with a 384² input (24×24
tokens per frame). Seed 0 passed the gate fixed in advance (holdout CC 0.5938 ≥ 0.5896), so seeds 1 and 2 followed.
Reference: the 256² runs `runs/vitg_s{0,1,2}` on the same frames (`runs/review/inv/summary_res384.json`).

| Metric | 256² | 384² | Δ [95 %] | Outcome |
|---|---|---|---|---|
| CC | 0.5854 | 0.5946 | **+0.0092 [+0.0030, +0.0163]** | effect |
| NSS | 3.361 | 3.442 | +0.082 [+0.042, +0.129] | effect |
| SIM | 0.451 | 0.454 | +0.003 [−0.004, +0.012] | inconclusive |
| KLD (lower is better) | 1.201 | 1.178 | −0.024 [−0.047, −0.003] | effect |

- Input resolution limits accuracy on DHF1K once the model is trained at that size. Per seed, CC rises by +0.0072,
  +0.0083 and +0.0120. Seed 2's larger gain partly reflects its reference, the weakest of the six 256² seeds
  (0.5829). Against the six-seed 256² mean (0.5861), the 384² mean is 0.0085 higher (descriptive).
- It is the largest improvement over the reference found in this investigation, about the size of the step from
  ViT-L to ViT-g (+0.0083, seed 0, section 3).
- NSS gains more than SIM. This fits F1, where finer localisation mattered more for NSS than for CC (section 7.1).
- The best epoch is the second in seeds 0 and 1 and the third in seed 2 (+0.0007 over its second).
- Cost: a training epoch takes 2.6 times as long as at 256² (1,563 s against 595 s, same GPU). The encoder needs 2.77
  times the FLOPs (A3, section 5.3).
- [Post-hoc, descriptive] 40 of 60 holdout videos gain. The per-video gain (mean of the three seed pairs) shows no
  clear association with the ground truth's fine detail (Spearman ρ with the 16² ceiling +0.05 [−0.22, +0.32]), its
  concentration (entropy −0.01 [−0.29, +0.26]), its centrality (−0.08 [−0.34, +0.19]), gaze change (−0.16 [−0.42,
  +0.12]) or motion (−0.25 [−0.49, +0.02]). With 60 videos these intervals are wide. They do not support the idea
  that the gain sits on the small, off-centre targets of section 7.4, but they cannot exclude it.

### 4.5 Training configuration (15D) [Completed, seeds 0–2, rule of 12E; six epochs: seed 0, descriptive]

Variants of R's training (learning rate 3e-4; loss KLD + 1.0 × CC term + 0.1 × NSS term), in the same runs as 15C
(`runs/inv_c_s{0,1,2}`, `summary_ablation.json`). CC is the mean over the three seeds.

| Variant of R's training | CC | ΔCC vs R [95 %] | ΔCC per seed | Outcome |
|---|---|---|---|---|
| Learning rate 1e-4 (×1/3) | 0.5820 | −0.0047 [−0.0075, −0.0020] | −0.003, −0.006, −0.005 | effect |
| Learning rate 9e-4 (×3) | 0.5862 | −0.0004 [−0.0039, +0.0024] | +0.000, −0.002, +0.001 | inconclusive |
| No CC term in the loss | 0.5861 | −0.0006 [−0.0024, +0.0012] | −0.001, −0.000, −0.000 | equivalent |
| No NSS term in the loss | 0.5870 | +0.0003 [−0.0022, +0.0025] | +0.001, −0.002, +0.002 | equivalent |
| Six epochs, cosine schedule over six (seed 0; best: second epoch) | 0.5862 | −0.0007 [−0.0044, +0.0030] | −0.001 | — (one seed) |

- **The recipe is robust upwards and the loss is partly redundant.**
  - Tripling the learning rate changes CC by −0.0004; the 90 % interval [−0.0031, +0.0019] just misses equivalence.
  - Dropping the CC term or the NSS term is equivalent in CC.
  - Without the NSS term, NSS itself falls by 0.020 [−0.034, −0.005], about 0.6 % of its value. The rule is defined
    on CC, so this is descriptive.
  - Only the lower learning rate costs more than the margin. It peaks at the same (second) epoch as R and is below R
    at the second and third epochs of every seed, so it is not merely slower to converge.
- **A longer schedule only prolongs overfitting** (seed 0). The six-epoch run scores 0.5600, 0.5862, 0.5854, 0.5791,
  0.5737 and 0.5694 after its six epochs, while its training loss keeps falling. Its best epoch is again the second.
- **The second epoch is the best in 27 of the 30 seed-and-variant pairs of 15C/15D other than V1.** R peaks there in
  all three seeds. Only V5 (seed 2) and the ×3 learning rate (seeds 1–2) peak at the third, and V1 peaks at the third
  in every seed.

---

## 5. Computational efficiency analysis

### 5.1 Cost of the model [Thesis, section 5 "Computational Cost"; extended by A3 in section 5.3]

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

### 5.3 Cost by input size, clip length and decoder (A3) [Completed; RTX A5000, bf16, protocol of 13D]

Encoder only (`runs/review/inv/a3_cost.json`). One pass serves as many native frames as the clip has sampled frames
(default tiling, stride 2).

| Encoder, input, clip | TFLOPs per clip | TFLOPs per output frame | Time per output frame, batch 1 / 8 | Peak memory, batch 8 |
|---|---|---|---|---|
| ViT-L, 256², 16 frames | 1.656 | 0.103 | 4.5 / 2.8 ms | 3.0 GB |
| ViT-g, 192², 16 frames | 2.629 | 0.164 | 7.5 / 3.7 ms | 7.1 GB |
| **ViT-g, 256², 16 frames (default)** | 5.088 | 0.318 | 9.4 / 7.0 ms | 8.2 GB |
| ViT-g, 384², 16 frames | 14.105 | 0.882 | 22.6 / 19.7 ms | 11.1 GB |
| ViT-g, 256², 32 frames | 12.065 | 0.377 | 9.4 / 8.4 ms | 10.5 GB |

Decoders on ViT-g taps (per 16-frame clip; times for a batch of 8):

| Decoder | Parameters | TFLOPs per clip | Forward | Training step | ΔCC (15C) |
|---|---|---|---|---|---|
| R (default) | 7.39 M | 0.168 | 54.7 ms | 177 ms | — |
| V1 linear readout | 0.006 M | < 0.001 | 7.2 ms | 12.7 ms | −0.072 |
| V3 no temporal mixing | 6.60 M | 0.145 | 56.6 ms | 183 ms | −0.024 |
| V4 taps 30/40 only, 1/16 | 2.72 M | 0.011 | 4.2 ms | 20.9 ms | −0.004 |
| V5 64 channels | 0.73 M | 0.012 | 14.8 ms | 75.7 ms | −0.004 |

V2 and V6 cost the same as R. A full training step (encoder forward without gradient, decoder forward and backward,
batch 8) takes 1,090 ms at 256² (peak 10.0 GB) and 4,212 ms at 384² (17.4 GB).

- A 384² input costs 2.77 times the encoder FLOPs of 256² and 2.4–2.8 times the time per output frame, for +0.0092
  CC (F3). A training step costs 3.9 times as much. A whole epoch cost 2.6 times as much, because data loading and the
  holdout pass do not grow with the input.
- A 32-frame clip costs 1.19 times as much per output frame as a 16-frame clip, for +0.0036 CC (thesis, seed 0).
- The decoder is 3 % of the FLOPs. V4 and V5 cut its FLOPs about fifteen-fold for −0.004 CC, which saves about 3 % of
  the whole model.
- Arithmetic, not a measurement: at 384², keeping all 16 positions of each window (half the passes, section 5.2) would
  cost 0.44 TFLOPs per output frame, 1.39 times today's 0.318. That assumes that the all-16 deficit can be removed
  (section 9), which is untested.

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

**Zero-shot ViT-g (15I)** [Completed]. The model is `runs/vitg_s0`, trained on DHF1K only. σ was fixed in advance at
8 × native width / 640. Every fifth frame is scored (UCF Sports: every frame). The STAViS test folds are pooled, since
each video lies in one fold. Values are per-video means with 95 % video-bootstrap intervals
(`runs/review/inv/summary_zeroshot.json`). "Fine-tuned" is the thesis's ViT-g seed-0 model fine-tuned on each
dataset, on the same videos (`runs/review/inv/summary_bias.json`). It was scored on every frame with its own tuned σ,
so the gain from fine-tuning is descriptive.

| Dataset | Videos | Gaze data | Zero-shot CC | NSS | Fine-tuned CC | Gain from fine-tuning [95 %] |
|---|---|---|---|---|---|---|
| UCF Sports (task-driven gaze) | 47 | eye tracking | 0.622 [0.587, 0.654] | 3.22 | 0.730 | +0.109 [+0.082, +0.136] |
| DIEM | 17 | eye tracking | 0.692 [0.639, 0.740] | 2.79 | 0.714 | +0.022 [−0.004, +0.045] |
| Coutrot1 | 60 | eye tracking | 0.614 [0.583, 0.644] | 3.00 | 0.648 | +0.035 [+0.019, +0.051] |
| Coutrot2 (meetings, speech) | 15 | eye tracking | 0.662 [0.632, 0.691] | 5.03 | 0.820 | +0.158 [+0.115, +0.200] |
| ETMD | 12 | eye tracking | 0.593 [0.576, 0.610] | 3.20 | 0.672 | +0.080 [+0.066, +0.093] |
| AVAD | 45 | eye tracking | 0.695 [0.656, 0.732] | 3.98 | 0.731 | +0.036 [+0.019, +0.055] |
| SumMe | 23 | eye tracking | 0.556 [0.528, 0.584] | 2.97 | 0.586 | +0.030 [+0.019, +0.040] |
| Hollywood-2 (Vig gaze) | 884 | eye tracking | 0.633 [0.626, 0.640] | 3.66 | 0.684 | +0.051 [+0.049, +0.054] |
| MVVA (multi-face scenes, 34 observers) | 294 | eye tracking | 0.726 [0.715, 0.737] | 3.90 | — | — |
| AViMoS public test | 150 | mouse tracking | 0.680 [0.665, 0.694] | 2.84 | — | — |

- With ViT-g, as with ViT-L, fine-tuning gains most on Coutrot2 (+0.158) and UCF Sports (+0.109), then on ETMD
  (+0.080) and Hollywood-2 (+0.051). Elsewhere it gains 0.02–0.04.
- The MVVA and AViMoS scores are not compared with published models. MVVA results in the literature use the authors'
  own split, which the release does not include, and no published AViMoS result uses this zero-shot protocol.
- CC is not comparable across datasets, because each dataset blurs its ground truth differently (the DHF1K holdout
  gives 0.587).

**Dataset bias in transfer (15M)** [Completed, descriptive; design and reading fixed before any zero-shot result;
`runs/review/inv/summary_bias.json`]. Three parameters are fitted per dataset: blur, an exponent γ and a centre
factor. They are chosen on one half of the videos and scored on the other (two-fold cross-fitting). The identity
transform reproduced the 15I per-video CC on every dataset (largest difference 2.5 × 10⁻⁶). The share is (adapted −
zero-shot) / (fine-tuned − zero-shot) on the same videos.

| Dataset | Zero-shot CC | Bias-adapted CC | Adapted − zero-shot [95 %] | Share of the fine-tuning gain [95 %] | Reading fixed in advance |
|---|---|---|---|---|---|
| UCF Sports | 0.622 | 0.632 | +0.010 [+0.005, +0.016] | 0.09 [0.05, 0.14] | content |
| DIEM | 0.692 | 0.701 | +0.010 [+0.003, +0.018] | 0.44 [−1.00, 2.15] | mixed; uninformative, since the fine-tuning gain's interval includes 0 |
| Coutrot1 | 0.614 | 0.619 | +0.006 [+0.003, +0.008] | 0.16 [0.11, 0.25] | content |
| Coutrot2 | 0.662 | 0.668 | +0.006 [−0.004, +0.015] | 0.04 [−0.02, 0.11] | content |
| ETMD | 0.593 | 0.598 | +0.005 [+0.000, +0.011] | 0.07 [0.00, 0.12] | content |
| AVAD | 0.695 | 0.701 | +0.007 [+0.001, +0.013] | 0.19 [0.02, 0.35] | content |
| SumMe | 0.556 | 0.557 | +0.000 [−0.003, +0.003] | 0.02 [−0.11, 0.11] | content |
| Hollywood-2 | 0.633 | 0.640 | +0.007 [+0.007, +0.008] | 0.14 [0.13, 0.16] | content |
| AViMoS | 0.680 | 0.690 | +0.010 [+0.007, +0.014] | no fine-tuned model | — |
| MVVA | 0.726 | 0.729 | +0.003 [+0.002, +0.005] | no fine-tuned model | — |

- **SURGE-D's transfer gap is mainly content, not a low-dimensional bias.** On seven of the eight datasets with a
  fine-tuned model, three bias parameters recover at most a fifth of the fine-tuning gain (0.02–0.19). DIEM is the
  exception in its point estimate only (0.44), and its interval is uninformative.
- The expectation [hypothesis] held for Coutrot2 (0.04) and UCF Sports (0.09). It did not hold for the free-viewing
  datasets, whose shares are as low (SumMe 0.02, Coutrot1 0.16, AVAD 0.19).
- This differs from images, where fewer than 20 dataset-specific parameters closed more than 75 % of the gap (Kümmerer,
  Khanuja & Bethge, ICCV 2025). The comparison is indicative only: three parameters here, a different reference (the
  fine-tuning gain), and fine-tuned models scored on every frame with their own σ.
- In absolute terms the bias fit adds 0.000–0.010 CC, about what the 384² input adds on DHF1K, against the 0.02–0.16
  that fine-tuning adds.
- Fitted parameters (descriptive): γ = 0.75, which flattens the maps, on UCF Sports, DIEM, ETMD, AVAD and Hollywood-2
  in both halves. Coutrot2 chose γ = 1.5 and 1.5 times the blur in both halves, and AViMoS's mouse-tracking maps chose
  a stronger centre (κ 4 and 8).
- An earlier run of the same script on MVVA and AViMoS served as a code check. The queue's run overwrote its files and
  is the one reported.

**Multi-face scenes (15K)** [Completed, descriptive; `runs/review/inv/k_mvva_faces.json`]. The frames are the MVVA
frames of the zero-shot run that show at least one talking and one silent face: 279 videos, 23,102 frames. Face boxes
and talking labels come from the MVVA release. As in the authors' demo script (`demo_face_talking.py`), a face counts
as talking only when its label is 1. An undocumented label 2 (0.7 % of face-frames, 41 videos) therefore counts as
silent. Values are per-video means, with 95 % video-bootstrap intervals for the last column.

| Share of | Talking faces | Silent faces | Talking faces' share of the face total |
|---|---|---|---|
| Frame area | 1.8 % | 2.7 % | 41.5 % |
| Human fixations | 38.7 % | 17.2 % | 66.7 % |
| Ground-truth density | 31.7 % | 14.0 % | 66.8 % [65.3, 68.3] |
| Predicted density | 24.4 % | 18.9 % | 54.6 % [52.8, 56.4] |

- The model puts the right total on faces: 43.3 % predicted against 45.6 % in the ground truth.
- It splits that total wrongly. It puts 7.2 points less on talking faces (95 % [−7.9, −6.5]) and 4.9 points more on
  silent ones ([+4.2, +5.6]). It favours talking faces beyond their area (54.6 % against 41.5 %), but much less than
  people do (66.8 %).
- Cross-fitted oracles, with weights chosen on one half of the videos and scored on the other (both halves chose the
  same weights):
  - Raising all faces ×1.5 adds +0.0046 CC [+0.0033, +0.0059].
  - Doubling the talking faces adds +0.0242 CC [+0.0207, +0.0276].
  - The difference, +0.0196 CC [+0.0163, +0.0229] (+0.17 NSS), estimates what knowing the speaker is worth to this
    model's map through re-weighting alone.
- The thesis could not test speaker cues on the STAViS sets, where faces are 12–15 pixels wide at 256×340 (14B).
  Neither global audio features (11D) nor sound-source maps (14F) helped on Coutrot2. MVVA is 1280×720 and comes with
  face tracks and talking labels, so the speaker gap can be measured there.

**Speaker detection (15N)** [Completed, descriptive; `runs/review/inv/n_mvva_asd{,_posthoc,_allframes}.json`].
Light-ASD (Liao et al., CVPR 2023), trained on AVA-ActiveSpeaker, scored MVVA's own face tracks from the video frames
and the soundtrack. The tracks made face detection unnecessary.
- **It ranks the faces of a frame well.** Against the talking labels, its AUC is 0.878 over 454,040 face-frames
  (0.885 per video). In the 112,723 frames with exactly one talking face, that face scores highest 88.3 % of the time,
  against 38.6 % by chance.
- **Its decisions are conservative.** At its threshold of 0 it calls 19.4 % of face-frames talking; the labels mark
  31.3 %.
- **As the speaker in the 15K oracle** (same frames, halves, grid and bootstrap; score > 0 counts as talking), it
  adds +0.0100 CC beyond the face oracle [+0.0076, +0.0123]. That is 0.51 [0.42, 0.60] of the label-based +0.0196.
  The pre-registered expectation, more than one half, is not established.
- **Post hoc, other thresholds.** At −0.5, where the detector's talking share matches the labels', it realises 0.59
  [0.51, 0.67] (+0.0117), and both halves chose this threshold. Light-ASD's probabilities as soft weights realise 0.53.
  Taking the top-scoring face of each frame realises 0.78. That rule presumes that someone speaks, which the 15K
  frames guarantee by selection.
- **Post hoc, every frame.** The 15K frames were selected with the labels, so the same re-weighting (faces called
  talking ×2) was applied to all 28,899 scored frames (294 videos). Every one of them shows a face.

| Speaker from | ΔCC, all frames | ΔNSS | ΔCC, the 5,797 frames outside 15K |
|---|---|---|---|
| Light-ASD, threshold 0 | +0.0121 [+0.0101, +0.0141] | +0.17 | +0.0007 [−0.0020, +0.0034] |
| Light-ASD, threshold −0.5 | +0.0134 [+0.0113, +0.0156] | +0.21 | +0.0010 [−0.0022, +0.0042] |
| Labels (reference) | +0.0198 [+0.0168, +0.0228] | +0.28 | −0.0010 [−0.0018, −0.0001] |

- The gain carries over to the whole dataset, diluted by the 15K frames' share (80 %). The other frames lose nothing
  measurable.
- **Caveats.** The ×2 weight was chosen on MVVA (cross-fitted), so this is a one-parameter speaker prior fitted on half
  of MVVA, not zero-shot transfer. MVVA is the only dataset here with face tracks and talking labels, and 80 % of its
  scored frames show a talking and a silent face. Datasets with fewer conversations would gain less.
- Four videos with nominal frame rates of 23.94 and 29.58–29.59 fps get almost no scores above 0 (0–3 %, against
  42–56 % labelled talking). Their soundtracks match the video length, and the cause was not investigated.

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

### 7.2 Scene cuts (15J descriptive, 15L test) [Completed]

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
- After cuts the ground truth is more central, and it relocates 10–14 frames (≈330–470 ms) after them. This
  matches the eye-movement literature:
  - every observer re-centres on a new scene; the first saccade comes ≈356 ms after scene onset, and the centring
    lasts about two saccades (Tseng, Carmi, Cameron, Munoz & Itti, JOV 2009);
  - motion and flicker were particularly predictive of the saccades made right after a scene onset, which vary
    little between observers (Carmi & Itti, Vision Research 2006).
- DIEM, which El-Jiz et al. (WACV 2026) found to be hurt by temporal fusion, has 20.7 cuts per video; its fold-1 test
  has 231. There too, gaze is more central after cuts (centre CC 0.44–0.47 in the first 10 frames, 0.39 after 60
  frames) and less dispersed (entropy 8.57–8.80 vs 8.99).

By centrality and entropy, these frames should be easier for a model with a centre bias, yet the model scores lower
on them. This analysis cannot tell whether cross-shot context causes the dip or post-cut gaze behaviour does. The
pre-registered test 15L addresses that.

**Test of the explanations (15L) [Completed; DIEM fold-1 test with 3 seeds (primary); DHF1K holdout, seed 0].**
Dense windows (the scored frame at position 8) were scored with and without clamping each window to the frame's own
shot. Clamped positions repeat the shot's first or last frame. `runs/review/inv/summary_cuts.json`.

| | Post-cut frames (0–14 after a cut): unclamped → clamped | ΔCC [95 %] | All frames: ΔCC [95 %] |
|---|---|---|---|
| DIEM (primary; 14 videos, 689 post-cut frames per seed) | 0.7065 → 0.6052 | **−0.101 [−0.139, −0.063]** | −0.0125 [−0.0217, −0.0049] |
| DHF1K holdout (17 videos, 176 post-cut frames) | 0.5858 → 0.4488 | −0.137 [−0.199, −0.079] | −0.0034 [−0.0057, −0.0015] |

By frames since the cut. These bins were registered before the result (`summary_cuts_bins.json`):

| ΔCC (clamped − unclamped) | 0–4 | 5–9 | 10–14 | ≥60 |
|---|---|---|---|---|
| DIEM, 3 seeds | −0.230 [−0.317, −0.142] | −0.064 [−0.089, −0.037] | −0.007 [−0.018, +0.004] | −0.000 [−0.002, +0.001] |
| DHF1K holdout, seed 0 | −0.201 [−0.283, −0.115] | −0.163 [−0.232, −0.101] | −0.041 [−0.102, +0.003] | −0.002 [−0.005, +0.002] |

**Outcome, by the pre-registered reading: an effect with ΔCC < 0, so explanation (i), contamination, is refuted.**
Frames from the previous shot do not mislead the model. They help it, by 0.10–0.14 CC on post-cut frames.
- With the unclamped dense tiling, post-cut frames score about as well as frames ≥60 after a cut: DIEM −0.004
  [−0.057, +0.045], DHF1K −0.010 [−0.058, +0.042].
- The registered secondary reading ("if the dip persists under clamping, explanation (iii) carries it") does not
  apply, because clamping itself creates the dip (−0.105 on DIEM, −0.146 on DHF1K).

**Why: gaze lags the cut [Completed; descriptive, specified after the result; ground truth only, no model;
`scripts/inv_cut_gaze.py`].** The ground truth k frames after a cut is compared (CC) with the ground truth of the
last frame before the cut. The control applies the same measure around pseudo-cuts in the middle of uncut shots.

| k (frames after the cut) | 0 | 2 | 4 | 6 | 8 | 10 | 14 | 20 |
|---|---|---|---|---|---|---|---|---|
| DIEM, at cuts (1,254 cuts, 75 videos) | 0.979 | 0.935 | 0.843 | 0.630 | 0.486 | 0.404 | 0.319 | 0.290 |
| DIEM, no cut (352 pseudo-cuts) | 0.976 | 0.931 | 0.881 | 0.835 | 0.791 | 0.754 | 0.690 | 0.614 |
| DHF1K holdout + val, at cuts (81 cuts, 33 videos) | 0.836 | 0.716 | 0.643 | 0.612 | 0.537 | 0.445 | 0.343 | 0.258 |
| DHF1K holdout + val, no cut (162 pseudo-cuts) | 0.863 | 0.736 | 0.662 | 0.610 | 0.570 | 0.535 | 0.493 | 0.430 |

- For the first 2–4 frames (DIEM) or 6 frames (DHF1K) after a cut, gaze stays as close to its pre-cut position as
  it does in uncut footage. Only then does it move to the new shot.
- Gaze is closer to its settled post-cut distribution (frames 30–44) than to the pre-cut one from frame 8 (DIEM) or
  11 (DHF1K) on. Without a cut, that crossing comes at frame 17 and 13.
- This time course matches the cost of clamping, which is largest at 0–4 frames and gone by 10–14. It also matches
  the first-saccade latency after a scene onset (≈356 ms, about 11 frames at 30 fps; Tseng et al., JOV 2009).
- For a few hundred milliseconds after a cut, the previous shot predicts where people look. The model uses this.

**Default tiling [Completed; descriptive, post hoc; DHF1K holdout, seed 0].** Part of the dip of the default tiling
reported above comes from the tiling itself. At 5–9 frames after a cut, the default tiling scores 0.026 [0.006, 0.051]
below the dense tiling (0.554 vs 0.579). At ≥60 frames after a cut, the two agree (+0.001 [−0.003, +0.005]).
[Hypothesis] In the default tiling, some post-cut frames sit early in their window and see less of the previous shot.

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
- **Larger test inputs keep NSS but lose CC** (320²: NSS +0.009, CC −0.018, KLD +0.05). The map is as high at
  fixated pixels as before, but its overall shape matches the ground truth less well (section 4.4).
- **The gain from training at 384² is spread across videos** (40 of 60 gain), with no clear concentration on the
  small, off-centre targets among the worst videos (post-hoc, descriptive; section 4.4).
- **The transfer gap is content, not dataset bias.** Three bias parameters recover 0.02–0.19 of the fine-tuning gain
  on seven of eight video datasets (15M). For images, fewer than 20 such parameters closed more than 75 % of the gap
  (section 6.2).
- **Heavy blur costs as much as reversing time** (σ 8: −0.052 CC), while noise, JPEG compression and greyscale
  cost at most 0.0033 (section 6.1).
- **The default tiling is near the accuracy-efficiency knee**, although it was chosen without a sweep (thesis).
- **Gaze after cuts is more central and more concentrated**, yet the model scores lower there in the default tiling.
- **Cross-shot context helps after cuts, contrary to the pre-registered hypothesis.** Removing it costs 0.10–0.14
  CC on post-cut frames, because gaze stays where it was for 2–6 frames after a cut (section 7.2).

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

Every row rests on completed experiments (DHF1K holdout unless stated). Types: **architecture** (a property of the
design), **training** (of the recipe), **input** (information the model does not receive), **data**, **evaluation**.

| # | Limitation | Evidence | Size | Type |
|---|---|---|---|---|
| L1 | No state across windows. Positions without past frames are poor, so the default tiling discards half of each window's outputs | 4.1, 5.2 | position 0: −0.077 CC against position 8; all 16 positions kept: −0.0063 at half the encoder passes | architecture; doubles the encoder cost |
| L2 | Input resolution limits accuracy | 4.4 (F3), 6.1 | 384²: +0.0092 [+0.0030, +0.0163], spread across videos; blur σ 8: −0.052 | input; the gain costs 2.77 times the encoder FLOPs |
| L3 | Tuned to one frame rate and one input size | 4.3, 4.4 | at test time, stride 1 / 4: −0.012 / −0.019; 224² / 320²: −0.014 / −0.018 | training (one rate, one size) |
| L4 | The speaker is under-weighted in multi-face scenes | 6.2 (15K, 15N) | talking faces get 54.6 % of the predicted face mass against 66.8 % of gaze; a speaker oracle adds +0.020 CC, a detector +0.012–0.013 (MVVA) | input (no speaker cue) |
| L5 | The decoder is data-limited | 1, 4.5 | best epoch the second in 27 of the 30 seed–variant pairs of 15C/15D (V1 excluded); 135 → 270 → 540 videos: 0.5645 → 0.5729 → 0.5783 (ViT-L) | data |
| L6 | Transfer to other datasets is limited by content | 6.2 (15I, 15M) | fine-tuning adds 0.02–0.16 CC; three bias parameters recover 0.02–0.19 of it on seven of eight datasets | data (content), not a low-dimensional bias |

**What a successor must keep (measured).**
- A non-linear decoder on the token grid: +0.072 CC over a linear readout (V1).
- Temporal convolutions in the decoder: +0.023 (V3a, 15C2). With them, clip-pooled norm statistics add nothing
  measurable (V3b equivalent), so each time step can be normalised on its own.
- Temporal input: +0.057 over a matched static model (E2). The benefit is three times larger in the most moving fifth
  of frames than in the stillest.
- A frozen encoder: LoRA adaptation lost in all three seeds (thesis).

**Checked and ruled out.**
- Cross-shot context after cuts: it helps (+0.10–0.14 CC on post-cut frames, 15L).
- Output resolution: a 64² output allows CC 0.999, and even the 16² token grid allows 0.91 (F1).
- Combining depths: the last tap alone −0.0015 (inconclusive); taps 30 and 40 at 1/16 only −0.004 (V2, V4).
- Loss design: without the CC term or the NSS term, equivalent (15D). A three times higher learning rate: −0.0004,
  inconclusive.
- Photometric robustness: noise, JPEG, contrast and greyscale change CC by at most 0.0033 (15H).
- A low-dimensional dataset bias: it explains little of the transfer gap (15M).
- Clip-pooled normalisation as a route for temporal information: equivalent to per-step normalisation (15C2).
- Reproducibility: A1 identical; A2 at most 0.0007 CC per epoch.
- Per-video variance: 74 % follows the ground truth's reliability and entropy, a property of the evaluation (7.3).

**Headroom on DHF1K.** Every single improvement measured or bounded so far raises holdout CC by at most about 0.01:
384² +0.009 (F3); ViT-L → ViT-g +0.008 (seed 0, thesis); 32 frames +0.004 (seed 0, thesis); a second frozen encoder
+0.004 (not established); a three-seed ensemble +0.003 (val); denser tilings at most +0.0025 (5.2). The large
measured gaps lie elsewhere: transfer to other datasets (0.02–0.16) and speakers in multi-face scenes (0.012–0.020).

## 10. Literature-validated research gaps

Each gap follows from a limitation of section 9. Section 10.7 lists the literature by topic, and novelty is judged
against it. Sources are peer-reviewed unless a preprint is named as such.

### 10.1 G1. State across windows of a frozen clip-level video encoder (L1)
- **Evidence.** A frame at the start of a window loses 0.077 CC. The loss halves with each past frame and is gone
  after ≈0.35–0.4 s (4.1). Keeping all 16 positions costs 0.0063 at half the encoder passes, while two or three times
  the passes gain at most 0.0025 (5.2). The decoder's temporal benefit comes from six 3×1×1 convolutions, which see
  zero padding at a window's first time step (15C2).
- **Motivation.** The encoder is 97 % of the FLOPs (5.1). Keeping all 16 positions halves the passes per output frame
  (0.063 → 0.032). It would also allow streaming with a latency of one window.
- **Related work.** Recurrent saliency models carry state on image backbones (ACLNet, CVPR 2018 and TPAMI 2021;
  UNISAL, ECCV 2020; SalEMA, BMVC 2019). MeMViT (CVPR 2022) caches memory inside the encoder. MoViNets (Kondratyuk et
  al., CVPR 2021) introduce a stream buffer, "a mechanism to cache feature activations on the boundaries of subclips",
  which replaces the causal padding of a convolution with "the last few frames from a previous subclip". Video Depth
  Anything (CVPR 2025) keeps its encoder frozen and carries content across windows, and an online variant (3DV 2026)
  caches latent features. XMem (ECCV 2022) keeps feature memory for segmentation.
- **Unresolved.** A light decoder that gives a window's first positions the past they lack, for a frozen clip-level
  encoder, without re-encoding frames. None was found for saliency.
- **Potential solutions [Proposed].**
  - (a) A stream buffer in the decoder's six temporal convolutions. At a window's first time step, each convolution
    reads the previous window's last time step instead of zeros. This adds no parameters. Through the six
    convolutions it reaches six time steps (12 positions) back, beyond the 5–6 positions after which the deficit is
    gone [Hypothesis: enough, if past decoder features can make up for the encoder's missing past]. GroupNorm per
    time step (V3b, equivalent) normalises carried and new features alike.
  - (b) If (a) is not enough, a zero-initialised cross-attention block through which a window's first positions
    attend to the previous window's features at its last positions.
  - Both are trained on pairs of consecutive windows.
- **Novelty.** Moderate. Caching across windows exists for recognition (MoViNets), depth and segmentation. For
  saliency from a frozen clip-level encoder, motivated by a measured positional deficit, it is new.
- **Validation [Proposed].** First (a) at test time on an existing checkpoint (under an hour). Then the trained all-16
  tiling with state against the default tiling without state, rule of 12E, three seeds. Equivalence (±0.003) at half
  the passes is success; a position-0 deficit below 0.01 shows that the mechanism works (13.2).
- **Expected.** Accuracy within ±0.003 of the default, at half the encoder passes.

### 10.2 G2. Resolution at an affordable cost (L2)
- **Evidence.** Training at 384² gains +0.0092 CC [+0.0030, +0.0163], spread across videos (4.4). Blur σ 8 costs
  0.052 (6.1). The 384² encoder needs 2.77 times the FLOPs (5.3).
- **Motivation.** It is the largest improvement found in this investigation, at almost three times the encoder cost.
- **Related work.** SALICON (Huang, Shen, Boix & Zhao, ICCV 2015, doi:10.1109/ICCV.2015.38) integrates information
  at several image scales for image saliency. In the AIM 2024 challenge (ECCV Workshops), the second-ranked team added
  "a spatial detail encoding branch based on a single high-resolution video frame" to low-resolution video context;
  the challenge report does not isolate its effect. FixRes (NeurIPS 2019) explains the train–test resolution
  discrepancy, and gaze is largely set by coarse content (Judd et al., JOV 2011).
- **Unresolved.** No controlled measurement of input resolution for video saliency from a frozen video encoder was
  found before F3, nor a design that obtains the gain below its cost.
- **Potential solutions [Proposed].**
  - (a) Combine with G1. Keeping all 16 positions halves the passes, so 384² would cost 1.39 times today's encoder
    FLOPs per output frame instead of 2.77.
  - (b) A single-frame high-resolution detail branch beside the 256² video encoder, as in the AIM 2024 entry. Its
    cost depends on the image encoder.
- **Novelty.** Low for resolution alone; moderate in combination with G1.
- **Validation [Proposed].** 384² with state against the 256² default (an effect of at least 0.003 expected) and
  against F3's 384² default (equivalence expected), with the cost per output frame measured by the A3 protocol (13.3).

### 10.3 G3. Rate- and size-robust decoding (L3)
- **Evidence.** At test time, other strides cost 0.012–0.019 CC and other sizes 0.014–0.018 (4.3, 4.4). SumMe and
  MVVA include 15 fps videos, where stride 2 spans 2.1 s.
- **Related work.** FixRes; FlexiViT (CVPR 2023), where random patch sizes give one set of weights that works across
  them; Multigrid training (CVPR 2020); action-recognition models are mostly robust to rate changes (Schiappa et al.,
  CVPR 2023). Cheon & Lee (2015) studied frame rate and saliency (title only).
- **Unresolved.** No study of the frame-rate robustness of deep video saliency models was found.
- **Potential solution [Proposed].** First, at no cost, choose the stride from the native frame rate (stride 1 at
  15 fps). Then train the decoder over random strides and sizes.
- **Novelty.** Low to moderate.
- **Validation [Proposed].** The zero-cost check on the 15 fps videos (13.4), then matched-condition comparisons at
  strides 1, 2 and 4 and sizes 224–384.
- **Expected.** Up to 0.012–0.019 off the training condition, nothing at it.

### 10.4 G4. Content in cross-dataset transfer (L6)
- **Evidence.** Fine-tuning adds 0.02–0.16 CC, and three bias parameters recover 0.02–0.19 of that on seven of eight
  datasets (15I, 15M).
- **Motivation.** These are the largest measured gaps. For images the opposite holds: bias dominates (Kümmerer,
  Khanuja & Bethge, ICCV 2025).
- **Related work.** UNISAL (ECCV 2020) and HD²S (IJCV 2021) train one model across datasets, with dataset-specific
  priors, smoothing and normalisation.
- **Unresolved.** Which content a DHF1K-trained model misses on each dataset, and whether more diverse training
  content narrows a gap that bias does not explain.
- **Potential solution [Proposed].** More diverse training content for the decoder: AViMoS's 1,000 training videos
  (G5), or several eye-tracking datasets with the frozen encoder. Where the gap is largest, content-specific cues: the
  speaker (Coutrot2, G6) and the viewing task (UCF Sports).
- **Novelty.** The finding that bias explains little of video transfer is new, and stands as an analysis result. The
  remedies are known in kind.
- **Validation [Proposed].** Zero-shot CC on the eight eye-tracking datasets with the 15I protocol, and the 15M share
  as a second outcome (13.5).

### 10.5 G5. Training data for the decoder (L5)
- **Evidence.** The best epoch is the second in 27 of the 30 seed–variant pairs of 15C/15D (V1 excluded), and loss
  or learning-rate changes do not help (4.5). The second doubling of DHF1K data gives 0.64 of the first doubling's
  gain (section 1).
- **Related work.** SALICON (CVPR 2015) collected mouse tracking for image saliency, and mouse data are acceptable for
  training (Tavakoli et al., CVPR 2017). AViMoS (AIM 2024, ECCV Workshops) has 1,500 videos under CC-BY-4.0, 1,000 of
  them for training. Lyudvichenko et al. (GraphiCon 2019) used mouse maps as an input, not for training.
- **Unresolved.** No test was found of pre-training a video saliency model on mouse tracking and evaluating it on eye
  tracking.
- **Potential solution [Proposed].** Train the decoder on DHF1K and AViMoS together, or pre-train on AViMoS and
  fine-tune on DHF1K.
- **Novelty.** Moderate.
- **Validation [Proposed].** DHF1K holdout (rule of 12E) and zero-shot transfer (descriptive) (13.5).
- **Expected.** Small on DHF1K: the learning curve suggests at most about +0.0035 per doubling of eye-tracking data,
  and less for mouse data. Unknown for transfer [Hypothesis: larger where the gap is content].

### 10.6 G6. Speaker-aware weighting in social scenes (L4)
- **Evidence.** A speaker oracle adds +0.020 CC on MVVA and Light-ASD +0.012–0.013 over all frames; frames without a
  talking face lose nothing measurable (15K, 15N). Coutrot2 has the largest fine-tuning gain (+0.158), of which bias
  recovers 0.04 (15M).
- **Related work.** Coutrot & Guyader (JOV 2014; ICIP 2014, where speaker diarization raises speakers above
  addressees); Liu et al. (CVPR 2017); Xu et al. (TIP 2018); MVVA (ECCV 2020); AVM-Net (IJCV 2023); TalkNet;
  Light-ASD.
- **Unresolved.** The speaker effect on a strong frozen encoder beyond MVVA. In the STAViS frames, faces are 12–15
  pixels wide, which rules out speaker detection on Coutrot2.
- **Novelty.** Low: the mechanism has been known since 2014. Its measurement on a frozen deep encoder is new.
- **Validation.** Done on MVVA, cross-fitted. Coutrot2 only with native-resolution frames.

**Not pursued, and why.** Shot-aware context resets (refuted, 15L). Dataset-bias calibration (15M). Decoder
redesigns of depth fusion, priors and loss terms (15C, 15D: at most 0.004). Audio features, sound-source maps, a
predictability gate, a second encoder and LoRA (thesis: none adopted). An epistemic (disagreement) readout of the
predictor remains untested; the thesis's prediction-error result was negative.

### 10.7 Literature checked, by topic

**Temporal information in video saliency.**
- Tangemann, Kümmerer, Wallis & Bethge (ECCV 2020, doi:10.1007/978-3-030-58604-1_40): on LEDOV, at least 75 % of
  the explainable information is captured by static features. A static model (DeepGaze MR) beat the video models of
  the time.
- El-Jiz, Kümmerer, Tangemann, Bethge, Bartels & Bannert (WACV 2026, doi:10.1109/WACV61042.2026.00572) compared a spatio-temporal model with an
  otherwise identical spatial model. Temporal fusion gained 0.20 bits per fixation on LEDOV and lost 0.07 bits on
  DIEM, which they attribute to hard cuts and rapid zooms that "disrupt current temporal fusion approaches". They do
  not test a remedy.
- Together, these papers suggest that temporal integration helps on coherent footage and hurts on edited footage.
  Neither studies a frozen video world-model encoder, positional effects inside a clip, or a remedy for cuts.
- Video saliency decoders often keep the time axis. THTD-Net (Moradi, Palazzo & Spampinato, VISAPP 2024,
  doi:10.5220/0012422800003660) reduces "temporal features' dimensions" gradually in its decoder. It performs
  comparably to multi-branch models on DHF1K, UCF Sports and Hollywood-2. 15C measures this design choice under
  control, for a frozen video encoder: removing the decoder's temporal mixing costs 0.024 CC (section 3).
- For SURGE-D on DHF1K, a matched model trained on static clips loses 0.057 CC (15E2, section 4.2). On edited
  footage such as DIEM, El-Jiz et al. found temporal fusion to hurt. No matched static control has been run there.
- For SURGE-D the cut part does not hold: 15L shows that context from before a cut helps the frames after it
  (section 7.2). Whatever hurts El-Jiz et al.'s temporal model on DIEM, it is not, for SURGE-D, the presence of
  cross-shot frames.

**Gaze around cuts (vision science).**
- Every observer re-centres on a new scene. The first saccade comes ≈356 ms after scene onset, and the centring
  lasts about two saccades (Tseng, Carmi, Cameron, Munoz & Itti, JOV 2009, doi:10.1167/9.7.4). The authors
  recommend a centre-biased evaluation baseline for the first saccades after a scene onset and an unbiased one
  afterwards. That is a time-since-cut centre prior, proposed for evaluation only.
- Motion and flicker were particularly predictive of the saccades made right after a scene onset, which vary little
  between observers (Carmi & Itti, Vision Research 2006, doi:10.1016/j.visres.2006.08.019).
- Inter-observer congruency drops in the five frames after a cut. The authors suggest shot metadata for saliency
  models but do not build such a model (Bruckert, Christie & Le Meur, Behavior Research Methods 2023,
  doi:10.3758/s13428-022-01949-7).
- No saliency model found so far treats shot boundaries explicitly. Sliding-window models (TASED-Net, ICCV 2019,
  doi:10.1109/ICCV.2019.00248; ViNet, IROS 2021, doi:10.1109/IROS51168.2021.9635989; SalFoM, ICPR 2024,
  doi:10.1007/978-3-031-78312-8_3; SURGE-D) feed cross-shot frames as context.
- **Gap withdrawn.** The candidate gap "reset temporal context at cuts" is refuted for SURGE-D. Resetting costs
  0.10–0.14 CC on post-cut frames, because gaze lags the cut by 2–6 frames (section 7.2). Any cut-aware design has
  to keep the previous shot for the first ≈10 frames. The time course itself (lag, then re-centring) is
  well described in vision science and is already exploited implicitly. No measurable post-cut deficit remains under
  the dense tiling.

**State carried across clips.**
- Recurrent video saliency models carry state from frame to frame on top of image backbones: ACLNet (Wang et al.,
  CVPR 2018, doi:10.1109/CVPR.2018.00514; TPAMI 2021, doi:10.1109/TPAMI.2019.2924417), UNISAL (Droste, Jiao & Noble, ECCV 2020) and SalEMA (Linardos et al., BMVC 2019). SalEMA
  found that an exponential moving average of an internal state gives results and maps similar to a ConvLSTM.
- MeMViT (Wu et al., CVPR 2022, doi:10.1109/CVPR52688.2022.01322) processes a video online as a sequence of short
  clips, attending to a cached memory of earlier clips. This gave 30× longer temporal support for 4.5 % more compute
  in action recognition. The memory sits inside the encoder, so it does not apply directly to a frozen encoder.
- MoViNets (Kondratyuk et al., CVPR 2021, doi:10.1109/CVPR46437.2021.01576) introduce a stream buffer, "a mechanism
  to cache feature activations on the boundaries of subclips". For causal convolutions it replaces the padding with
  "the last few frames from a previous subclip", so that 3D CNNs process streams of any length with constant memory.
  In SURGE-D the decoder's temporal benefit comes from its six temporal convolutions (15C2), so a buffer there
  would need no change to the encoder.
- LSTD-Net (Moradi et al., CVIU 2026, doi:10.1016/j.cviu.2026.104740) adds long- and short-term temporal modelling
  to a UMT foundation encoder. Its full text could not be accessed, so whether it carries state across clips is
  unverified.
- A preprint (Wulff, Abawi, Allgeuer & Wermter, arXiv 2504.05913, 2025) reports that short-term context helps up to
  a threshold, while longer-term context keeps helping. Within a clip, SURGE-D's benefit from past context also
  saturates, after ≈0.4 s (section 4.1).
- Video Depth Anything (Chen et al., CVPR 2025, doi:10.1109/CVPR52734.2025.02126) keeps its encoder frozen "to
  reduce training costs and preserve well-learned features". For long videos it feeds key frames from earlier windows
  into the current window, which "incorporates content from earlier windows into the current window with minimal
  computation burden". An online variant caches latent features during inference instead, and masks frames during
  training (Feiden et al., 3DV 2026, doi:10.1109/3DV69130.2026.00167).
- In video object segmentation, XMem (Cheng & Schwing, ECCV 2022, doi:10.1007/978-3-031-19815-1_37) keeps feature
  memory stores across long videos, inspired by the Atkinson–Shiffrin model of human memory.
- These designs carry content across windows for recognition, depth and segmentation. No
  peer-reviewed study was found of a light decoder that carries state across clips of a frozen clip-level video
  encoder for saliency. Sections 4.1 and 5.2 quantify what such a state would have to recover: up to 0.077 CC at the
  start of a window, and 0.0063 CC overall if all 16 positions were kept.

**Frozen backbones and readouts.**
- DeepGaze II (Kümmerer, Wallis, Gatys & Bethge, ICCV 2017, doi:10.1109/ICCV.2017.513) and DeepGaze IIE (Linardos,
  Kümmerer, Press & Bethge, ICCV 2021, doi:10.1109/ICCV48922.2021.01268) read out frozen image backbones.
- Video Depth Anything (CVPR 2025, doi:10.1109/CVPR52734.2025.02126) replaces the head of an image depth model with
  an efficient spatio-temporal head and keeps the encoder frozen.
- Layer-wise analyses of ViTs find mid-to-late layers best for localised tasks (Walmer et al., CVPR 2023,
  doi:10.1109/CVPR52729.2023.00723).
- Dense prediction from plain ViTs.
  - DPT (Ranftl, Bochkovskiy & Koltun, ICCV 2021, doi:10.1109/ICCV48922.2021.01196) assembles tokens from several
    stages of the transformer into image-like representations at several resolutions. A convolutional decoder then
    combines them.
  - For detection, ViTDet (Li, Mao, Girshick & He, ECCV 2022, doi:10.1007/978-3-031-20077-9_17) found it "sufficient
    to build a simple feature pyramid from a single-scale feature map".
  - SURGE-D's decoder builds a ViTDet-style pyramid from four taps, as DPT does. 15C points the way ViTDet did, for
    saliency from a frozen video encoder: the decoder fed only the last tap loses 0.0015 (inconclusive), and
    without the shallower taps and the top-down path it loses 0.004 (section 3).
- No peer-reviewed probe of a frozen video world-model encoder for saliency was found. The closest, PredJSal (NTIRE
  2026), is an arXiv preprint.

**Data scale and mouse tracking.**
- SALICON (Jiang et al., CVPR 2015, doi:10.1109/CVPR.2015.7298710) established mouse-tracking pre-training for image saliency.
- Mouse data agree only partly with eye tracking and are less reliable for evaluation, but acceptable for training
  (Tavakoli et al., CVPR 2017, doi:10.1109/CVPR.2017.673).
- AViMoS (AIM 2024, ECCV Workshops, doi:10.1007/978-3-031-91856-8_11) offers 1,500 videos of crowdsourced mouse
  tracking, validated against eye tracking with a 300 ms lag correction.
- The closest video work found uses mouse data as an input, not for training (Lyudvichenko et al., GraphiCon 2019,
  CEUR Workshop Proceedings vol. 2485; arXiv 1907.00480).
  - A SAM-ResNet with an LSTM takes the frames plus a saliency map from a few observers as an extra prior, and
    refines that map.
  - It was trained on eye-tracking data (DHF1K and SAVAM), with three observers' eye-tracking maps as the prior.
    At test time it received mouse-tracking maps of ten observers instead.
- No test was found of pre-training a video saliency model on mouse tracking and evaluating it on eye-tracking
  benchmarks.

**Faces and speakers.**
- Gaze follows the speaker in conversations (Coutrot & Guyader, JOV 2014, doi:10.1167/14.8.5).
- The same authors built an audiovisual saliency model that uses speaker diarization to raise the saliency of the
  speakers above that of the addressees. Against eye tracking, it "significantly outperforms visual attention models
  using an equal and constant saliency value for all faces" (Coutrot & Guyader, ICIP 2014,
  doi:10.1109/ICIP.2014.7025219). The re-weighting of 15K and 15N is this mechanism, measured on top of a frozen deep
  video encoder.
- In multiple-face videos, "the attention of different subjects consistently focuses on a single face in each frame"
  (Liu, Zhang, Xu & He, CVPR 2017, doi:10.1109/CVPR.2017.343). Their model predicts the salient face and its
  transitions across frames (CNN with a multiple-stream LSTM).
- Speaking and head turning are "extremely attractive to visual attention in videos". A hidden Markov model over
  faces used them, with face size, centre bias and head pose, to predict which face is looked at (Xu, Liu, Hu & He,
  IEEE TIP 2018, doi:10.1109/TIP.2018.2837106).
- Attention to conversational clips has also been modelled from audio and video together, with social value
  modulating selection (Boccignone et al., ECCV Workshops 2018, doi:10.1007/978-3-030-11012-3_27).
- MVVA (Liu et al., ECCV 2020, doi:10.1007/978-3-030-58565-5_25) showed that attention in multi-face videos differs
  with sound when turn-taking occurs.
- AVM-Net (Qiao et al., IJCV 2023, doi:10.1007/s11263-023-01950-3) learns saliency and sound-source localisation
  jointly on MVVA.
- Active-speaker detectors exist: TalkNet (ACM MM 2021, doi:10.1145/3474085.3475587) and Light-ASD (CVPR 2023,
  doi:10.1109/CVPR52729.2023.02196).
- Explicit speaker modelling is therefore established for multi-face video. What it adds on top of a strong frozen
  visual encoder had not been measured. For SURGE-D, 15K now gives a first estimate: on MVVA, a speaker oracle adds
  +0.020 CC beyond a face oracle through re-weighting alone (section 6.2). With Light-ASD in place of the labels
  (15N), re-weighting adds +0.012–0.013 CC over all MVVA frames, with the weight fitted on MVVA.

**Prediction error as attention.**
- Bayesian surprise (Itti & Baldi, Vision Research 2009, doi:10.1016/j.visres.2008.09.007) is a change of belief, not a prediction error.
- Prediction-error curiosity is captured by stochastic "noisy TV" content (Burda et al., ICLR 2019); ensemble
  disagreement avoids this (Pathak, Gandhi & Gupta, ICML 2019).
- A V-JEPA prediction error detects physically impossible events (Garrido et al., 2025, arXiv preprint).
- The thesis's negative result (error anti-correlated with fixations) is consistent with the noisy-TV account. A
  disagreement-based (epistemic) readout has not been tested against gaze.

**Input size, fine detail and frame rate.**
- Image classifiers trained with random-resized crops see objects at another apparent size at test time. A cheap
  fine-tuning at the test resolution corrects this (Touvron, Vedaldi, Douze & Jégou, NeurIPS 2019). SURGE-D loses CC
  both below and above its training size (section 4.4). Trained at 384², it gains +0.0092 CC (F3, section 4.4). The
  test-time losses are therefore a train–test mismatch, and a larger input helps once the model is trained for it.
- "Fixations from lower resolution images can predict fixations on higher resolution images", and inter-observer
  consistency rises with resolution "until around 16-64 pixels (1/2 to 2 cycles per degree) after which consistency
  remains relatively constant" (Judd, Durand & Torralba, JOV 2011, doi:10.1167/11.4.14). Where people look is thus
  largely set by coarse content. Yet SURGE-D loses 0.052 CC under blur σ 8 (section 6.1) and gains 0.009 CC from a
  384² input (F3). The ground truth comes from sharp video, so part of the blur loss is information the blurred input
  no longer contains.
- Multi-scale input in saliency models:
  - SALICON (Huang, Shen, Boix & Zhao, ICCV 2015, doi:10.1109/ICCV.2015.38) fine-tunes object-recognition networks
    for image saliency and integrates "information at different image scales".
  - In the AIM 2024 video saliency challenge (Moskalenko et al., ECCV Workshops, doi:10.1007/978-3-031-91856-8_11),
    the second-ranked team added "a spatial detail encoding branch based on a single high-resolution video frame" to
    low-resolution video context. The challenge report does not isolate the effect of that branch.
  - No controlled measurement of input resolution for video saliency from a frozen video encoder was found before
    F3.
- SURGE-D loses 0.012–0.019 CC when its frame stride differs from training (section 4.3). Schiappa et al. (CVPR 2023,
  below) found action-recognition models mostly robust to such changes.
- Frame rate in video saliency: Cheon & Lee (Signal Processing: Image Communication 2015,
  doi:10.1016/j.image.2015.05.010) analysed gaze patterns and evaluated saliency models at several temporal
  resolutions. Its abstract could not be accessed, so only its title is used here. No study of the frame-rate
  robustness of current deep video saliency models was found.
- Training over several sizes or rates:
  - FlexiViT (Beyer et al., CVPR 2023, doi:10.1109/CVPR52729.2023.01393): "simply randomizing the patch size at
    training time leads to a single set of weights that performs well across a wide range of patch sizes".
  - Multigrid training (Wu, Girshick, He, Feichtenhofer & Krähenbühl, CVPR 2020, doi:10.1109/CVPR42600.2020.00023)
    uses "variable mini-batch shapes with different spatial-temporal resolutions". It trained a SlowFast network 4.5×
    faster while improving Kinetics-400 accuracy by 0.8 points.

**Dataset bias and training on several datasets.**
- UNISAL (Droste, Jiao & Noble, ECCV 2020, doi:10.1007/978-3-030-58558-7_25) names the domain shift between image
  and video saliency data as the main obstacle to one model for both. It trains a light encoder–RNN–decoder on both
  kinds of data, with domain-adaptive priors, fusion and smoothing and a "Bypass-RNN". With one set of parameters it
  reached state-of-the-art results on DHF1K, Hollywood-2 and UCF Sports, and was competitive on SALICON and MIT300.
- For images, models lose about 40 % of their performance when moved to another dataset, and greater training
  diversity does not close the gap, "with close to 60 % attributed to dataset-specific biases" (Kümmerer, Khanuja &
  Bethge, ICCV 2025, doi:10.1109/ICCV51701.2025.02050). Fewer than 20 dataset-specific parameters (multi-scale
  structure, centre bias, fixation spread), adapted to the new data, account for more than 75 % of the gap, much of
  it with as few as 50 samples.
- HD²S (Bellitto, Proietto Salanitri, Palazzo, Rundo, Giordano & Spampinato, IJCV 2021,
  doi:10.1007/s11263-021-01519-y) specialises a video saliency model to each dataset through domain-specific priors,
  smoothing and batch normalisation. For datasets without annotations, it learns general features with gradient
  reversal at several scales, and then performs comparably to supervised state-of-the-art models.
- Video datasets were not part of Kümmerer et al.'s study. For SURGE-D, three such parameters recover only 0.02–0.19
  of the fine-tuning gain on seven of eight video datasets (15M, section 6.2). Its transfer gap is mainly content.

**Robustness and evaluation.**
- Common corruptions: Hendrycks & Dietterich (ICLR 2019).
- Temporal perturbations of video models (sampling rate, reversal, jumbling, freezing): Schiappa et al. (CVPR 2023,
  doi:10.1109/CVPR52729.2023.01412)
  found action-recognition models robust to them on most datasets but not on SSv2, where actions can be reversed.
  No comparable study of video saliency models was found.
- Gaze itself changes under image transformations (Che et al., IEEE TIP 2020, doi:10.1109/TIP.2019.2945857).
- Metric behaviour: Bylinskii et al. (TPAMI 2019, doi:10.1109/TPAMI.2018.2815601); Kümmerer et al. (ECCV 2018;
  PNAS 2015, doi:10.1073/pnas.1510393112).

## 11. Ranked potential research contributions

Criteria, in order: the measured or bounded gain on DHF1K; the gain elsewhere; novelty against section 10; the cost
to test; the risk. GPU times are estimates from the measured epoch times (single RTX A5000).

| Rank | Contribution | DHF1K | Elsewhere | Novelty | Cost to test | Main risk |
|---|---|---|---|---|---|---|
| 1 | Stateful decoder with a 384² input (G1 + G2) | +0.009 measured for 384² (F3); the state bounded to ±0.003 | untested (13.6 checks whether 384² transfers) | moderate | under an hour for the test-time check; ≈ 4 h (three seeds at 256²), then ≈ 10 h at 384² | the state may not close the positional deficit, if the deficit lies in the encoder's features |
| 2 | More diverse training content: mouse tracking and other datasets (G5 + G4) | small: at most ≈ +0.0035 per doubling | the transfer gap of 0.02–0.16 is content, so possibly large | moderate | ≈ 9 h | mouse maps differ from gaze (AViMoS chose a stronger centre in 15M); for images, more diverse data did not close the gap |
| 3 | Rate- and size-robust decoding (G3) | none at the training condition | up to 0.012–0.019 off it | low to moderate | hours for the zero-cost check | datasets may lie close to the training rate |
| 4 | Speaker-aware weighting (G6) | none (few conversations in DHF1K) | MVVA +0.012–0.013 with a detector | low | done for MVVA | needs resolvable faces |
| 5 | Analysis: where SURGE-D's accuracy comes from (sections 3–7), and a video transfer gap that is content, not bias (15M) | — | — | moderate | done | 15M uses three parameters and one encoder |

## 12. Recommended research direction

**Recommendation: a stateful, higher-resolution decoder for a frozen video encoder (rank 1), with more diverse
training content (rank 2) as the second line.**

- SURGE-D's accuracy comes from the frozen encoder's spatio-temporal features, read by a non-linear decoder with
  temporal convolutions (V1, V3, 15C2, E2). Decoder details move CC by at most 0.004 (15C, 15D). Further gains must
  come from what the encoder sees and from how much of its output is used.
- Both are measured. At 256² the encoder sees too little detail (L2: +0.009 at 384²). The default tiling throws away
  half of the encoder's outputs, because they lack a past (L1).
- The two fit together. A state that makes all 16 positions usable halves the encoder passes, which pays for half the
  cost of the larger input: 1.39 times today's encoder FLOPs per output frame instead of 2.77.
- The state has a parameter-free first form. 15C2 shows that the decoder's temporal benefit comes from its six
  temporal convolutions, and that per-step normalisation costs nothing. A stream buffer in those convolutions (as in
  MoViNets) can be tried on an existing checkpoint within an hour, before anything is trained.
- **Expected outcome [Hypothesis].** About +0.009 CC on DHF1K at about 1.4 times today's encoder cost, and a streaming
  mode with a latency of one window. The novelty lies in the state mechanism; the resolution result is a measurement.
- **Stop rule.** If the state does not bring the all-16 tiling to equivalence with the default at 256² (13.2), drop
  it and report 384² as a resolution result with its cost.
- The second line addresses the largest measured gaps (transfer, 0.02–0.16), with an uncertain outcome. It changes the
  training content, not the model, so it combines with the first.

**What the evidence advises against.** Further decoder variants, loss tweaks, audio features, surprise signals, a
second encoder and dataset-bias calibration: each was tested, and none gave an established gain.

**On beating the state of the art.** SURGE-D already exceeds SalFoM on DHF1K val under an identical protocol (+0.036
CC, section 8). The proposed direction would add about 0.009 on the holdout. An official claim still needs the DHF1K
test-server result, and recent models whose results could not be verified were not compared.

## 13. Proposed experiments for validating the improvement

Sections 13.2–13.7 are [Proposed]. Each decision rule is to be written into the research log and committed before
the first run. Suggested order: the test-time steps first (13.2 step 0, 13.4, 13.6), then the
trained parts of 13.2, 13.3, 13.5 and 13.7. The test-time steps take from under an hour (13.2 step 0) to
several hours (13.6).

### 13.1 Scene cuts (15L) [Completed; result in section 7.2]

**Outcome.** ΔCC = −0.101 [−0.139, −0.063] on DIEM (3 seeds), an effect with ΔCC < 0, so (i) is refuted. The
registered reading named (ii) or (iii) as the alternatives. The data support neither:
- **Against (ii):** past frames from the previous shot help.
- **Against (iii):** under the dense tiling, post-cut frames show no significant dip.

The explanation that fits is a fourth one, found by a descriptive follow-up on the ground truth: gaze lags the cut,
so the previous shot is informative. Shot-aware windows would harm the model, not help it.

The registered design, kept for the record. Three explanations competed for the post-cut dip:
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

### 13.2 P1. Cross-window state at 256² (G1)
- **Step 0 (test time, seed 0, descriptive).** The existing checkpoint with a stream buffer in its six temporal
  convolutions, under the all-16 tiling: at a window's first time step, each convolution reads the previous window's
  last time step instead of zeros. Against the same tiling without the buffer, per position as in 4.1. It shows
  within an hour whether past decoder features help at all.
- **P1a (trained).** Decoder R with GroupNorm per time step (V3b, equivalent to R in 15C2) and the stream buffer.
  The main recipe, with each sample a pair of consecutive windows (hop 16 positions), the loss on both and gradients
  through the buffer within the pair. Seeds 0–2.
- **P1b (only if P1a misses equivalence).** P1a plus a zero-initialised cross-attention block at the 1/16 level,
  through which positions 0–7 attend to the previous window's features at its last eight positions. At
  initialisation it equals P1a.
- **Test.** The all-16 tiling, with the state carried through the whole video.
- **Primary comparison.** Against the default tiling of `runs/vitg_s{0,1,2}`, rule of 12E. Equivalence means the
  state removes the need for overlapping windows, at half the passes. An effect below 0 means it does not suffice.
- **Mechanism check.** Per-position CC as in 4.1: the 0→8 deficit should fall below 0.01.
- **Gate.** Seeds 1 and 2 run only if seed 0's all-16 CC is at least 0.5836 (the default minus 0.003).
- **Cost.** About 80 minutes per seed for P1a (estimate).

### 13.3 P2. State with a 384² input (G1 + G2; only if 13.2 succeeds)
- The successful P1 model at 384², seeds 0–2. Against the 256² default an effect is expected (at least 0.003);
  against F3's 384² default tiling, equivalence. Cost per output frame with the A3 protocol.
- A second arm, if the cost has to fall further: a single-frame high-resolution detail branch beside the 256²
  encoder, compared with F3's result.
- **Cost.** About 3.3 hours per seed (estimate).

### 13.4 P3. Stride from the native frame rate (G3; test time)
- On the 15 fps videos of SumMe and MVVA, stride 1 against stride 2: the fine-tuned SumMe fold models, and the
  zero-shot `runs/vitg_s0` on MVVA. Rule of 12E per dataset, on those videos.

### 13.5 P4. More diverse training content (G4, G5)
- The decoder trained on DHF1K and AViMoS training videos together (mouse maps; AViMoS is CC-BY-4.0, so attribution is
  required), seeds 0–2. A second arm pre-trains on AViMoS and fine-tunes on DHF1K.
- Outcomes: DHF1K holdout (rule of 12E); zero-shot CC on the eight eye-tracking datasets with the 15I protocol, and the
  15M share (descriptive).
- [Hypothesis] Transfer improves most where the missing content is general (ETMD, Hollywood-2), less where it needs a
  task or a speaker (UCF Sports, Coutrot2).

### 13.6 P5. Does the 384² gain transfer? (G2; test time)
- The three F3 models zero-shot on the eight eye-tracking datasets with the 15I protocol, against the 256² models of
  the same seeds. Rule of 12E per dataset.

### 13.7 P6. A matched static control on DIEM
- Tests, for SURGE-D, El-Jiz et al.'s finding that temporal fusion hurts on DIEM (WACV 2026): the fine-tuned video
  model against a static model fine-tuned the same way (the E2 recipe), DIEM fold 1, three seeds, rule of 12E.
