# SURGE-D research log

Rules are written here **before** the runs they govern. Each result entry cites the run folder and log line.
Times are local (UTC+3 machine clock).

## 0. Setup (2026-11-05)

- Code: this repository only (`surge/`, `scripts/`). Nothing is imported from GRAFT / GRAFT-A (user instruction, 5 Nov).
- Data read from the original layouts: DHF1K `D:/DHF1K/{frames,annotations/annotation}`; UCF Sports
  `D:/Datasets/UCF_Sports/extracted`; SALICON `D:/Datasets/SALICON/extracted`.
- Splits (`configs/splits/`, fixed seed 20261105): DHF1K train 540 / **holdout 60** (from the 600 training videos,
  used for all model selection) / val 100 (601–700, the official validation set, used as the test set and scored
  only for the final comparisons). UCF Sports: 93 train / 10 holdout / 47 test.
- Hardware: 1× RTX A5000 24 GB. One GPU job at a time. `scripts/safety_monitor.ps1` runs before any GPU job; it
  only writes `SAFETY_STOP.txt`, which every SURGE job polls to save `last.pt` and exit. No job is ever force-killed.
- Metrics (`surge/metrics.py`): CC, NSS, SIM, KLD (MATLAB eps), AUC-Judd (AUC_Judd.m procedure), shuffled AUC
  (negatives = fixations pooled from 10 random frames of 10 other videos of the same split, per video).
  Aggregation: mean over frames within a video, then mean over videos.

## 1. Metric validation

Rule: the evaluator must reproduce independent scores of the same saliency maps. Reference maps: SalFoM's DHF1K-val
predictions on this machine (`D:/Research/baselines/salfom_val/maps`), whose per-frame scores were also computed with
the TMFI-Net evaluation code (`scores/*.json`, keys `tmfi_*`). Pass: |ΔCC|, |ΔSIM| ≤ 0.002, |ΔNSS|, |ΔKLD| ≤ 0.01 on
the mean over the 100 videos.

**Result (2026-11-05 13:40): PASS.** SalFoM's DHF1K-val maps (100 videos, 60,273 frames) scored by `scripts/evaluate.py`
(`runs/metric_check/`): CC 0.5639, NSS 3.3264, SIM 0.4130, KLD 1.2971, AUC-J 0.9243, sAUC 0.7406.
- vs an independent evaluator on this machine (GRAFT-A's, run earlier on the same maps; used only as a reference):
  CC, NSS, SIM, KLD identical per video (max |Δ| < 5e-5); AUC-J identical in mean (max per-video |Δ| 0.0007);
  sAUC +0.0023 in mean (random negative sampling differs; sAUC is protocol-dependent, so compare it only within SURGE).
- vs the TMFI-Net evaluation code (per-frame scores stored with the maps): CC 0.5642 (Δ −0.0003), NSS 3.3268
  (Δ −0.0004); its SIM/KLD/AUC-J use other normalizations (0.4347 / 1.2552 / 0.9271).
- The first version of AUC-J lacked AUC_Judd.m's tie-breaking jitter (−0.010 on these 8-bit PNG maps) and the first
  sAUC pooled all negatives (0.7752); both were fixed to the MATLAB procedures before any model was scored.
- SalFoM's val CC here, 0.564, is below its published *test* CC 0.569 (different split).

## 2. Phase 0, gate G1: backbone shoot-out (rules written 2026-11-05 12:58, before any G1 run)

- Runs: `configs/g1_{vjepa2_l,umt_l,vjepa21_l}.yaml`. Stage 1 (frozen backbone), identical decoder (6.99M params),
  10 epochs × 8 clips × 540 videos, batch 8, AdamW 3e-4 cosine, seed 0, 16 frames at stride 2.
  Input 256×256 (V-JEPA) / 224×224 (UMT-L: its fixed position table needs 224 and 16 frames).
- Metric: internal holdout CC at each run's best epoch (every 5th frame, σ = 0). NSS reported.
- Decision: the backbone with the highest holdout CC wins. V-JEPA 2.1-L (the proposal's choice) is kept if it is
  within 0.005 CC of the winner. Otherwise the winner replaces it for all later steps.
- Order (one job at a time): vjepa2_l, umt_l, vjepa21_l (V-JEPA 2.1 weights still downloading).

## 3. Phase 0, gate G2: surprise go/no-go (rules written before running it)

- Script `scripts/g2_surprise.py`, V-JEPA 2-L with its released predictor (V-JEPA 2.1's released predictor targets the
  ViT-G teacher space, not future frames, so it cannot give a forecasting error).
- Maps on the 60 holdout videos, windows of 16 frames (8 context + 8 target), at 4 fps and 15 fps: raw surprise,
  contextual surprise (true past − past from another video), and references: token atypicality, frame difference,
  the empirical center prior (mean training density).
- Scores: NSS and information gain (bits/fixation) over the center prior, on target frames.
- **Go** if some surprise variant beats *both* atypicality and frame difference on NSS *and* has positive information
  gain over the center prior. Otherwise SCM stays off (default) and the result is reported as a negative finding.

Note added 2026-11-05 13:10, before the real G2 run: a CPU test of `scripts/g2_surprise.py` on 2 holdout videos
(0028, 0031; stride 8) showed positive information gain for *every* map, including maps with negative NSS, so the
information-gain condition does not discriminate (the 50/50 mixture already beats the sharply peaked empirical prior).
The rule is unchanged; in practice the NSS conditions (beat both atypicality and frame difference) are binding.

## 4. Main program (rules written 2026-11-05 13:25, before G1 has finished)

Backbone = G1 winner (section 2). Seeds 0, 1, 2. All selection on the internal holdout; DHF1K val is scored once.

- **B1** (Stage 1, frozen): the G1 run is seed 0; seeds 1, 2 are added with the same config.
- **M** (Stage 2, LoRA r=16 on q/k/v/proj of the last 12 blocks, lr 1e-4; decoder lr 1e-5, initialized from B1 of the
  same seed; 10 epochs, batch 4).
- **LoRA rule:** M replaces B1 as the main model if the 3-seed mean holdout CC gain is ≥ +0.003 and M wins in at least
  2 of 3 seeds. Otherwise B1 is the main model and LoRA is reported as not helping.
- **Ablations** (Stage 1, seed 0, 10 epochs, vs B1 seed 0, descriptive): no center prior; no auxiliary heads;
  SALICON co-training (10,000 images per epoch); stride 4 (≈7.5 fps); rectangular 256×448 input (V-JEPA 2.1 only).
  An ablation is called a difference only if |ΔCC| ≥ 0.003 on the holdout.
- **Blur:** σ tuned on the holdout for the final model (`scripts/tune_sigma.py`), then frozen.
- **Final DHF1K val scoring, once:** main model seed 0 (single) and the 3-seed ensemble (mean of probabilities), all six
  metrics. Reference: SalFoM's val maps scored by the same evaluator (section 1); paired bootstrap over the 100 videos
  (10,000 resamples) for ΔCC and ΔNSS. A "beats SalFoM" claim needs the 95% CI of ΔCC above 0.
- **UCF Sports:** fine-tune the main DHF1K model (seed 0) on the 93 UCF training videos (Stage of the main model,
  10 epochs, lr ×0.3), select on the 10 holdout videos, score the 47 test videos once. Published references:
  ViNet-E CC 0.744, TMFI-Net 0.707, SalFoM 0.631.

Observation 2026-11-05 14:00 (G1 vjepa2_l, epochs 0–5): holdout CC 0.557, 0.573, **0.578**, 0.567, 0.565, 0.551 while
training CC rises 0.573 → 0.759. The decoder overfits the 540 training videos after about 2–3 epochs. G1 is unaffected
(best epoch per run). The main-program schedule/regularization will be fixed from the three G1 curves and written
here before any main run starts.

### G1 result (2026-11-05 17:30)

Holdout (60 videos, every 5th frame, σ = 0), best epoch of each 10-epoch Stage-1 run, seed 0:

| Backbone | Best epoch | CC | NSS | SIM | KLD | CC by epoch 0–9 |
|---|---|---|---|---|---|---|
| **V-JEPA 2 ViT-L** (HF, 256²) | 2 | **0.5782** | **3.314** | **0.4432** | **1.218** | .557 .573 .578 .567 .565 .551 .543 .539 .526 .522 |
| V-JEPA 2.1 ViT-L (distilled, 256²) | 1 | 0.5692 | 3.295 | 0.4377 | 1.245 | .549 .569 .568 .563 .553 .544 .539 .535 .527 .522 |
| UMT-L (K710→K400, 224²) | 1 | 0.5464 | 3.101 | 0.4157 | 1.323 | .527 .546 .542 .530 .516 .512 .504 .492 .484 .480 |

**Decision: V-JEPA 2 ViT-L** is the backbone for all later steps. V-JEPA 2.1 is 0.009 CC behind, outside the 0.005
margin, so the proposal's choice is not kept. Caveat: one seed per backbone; part of the gap may be seed noise.
All three runs peak at epoch 1–2 and then overfit (training CC keeps rising).

### Amendment to section 4, written 2026-11-05 17:32, before any main-program run

- Schedule for all main-program runs (B1, M, ablations): **3 epochs**, cosine, warm-up 0.25 epoch, best epoch on the
  holdout. Reason: every G1 curve peaks at epoch 1–2 of the 10-epoch schedule. Consequently the G1 run is *not* reused
  as B1 seed 0; B1 seeds 0–2 are new 3-epoch runs.
- The rectangular-input ablation is dropped (the HF V-JEPA 2 port assumes a square token grid).
- Queue: `scripts/queue_main.sh` (BB=vjepa2_l, 256², EP=3): B1 s0–2, M s0–2, then ablations no-prior, no-aux,
  stride 4, SALICON co-training. Starts after G2.

### G2 result (2026-11-05 18:05): NO-GO

`runs/g2/g2_results.json`; 60 holdout videos, first frame of each target tubelet, σ = 16 px, mean over videos.

| Map | NSS ~4 fps (stride 8) | NSS 15 fps (stride 2) | CC 4 fps | CC 15 fps |
|---|---|---|---|---|
| Surprise, raw (V-JEPA 2-L predictor error) | −0.190 | −0.502 | −0.043 | −0.099 |
| Surprise, contextual (true − other-video past) | 0.116 | −0.366 | 0.018 | −0.072 |
| Token atypicality | 0.454 | 0.650 | 0.081 | 0.123 |
| Frame difference | 0.810 | 0.803 | 0.149 | 0.148 |
| Empirical center prior | 1.464 | 1.479 | 0.289 | 0.292 |

No surprise variant beats atypicality or frame difference at either rate; raw surprise is *anti*-correlated with
fixations (consistent with the replication's negative correlation with the screen center). Information gain over the
prior is negative for every map on the full holdout (the 2-video CPU test was not representative). **SCM stays off.**

### B1 and LoRA (M) results, holdout CC at each run's best epoch (2026-11-05 20:40)

| Seed | B1 (frozen) | M (LoRA, from B1 of the same seed) | M − B1 |
|---|---|---|---|
| 0 | 0.5772 (ep 1) | 0.5743 (ep 0) | −0.0029 |
| 1 | 0.5801 (ep 1) | 0.5791 (ep 0) | −0.0010 |
| 2 | 0.5776 (ep 1) | (running) | |

**LoRA rule outcome (decided now, because it cannot change):** M must win in ≥ 2 of 3 seeds; it has lost 2 of 2, so
**B1 (frozen V-JEPA 2 ViT-L) is the main model**; seed 2 of M is still run and reported. The proposal's "LoRA belongs
in the main plan" is not supported here. Final evaluation is queued as `MAIN=main_b1 scripts/final_eval.sh` (waits for
the ablations).

### LoRA seed 2 and ablations (2026-11-05 23:20)

M seed 2: best holdout CC 0.5739 (B1 seed 2 0.5776, Δ -0.0037). **M loses in all 3 seeds; mean Δ -0.0025.**

Ablations (Stage 1, seed 0, 3 epochs; vs B1 seed 0 = CC 0.5772, NSS 3.329; a difference needs |ΔCC| ≥ 0.003):

| Variant | Best epoch | CC | ΔCC | NSS | SIM | KLD | Verdict |
|---|---|---|---|---|---|---|---|
| No center prior | 1 | 0.5772 | -0.0000 | 3.337 | 0.4556 | 1.241 | no difference |
| No auxiliary heads | 1 | 0.5757 | -0.0015 | 3.330 | 0.4537 | 1.242 | no difference |
| Stride 4 (≈7.5 fps, 2-s window) | 1 | 0.5803 | +0.0031 | 3.333 | 0.4524 | 1.217 | difference |
| + SALICON co-training (10k images/epoch) | 2 | 0.5801 | +0.0028 | 3.350 | 0.4471 | 1.211 | no difference |

Readings: the learned prior, the auxiliary heads and SALICON co-training add nothing measurable on top of frozen
V-JEPA 2 features (single seed). Stride 4 is +0.0031, exactly at the threshold, with half the inference cost per
frame: a candidate for a multi-seed follow-up, not a claim. The main model is unchanged (B1, stride 2).

## 5. Final results

### DHF1K val (official 100 videos, 60,273 frames), scored once (2026-11-06 00:15)

Main model = B1 (frozen V-JEPA 2 ViT-L + SURGE-D decoder, 16 frames at stride 2, 256²). Blur tuned on the holdout:
σ = 12 px (single), σ = 8 px (ensemble). Same evaluator for all rows. `runs/final/`.

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM (its released DHF1K weights, maps from this machine) | 0.5639 | 3.326 | 0.4130 | 1.297 | 0.9243 | 0.7406 |
| **SURGE-D, seed 0** | **0.5836** | **3.349** | **0.4492** | **1.222** | **0.9283** | **0.7636** |
| **SURGE-D, 3-seed ensemble** | **0.5910** | **3.418** | **0.4486** | **1.187** | **0.9302** | **0.7614** |

Paired bootstrap over videos (10,000 resamples), SURGE-D − SalFoM, 95% CI:

| | ΔCC | ΔNSS | ΔSIM | ΔKLD | ΔAUC-J | ΔsAUC |
|---|---|---|---|---|---|---|
| Seed 0 | +0.0197 [+0.0095, +0.0305] | +0.0225 [−0.0499, +0.0967] | +0.0362 [+0.0291, +0.0436] | −0.0747 [−0.1021, −0.0486] | +0.0040 [+0.0022, +0.0060] | +0.0230 [+0.0166, +0.0298] |
| Ensemble | +0.0271 [+0.0171, +0.0376] | +0.0910 [+0.0218, +0.1637] | +0.0356 [+0.0285, +0.0430] | −0.1101 [−0.1359, −0.0858] | +0.0060 [+0.0042, +0.0079] | +0.0208 [+0.0143, +0.0277] |

**Rule outcome: SURGE-D beats SalFoM on DHF1K val** (ΔCC CI above 0) as a single model and as an ensemble; the single
model ties SalFoM on NSS, the ensemble is better on all six metrics. Published val references (other evaluators):
TMFI-Net CC 0.554 / NSS 3.201, ViNet-E 0.549 / 3.134. Not a test-server result: DHF1K test labels are held out.

### UCF Sports test (47 videos, 2,929 frames), scored once (2026-11-06 00:27)

Fine-tuned from B1 seed 0 on 93 UCF training videos (Stage 1, 3 epochs, lr 1e-4), best epoch on the 10 holdout videos
(CC 0.7616, epoch 1), σ = 12 px tuned on that holdout. `runs/final/ucf_test*.json`.

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D zero-shot (DHF1K model, no UCF training) | 0.6077 | 3.128 | 0.5033 | 1.113 | 0.9223 | 0.7453 |
| **SURGE-D fine-tuned** | **0.7252** | **3.916** | **0.5799** | **0.773** | **0.9406** | 0.8089 |
| *Published (other evaluators):* ViNet-E | 0.744 | 4.156 | 0.587 | — | 0.941 | — |
| ViNet-A | 0.734 | 4.108 | 0.586 | — | 0.940 | — |
| STSANet | 0.721 | 3.927 | 0.560 | — | 0.936 | — |
| THTD-Net | 0.711 | 3.840 | 0.565 | — | 0.933 | — |
| TMFI-Net | 0.707 | 3.863 | 0.565 | — | 0.936 | — |
| SalFoM | 0.631 | 3.543 | 0.516 | — | 0.928 | — |

Reading: SURGE-D beats SalFoM (+0.094 CC) and TMFI-Net on UCF Sports, but **not** ViNet-E / ViNet-A (−0.019 / −0.009 CC),
so it is not state of the art on UCF. Caveat: published numbers come from other evaluation code; this evaluator's CC
matches TMFI-Net's code to 0.0003 (section 1), and the 47-video split is the standard one.

## 6. Summary of the implementation (2026-11-06)

1. **G1:** frozen V-JEPA 2 ViT-L beats V-JEPA 2.1 ViT-L (−0.009 CC) and UMT-L (−0.032) on the holdout; the proposal's
   V-JEPA 2.1 choice was replaced by the pre-registered rule.
2. **G2:** V-JEPA 2 predictor surprise is not a fixation cue (anti-correlated at 15 fps); SCM stays off.
3. **LoRA** loses to the frozen backbone in 3/3 seeds (mean −0.0025 CC); the frozen model is the main model.
4. **Ablations** (1 seed): center prior, aux heads, SALICON co-training: no measurable effect; stride 4: +0.0031 (borderline).
5. **DHF1K val:** SURGE-D beats SalFoM, single model ΔCC +0.020 [+0.010, +0.031], 3-seed ensemble +0.027
   [+0.017, +0.038]; the ensemble is better on all six metrics.
6. **UCF Sports:** CC 0.725, above SalFoM and TMFI-Net, below ViNet-E/A.
7. Safety: the monitor process (PID 21532) ran throughout and never tripped; no SAFETY_STOP.txt; no job was
   force-killed. **Its log is silent from 13:05 to 00:30**: my own `tail -F` watcher processes (left behind when
   their watches expired) held `runs/safety.log` open, so its writes failed. Temperatures after 13:05 are therefore
   not recorded (last summary 84 °C, max 86 °C). Checked afterwards: Windows System log 12:50–00:30 has 0 nvlddmkm /
   WHEA events and 0 unexpected-shutdown events. Fix: the monitor's log writes now retry, and the orphaned
   watchers were stopped.

## 7. STAViS audio-visual datasets: DIEM, Coutrot1, Coutrot2, ETMD, AVAD, SumMe (rules written 2026-11-06, before any run)

Requested by the user after section 6 ("what about the others?"). Not evaluated before because the proposal's plan named
only DHF1K, Hollywood-2 and UCF; that omission was not written down at the time. Other datasets on disk:
**Hollywood-2** has no gaze data here, so it cannot be scored. **MVVA** has no official split (and video 343 has no
maps), so any number would not be comparable to published ones; not run now. **AViMoS** is mouse tracking; not run now.

**Data.** Original STAViS-preprocessed layouts only (`D:/Datasets/<DB>/extracted/`): frames `img_NNNNN.jpg`
(256 × 340), maps `eyeMap_NNNNN.jpg` and fixations `fixMap_NNNNN.mat` (binary `eyeMap`, 480 × 640). The GRAFT `graft/`
layouts are not used. Folds: the official STAViS fold lists (`D:/Datasets/STAViS_folds`): 3 folds per dataset, DIEM a
single 64 / 17 split. Per video, frames 1 … min(listed count, frames that have image, map and fixation file) are used
(the lists sometimes say fewer frames than are on disk; one AVAD video has one frame without a map).

**Scoring.** Same evaluator as sections 1–5, all six metrics, every frame. Frames whose fixation map is empty (no gaze
recorded) are excluded from all metrics (`Video.skip_empty`; the DHF1K and UCF results are unaffected). sAUC negatives
come from other videos of the same test fold. Per-video mean, then mean over the fold's test videos; a dataset's
number is the **mean over its folds** (DIEM: its one split). Audio is never used: SURGE-D is a visual-only model.

**A. Zero-shot.** The main DHF1K model (B1 seed 0, `runs/main_b1_s0/best.pt`), σ = 12 px (its DHF1K-holdout σ), no
training on these datasets. Each fold's test list is scored, so every test video is seen once.

**B. Fine-tuned, per fold.** Initialized from B1 seed 0; Stage 1 (frozen backbone), 3 epochs, warmup 0.25, lr 1e-4,
center-prior lr 3e-4 (the UCF recipe of section 5). Training clips per epoch ≈ 1,500: clips_per_video =
ceil(1500 / number of training videos). Holdout: 10 % of the fold's training list (at least 1 video), drawn with
`random.Random("20261106-<db>-<fold>")` (`configs/splits/stavis_*_holdout.txt`, written before training). Best epoch
by holdout CC (every 2nd frame; DIEM every 5th); σ tuned on the same holdout (grid 0,4,8,12,16,24; CC). Each test
fold is scored **once**, after training of that fold has finished.

**Comparison.** Published numbers on these folds come from other evaluation code and cannot be bootstrapped against;
they are reported side by side, each re-checked in its source paper before it enters the table. A claim "beats X on
dataset D" needs a higher fold-mean CC than X's published CC. Visual-only and audio-visual published models are
listed separately: the audio-visual ones use information SURGE-D does not see.

**Published references, re-checked in the papers' own tables on 2026-11-06 (before any SURGE-D number on these data
was seen).** CC / NSS / AUC-J / SIM, fold means. Sources: DiffSal (CVPR 2024) Table 6, arXiv 2403.01226;
ViNet-S/A/E (arXiv 2502.00397) Tables II–III; DTFSal (arXiv 2504.10070) Table 1, which also reports a visual-only
variant DTFSal(V). Only the strongest rows are copied; the full tables are in the papers.

| Dataset | Best visual-only (published) | Best audio-visual (published) |
|---|---|---|
| DIEM | ViNet-E 0.701 / 2.840 / 0.913 / 0.566 | RAVF 0.685 / 2.78 / 0.911 / 0.559; DTFSal 0.682 / 2.98 / 0.921 / 0.562 |
| Coutrot1 | DTFSal(V) 0.623 / 3.21 / 0.904 / 0.505; ViNet-E 0.614 / 3.085 / 0.905 / 0.465 | DTFSal 0.661 / 3.31 / 0.911 / 0.523 |
| Coutrot2 | ViNet-A 0.862 / 6.8 / 0.961 / 0.638 | DTFSal 0.845 / 6.72 / 0.971 / 0.649 |
| ETMD | DTFSal(V) 0.653 / 3.57 / 0.939 / 0.499; ViNet-E 0.632 / 3.437 / 0.943 / 0.468 | DTFSal 0.667 / 3.82 / 0.957 / 0.531 |
| AVAD | DTFSal(V) 0.742 / 4.21 / 0.938 / 0.559; ViNet-E 0.729 / 4.167 / 0.938 / 0.547 | DTFSal 0.746 / 4.36 / 0.939 / 0.595 |
| SumMe | DTFSal(V) 0.561 / 2.91 / 0.911 / 0.441 (ViNet-S/A/E do not report SumMe) | DTFSal 0.573 / 3.20 / 0.927 / 0.449; DiffSal 0.572 / 3.14 / 0.921 / 0.447 |

Caveats found while checking: the papers disagree on baselines they both copy (STAViS on Coutrot2: 0.734 in DiffSal and
DTFSal, 0.652 in ViNet-S/A; ViNet on Coutrot2: 0.754 vs 0.724), so cross-paper numbers carry evaluator and protocol
noise of a few hundredths. ViNet-S/A state 3-fold averages for Coutrot1/2, AVAD, ETMD and the standard DIEM split;
DiffSal and DTFSal follow the STAViS protocol. None of these papers reports sAUC or KLD on these datasets.

## 8. DHF1K test-set submission (decided 2026-11-06, before generating any test map)

The user asked to submit to the DHF1K authors (results by email to the address in the DHF1K README; at most one
submission per week; test annotations for videos 701–1000 are held out). **Submitted model:** the 3-seed ensemble of
B1 (`runs/main_b1_s{0,1,2}/best.pt`), σ = 8 px (its DHF1K-holdout σ, `runs/final/sigma_ens3.json`): the configuration
that scored best on DHF1K val (section 5). No choice is made on test feedback.

**Format** (DHF1K README): `SURGE-D/<video>/<frame>.png`, 8-bit grayscale at native frame size, frame names as in
`D:/DHF1K/frames` (`0001.png` …). The frames on disk came from the same extraction for train/val and test; for videos
1–700 every frame count equals the annotation count (checked 2026-11-06, 0 mismatches). Script:
`scripts/predict_dhf1k_test.py`. **Format check before sending:** the same script on val videos 601–605, scored with
`scripts/evaluate.py --maps`, must reproduce the ensemble's val per-video CC to within PNG rounding (≤ 0.002).

## 9. Hollywood-2 with the Vig, Dorr & Cox gaze data (rules written 2026-11-06, before any map was built)

The standard Hollywood-2 saliency ground truth (Mathe & Sminchisescu, 16–19 viewers, shot-split by the DHF1K authors)
is still unavailable (site login fails to send verification email; DHF1K Drive link 404; the user has emailed the
authors). The user asked to use, until then, the open **Vig, Dorr & Cox (ECCV 2012)** gaze recordings
(figshare 93951, CC BY; md5 of every file checked against figshare): 4 viewers who watched all 1,707 clips plus a
pooled fifth "subject" (6 people, non-overlapping parts), 1000 Hz, recorded during the action-recognition task.
**These numbers are not comparable to published Hollywood-2 results** (different and fewer viewers, whole clips not
shots) and are reported only as "Hollywood-2 (Vig gaze)".

**Frames.** Every decoded frame of the 823 train + 884 test action clips (`scripts/prep_hollywood_frames.py`; 487,556
frames; OpenCV decoding). Gaze files state the same resolution as the clips.

**Maps** (`scripts/prep_hollywood_gaze.py`). Frame f (1-based) covers [(f−1)/fps, f/fps) s. Per viewer and frame: gaze
samples with confidence 1.0 that are not inside a saccade (onset–offset from the `.sacc` file); if any remain, the
viewer contributes one fixation point = the median x, y of those samples (rounded, clipped to the frame). Fixation map:
binary, native resolution, ≤ 5 points. Density map: that binary map blurred with a Gaussian of σ = 20 px (the
ACLNet / DHF1K recipe), stored as 8-bit PNG scaled to max 255. Frames with no point are excluded from scoring
(`skip_empty`), as for the STAViS data.

**Protocol.** Split = the official one (823 train, 884 test). Holdout: 10 % of the training clips (82), drawn with
`random.Random("20261106-hollywood-vig")`. A. Zero-shot: B1 seed 0, σ = 12. B. Fine-tuned: the section 7 recipe
(Stage 1, 3 epochs, warmup 0.25, lr 1e-4, clips_per_video = ceil(1500 / training clips), best epoch by holdout CC on
every 5th frame, σ tuned on the holdout). The 884 test clips are scored once per model. Runs after the STAViS queue and
the DHF1K test maps (one GPU job at a time).

Maps built 2026-11-06 (`scripts/prep_hollywood_gaze.py`): 1,707 clips, 487,556 frames; 99.8 % of frames have at least
one point, mean 4.53 points per frame. Gaze md5s all match figshare. A visual check of overlays on three clips shows
points on faces and the acting person, as expected for this task.

### Section 7 results (queue finished 2026-11-06 13:45; `runs/stavis/`, summary in `runs/stavis/summary.json`)

Fold means (DIEM: its one split). Every test fold scored once. Fine-tuned σ per fold (holdout): DIEM 4; Coutrot1
12/8/12; Coutrot2 12/12/12; ETMD 12/8/16; AVAD 12/12/8; SumMe 12/16/8. As on DHF1K, holdout CC peaked after epoch
1–2 on most folds (best epoch kept).

| Dataset | Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|---|
| DIEM | SURGE-D zero-shot | 0.675 | 2.730 | 0.559 | 0.815 | 0.916 | 0.713 |
| | **SURGE-D fine-tuned** | **0.706** | 2.790 | 0.568 | 0.708 | 0.920 | 0.728 |
| | best visual (ViNet-E) / best AV (RAVF; DTFSal) | 0.701 / 0.685; 0.682 | 2.840 / 2.78; 2.98 | 0.566 / 0.559; 0.562 | — | 0.913 / 0.911; 0.921 | — |
| Coutrot1 | SURGE-D zero-shot | 0.600 | 2.912 | 0.480 | 1.136 | 0.899 | 0.742 |
| | **SURGE-D fine-tuned** | **0.632** | 3.164 | 0.489 | 0.996 | 0.907 | 0.762 |
| | best visual (DTFSal(V)) / best AV (DTFSal) | 0.623 / 0.661 | 3.21 / 3.31 | 0.505 / 0.523 | — | 0.904 / 0.911 | — |
| Coutrot2 | SURGE-D zero-shot | 0.645 | 4.796 | 0.428 | 1.069 | 0.958 | 0.723 |
| | SURGE-D fine-tuned | 0.816 | 6.564 | 0.599 | 0.694 | 0.966 | 0.758 |
| | best visual (ViNet-A) / best AV (DTFSal) | 0.862 / 0.845 | 6.8 / 6.72 | 0.638 / 0.649 | — | 0.961 / 0.971 | — |
| ETMD | SURGE-D zero-shot | 0.588 | 3.169 | 0.468 | 1.173 | 0.939 | 0.718 |
| | **SURGE-D fine-tuned** | **0.662** | 3.608 | 0.502 | 0.965 | 0.948 | 0.759 |
| | best visual (DTFSal(V)) / best AV (DTFSal) | 0.653 / 0.667 | 3.57 / 3.82 | 0.499 / 0.531 | — | 0.939 / 0.957 | — |
| AVAD | SURGE-D zero-shot | 0.692 | 3.919 | 0.541 | 0.985 | 0.931 | 0.760 |
| | SURGE-D fine-tuned | 0.718 | 4.105 | 0.547 | 0.856 | 0.938 | 0.780 |
| | best visual (DTFSal(V)) / best AV (DTFSal) | 0.742 / 0.746 | 4.21 / 4.36 | 0.559 / 0.595 | — | 0.938 / 0.939 | — |
| SumMe | SURGE-D zero-shot | 0.551 | 2.931 | 0.442 | 1.297 | 0.918 | 0.729 |
| | **SURGE-D fine-tuned** | **0.581** | 3.151 | 0.443 | 1.174 | 0.926 | 0.764 |
| | best visual (DTFSal(V)) / best AV (DTFSal) | 0.561 / 0.573 | 2.91 / 3.20 | 0.441 / 0.449 | — | 0.911 / 0.927 | — |

**Rule outcome (fold-mean CC above the published CC):**
- Versus the best published **visual-only** model: higher on **DIEM (+0.005), Coutrot1 (+0.009), ETMD (+0.009),
  SumMe (+0.020)**; lower on **Coutrot2 (−0.046)** and **AVAD (−0.024)**.
- Versus the best published **audio-visual** model (which also uses the soundtrack): higher on **DIEM (+0.021)** and
  **SumMe (+0.008)**; lower on Coutrot1, Coutrot2, ETMD, AVAD.
- Margins under ~0.01 (DIEM, Coutrot1, ETMD vs visual; SumMe vs AV) are inside the cross-paper disagreement noted
  above, so they are ties at best, not clear wins. The clear results: SumMe vs visual-only (+0.020) is a win; Coutrot2
  and AVAD are clear losses. Other metrics are mixed: SURGE-D's NSS and SIM are below the best published on most
  datasets even where its CC is higher.
- Coutrot2 (4-person meetings, gaze follows the speaker) is where the gap is largest and where fine-tuning helped most
  (+0.171 CC over zero-shot); a visual-only model cannot know who is talking.

### Section 8 outcome (2026-11-06)

Format check passed: maps written by `predict_dhf1k_test.py` for val videos 601–605 reproduce the ensemble's val
per-video CC within 0.0001 (`runs/dhf1k_submission/format_check.*`). Test maps: 300 folders 0701–1000, 176,192 PNGs
(8-bit, 640 × 360), frame names identical to `D:/DHF1K/frames` for every video. Archive `D:/DHF1K_submission/SURGE-D.zip`,
3,472,705,435 bytes, zip test passed, SHA-256 68794038688eb9e4efdff01e68d19229755d676e5cc1853c36d6f1fa8950e20b.
Sending is done by the user (upload + email to the DHF1K authors).

### Section 9 results (queue finished 2026-11-06 17:19; `runs/hollywood_vig/`, `runs/hollywood_vig_ft/`)

Hollywood-2 (Vig gaze) test: 884 clips, 257,733 scored frames. **Not comparable to published Hollywood-2 results.**

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D zero-shot (B1 seed 0, σ = 12) | 0.632 | 3.595 | 0.503 | 1.061 | 0.951 | 0.780 |
| **SURGE-D fine-tuned** (best epoch 2, σ = 12) | **0.680** | **3.898** | **0.518** | **0.925** | **0.956** | **0.817** |

Fine-tuning: holdout CC by epoch 0.706 / 0.704 / 0.708 on every 5th frame, so best.pt = epoch 2 (the config's
UCF paths are unused; the `hollywood_vig` loader reads its own split and holdout files); σ grid on the holdout 0–24 → 12 (CC 0.718). Fine-tuning improves every metric (+0.048 CC). Training CC
kept rising (0.683 → 0.730) while holdout CC stayed flat, the same early-saturation pattern as on DHF1K and UCF.

## 10. Improvements: per-dataset seed ensembles and the V-JEPA 2 ViT-g backbone (rules written 2026-11-07, before any run)

The user asked for both improvements. One GPU job at a time; order: 10B (ViT-g, DHF1K holdout gate) first, then 10A.

**10A. Seed ensembles on every fine-tuned dataset (ViT-L, independent of the 10B outcome).** For UCF Sports, each of
the 16 STAViS folds and Hollywood-2 (Vig gaze): two more fine-tunes, member k ∈ {1, 2} initialised from
`runs/main_b1_s{k}/best.pt` with `train.seed=k`, everything else identical to the seed-0 run of that dataset (recipe,
holdout, clips per video, eval frame step, best epoch by holdout CC). Ensemble = mean of the three members' raw
predictions (as for DHF1K); σ tuned for the ensemble on the same holdout and grid (UCF frame step 1, STAViS 2 (DIEM 5),
Hollywood 5). Each test set is scored once with the ensemble. Reported next to the seed-0 single model; for STAViS
the dataset number is still the fold mean. No other change is made after seeing test numbers.

**10B. V-JEPA 2 ViT-g/16** (`facebook/vjepa2-vitg-fpc64-256`, MIT, 40 layers, width 1408, frozen; taps 10/20/30/40,
the same relative depths as ViT-L's 6/12/18/24). Run G-1 = B1 with `model.backbone=vjepa2_g`, seed 0, every other
setting unchanged (if it does not fit in 24 GB at batch 8, batch 4 with the learning rates unchanged, noted). Gate:
**adopt ViT-g only if G-1's best holdout CC (DHF1K holdout 60, every 5th frame) ≥ 0.5833** = the ViT-L 3-seed mean
(0.5772, 0.5801, 0.5776 → 0.5783) + 0.005, i.e. above the ViT-L seed spread (0.003). If it fails: report the holdout
number only; DHF1K val is not touched. If it passes: train seeds 1 and 2, tune σ on the holdout (single and
ensemble), score DHF1K val once each, and fine-tune on UCF (seed 0, the section-5 recipe) and score UCF test once.
Whether ViT-g then replaces ViT-L on the STAViS and Hollywood data is left to the user (it costs ~4× the GPU time).

### Section 10B gate (2026-11-07 08:25; `runs/vitg_s0`)

G-1 (ViT-g, seed 0) fit at batch 8 (10.0 GB, ~10 min per epoch + 6 min holdout; backbone 1,034.6M, decoder 7.39M).
Holdout CC by epoch 0.5658 / **0.5866** / 0.5844 → best 0.5866 (epoch 1) ≥ 0.5833: **gate passed** (+0.008 over the
ViT-L 3-seed mean, above every ViT-L seed). Seeds 1 and 2, σ tuning, DHF1K val and UCF follow as written.

### Section 10B results: DHF1K val (2026-11-07 11:35; `runs/vitg/`), scored once each

Holdout bests: seed 0 0.5866, seed 1 0.5867, seed 2 0.5829 (ViT-L: 0.5772, 0.5801, 0.5776). σ tuned on the holdout:
8 (single) and 8 (ensemble).

| DHF1K val (100 videos) | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM (released weights, same evaluator) | 0.5639 | 3.326 | 0.4130 | 1.297 | 0.9243 | 0.7406 |
| SURGE-D ViT-L single (σ 12) | 0.5836 | 3.349 | 0.4492 | 1.222 | 0.9283 | 0.7636 |
| SURGE-D ViT-L 3-seed ensemble (σ 8) | 0.5910 | 3.418 | 0.4486 | 1.187 | 0.9302 | 0.7614 |
| **SURGE-D ViT-g single** (σ 8) | **0.6003** | **3.457** | **0.4531** | **1.170** | **0.9308** | **0.7709** |
| **SURGE-D ViT-g 3-seed ensemble** (σ 8) | **0.6029** | **3.486** | **0.4574** | **1.160** | **0.9314** | 0.7680 |

Paired bootstrap over videos (`runs/vitg/cmp_*.txt`), ΔCC [95 % CI]:
- ViT-g single vs ViT-L single: +0.0167 [+0.0116, +0.0222]; better on all six metrics, every interval excludes 0
  (SIM barely: +0.0039 [+0.0000, +0.0077]).
- ViT-g ensemble vs ViT-L ensemble: +0.0118 [+0.0077, +0.0163]; better on all six, every interval excludes 0.
- vs SalFoM: single +0.0364 [+0.0259, +0.0475], ensemble +0.0390 [+0.0286, +0.0498]; better on all six metrics.

ViT-g is the stronger backbone on DHF1K. The 3-seed ensemble adds less on ViT-g (+0.003 CC) than on ViT-L (+0.007).

### Section 10 results: UCF Sports test (47 videos), scored once each (2026-11-07 12:10)

| UCF Sports test | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L single (section 5; σ from its holdout) | 0.7252 | 3.916 | 0.5799 | 0.773 | 0.9406 | 0.8089 |
| SURGE-D ViT-g single (10B; holdout best 0.7660, epoch 1; σ 8) | 0.7305 | 4.061 | 0.5831 | 0.747 | 0.9442 | 0.8145 |
| SURGE-D ViT-L 3-seed ensemble (10A; members' holdout 0.7616 / 0.7579 / 0.7602; σ 8) | **0.7331** | 4.028 | **0.5833** | 0.749 | 0.9430 | **0.8153** |

Paired bootstrap ΔCC vs ViT-L single: ViT-g +0.0053 [−0.0055, +0.0178] (not significant; better on 23/47 videos);
ViT-L ensemble +0.0080 [+0.0039, +0.0123]. Published: ViNet-E 0.744, ViNet-A 0.734, TMFI-Net 0.707, SalFoM 0.631.
The ensemble ties ViNet-A (−0.001) and stays below ViNet-E (−0.011): still not state of the art on UCF.

### Section 10A results: STAViS seed ensembles (2026-11-07 19:21; `runs/ens/stavis_*.json`, `runs/ens/stavis_summary.json`)

ViT-L, 3 fine-tuned members per fold, σ tuned per fold on the holdout; dataset number = fold mean
(`scripts/summarize_stavis_ens.py`). Single = the seed-0 model of section 7.

| Dataset | Single CC | **Ensemble** CC | Δ | Ens. NSS | Ens. SIM | Ens. KLD | Ens. AUC-J | Ens. sAUC | best visual / best AV CC | Ens. − visual / − AV |
|---|---|---|---|---|---|---|---|---|---|---|
| DIEM | 0.706 | **0.714** | +0.009 | 2.832 | 0.575 | 0.688 | 0.921 | 0.731 | 0.701 / 0.685 | **+0.013 / +0.029** |
| Coutrot1 | 0.632 | **0.641** | +0.009 | 3.183 | 0.491 | 0.977 | 0.909 | 0.769 | 0.623 / 0.661 | **+0.018** / −0.020 |
| Coutrot2 | 0.816 | 0.807 | −0.009 | 6.606 | 0.590 | 0.709 | 0.966 | 0.758 | 0.862 / 0.845 | −0.055 / −0.038 |
| ETMD | 0.662 | **0.670** | +0.008 | 3.649 | 0.505 | 0.940 | 0.950 | 0.763 | 0.653 / 0.667 | **+0.017** / +0.003 |
| AVAD | 0.718 | 0.729 | +0.011 | 4.167 | 0.554 | 0.821 | 0.941 | 0.782 | 0.742 / 0.746 | −0.013 / −0.017 |
| SumMe | 0.581 | **0.593** | +0.012 | 3.232 | 0.448 | 1.138 | 0.928 | 0.769 | 0.561 / 0.573 | **+0.032 / +0.020** |

- The ensemble raises fold-mean CC on 5/6 datasets (+0.008 to +0.012) and lowers it on Coutrot2 (−0.009); NSS, KLD
  and sAUC improve on all six.
- Versus the best published **visual-only** model (section 7 rule, margins < 0.01 = ties): the ensemble is now clearly
  higher on **DIEM, Coutrot1, ETMD and SumMe** (+0.013 to +0.032; three of these were ties with the single model) and
  still lower on Coutrot2 and AVAD.
- Versus the best **audio-visual** model: higher on DIEM (+0.029) and SumMe (+0.020), tie on ETMD (+0.003), lower on
  Coutrot1, Coutrot2, AVAD.
- NSS stays below the best published on DIEM, Coutrot1 and AVAD; above it on ETMD (vs visual) and SumMe.

### Section 10A results: Hollywood-2 (Vig gaze) ensemble (queue finished 2026-11-07 21:57; `runs/ens/hollywood_*`)

Members' holdout CC 0.7076 (seed 0) / 0.7110 / 0.7067; ensemble σ 12 (tuned on the holdout). 884 test clips, scored
once. **Not comparable to published Hollywood-2 results.**

| Hollywood-2 (Vig gaze) test | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D fine-tuned, single (seed 0) | 0.6798 | **3.898** | **0.5178** | 0.925 | 0.9564 | 0.8167 |
| SURGE-D fine-tuned, 3-seed ensemble | **0.6841** | 3.875 | 0.5073 | **0.918** | **0.9574** | **0.8192** |

Paired bootstrap (ensemble − single): CC +0.0043 [+0.0036, +0.0050], KLD −0.008, AUC-J +0.001, sAUC +0.003 (all
intervals exclude 0); NSS −0.024 and SIM −0.011 (also significant, worse). Mixed: the gain is smallest here of all
datasets, and the averaged maps lose some peak sharpness (NSS, SIM) with only ≤ 5 viewers per frame.

### Section 10 summary (2026-11-07)

- **ViT-g (10B)** passed its gate and is the better backbone on DHF1K: val CC 0.600 single / 0.603 ensemble vs
  ViT-L 0.584 / 0.591, better on all six metrics (paired intervals exclude 0); vs SalFoM +0.036 / +0.039 CC. On UCF
  its gain (+0.005 CC) is not significant. ViT-g is not yet run on STAViS or Hollywood (user's decision).
- **Seed ensembles (10A, ViT-L)**: UCF 0.7331 (+0.008, ties ViNet-A, below ViNet-E); STAViS CC up on 5/6 datasets,
  clear visual-only wins on DIEM, Coutrot1, ETMD, SumMe, still behind on Coutrot2 and AVAD; Hollywood (Vig) +0.004 CC
  with NSS/SIM slightly lower.
- The DHF1K test maps in `D:/DHF1K_submission/SURGE-D.zip` are still the ViT-L ensemble.

## 11. Next round: train-on-600, DHF1K test maps with ViT-g, 32-frame clips, audio, ViT-g on the other datasets (rules written 2026-11-08, before any run)

The user asked for everything that does not depend on the DHF1K reply or the IMAR reply. One GPU job at a time, in
this order: 11A, 11B, 11C, 11D, 11E (`scripts/queue_next.sh`).

**11A. Train on all 600 DHF1K training videos (ViT-g).** Split `dhf1k_train600.txt` = train540 ∪ holdout60. Seeds
0–2, B1 recipe with `model.backbone=vjepa2_g`, 3-epoch schedule; no data is left for selection, so the checkpoint is
fixed in advance: **epoch_1.pt** (the best epoch of all three ViT-g seeds in section 10B). σ = 8 (section 10B's
tuned value, single and ensemble). Holdout scores printed during training are on training data and are not used.
Scored once on DHF1K val: seed 0 and the 3-seed ensemble, paired-bootstrap vs section 10B. **The 600 models replace
the 540 models for the DHF1K test maps only if the ensemble's ΔCC interval vs the 10B ensemble is above 0 and it is
not worse on more than 2 of 6 metrics;** otherwise the 10B ensemble is used.

**11B. DHF1K test maps with ViT-g.** The 11A winner (3-seed ensemble, σ 8). Format check first, as in section 8
(val 601–605 through `predict_dhf1k_test.py`, |ΔCC| ≤ 0.002 vs that ensemble's val per-video CC), then 701–1000,
zipped as `D:/DHF1K_submission/SURGE-D-vitg.zip` (root folder `SURGE-D/`). Sending stays with the user.

**11C. 32-frame clips (ViT-g, train540).** Run T32-1 = 10B G-1 with `data.T=32` (stride 2, so 64 frames ≈ 2.6 s;
the output still covers the central half). Batch 8, else 4 on out-of-memory. Gate: best holdout CC ≥ **0.5904** =
ViT-g T=16 3-seed mean (0.5866, 0.5867, 0.5829 → 0.5854) + 0.005. If it fails: holdout only, val untouched. If it
passes: seeds 1–2, σ tuned on the holdout, val scored once (single and ensemble), compared with 10B. It does not
change 11B.

**11D. Audio on the STAViS datasets (ViT-L).**
- Audio: the STAViS release's own WAVs (`<DB>/extracted/video_audio/<name>/<video>/<video>.wav`), mixed to mono,
  resampled to 16 kHz. Videos without a WAV get zero audio features.
- Encoder, frozen: AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, BSD-3-Clause), run on consecutive 10.24 s chunks;
  patch tokens averaged over frequency give a 768-d feature every 10 ms column. Feature of video frame k (1-based,
  time (k − 0.5)/fps, fps from the fold list) = mean of the columns whose centre lies within ±0.5 s.
  Cached once per video.
- Fusion (new decoder module, on P8, P16a, P16b): per tubelet, the audio feature (mean of its two frames) → LayerNorm →
  linear to 256 → (i) FiLM (γ, β) and (ii) an audio–visual correlation map σ(⟨Wq·F(x), Wk·a⟩/√d) gating a residual
  1×1 conv. The FiLM and residual output layers are zero-initialised, so at the start the model is exactly the visual
  model.
- Training: per fold, the section 7 seed-0 recipe (init `runs/main_b1_s0`, Stage 1, 3 epochs, same holdout, clips per
  video, frame step); σ tuned on the holdout; each test fold scored once. Baseline = the section 7 single visual model
  (same init, seed, recipe).
- Control: the same audio model scored with audio rotated among the fold's test videos (video i gets video i+1's
  sound). If the model uses sound, matched audio must beat rotated audio.
- Claim rule: **"audio helps on dataset X" only if its fold-mean ΔCC vs the visual model ≥ +0.010 and matched −
  rotated ≥ +0.005.** Coutrot2 and AVAD, the two clear losses, are where it is expected to matter.

**11E. ViT-g on UCF, STAViS and Hollywood-2 (Vig gaze).** Same recipes as section 10A, with members k = 0, 1, 2
initialised from `runs/vitg_s{k}/best.pt` (`model.backbone=vjepa2_g`). σ tuned on each holdout; scored once each:
the seed-0 single and the 3-seed ensemble. Compared with the ViT-L single (section 7/9/5) and ViT-L ensemble
(section 10A).

Audio features built 2026-11-08 (`scripts/prep_stavis_audio.py`, CPU, `runs/prep_stavis_audio.log`): DIEM 78/84,
Coutrot1 60/60, Coutrot2 15/15, ETMD 12/12, AVAD 45/45, SumMe 23/25 videos with audio (the counts the STAViS release
ships); WAV durations match frame count / fps.

### 11A results (2026-11-08)

Trained 07:06–09:45 (three seeds, epoch_1.pt each), DHF1K val, σ 8:

| Model | CC | NSS | SIM | KLD | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| ViT-g train600, seed 0 | 0.6002 | 3.495 | 0.4594 | 1.173 | 0.9307 | 0.7647 |
| ViT-g train600, 3-seed ensemble | 0.6044 | 3.500 | 0.4558 | 1.156 | 0.9315 | 0.7697 |
| ViT-g train540 ensemble (10B) | 0.6029 | 3.486 | 0.4574 | 1.160 | 0.9314 | 0.7680 |

- Ensemble 600 vs 540 (paired bootstrap): ΔCC +0.0015 [−0.0006, +0.0037]; NSS +0.014 [−0.000, +0.028]; SIM −0.0017
  [−0.0032, −0.0001]; KLD −0.004 (n.s.); AUC-J +0.0001 (n.s.); sAUC +0.0017 [−0.0000, +0.0033].
- Single 600 vs 540 seed 0: CC −0.0001 (n.s.); NSS +0.039 and SIM +0.006 significant; sAUC −0.006 significant.
- 600 ensemble vs SalFoM: CC +0.041 [+0.030, +0.052], all six metrics significantly better.
- **Decision (rule above): the ΔCC interval includes 0, so the 540 ensemble is kept for the test maps
  (`runs/vitg600/use600.txt` = 0).** The 60 extra videos (+11 %) change the val scores within noise.

### 11B format check (2026-11-08)

Val 601–605 through `predict_dhf1k_test.py` (10B ensemble, σ 8): per-video |ΔCC| ≤ 8e-5 against the evaluation
pipeline (rule: ≤ 0.002), so the format check passed. Test maps (10B ViT-g 3-seed ensemble, σ 8): 300 folders
0701–1000, 176,192 PNGs (the same count as the section 8 archive). Archive `D:/DHF1K_submission/SURGE-D-vitg.zip`
(root folder `SURGE-D/`), 2,661,220,447 bytes (smaller than the ViT-L archive: smoother maps compress better), zip test
passed, SHA-256 0fe4e2e75853ee888251462f8d216918466bf7ed971b19dad50b39b86155fc4f. Finished 13:04. Sending stays with
the user; this archive supersedes `SURGE-D.zip` (expected val CC 0.6029 vs 0.5911).

### 11C results (2026-11-08)

T32-1 (ViT-g, seed 0, T = 32, batch 8, 16 GB, about 30 min per epoch): holdout CC 0.5628 / 0.5856 / **0.5902** at
epochs 0 / 1 / 2 (NSS 3.407, SIM 0.452, KLD 1.181 at epoch 2). **Gate (≥ 0.5904) failed by 0.0002, so seeds 1–2 are
not run and val is not touched.** For context only (not a test): the T = 16 seed-0 run reached 0.5866 and peaked at
epoch 1, while T = 32 was still rising at its last epoch, so a longer schedule might change this. That would be a new
pre-registered run, not a re-reading of this one.
