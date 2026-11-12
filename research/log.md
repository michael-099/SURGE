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

### 11D results (2026-11-08)

16 audio fine-tunes, 14:33–19:50 (`runs/stavis_av_*`, scores in `runs/av/`, fold means in `runs/av/summary.json`).
Fold-mean CC; "visual" = the section 7 single model (same init, seed and recipe):

| Dataset | Visual | Audio (matched) | Audio (rotated) | Audio − visual | Matched − rotated | Audio helps? |
|---|---|---|---|---|---|---|
| DIEM | 0.706 | 0.706 | 0.706 | +0.0008 | +0.0002 | no |
| Coutrot1 | 0.632 | 0.633 | 0.630 | +0.0010 | +0.0031 | no |
| Coutrot2 | 0.816 | 0.816 | 0.797 | −0.0003 | **+0.0183** | no |
| ETMD | 0.662 | 0.661 | 0.661 | −0.0014 | −0.0001 | no |
| AVAD | 0.718 | 0.718 | 0.717 | +0.0002 | +0.0013 | no |
| SumMe | 0.581 | 0.579 | 0.580 | −0.0021 | −0.0011 | no |

**The claim rule holds on no dataset: this audio fusion does not improve the visual model.** Other metrics move by
≤ 0.1 NSS, ≤ 0.004 SIM/sAUC. One signal: on Coutrot2 (one-to-one conversations, where the speaker is the target) the
model does use the sound, since the wrong video's sound costs 0.018 CC, but the right sound only matches the visual model,
which already finds faces and speakers from the image. Coutrot2 has 15 videos (5 per test fold), so this is suggestive
only. The gap to the audio-visual methods on Coutrot2 and AVAD (section 7) is therefore not closed by a frozen AST
encoder with zero-initialised FiLM/correlation fusion and 3 epochs of fine-tuning on these small sets.

### 11E results: UCF Sports test (47 videos), 2026-11-08 20:39

ViT-g members fine-tuned from `runs/vitg_s{0,1,2}/best.pt` (section 5 recipe); σ 8 for single and ensemble (holdout).

| UCF Sports test | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| ViT-g single (seed 0) | 0.7305 | 4.061 | 0.5830 | 0.747 | 0.9442 | 0.8145 |
| ViT-g 3-seed ensemble | **0.7350** | **4.085** | **0.5857** | **0.736** | **0.9449** | **0.8173** |
| ViT-L 3-seed ensemble (10A) | 0.7331 | 4.028 | 0.5833 | 0.749 | 0.9430 | 0.8153 |

- The seed-0 run reproduces section 10B's ViT-g UCF result exactly (0.7305): the pipeline is deterministic.
- ViT-g ensemble vs ViT-L ensemble: ΔCC +0.0019 [−0.0071, +0.0125] (n.s.); only NSS significant (+0.057).
  vs ViT-L single: ΔCC +0.0098 [−0.0001, +0.0211] (borderline).
- Published: ViNet-E 0.744, ViNet-A 0.734. The ViT-g ensemble ties ViNet-A (+0.001) and stays below ViNet-E
  (−0.009). Still not state of the art on UCF; the ranking is unchanged.

### 11E results: STAViS datasets (queue 2026-11-08 20:39 → 2026-11-09 14:48; `runs/g_other/`)

48 ViT-g fine-tunes (16 folds × seeds 0–2, section 7 recipe, init `runs/vitg_s{k}/best.pt`), σ tuned per fold on the
holdout (ensemble: 8–16). Fold-mean CC (`scripts/summarize_stavis_vitg.py`, `runs/g_other/stavis_summary.json`);
paired bootstrap over each dataset's pooled test videos (each video is in exactly one test fold;
`runs/g_other/cmp_<dataset>_*.txt`).

| Dataset | ViT-L single | ViT-L ens. | ViT-g single | **ViT-g ens.** | ens. NSS | ens. SIM | ens. KLD | ens. AUC-J | ens. sAUC | g ens. − L ens. CC [95 % CI] | vs best visual / best AV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DIEM (17) | 0.706 | 0.714 | 0.714 | **0.729** | 2.878 | 0.584 | 0.664 | 0.923 | 0.735 | **+0.014 [+0.003, +0.027]** | **+0.027 / +0.043** |
| Coutrot1 (60) | 0.632 | 0.641 | 0.648 | **0.657** | 3.302 | 0.504 | 0.947 | 0.911 | 0.774 | **+0.017 [+0.005, +0.031]** | **+0.034** / −0.004 |
| Coutrot2 (15) | 0.816 | 0.807 | **0.820** | 0.819 | 6.673 | 0.601 | 0.683 | 0.966 | 0.759 | +0.012 [−0.007, +0.032] | −0.043 / −0.026 |
| ETMD (12) | 0.662 | 0.670 | 0.672 | **0.676** | 3.687 | 0.510 | 0.930 | 0.950 | 0.765 | **+0.006 [+0.002, +0.010]** | **+0.023** / +0.009 |
| AVAD (45) | 0.718 | 0.729 | 0.731 | **0.740** | 4.306 | 0.565 | 0.795 | 0.942 | 0.782 | **+0.011 [+0.005, +0.019]** | −0.002 / −0.006 |
| SumMe (23) | 0.581 | **0.593** | 0.586 | 0.590 | 3.261 | 0.448 | 1.148 | 0.927 | 0.763 | −0.003 [−0.010, +0.004] | **+0.029 / +0.017** |

- **ViT-g ensemble vs ViT-L ensemble:** higher CC on 5/6 datasets, significant on DIEM, Coutrot1, ETMD and AVAD
  (with NSS, SIM, KLD also significantly better on Coutrot1, AVAD and mostly DIEM/ETMD); n.s. on Coutrot2; SumMe
  is a tie in CC and sAUC is worse (−0.006, significant).
- **ViT-g single vs ViT-L single:** CC higher on all 6; significant on Coutrot1 (+0.016), ETMD (+0.010), AVAD
  (+0.013). On 5/6 datasets the single ViT-g model matches or beats the ViT-L ensemble.
- **Versus published (section 7 rule, margins < 0.01 = ties):**
  - best visual-only: clearly higher on DIEM, Coutrot1, ETMD, SumMe (+0.023 to +0.034); **AVAD is now a tie
    (−0.002, was −0.013)**; Coutrot2 still a clear loss (−0.043).
  - best audio-visual: clearly higher on DIEM (+0.043) and SumMe (+0.017); ties on Coutrot1 (−0.004), ETMD
    (+0.009) and AVAD (−0.006); Coutrot2 still a clear loss (−0.026).
- The one dataset a visual model cannot fix is Coutrot2 (who is talking); 11D showed that the audio module as built
  does not fix it either.

### 11E results: Hollywood-2 (Vig gaze), queue finished 2026-11-09 21:08

ViT-g members fine-tuned from `runs/vitg_s{k}/best.pt` (section 9 recipe); members' holdout CC 0.7103 (seed 0,
epoch 2) / 0.7143 (epoch 1) / 0.7104 (epoch 0); σ 12 for single and ensemble (holdout). 884 test clips, scored once.
**Not comparable to published Hollywood-2 results.**

| Hollywood-2 (Vig gaze) test | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| ViT-L single (section 9) | 0.6798 | 3.898 | 0.5178 | 0.925 | 0.9564 | 0.8167 |
| ViT-L 3-seed ensemble (10A) | 0.6841 | 3.875 | 0.5073 | 0.918 | 0.9574 | 0.8192 |
| ViT-g single (seed 0) | 0.6844 | **3.933** | **0.5214** | 0.917 | 0.9569 | 0.8180 |
| ViT-g 3-seed ensemble | **0.6882** | 3.907 | 0.5104 | **0.911** | **0.9577** | **0.8199** |

Paired bootstrap over clips (`runs/g_other/cmp_hollywood_*.txt`):
- ViT-g single vs ViT-L single: CC +0.0046 [+0.0032, +0.0060]; better on all six metrics, every interval excludes 0.
- ViT-g ensemble vs ViT-L ensemble: CC +0.0041 [+0.0030, +0.0052]; better on five metrics (sAUC +0.0007 n.s.).
- ViT-g ensemble vs ViT-g single: CC +0.0038, KLD, AUC-J, sAUC better; NSS −0.026 and SIM −0.011 worse (the same
  pattern as with ViT-L: averaging blunts the peaks that ≤ 5 viewers per frame reward).

### Section 11 summary (2026-11-09)

- **11A** train600: no gain (ensemble ΔCC +0.0015, interval includes 0); the 540 models are kept.
- **11B** DHF1K test maps from the ViT-g ensemble: `D:/DHF1K_submission/SURGE-D-vitg.zip` (SHA-256 0fe4e2e7…);
  supersedes `SURGE-D.zip`. Sending is the user's.
- **11C** 32-frame clips: holdout 0.5902, gate 0.5904 → not adopted.
- **11D** audio (AST + FiLM/correlation fusion): no improvement on any STAViS dataset.
- **11E** ViT-g everywhere: best SURGE-D numbers on UCF (0.735), 5/6 STAViS datasets (AVAD now ties the best published
  visual model) and Hollywood (Vig) (0.688); significant over ViT-L on DIEM, Coutrot1, ETMD, AVAD and Hollywood.
- The safety monitor ran throughout (max 87 °C); no SAFETY_STOP; nothing was force-killed.

## 12. Review follow-up (rules written 2026-11-10 18:30, before any run)

An area-chair-style review of the thesis (`Paper/msc-thesis/review/area-chair-review.md`) found claims that need
evidence that had not been collected. The user asked for the appropriate follow-up overnight. Order: 12A, 12B (GPU,
existing models), 12C and 12F (CPU), 12D (GPU, new models), then 12E and the ViT-L part of 12D as time allows. One
GPU job at a time under the safety monitor. Results are reported in the thesis as review follow-up results, with the
date of this section.

**12A. Seeds 1 and 2 scored singly on DHF1K val.** `runs/main_b1_s{1,2}` (ViT-L, σ 12) and `runs/vitg_s{1,2}`
(ViT-g, σ 8), each with the σ of the seed-0 single model of the same encoder (sections 5 and 10B), no new tuning.
Scored once, all frames, six metrics. Reported: each seed's paired bootstrap against SalFoM (released maps) and the
mean ± SD over the three seeds. Reading: "a single model is better than SalFoM" holds for an encoder only if the
ΔCC interval of every seed lies above 0.

**12B. Frozen UMT-L with the SURGE-D decoder on DHF1K val.** `runs/g1_umt_l/best.pt` (gate G1: seed 0, best of 10
epochs, 224² input). σ tuned on the holdout (grid 0–24, CC, every 5th frame); scored once on val, six metrics.
Paired bootstrap against SalFoM and against SURGE-D ViT-L seed 0. Reading: if the ΔCC interval against SalFoM lies at
or above 0, the gain over SalFoM is not specific to V-JEPA 2 (a frozen encoder of SalFoM's own family with this
decoder already matches it); if it lies below 0 and below SURGE-D ViT-L, the encoder is the main source of the gain.
Caveats: one seed, a 10-epoch run, 224² input.

**12C. SalFoM with the same blur grid.** SalFoM's released val maps blurred with σ ∈ {0, 4, 8, 12, 16, 24} native
pixels (σ 0 = as released), scored on val (all frames, six metrics). The σ with the best val CC is an upper bound
for SalFoM, because it is selected on the test set. Paired bootstrap of each SURGE-D val result (ViT-L and ViT-g,
single and ensemble) against SalFoM at that σ. Reading: "better than SalFoM" stands for a model only if its ΔCC
interval against the best-σ SalFoM lies above 0.
*Amendment, 18:40, before any blurred SalFoM score was seen:* to save CPU time, the grid is scored with CC, NSS, SIM
and KLD only (the selection uses CC); all six metrics are computed for σ 0 (the existing
`runs/metric_check/salfom_val_all.json`) and for the selected σ.

**12D. Causal SURGE-D (the identical-protocol control of RQ1).**
- Definition: the map of frame f is the model's output at the last sampled position of the window f−30, f−28, …, f
  (16 frames, stride 2; indices below 0 clamped to frame 0). No frame after f is used, as in SalFoM.
- Training: the B1 recipe unchanged (ViT-g: `model.backbone=vjepa2_g`, 256², 3 epochs, warm-up 0.25 epoch, 8 clips
  per video, batch 8, lr 3e-4, prior lr 1e-3, decoder from scratch), except that each clip is supervised on its last
  sampled frame only (target frame drawn uniformly over the video; auxiliary heads on the same frame). Seeds 0, 1, 2.
- Selection: best epoch by holdout CC, scored causally on every 10th frame. σ tuned on the holdout (grid 0–24, causal,
  every 10th frame) for seed 0 (also used for seeds 1 and 2) and for the three-seed ensemble.
- Test: DHF1K val scored once, causally, on every 10th frame (frames 0, 10, 20, … of each video), six metrics.
  SalFoM is scored on the same frames, as released and at the 12C σ.
- **Primary test:** paired bootstrap over the 100 videos, causal ViT-g ensemble against SalFoM at its 12C σ. RQ1 is
  answered "better under an identical protocol" if the ΔCC interval lies above 0, and "not shown under an identical
  protocol" otherwise.
- Secondary: causal single models against SalFoM; causal against offline SURGE-D on the same frames (offline models
  rescored on the same frames if time allows, otherwise per-video means over all frames). ViT-L causal (same recipe
  with `vjepa2_l`) only after 12E.

**12E. Component ablations with three seeds (ViT-L, if time allows).** Seeds 1 and 2 of no centre prior, no
auxiliary heads, SALICON co-training and stride 4 (B1 recipe of section 4, seed changed). The holdout per-video
scores of all 15 models (B1 seeds 0–2 and the 4 × 3 ablation runs) are recomputed with `scripts/evaluate.py` (holdout,
every 5th frame, σ 0, as during training). Effect per factor: the mean over seeds of the per-video ΔCC to B1 of the
same seed, with a two-level bootstrap over seeds and videos. Outcome: **effect** if the 95 % interval excludes 0 and
|Δ| ≥ 0.003; **equivalent** if the 90 % interval lies within ±0.003; otherwise **inconclusive**.

**12F. Gate G2 intervals (CPU).** 95 % video-bootstrap intervals for each map's NSS and CC at both frame rates, and
for the paired differences surprise − frame difference and surprise − token atypicality, from
`runs/g2/g2_results.json`. Reading: "anti-correlated" holds for a map and rate only if its NSS interval lies below 0;
"weaker than frame difference" holds only if the paired NSS interval lies below 0.

### 12A, 12B, 12C and 12F results (2026-11-10, queue A finished 19:56; SalFoM grid on the CPU; `runs/review/`)

**12F (G2 intervals, 60 holdout videos).** Raw surprise NSS −0.190 [−0.332, −0.045] at ~4 fps and −0.502
[−0.620, −0.385] at ~15 fps (below 0 at both rates). Contextual surprise +0.116 [−0.043, +0.275] at ~4 fps (not
distinguishable from 0) and −0.366 [−0.505, −0.225] at ~15 fps. Every paired NSS and CC difference surprise − frame
difference and surprise − atypicality is below 0 at both rates (e.g. raw − frame difference NSS −1.000 [−1.183,
−0.817] at ~4 fps). `runs/review/g2_intervals.json`.

**12A (seeds singly on val, σ of the seed-0 model).** CC / NSS:
- ViT-L: s0 0.5836 / 3.349, s1 0.5805 / 3.307, s2 0.5872 / 3.371; mean 0.5838 ± 0.0034 (SD).
- ViT-g: s0 0.6003 / 3.457, s1 0.5948 / 3.427, s2 0.5957 / 3.472; mean 0.5969 ± 0.0030. Seed 0, the reported
  single model (fixed in advance), is the best of the three ViT-g seeds on val.
- vs SalFoM as released: ΔCC above 0 for every seed of both encoders (ViT-L s1 +0.0165 [+0.0067, +0.0267], s2
  +0.0233 [+0.0128, +0.0344]; ViT-g s1 +0.0309 [+0.0207, +0.0415], s2 +0.0318 [+0.0208, +0.0433]). NSS not
  significant for any ViT-L seed; significant for every ViT-g seed.

**12B (frozen UMT-L + SURGE-D decoder, σ 8 from the holdout).** Val CC 0.5450, NSS 3.100. vs SalFoM ΔCC −0.0189
[−0.0286, −0.0095] (below 0); SURGE-D ViT-L s0 − UMT-L ΔCC +0.0386 [+0.0254, +0.0526]. Reading (rule above): the
interval is below 0 and below SURGE-D ViT-L, so the encoder (V-JEPA 2 rather than UMT) is the main source of the
gain over SalFoM; a frozen encoder of SalFoM's own family with this decoder does not reach SalFoM.

**12C (SalFoM with the blur grid).** Val CC by σ: 0 → 0.5639, 4 → 0.5660, 8 → 0.5703, 12 → 0.5737, **16 → 0.5741**,
24 → 0.5647; σ 16 (all six: NSS 3.265, SIM 0.4009, KLD 1.280, AUC-J 0.9277, sAUC 0.7557) is the upper bound.
SURGE-D − SalFoM(σ 16), ΔCC:
- ViT-L: s0 +0.0095 [−0.0008, +0.0203], s1 +0.0063 [−0.0031, +0.0160], s2 +0.0131 [+0.0027, +0.0241]; ensemble
  +0.0169 [+0.0069, +0.0273].
- ViT-g: s0 +0.0262 [+0.0160, +0.0371], s1 +0.0207 [+0.0109, +0.0310], s2 +0.0216 [+0.0106, +0.0331]; ensemble
  +0.0287 [+0.0186, +0.0394].
- Reading (rule above): the claim stands for every ViT-g model and for the ViT-L ensemble; it does **not** stand for
  ViT-L single models (two of three seeds, including the reported seed 0, include 0). ViT-g s0 and the ViT-g
  ensemble are better on all six metrics; ViT-g s2 and the ViT-L ensemble are not significant on sAUC.

### 12D results, ViT-g (2026-11-11, queue B; `runs/causal_g_s{0,1,2}`, `runs/review/`)

- Training: seeds 0, 1, 2 trained 21:13–00:07; holdout CC (causal, every 10th frame) by epoch: s0 0.5664 / **0.5797** /
  0.5731, s1 0.5703 / **0.5856** / 0.5764, s2 0.5733 / **0.5820** / 0.5740; best epoch is the second for all three.
  σ from the holdout: 8 for seed 0 and 8 for the ensemble.
- Val, causal, every 10th frame (100 videos, 6,076 frames), six metrics CC / NSS / SIM / KLD / AUC-J / sAUC:
  - causal s0 0.5978 / 3.425 / 0.4569 / 1.170 / 0.9303 / 0.7636; s1 0.5913; s2 0.5925 (single-seed mean 0.5939,
    SD 0.0035); ensemble 0.6021 / 3.462 / 0.4567 / 1.156 / 0.9314 / 0.7657.
  - SalFoM on the same frames: released 0.5622 / 3.302 / 0.4127 / 1.302 / 0.9230 / 0.7387; σ 16 (12C) 0.5729 / 3.244 /
    0.4009 / 1.283 / 0.9267 / 0.7540.
- **Primary test:** causal ViT-g ensemble − SalFoM(σ 16), ΔCC **+0.0292 [+0.0188, +0.0400]**, above 0. Reading (rule
  above): RQ1 is answered "better under an identical protocol" (same videos, same frames, same evaluator, no frame
  after the predicted one for either model, SalFoM's blur at its best value on val).
- Secondary: every causal single seed is above SalFoM(σ 16) on CC (s0 +0.0249 [+0.0142, +0.0362], s1 +0.0184
  [+0.0080, +0.0294], s2 +0.0196 [+0.0082, +0.0314]). The ensemble is better on all six metrics (sAUC +0.0116
  [+0.0048, +0.0189]); s0 and s1 also on all six; s2 not significant on sAUC (+0.0032 [−0.0050, +0.0115]).
  Against SalFoM as released: ensemble +0.0399 [+0.0291, +0.0511].
- Causal against offline on the same frames (offline ViT-g `runs/vitg_s{0,1,2}`, σ 8, rescored on the same 6,076
  frames, finished 03:15): offline s0 CC 0.5976, ensemble 0.6005 (all frames: 0.6003 and 0.6029). Causal − offline:
  - s0: CC +0.0003 [−0.0044, +0.0049], NSS −0.0022 [−0.0318, +0.0272], SIM +0.0043 [+0.0011, +0.0076], KLD −0.0074
    [−0.0192, +0.0051], AUC-J +0.0006 [−0.0003, +0.0015], sAUC −0.0049 [−0.0084, −0.0014].
  - ensemble: CC +0.0016 [−0.0015, +0.0046], NSS +0.0042 [−0.0155, +0.0226], SIM −0.0002 [−0.0025, +0.0020], KLD
    −0.0099 [−0.0179, −0.0017], AUC-J +0.0010 [+0.0004, +0.0016], sAUC +0.0000 [−0.0023, +0.0023].
  - Reading (no threshold was set for this secondary comparison): removing the frames after the predicted one cost
    no measurable CC; the other metrics moved by small amounts in both directions. The offline ensemble − SalFoM(σ 16)
    on these frames is +0.0276 [+0.0174, +0.0384], so the advantage over SalFoM does not come from future context.
    Extra CPU comparisons (not in the queue): offline s0 and ensemble against SalFoM released and σ 16 on these
    frames, `runs/review/cmp_offline_g_*_step10_vs_salfom*.json`.

### 12E results (2026-11-11, queue B; `runs/abl_{noprior,noaux,salicon,stride4}_s{1,2}`, `runs/review/holdout/`)

- Training: the eight runs trained 03:15–07:30 (configs identical to the seed-0 ablations except `train.seed` and
  `run_dir`); the holdout rescoring of the 15 models (every 5th frame, σ 0) ran 07:30–08:20. The recomputed holdout
  CC equals the value recorded during training for all 15 models (largest absolute difference 0.0, at float precision).
- Holdout CC per seed (s0 / s1 / s2): reference B1 0.5772 / 0.5801 / 0.5776; no prior 0.5772 / 0.5792 / 0.5754; no
  aux 0.5757 / 0.5790 / 0.5773; SALICON 0.5801 / 0.5801 / 0.5775; stride 4 0.5803 / 0.5838 / 0.5795.
- `scripts/ablation_seeds.py --out runs/review/ablation_seeds.json` (two-level bootstrap, 10,000 resamples, δ 0.003),
  mean ΔCC to B1 of the same seed, 95 % and 90 % intervals, per-seed ΔCC → outcome:
  - no centre prior: −0.0010 [−0.0026, +0.0005], 90 % [−0.0024, +0.0003]; −0.0000, −0.0009, −0.0022 → **equivalent**
  - no auxiliary heads: −0.0010 [−0.0023, +0.0003], 90 % [−0.0020, +0.0001]; −0.0015, −0.0011, −0.0003 → **equivalent**
  - SALICON co-training: +0.0009 [−0.0039, +0.0058], 90 % [−0.0031, +0.0049]; +0.0028, −0.0000, −0.0001 →
    **inconclusive**
  - stride 4: +0.0029 [−0.0009, +0.0068], 90 % [−0.0003, +0.0062]; +0.0031, +0.0037, +0.0020 → **inconclusive**
- Reading (rule above): the centre prior and the auxiliary heads are equivalent to their absence within ±0.003 CC.
  SALICON's seed-0 gain did not recur. Stride 4 improved with all three seeds but its mean is below δ and its 95 %
  interval includes 0, so the seed-0 "difference at the threshold" does not hold up as an effect. Outside the three
  outcomes of the rule (no non-inferiority outcome was defined), its 90 % lower bound −0.0003 excludes a loss of δ.
- Thesis updated: `tab:ablation` (components block, three seeds), §5.6 Components, RQ3 discussion, Ch6 RQ3,
  contribution 3 and future work, Ch1 limitations and contribution bullet 3, abstract; §4 statistics describes the
  two-level bootstrap.

### 12D addendum, ViT-L (written 2026-11-11 08:33, before any causal ViT-L result was seen)

As for ViT-g, the offline ViT-L models are rescored on the same every-10th validation frames once queue B has
finished (one GPU job at a time): `runs/main_b1_s0` with σ 12 and the three-seed ensemble `runs/main_b1_s{0,1,2}`
with σ 8 (the σ of section 5, no new tuning), six metrics. Paired bootstraps: causal − offline (s0 and ensemble) and
offline − SalFoM (released and σ 16) on these frames. Secondary and descriptive, no threshold, as for ViT-g. The
primary reading of causal ViT-L is the one of 12D (ensemble − SalFoM σ 16, ΔCC interval above 0).

### 12G. Ensemble − single on DHF1K val (added 2026-11-11 08:50; CPU, descriptive, from existing score files)

Not in the plan of section 12; it checks the thesis wording "seed ensembles added a reliable gain" (review item T12)
with `scripts/compare.py` on the stored per-video val scores (no new scoring, no decision depends on it).
- ViT-L ensemble (`runs/final/val_ens3.json`, σ 8) − seed 0 (`val_single_s0.json`, σ 12): CC +0.0074 [+0.0050,
  +0.0099], NSS +0.0686 [+0.0514, +0.0871], KLD −0.0354 [−0.0440, −0.0273], AUC-J +0.0019 [+0.0014, +0.0025], SIM
  −0.0006 [−0.0024, +0.0013], sAUC −0.0022 [−0.0040, −0.0005].
- ViT-g ensemble (`runs/vitg/val_ens3.json`, σ 8) − seed 0 (σ 8): CC +0.0025 [+0.0002, +0.0049], NSS +0.0293
  [+0.0141, +0.0454], SIM +0.0043 [+0.0025, +0.0063], KLD −0.0099 [−0.0164, −0.0035], AUC-J +0.0006 [+0.0002,
  +0.0010], sAUC −0.0029 [−0.0046, −0.0012].
- Reading: the ensemble gain in CC is above 0 for both encoders (barely for ViT-g); sAUC falls slightly for both.
  `runs/review/cmp_ens3_vs_single_{l,g}_val.{json,txt}`.

### 12D results, ViT-L (2026-11-11, queue B, then queue C for the addendum; `runs/causal_l_s{0,1,2}`, `runs/review/`)

- Training: seeds 0, 1 and 2 trained 08:20–09:45. Holdout CC (causal, every 10th frame) by epoch, best in bold:
  - s0: 0.5577 / **0.5721** / 0.5644
  - s1: 0.5629 / **0.5737** / 0.5583
  - s2: 0.5556 / **0.5698** / 0.5571
  - The second epoch is best for all three.
- σ from the holdout (09:45–10:05): 8 for seed 0 and 8 for the ensemble. For seed 0, σ 8 and σ 12 tie at four decimals
  (0.57410 and 0.57411). `scripts/tune_sigma.py` breaks such ties toward the smaller σ.
- Val, causal, every 10th frame (100 videos, 6,076 frames; scored 10:05–10:56). Metrics in the order CC / NSS / SIM /
  KLD / AUC-J / sAUC:
  - causal s0: 0.5842 / 3.346 / 0.4511 / 1.212 / 0.9281 / 0.7600
  - causal s1 and s2: CC 0.5767 and 0.5792. Single-seed mean 0.5801, SD 0.0038.
  - ensemble: 0.5900 / 3.390 / 0.4495 / 1.189 / 0.9294 / 0.7607
- **Primary reading (the 12D rule, as for ViT-g):** causal ViT-L ensemble − SalFoM(σ 16), ΔCC **+0.0171
  [+0.0071, +0.0275]**, above 0. So "better under an identical protocol" also holds with the ViT-L encoder, which is
  the size of SalFoM's.
- Secondary:
  - Single seeds − SalFoM(σ 16): s0 +0.0113 [+0.0009, +0.0221]; s1 +0.0038 [−0.0062, +0.0142]; s2 +0.0063
    [−0.0049, +0.0180]. As offline (12C), a single ViT-L model is not reliably better than SalFoM at its best blur.
  - The ensemble is better on all six metrics: AUC-J +0.0028 [+0.0011, +0.0045], sAUC +0.0066 [+0.0002, +0.0135].
  - Against SalFoM as released, CC is above 0 for every seed (s0 +0.0220, s1 +0.0145, s2 +0.0170) and for the
    ensemble (+0.0278 [+0.0175, +0.0384]).
  - Cost: a single causal ViT-L model needs 1.82 TFLOPs per output frame, close to SalFoM's encoder at 1.90
    (section 5 of the thesis). The ensemble needs three times as much.
- 12D addendum: the offline ViT-L models were rescored on the same 6,076 frames (`scripts/queue_review_c.sh`,
  10:57–11:20): `runs/main_b1_s0` at σ 12 and the ensemble `runs/main_b1_s{0,1,2}` at σ 8, both with no new tuning.
  - Offline s0: 0.5812 / 3.321 / 0.4486 / 1.229 / 0.9274 / 0.7611.
  - Offline ensemble: 0.5888 / 3.390 / 0.4481 / 1.192 / 0.9293 / 0.7592.
  - On all frames these models score CC 0.5836 and 0.5910.
- Causal − offline on these frames:
  - s0: CC +0.0030 [−0.0014, +0.0077], NSS +0.0249 [−0.0041, +0.0560], SIM +0.0025 [−0.0007, +0.0060], KLD
    −0.0172 [−0.0293, −0.0052], AUC-J +0.0007 [−0.0001, +0.0016], sAUC −0.0011 [−0.0041, +0.0019].
  - ensemble: CC +0.0012 [−0.0022, +0.0047], NSS +0.0007 [−0.0203, +0.0225], SIM +0.0014 [−0.0009, +0.0039], KLD
    −0.0025 [−0.0120, +0.0069], AUC-J +0.0001 [−0.0006, +0.0008], sAUC +0.0014 [−0.0009, +0.0040].
  - Reading (descriptive, no threshold, as for ViT-g): dropping the frames after the predicted one cost no measurable
    accuracy with ViT-L either. The only interval that excludes 0 is seed 0's KLD, and it favours the causal model.
- Offline − SalFoM on these frames:
  - against SalFoM(σ 16): s0 CC +0.0083 [−0.0022, +0.0191], interval includes 0; ensemble +0.0159 [+0.0058, +0.0264].
  - against SalFoM as released: s0 +0.0189 [+0.0086, +0.0297]; ensemble +0.0265 [+0.0164, +0.0371].
  - Same picture as on all frames (12C). Files: `runs/review/offline_l_val_step10_{s0,ens3}.json`,
    `cmp_causal_vs_offline_l_{s0,ens3}.*`, `cmp_offline_l_{s0,ens3}_step10_vs_salfom{,_best}.*`.
- Thesis updated:
  - ViT-L blocks in `tab:causal`; a ViT-L block in `tab:app-causal`; ViT-L columns in `tab:app-causal-offline`;
  - a ViT-L paragraph in §5.5.2; RQ1 in §5.7 and §6.1;
  - contribution 1 in §1.8; the abstract; future work ("Efficient causal prediction"); the val ledger; Appendix B.

## 13. Paper update: learning curve, incremental information of the prediction error, a masked readout, measured timing (rules written 2026-11-11 20:58, before any run)

The user asked to "update the paper and improve it". Besides the text revision, four items that the review left open
(its Section 11.2: F8, F1b, F9, F6) are run, because each one decides a sentence that the thesis currently hedges.
Order: 13A (GPU, new models), then 13B and 13C (one zero-shot GPU job, then CPU), then 13D (GPU, nothing else
running). One GPU job at a time under the safety monitor. Nothing here touches DHF1K val or any test set.

**13A. Learning curve over the number of training videos (review F8; tests the data-limitation hypothesis).**
- Hypothesis H (thesis §5.7, §6.1): with the frozen encoder, accuracy at 540 training videos is limited by the amount
  of training data; this was offered as the reason why no decoder component matters.
- Arms: 135, 270 and 540 training videos, V-JEPA 2 ViT-L (the encoder of the component ablations), seeds 0, 1, 2.
  The 540-video arm is the existing `runs/main_b1_s{0,1,2}`.
- Subsets: for seed s the 540 training ids are shuffled with `random.Random(13000 + s)`; the first 270 are
  `configs/splits/dhf1k_train270_s{s}.txt` and the first 135 are `dhf1k_train135_s{s}.txt` (nested within a seed, a
  new draw per seed, so that the seed-level variation includes the choice of videos). Script
  `scripts/make_lc_splits.py`.
- Recipe: B1 unchanged (256², 3 epochs, warm-up 0.25 epoch, batch 8, lr 3e-4, prior lr 1e-3, decoder from scratch,
  `train.seed=s`), except `data.train_split` and the number of clips per video: 16 for 270 videos and 32 for 135, so
  that every arm draws 4,320 clips per epoch and takes the same 1,620 optimiser steps. Only the number of distinct
  videos changes. Runs `runs/lc{270,135}_s{0,1,2}`.
- Scoring: best epoch by holdout CC, as in B1. The holdout per-video scores are recomputed with
  `scripts/evaluate.py` (60 videos, every 5th frame, σ 0) into `runs/review/holdout/lc*.json`, as in 12E.
- Analysis (`scripts/learning_curve.py`): holdout CC per arm (mean and SD over seeds); the paired differences
  Δ(270 − 540), Δ(135 − 270) and Δ(135 − 540) per video and seed, with the two-level bootstrap of 12E (seeds, then
  videos; 10,000 resamples) and the same margin δ = 0.003. Outcomes as in 12E: **effect** if the 95 % interval
  excludes 0 and |Δ| ≥ δ; **equivalent** if the 90 % interval lies within ±δ; otherwise **inconclusive**.
- Reading, on the primary comparison Δ(270 − 540):
  - *H refuted in this range* if it is **equivalent**: halving the training set changes holdout CC by less than
    0.003, so the amount of data is not what limits the model. The thesis then drops the explanation.
  - *H supported in this range* if it is an **effect** with a negative sign: the curve is still rising at 540 videos.
    The thesis then states it as supported for ViT-L between 135 and 540 videos, not as proven beyond 540.
  - Otherwise *inconclusive*; the thesis keeps H as a hypothesis and reports the interval.
  - Δ(135 − 270) and the gain per doubling of the data are reported for the shape of the curve only.
- Caveats, written now: three seeds; one encoder; equal steps, so each video of a smaller arm is sampled more often;
  the holdout is also the set on which each run's epoch is chosen (the same for every arm).

**13B. Incremental information of the prediction error (review F1b; RQ2's "beyond simple cues").**
- Maps: `scripts/g2_maps.py` recomputes the maps of gate G2 (section 3: V-JEPA 2 ViT-L and its predictor, 8 context
  + 8 target frames, strides 8 and 2, first frame of each target tubelet, the 60 holdout videos) and stores, per
  scored frame, the 16×16 token maps of raw surprise, contextual surprise, token atypicality and frame difference.
  Check: the NSS of the stored maps (σ 16) must reproduce the means of `runs/g2/g2_results.json` to within 0.01 for
  raw surprise, atypicality and frame difference.
- Model family: on a grid of 45×80 cells (8×8 px), p(cell) = softmax(w_c · log prior + Σ_k w_k · z_k), where prior is
  the empirical centre prior of the 540 training videos (`runs/g2/center_prior.npy`), summed per cell, and z_k is cue
  k upsampled and blurred as in G2 (σ 16 px), averaged per cell and standardised over the cells of the frame.
- Score: information gain over the centre prior in bits per fixation, i.e. the mean over a frame's fixations of
  log2 p(cell) − log2 prior(cell); per video the mean over its scored frames; overall the mean over videos.
- Fit: the weights maximise the mean per-video information gain of the fitting half (concave; L-BFGS from w_c = 1 and
  all other weights 0). Cross-fitting: holdout videos sorted by id and assigned alternately to halves A and B; the
  weights fitted on one half are scored on the other, so every video has an out-of-fold score for every model.
- Quantities: baseline B = {centre prior, frame difference, atypicality}; full F = B + surprise; incremental gain
  U = IG(F) − IG(B) per video, mean over the 60 videos with a 95 % and a 90 % video-bootstrap interval (10,000
  resamples, fitted weights held fixed). For scale, the same leave-one-out gain is reported for frame difference
  (F minus the model without it) and for atypicality, and IG(B) itself.
- Primary comparison: raw surprise at stride 8 (about 4 fps, the rate closest to the predictor's pre-training).
  Secondary, descriptive: raw surprise at stride 2 and contextual surprise at both rates (no multiplicity
  correction; the RQ2 answer uses the primary only).
- Reading:
  - **incremental information** if the 95 % interval of U lies above 0. The sign of the fitted surprise weight is
    reported with it: a negative weight means that fixations avoid high-error regions, which is information but not
    the "surprise attracts gaze" hypothesis.
  - **negligible** if the 90 % interval lies within ±0.01 bits per fixation. The margin is fixed now; a cue z adds
    about NSS²/(2 ln 2) bits at its best weight, so 0.01 bits corresponds to a conditional NSS of about 0.12.
  - otherwise **inconclusive**, reported with the interval.

**13C. A readout closer to the predictor's pre-training (review F9; same GPU job as 13B).**
- Why: the predictor was trained to fill in masked spatial blocks that extend over the whole clip in time, from the
  visible remainder. G2 asked it to predict the second half of a clip from the first half, which is outside that
  training task (thesis §5.4 gives this as a possible reason for the negative result).
- Readout "masked-tube error": the 16×16 token grid is cut into sixteen 4×4 blocks. In each of four passes, the four
  blocks with the same (row parity, column parity) are masked in all 8 tubelets (25 % of the tokens; every masked
  block has visible neighbours). The encoder is run on the visible tokens only (its layers with the position ids of
  the visible tokens), the predictor predicts the masked tokens, and the error is the mean absolute difference to the
  layer-normalised features of the full clip, as in G2. Four passes give one error per token; the map of a frame is
  the error at its tubelet.
- Checks before use, in float32 on one clip: (i) with all tokens visible the masked encoder reproduces the model's
  own encoder output; (ii) with the first 1,024 tokens visible it reproduces the encoding of the first 8 frames
  alone. Tolerance 1e-3 (maximum absolute difference). If a check fails, 13C is not run and that is reported.
- Windows and scored frames: exactly those of G2 (so the other maps and the fixations are shared).
- Scores and readings: standalone NSS and CC with video-bootstrap intervals and the paired differences to frame
  difference and atypicality (as 12F), and the incremental gain U of 13B with this error in place of the surprise,
  primary at stride 8. Same readings as 12F and 13B, stated per readout.
- Caveat, written now: 25 % masking leaves more context than pre-training did (where most tokens were masked), so
  this readout is closer to pre-training in the geometry of the mask, not identical to it.

**13D. Measured inference time (review F6; GPU, last, with no other job running).**
- `scripts/time_inference.py`: SalFoM (the released `Model.py` and DHF1K weights; 16 frames at 224², one map per
  pass) and SURGE-D ViT-L and ViT-g (trained seed-0 models; 16 frames at 256²; one pass gives 16 maps offline and one
  map causally). Batch 1 and batch 8, bfloat16 autocast (as in the evaluation of all three), random input already on
  the GPU, so frame decoding, resizing and the final blur are excluded for every model. 10 warm-up passes, then 50
  timed passes with `torch.cuda.synchronize()`; median and inter-quartile range; peak GPU memory.
- Reported: milliseconds per pass and per output frame (SalFoM: per pass; SURGE-D offline: per pass / 16; SURGE-D
  causal: per pass), next to the FLOP estimates.
- Reading (the review's criterion, written in the review before this run): the offline efficiency statement is
  refuted if SalFoM's measured time per output frame is within a factor of 2 of SURGE-D ViT-g offline. The thesis's
  cost statement then uses the measured ratios, whatever they are.

### 13B/13C addendum: the predictor's mask token (written 2026-11-11 22:05 and committed as 7fe4ff1 at 22:07, before any GPU run of `g2_maps.py` and before any gaze-based score of the new maps)

**What was found.** A CPU smoke test of `scripts/g2_maps.py` (one holdout video, two windows; it computes prediction
errors only and reads no gaze data) included a diagnostic over the predictor's mask tokens. The facts below were
checked in the weights and in the code, not inferred:
- The released predictor has 10 mask tokens (`predictor.embeddings.mask_tokens`, shape (10, 1, 1, 384)). In both
  checkpoints used here (`facebook/vjepa2-vitl-fpc64-256`, `facebook/vjepa2-vitg-fpc64-256`) only token 0 is non-zero
  (L2 norm 0.769 and 0.710); tokens 1–9 are exactly zero, i.e. still at their zero initialisation.
- The Hugging Face port (transformers 5.22.1, `models/vjepa2/modeling_vjepa2.py`) reads `mask_index = 1` in
  `VJEPA2PredictorEmbeddings.forward`, and `VJEPA2Predictor.forward` never passes another value. Every predictor call
  through the port therefore uses a zero vector as the mask token.
- Meta's training code (facebookresearch/vjepa2, commit 204698b) calls the predictor with `mask_index=i`
  (`src/utils/wrappers.py`, `PredictorMultiSeqWrapper`), where i enumerates the groups of clips that have the same
  number of frames. The collator keys these groups by frame count (`src/masks/multiseq_multiblock3d.py`), and the
  released configurations give every dataset the same frame count (`dataset_fpcs: [16, 16, 16]`; cooldown
  `[64, 64, 64]`), so i = 0 throughout pre-training. This agrees with the weights: only token 0 was ever trained.
- Smoke test (video 0028, two windows): mean masked-tube error 0.565 with token 0 and 0.595 with each of tokens 1–9
  (identical to each other, as they must be).

**Consequence for what is already reported.** Gate G2 (section 3; result of 2026-11-05) and its intervals (12F) used
the port's default, i.e. an untrained mask token. The numbers stand as a record of that readout. They are not a test
of the predictor as it was trained, and the NO-GO decision was taken on that readout. SURGE-D itself is not
affected: it uses the encoder only (`skip_predictor=True`); no trained model, score or comparison changes.

**Decisions (fixed now; no gaze score with token 0 exists yet).**
1. `g2_maps.py` stores every prediction-error map with both tokens: `surprise_raw`, `surprise_ctx`, `masked_tube`
   with token 1 (the readout of G2) and `surprise_raw_t0`, `surprise_ctx_t0`, `masked_tube_t0` with token 0 (the
   trained token). Encodings, targets, windows and the wrong-context rule are shared; only the predictor call
   differs. The 5 % rule of the smoke-test diagnostic is dropped; the diagnostic only records the mean masked-tube
   error of all 10 tokens on the first stride-8 window of every 6th video.
2. The reproduction check of 13B stays on the token-1 maps: they must reproduce G2 within 0.01 NSS.
3. **Token 0 is the primary readout of 13B and 13C**, because RQ2 asks about the predictor's error and the predictor
   as trained uses token 0. Primary comparisons: U of `surprise_raw_t0` at stride 8 (13B) and U of `masked_tube_t0`
   at stride 8 (13C), with the readings of 13B. Token 1 is reported beside each, descriptive.
4. **Gate G2 is re-scored with token 0 by its own rule** (section 3, as coded in `g2_surprise.py`): a variant passes
   if its mean NSS is above that of both token atypicality and frame difference and its information gain (the 50/50
   mixture with the centre prior, over the prior) is positive; GO if any of the four variants (raw, contextual; both
   rates) passes. 12F's intervals and readings are repeated for the token-0 maps. The masked-tube error is not part
   of the gate (it did not exist when the gate was defined); it is reported under 13C.
   - NO-GO again: the thesis reports the token-0 readout as the result of the gate, keeps the token-1 numbers as the
     original run, and says that the decision does not depend on the mask token.
   - GO: the thesis says that the NO-GO decision was taken on a faulty readout and that the surprise branch was
     therefore never tested; RQ2's answer is rewritten to what the token-0 data show. No surprise branch is built in
     this round (a new model with new comparisons on val); it is listed as future work.
5. Descriptive only, no reading attached: the gain of each single cue over the fitted prior, IG(P + cue) − IG(P); the
   mean value of each stored error map; the mean per-frame correlation between the token-0 and token-1 map of each
   readout.
6. Whatever the outcome, the thesis states in its method and results that the original gate used the port's default
   token, and Appendix B lists the finding as a deviation.

**13D addendum (same time, before the run).** The criterion is applied at each batch size (1 and 8): the offline
efficiency statement counts as refuted if the ratio SalFoM time per output frame / SURGE-D ViT-g offline time per
output frame is below 2 at either batch size.

**Pipeline smoke test with gaze scores (run 2026-11-11 22:13, after commit 7fe4ff1; recorded here 2026-11-12 08:25,
before the full run).** After the addendum above was committed, `g2_maps.py` and `g2_incremental.py` were run end to
end on the CPU on four holdout videos (0028, 0031, 0043, 0048; 32 scored frames per rate) to test the pipeline. That
run printed gaze-based scores of these four videos, including scores of the token-0 maps; its output is kept as
`runs/review/g2_incremental_smoke4.txt`. Nothing was read from it and nothing was changed after it: the rules,
margins, primary comparisons and decisions of 13B, 13C and the addendum are those of commit 7fe4ff1. The only code
change since is a `.detach()` where `g2_maps.py` records the mask-token norms (no effect on any map). The four
videos are among the 60 of the full analysis, which recomputes their maps.

*Correction to the sentence "Nothing was read from it" (added 2026-11-12 08:33, while the full run was in progress
and before any of its scores existed).* At about 08:28, six minutes after the full run had been launched, the JSON
of this four-video smoke test was printed in full to learn its structure for the table generator
(`scripts/review_tables.py g2`), so its numbers were displayed, including the token-0 scores of the four videos. No
rule, margin, primary comparison, decision or line of analysis code was changed after that: `g2_maps.py` and
`g2_incremental.py` are as launched, and the rules are those of commit 7fe4ff1. The generator was then tested on that
file with every digit masked. At about 08:37, still before any score of the full run existed, the generator's output
for the same four-video file was displayed once more without the mask (the same numbers as at 08:28). Again nothing
was changed after it.

### 13A result: learning curve (queue D finished 2026-11-11 23:47; `runs/review/learning_curve.{json,txt}`)

Runs `runs/lc{270,135}_s{0,1,2}` (1,620 steps each), holdout rescored into `runs/review/holdout/lc*.json`.

| training videos | holdout CC, seeds 0 / 1 / 2 | mean ± SD | NSS | SIM | KLD | train CC, last epoch (mean) |
|---|---|---|---|---|---|---|
| 135 | 0.5606 / 0.5639 / 0.5690 | 0.5645 ± 0.0042 | 3.227 | 0.4253 | 1.263 | 0.718 |
| 270 | 0.5686 / 0.5762 / 0.5739 | 0.5729 ± 0.0039 | 3.286 | 0.4419 | 1.238 | 0.702 |
| 540 | 0.5772 / 0.5801 / 0.5776 | 0.5783 ± 0.0016 | 3.323 | 0.4448 | 1.227 | 0.683 |

| difference (paired by seed and video) | per seed | mean | 95 % | 90 % | outcome |
|---|---|---|---|---|---|
| CC(270) − CC(540), primary | −0.0087, −0.0038, −0.0036 | −0.0054 | [−0.0112, +0.0011] | [−0.0101, −0.0001] | **inconclusive** |
| CC(135) − CC(270) | −0.0079, −0.0123, −0.0049 | −0.0084 | [−0.0164, −0.0004] | [−0.0150, −0.0018] | effect |
| CC(135) − CC(540) | −0.0166, −0.0162, −0.0086 | −0.0138 | [−0.0238, −0.0048] | [−0.0219, −0.0063] | effect |

**Reading by the rule of 13A: inconclusive.** The primary difference is negative in every seed and larger than the
margin, but its 95 % interval includes 0, so H is not "supported in this range"; its 90 % interval reaches far beyond
±0.003, so H is not "refuted in this range" either. The thesis keeps H as a hypothesis and reports the interval.
Shape only (no outcome attached): the second doubling (270 → 540) brought 0.64 of the gain of the first (135 → 270);
the best epoch was the 2nd in all three 540-video runs, the 2nd, 2nd and 1st with 270 videos and the 2nd, 1st and 2nd
with 135; the training CC of the last epoch rises as the set shrinks (0.683, 0.702, 0.718) while holdout CC falls.
Caveats as written in the rule (three seeds; ViT-L only; equal steps; the holdout also selects the epoch). DHF1K has
no further annotated training videos beyond the 60 of the holdout, so the curve cannot be extended on this dataset.

### 13D first run, and the rule for a repeat (written 2026-11-12 09:22, before the repeat is started)

**First run** (queue E, 09:04:00 to 09:11:38; `runs/review/timing.{json,txt}`; NVIDIA RTX A5000, torch 2.14.1+cu130,
bfloat16 autocast, 10 warm-up and 50 timed passes; median [inter-quartile range]):

| model | batch | ms per pass | encoder alone, ms | ms per output frame | peak GPU memory |
|---|---|---|---|---|---|
| SalFoM (released code and weights; 402,945,443 parameters; 0 missing / 0 unexpected keys) | 1 | 227.2 [226.3, 227.9] | 207.3 | 227.15 | 3.58 GB |
| | 8 | 1624.6 [1608.0, 1634.2] | 1605.6 | 203.07 | 14.20 GB |
| SURGE-D ViT-L (`runs/main_b1_s0`) | 1 | 86.6 [84.3, 133.6] | 74.2 | 5.41 offline, 86.6 causal | 2.08 GB |
| | 8 | 409.5 [408.2, 410.1] | 356.9 | 3.20 offline, 51.2 causal | 3.85 GB |
| SURGE-D ViT-g (`runs/vitg_s0`) | 1 | 165.6 [163.7, 186.5] | 165.3 | 10.35 offline, 165.6 causal | 6.16 GB |
| | 8 | 987.9 [985.9, 988.9] | 931.8 | 7.72 offline, 123.5 causal | 8.21 GB |

Ratios, batch 1 / batch 8: SalFoM / ViT-g offline 21.95 / 26.31; SalFoM / ViT-L offline 41.98 / 63.48; ViT-g causal /
SalFoM 0.73 / 0.61; ViT-L causal / SalFoM 0.38 / 0.25. By the criterion of 13D and its addendum the offline
efficiency statement **holds** in this run (both ratios far above 2).

**Deviation from the protocol.** 13D says "nothing else running". While this run was in progress, short shell
commands were started from this session on the same machine (reading the queue log, the result files of 13B/13C and
the clock). They do not use the GPU, but the wide inter-quartile range of ViT-L at batch 1 ([84.3, 133.6] ms around a
median of 86.6) shows that some passes were disturbed. The decision is not in doubt (21.95 and 26.31 against a
threshold of 2); the individual times may be.

**Rule for the repeat (fixed now).** The same command with the unchanged script, written to
`runs/review/timing_run2.{json,txt}`, with no process started from this session between its launch and its end (file
reads and edits through the editor only). The thesis reports the repeat, as the run that meets the protocol; the first
run stays in the repository and in the table above. The criterion is applied to both runs: the offline efficiency
statement counts as refuted if the ratio SalFoM / SURGE-D ViT-g offline is below 2 at either batch size in either
run. If any median time per output frame differs by more than 10 % between the two runs, the thesis says so and
gives both values.
