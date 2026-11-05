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
