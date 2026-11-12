# SURGE-D against state-of-the-art video saliency models: an evidence-based analysis of its advantages and trade-offs

**Companion to:** `report/critical-review.md` (same folder).
**Date:** 2026-11-12.

> **Note on independence.** This analysis was written by the same AI assistant that built SURGE-D and drafted the
> thesis. It is not an independent assessment. To counter that bias:
> - Every competitor number below was taken from the model's own paper wherever possible and labelled with its
>   source.
> - SURGE-D's headline numbers were checked against the repository's run output files.
> - Advantages that are only derived, not measured, are marked as such.

---

## 0. Paper identification (required step)

| Field | Value |
|---|---|
| Title | *Video Saliency Prediction with a Frozen Self-Supervised Video Encoder* (`Paper/msc-thesis/main.tex`, title page) |
| Model | **SURGE-D**: a frozen V-JEPA 2 encoder with a trained multi-scale (2+1)D decoder. The name comes from the original "SURprise-Gated Encoder–decoder" design; "D" marks the reduced design kept after the surprise branch failed its gate (§4.4, p. 30). |
| Document type | MSc thesis (draft). Addis Ababa University, Addis Ababa Institute of Technology (AAiT), School of Information Technology and Engineering (SiTE). Submission date set to November 2026. **Not peer-reviewed.** |
| Version analysed | `Paper/msc-thesis/main.pdf` built at commit `b6e499c`, 114 pages. Repository `github.com/michael-099/SURGE`. |
| Author | The title-page fields (student, advisor, department) are still placeholders; the repository owner is GitHub user `michael-099`. |
| Code | The repository: model (`surge/`), training and evaluation scripts (`scripts/`), configurations and splits (`configs/`), decision log (`research/log.md`), and the commands in App. B (pp. 93–96). |
| Weights | Encoder: the public V-JEPA 2 checkpoints `facebook/vjepa2-vitl-fpc64-256` and `facebook/vjepa2-vitg-fpc64-256` (MIT licence, thesis p. 31). Trained decoders: **not distributed**; the checkpoints are 26.7 MB (ViT-L) and 28.2 MB (ViT-g), measured. |
| Claims relevant to SOTA | (i) Better than SalFoM on DHF1K validation on all six metrics, offline and causally. (ii) About 1/9 of SalFoM's encoder compute per output frame offline. (iii) Competitive on UCF Sports. (iv) Above the best verified peer-reviewed results on 4 of 6 audio-visual datasets, without audio. (v) Strong results with only 7 M trained parameters. |

---

## 0b. Comparison scope (defined explicitly)

**Task.** Video saliency prediction: per-frame human eye-fixation density. Image saliency, salient-object detection
and mouse-tracking benchmarks are out of scope.

**SURGE-D variants compared.**

| Variant | Role | Parameters |
|---|---|---|
| ViT-g single (seed 0) | Primary single model | 1,042 M total, 7.39 M trained |
| ViT-g three-seed ensemble | Primary ensemble | as above, × 3 decoders |
| ViT-L single and ensemble | Encoder the size of SalFoM's | 333 M total, 6.99 M trained |
| Causal ViT-g / ViT-L | Past-only setting | as above |

Published models are mostly single models; ViNet-E is a two-model ensemble. Ensembles are therefore labelled
wherever they appear.

**Competitors included.** Peer-reviewed video saliency models from 2019–2026 that report results on at least one of
the thesis's benchmarks and whose tables could be read:
- TASED-Net (ICCV 2019), UNISAL (ECCV 2020), STAViS (CVPR 2020)
- ViNet (IROS 2021), HD2S (IJCV 2021)
- STSANet (TMM 2023), TMFI-Net (TCSVT 2023), TinyHD (WACV 2023), CASP-Net (CVPR 2023)
- THTD-Net (VISAPP 2024), DiffSal (CVPR 2024), SalFoM (ICPR 2024; LNCS 2025)
- ViNet-S/A/E (ICASSP 2025), UniST (TCSVT 2025)
- AVRSP (TETCI 2026): its numbers are taken from the thesis and were not re-verified.

**Competitors excluded, and why.**
- TM2SP (TCSVT 2025), MSFF-Net (TMM 2024), LSTD-Net (CVIU 2026), FOSal, and the model of Mao et al.: their tables
  were not accessible. Where they report DHF1K, it is mostly the test split.
- NTIRE 2026 challenge entries, including PredJSal: different dataset (mouse tracking) and protocol. PredJSal is
  discussed only for architecture (§3).
- Preprint-only models (TSFP-Net, DTFSal, ViSAGE).
- DcF-Net (2026): reported DHF1K CC 0.570, but venue and split unverified.

**Comparability classes** (each table below states its class).

| Class | Setting | How it compares with SURGE-D |
|---|---|---|
| **A** | DHF1K validation, SalFoM re-scored by the same evaluator from its released weights | **Directly comparable.** Paired per-video bootstrap available. |
| **B** | DHF1K validation, published numbers from other evaluation code | Same split, **approximately comparable**. Published validation numbers for the same model differ by up to 0.03 CC: TMFI-Net is 0.554 in ViNet-S but 0.524 in UniST and DiffSal. |
| **C** | DHF1K **test** (official server) | **Not comparable.** SURGE-D's test scores are pending. Shown for context only. |
| **D** | UCF Sports test, and the six STAViS audio-visual datasets (official folds) | Same splits, other evaluators: **approximately comparable**. Published evaluations of the same model differ by up to 0.08 CC on Coutrot2 (STAViS: 0.734 in DiffSal against 0.652 in ViNet-S). Most competitors on the audio-visual sets use audio; SURGE-D does not. |
| **E** | Hollywood-2 | **Not comparable.** SURGE-D uses the openly licensed Vig et al. gaze (≤5 observers, whole clips); the literature uses the Mathe and Sminchisescu ground truth (16–19 observers, shot-segmented). |
| **F** | Efficiency | Approximately comparable only. GPUs differ (RTX A5000 for SURGE-D and SalFoM, RTX 4090 for ViNet), precisions differ, and counting conventions differ (FLOPs = 2 × MACs; some tables label MACs as "FLOPs"; attention products are omitted by some counters). |

**Metric conventions.**
- The thesis's **sAUC** draws negatives from one frame of each of ten other videos (§4.8, p. 39). It is comparable
  only within the thesis.
- **KLD** is rarely reported by competitors.
- AUC-J, NSS, CC and SIM come from different code bases. The thesis's evaluator matches TMFI-Net's code within
  0.0003 CC and 0.0004 NSS on SalFoM's maps (p. 43).

**Evidence labels used in all tables.**

| Label | Meaning |
|---|---|
| **[O]** | Model's original paper, read for this analysis (arXiv or open version of the peer-reviewed paper). |
| **[S: X]** | Secondary: reported in paper X. |
| **[T]** | Reported in the thesis. |
| **[T✓]** | Reported in the thesis and checked against the repository's run output files. |
| **[M]** | Measured in this project: the thesis's timing, FLOP counting and parameter counting of SalFoM's released model, or the on-disk sizes measured for this report. |
| **[D]** | Derived arithmetic from labelled values. |
| **NR** | Not reported in the sources consulted. |

---

## 1. Prediction performance

### 1.1 DHF1K validation (the thesis's test set): classes A and B

**Against SalFoM, same evaluator, paired (class A)** [T✓; T5.6, T5.7, TA.1, TA.3, pp. 50–51, 85–86].

| SURGE-D | ΔCC vs SalFoM (released) | ΔCC vs SalFoM best blur (σ = 16, chosen on the test set) | Other metrics |
|---|---|---|---|
| ViT-g single | **+0.036 [+0.026, +0.048]** (+6.5 %) | **+0.026 [+0.016, +0.037]** (+4.6 %) | All six metrics better, every interval excludes 0 |
| ViT-g ensemble | **+0.039 [+0.029, +0.050]** (+6.9 %) | **+0.029 [+0.019, +0.039]** (+5.0 %) | All six better; higher CC on 79 of 100 videos |
| ViT-L single | **+0.020 [+0.010, +0.031]** (+3.5 %) | +0.0095 [−0.0008, +0.0203] (not significant) | NSS not significant against the released model |
| ViT-L ensemble | **+0.027 [+0.017, +0.038]** | **+0.017 [+0.007, +0.027]** | All six better against the released model |

Relative values are computed from the CC levels in T5.5 [D]. Absolute gains on the other metrics, ViT-g single
against SalFoM released: NSS +0.130 (+3.9 %), SIM +0.040 (+9.7 %), KLD −0.127 (−9.8 %), AUC-J +0.0065, sAUC +0.030
(TA.1).

**Against published validation numbers (class B).** Best published value per metric among the peer-reviewed sources
read:

| Metric | Best published (model, source) | SURGE-D ViT-g single | Δ (relative) | SURGE-D ViT-g ensemble | Δ (relative) |
|---|---|---|---|---|---|
| CC | 0.554 (TMFI-Net) [S: ViNet-S] | 0.6003 | +0.046 (+8.4 %) | 0.6029 | +0.049 (+8.8 %) |
| NSS | 3.201 (TMFI-Net) [S: ViNet-S] | 3.457 | +0.256 (+8.0 %) | 3.486 | +0.285 (+8.9 %) |
| SIM | 0.428 (TMFI-Net) [S: ViNet-S] | 0.4531 | +0.025 (+5.9 %) | 0.4574 | +0.029 (+6.9 %) |
| AUC-J | 0.924 (TMFI-Net, THTD-Net) [S: ViNet-S] | 0.9308 | +0.007 (+0.7 %) | 0.9314 | +0.007 (+0.8 %) |

The numbers in this table are taken from the newest peer-reviewed models' own papers: ViNet-E 0.549 [O], UniST
0.541 [O] and DiffSal (video only) 0.533 [O]. The SURGE-D ViT-L single model (0.5836) is also above every published
validation CC in this set.

**Verdict on DHF1K validation.** SURGE-D ViT-g is the best result among all *peer-reviewed* validation numbers that
could be read, on CC, NSS, SIM and AUC-J. The margin over the best published CC (+0.046 to +0.049) exceeds the known
cross-paper spread (≤0.03). Only the SalFoM comparison is same-evaluator and paired, and it survives the baseline's
best blur.

### 1.2 DHF1K test (official server): class C, context only

SURGE-D has **no test-set result** (predictions prepared, scores pending; p. 52). The best verified test results are
SalFoM's CC 0.569 and NSS 3.353 [O], then THTD-Net (0.547) and TMFI-Net (0.546) [S: SalFoM] (Table 8b). Whether
SURGE-D leads on the official split is **not established**.

### 1.3 UCF Sports (standard 103/47 split): class D

| Comparison | CC | NSS | SIM | AUC-J |
|---|---|---|---|---|
| SURGE-D ViT-g ensemble against ViNet-E (two-model ensemble, best published) [T; O] | 0.735 vs 0.744: **−0.009 (−1.2 %)** | 4.085 vs 4.156: −0.071 (−1.7 %) | 0.5857 vs 0.587: −0.001 | 0.9449 vs 0.941: **+0.004** |
| SURGE-D ViT-g single against ViNet-A (best single published) | 0.7305 vs 0.734: −0.0035 | 4.061 vs 4.108: −0.047 | 0.5831 vs 0.586: −0.003 | 0.9442 vs 0.940: +0.004 |

Against older SOTA, SURGE-D ViT-g single beats:
- TMFI-Net (0.707) by +0.024 CC (+3.3 %) and THTD-Net (0.711) by +0.020 [S: ViNet-S];
- UniST (0.706) [O];
- SalFoM (0.631) [O] by +0.100 (+15.8 %).

**Verdict on UCF Sports.** Competitive, but **not SOTA**. ViNet-E and ViNet-A, whose encoders are pretrained for
action recognition or action localisation, are level or ahead on CC, NSS and SIM. SURGE-D is ahead on AUC-J.

### 1.4 Six audio-visual datasets (STAViS folds): class D. SURGE-D uses no audio.

CC of the ViT-g ensemble against the strongest verified published models (fold means; Table 8d):

| Dataset | SURGE-D ViT-g ens. | ViNet-E (visual ens.) [O] | DiffSal (audio-visual) [O] | Δ vs best published | Δ relative |
|---|---|---|---|---|---|
| DIEM | **0.729** | 0.701 | 0.660 | +0.028 vs ViNet-E (AVRSP 0.685 [S: thesis]) | +4.0 % |
| Coutrot1 | **0.657** | 0.614 | 0.638 | +0.019 vs DiffSal | +3.0 % |
| Coutrot2 | 0.819 | 0.854 (ViNet-A **0.862**) | 0.835 | **−0.043** vs ViNet-A | −5.0 % |
| ETMD | **0.676** | 0.632 | 0.652 | +0.024 vs DiffSal | +3.7 % |
| AVAD | **0.740** | 0.729 | 0.738 | +0.002 vs DiffSal (a tie) | +0.3 % |
| SumMe | **0.590** | NR | 0.572 | +0.018 vs DiffSal | +3.1 % |

**Other metrics** [T; O]:
- **NSS**: above DiffSal on all six datasets (+0.7 % on ETMD to +8.6 % on DIEM), and above ViNet-E on four of five
  (below on Coutrot2).
- **AUC-J**: above both on every dataset where they report.
- **SIM is mixed**: lower than DiffSal on Coutrot1 (−0.011), Coutrot2 (−0.024) and AVAD (−0.006), and lower than
  ViNet-E on Coutrot2.

**The single-model comparison** is the fairer one against single published models. SURGE-D ViT-g single CC
(0.714 / 0.648 / 0.820 / 0.672 / 0.731 / 0.586) against the single DiffSal: +0.054, +0.010, −0.015, +0.020, −0.007
and +0.014.

**Verdict on the audio-visual datasets.** These results are strong for a visual-only model:
- best published CC and NSS on DIEM, Coutrot1, ETMD and SumMe;
- a tie on AVAD;
- a clear deficit on Coutrot2, where gaze follows the speaker.

The margins (0.02–0.03) are about the size of the cross-paper disagreement. These are therefore **competitive
results, not an established state of the art**, as the thesis itself says (p. 55).

### 1.5 Hollywood-2: class E, not comparable

SURGE-D ViT-g ensemble: CC 0.688 on the Vig gaze [T]. The published numbers on the standard ground truth (UniST
0.777 [O], ViNet-E 0.766 [O], TMFI-Net 0.739 [S], SalFoM 0.709 [O]) **must not** be compared with it. SURGE-D's
standing on Hollywood-2 is unknown.

### 1.6 Consistency across datasets and metrics

| Dataset | CC | NSS | SIM | AUC-J | KLD | Standing |
|---|---|---|---|---|---|---|
| DHF1K val | best | best | best | best | best vs SalFoM (others NR) | leading (class B), paired win over SalFoM (A) |
| DHF1K test | — | — | — | — | — | unknown |
| UCF Sports | −0.009 vs ViNet-E | lower | ≈ | higher | NR | competitive, not leading |
| DIEM, Coutrot1, ETMD, SumMe | best | best | mixed | best | NR | leading but within cross-paper spread |
| AVAD | tie | best | lower than DiffSal | best | NR | tie |
| Coutrot2 | lower | lower | lower | ≈ | NR | weaker |
| Hollywood-2 | — | — | — | — | — | not comparable |

The gains are **consistent on free-viewing data** (DHF1K, DIEM, Coutrot1, SumMe, ETMD) and **absent on task- or
speaker-driven gaze** (UCF Sports, Coutrot2). On SIM the advantage is the least consistent.

---

## 2. Computational efficiency

### 2.1 Measured and reported costs (class F)

All SURGE-D and SalFoM numbers: RTX A5000, bf16 autocast, model forward pass only, random input already on the GPU.
"Third of three runs" refers to the thesis's timing protocol [M; T5.12–5.13, pp. 57–58]. "Per output frame" divides by
the number of maps a forward pass yields.

| Model | Total params | Trained params | On-disk weights | Compute per output frame | Time per output frame (fps), batch 1 / 8 | Peak GPU memory, batch 1 / 8 | Input | Maps per pass |
|---|---|---|---|---|---|---|---|---|
| SURGE-D ViT-L, offline | 333.0 M [M] | 6.99 M | 1,243.5 MB encoder¹ + 26.7 MB decoder [M] | 114 GFLOPs ≈ 57 GMACs [T][D] | 5.5 ms (181) / 3.2 ms (313) [M] | 2.08 / 3.85 GB [M] | 16 frames, stride 2 (~1 s), 256×256 squashed | 16 |
| SURGE-D ViT-g, offline | 1,041.9 M [M] | 7.39 M | 3,946.6 MB encoder¹ + 28.2 MB decoder [M] | 328 GFLOPs ≈ 164 GMACs | 10.3 ms (97) / 7.7 ms (130) | 6.16 / 8.21 GB | same | 16 |
| SURGE-D ViT-L, causal | 333.0 M | 6.99 M | as above | 1,822 GFLOPs ≈ 911 GMACs | 88.5 ms (11.3) / 51.2 ms (19.5) | as above | 16 past frames | 1 |
| SURGE-D ViT-g, causal | 1,041.9 M | 7.39 M | as above | 5,255 GFLOPs ≈ 2,628 GMACs | 165.5 ms (6.0) / 123.5 ms (8.1) | as above | 16 past frames | 1 |
| SalFoM (released) | 402.9 M [M] | 402.9 M (end-to-end fine-tuning [O]) | 1,537.4 MB [M] | ≥2,866 GFLOPs ≈ 1,433 GMACs (encoder only)² [M] | 227.0 ms (4.4) / 202.4 ms (4.9) [M] | 3.58 / 14.20 GB [M] | 16 consecutive frames, 224×224 [O] | 1 |
| TMFI-Net | 61.34 M [S: ViNet-S] | NR | 234 MB [S] | NR | NR | NR | NR | 1 |
| THTD-Net | 57.67 M [S: ViNet-S] | NR | 220 MB [S] | NR | NR | NR | NR | 1 |
| STSANet | 168.56 M [S: ViNet-S] | NR | 643 MB [S] | NR | NR | NR | NR | 1 |
| ViNet-E | 48.19 M [O] | NR | 183.84 MB [O] | NR | ~90 fps, batch 1, RTX 4090 [O] | NR | 32 frames | 1 |
| ViNet-A | 38.69 M [O] | NR | 147.6 MB [O] | NR | ~120 fps, batch 1, RTX 4090 [O] | NR | 32 of a 64-frame window | 1 |
| ViNet-S | 9.5 M [O] | NR | 36.24 MB [O] | NR | ~200 fps (batch 1) / 1,070 fps (batch 8), RTX 4090 [O] | NR | 32 consecutive frames | 1 |
| DiffSal (audio-visual) | 76.57 M [O] | NR | NR | 187.31 G³ [O] | NR | NR | 16 frames, 224×384, plus audio | 1 |
| DiffSal (video only) | 70.54 M [O] | NR | NR | 161.06 G³ [O] | NR | NR | 16 frames, 224×384 | 1 |
| CASP-Net | 51.62 M [S: DiffSal] | NR | NR | 283.35 G³ [S] | NR | NR | NR | NR |
| ViNet (2021) | 31.1–33.97 M (sources disagree) [S] | NR | 124 MB [S: ViNet-S] | 115.3 GMACs [S: TinyHD] | NR | NR | NR | 1 |
| TASED-Net | 21.26–21.5 M [S] | NR | 82 MB [S] | 91.8 GMACs [S: TinyHD] | NR | NR | NR | 1 |
| HD2S | 29.8 M [S: TinyHD] | NR | NR | 11.1 GMACs [S: TinyHD] | NR | NR | NR | NR |
| STAViS | 20.76 M [S: DiffSal] | NR | NR | 15.31 G³ [S] | NR | NR | NR | NR |
| UNISAL | 3.71–4.06 M [S] | NR | 15.5 MB [S: ViNet-S] | 1.2 GMACs [S: TinyHD] | NR | NR | NR | NR |
| TinyHD-S (single output) | 3.94 M [O] | NR | NR | 5.57 GMACs [O] | NR | NR | 16 frames, 192×256 | 1 |
| TinyHD-M (multi-output) | 3.92 M [O] | NR | NR | 0.50 GMACs (7.95 per 16-frame clip) [O] | NR | NR | 16 frames, 192×256 | 16 |
| UniST | NR [O] | NR | NR | NR | NR | NR | 16 frames, 224×384 | central frames |

**Notes on the table.**
1. The V-JEPA 2 `model.safetensors` files also contain the predictor, which SURGE-D does not use. Weights are fp32.
2. SalFoM's figure counts the encoder only, with attention products included; the decoder is not counted.
3. DiffSal's table labels these values "FLOPs". For TASED-Net (91.80) and ViNet (115.31) they equal TinyHD's
   **GMACs** per output frame (1,468.0/16 and 1,844.5/16). DiffSal's column therefore most likely counts
   multiply–accumulates. Whether DiffSal's figure includes all four denoising steps is not stated.

SURGE-D's FLOPs follow the thesis's convention (one multiply–add = 2 FLOPs), so GMACs ≈ GFLOPs / 2 [D].

### 2.2 What SURGE-D does better on cost, and what it does not

1. **The offline per-frame cost is low for its accuracy (the amortisation advantage).** One pass over a 16-frame clip
   yields maps for 16 original frames (Alg. 4.1, p. 39). Per output frame:
   - SURGE-D ViT-L (≈57 GMACs) is **cheaper than TASED-Net (92), ViNet (115), DiffSal (161–187) and CASP-Net (283)**,
     and **≈25× cheaper than SalFoM's encoder** (≥1,433).
   - SURGE-D ViT-g (≈164 GMACs) costs about the same as DiffSal (video only, 161) while scoring CC 0.600 against 0.533
     on DHF1K validation.
   - Measured time against SalFoM on the same GPU: **22–26× faster per output frame** (ViT-g) and 41–64× (ViT-L)
     across three runs, criterion fixed in advance [T, p. 58].
2. **Offline throughput is real-time on one workstation GPU.** At batch 1, ViT-g runs at 97 fps and ViT-L at 181 fps
   per output frame. ViNet-E (90 fps) and ViNet-S (~200 fps) are in the same class, measured on a faster RTX 4090
   [O], so the numbers are not directly comparable.
3. **Ensembles could be cheap, but this has not been demonstrated** [D]. All seeds of a SURGE-D ensemble share
   *identical* frozen encoder weights, so an ensemble needs one encoder pass and three decoder passes. Measured pass
   time minus encoder time gives the decoder share [M; T5.13 run 3]:

   | Configuration (estimates) | Compute per output frame | Time per output frame, batch 1 / 8 | Relative to SalFoM |
   |---|---|---|---|
   | ViT-g ensemble, offline | ≈ 350 GFLOPs (≈175 GMACs), +6.4 % over one model | ≈ 11.7 / 8.6 ms | — |
   | ViT-g ensemble, causal (the configuration that beats SalFoM by +0.029 CC under the past-only protocol) | ≈ 5.6 TFLOPs (≈1.95× SalFoM's encoder) | ≈ 188 / 138 ms | **0.83× / 0.68× of SalFoM's time** |

   The thesis's code runs each member's encoder separately, so the measured ensemble cost is three times a single
   model. **This qualifies M5 of `critical-review.md`:** its figures (5.5× FLOPs, 1.8–2.2× time for the causal
   ensemble) describe the implementation as run, not the design.
4. **Parameter count and memory are where SURGE-D is worse.**
   - It is 5–7× larger than TMFI-Net, ViNet-E or THTD-Net (333 M, against 48–61 M), and the ViT-g model is 17–22×
     larger. Lightweight models (UNISAL, TinyHD, ViNet-S: 4–9.5 M) are smaller by about two orders of magnitude.
   - Its encoder files (1.24–3.95 GB) are far larger than any competitor's except SalFoM's (1.54 GB).
   - "7 M trained parameters" describes the training cost, **not the deployed size**.
5. **The causal (streaming) mode is not real-time.** Single causal models run at 6–20 fps on the A5000, whereas ViNet-S
   and ViNet-A, which are causal by design, run at 120–200 fps on an RTX 4090 [O]. The offline mode must also buffer up
   to 15 future frames (~0.5 s at 30 fps) before it can output a frame. **SURGE-D's efficiency advantage is an
   offline, batch-processing advantage.**
6. **Lightweight models dominate on compute.** TinyHD-M (0.5 GMACs per frame, multi-output, DHF1K *test* CC 0.493)
   and UNISAL (1.2 GMACs, test CC 0.490) are 50–300× cheaper per frame, at much lower accuracy (validation CC 0.49 in
   UniST's table, against 0.58–0.60 for SURGE-D).

### 2.3 Training cost and hardware

| Model | Training hardware | Training schedule | What is trained | Reported time |
|---|---|---|---|---|
| SURGE-D | 1× RTX A5000 (24 GB); ViT-g fits in 10 GB at batch 8 (p. 43) | 3 epochs × 4,320 clips (540 videos × 8 clips), batch 8 | decoder only (7 M) | **~15 min (ViT-L), ~31 min (ViT-g)** per model, plus holdout evaluation (p. 42) [T] |
| SalFoM | NR | batch 1, Adam 1e-5, early stopping on the validation set [O] | encoder + decoder (403 M) | NR |
| DiffSal | 4× RTX 4090 [O] | ≤ 5 epochs, batch 20; video branch first pretrained on DHF1K [O] | full model (76.6 M) | NR |
| UniST | 2× RTX 4090 [O] | batch 6, Adam 1e-4; epochs NR [O] | full model | NR |
| TinyHD | NR | 200 epochs, SGD, batch 12, plus Kinetics-400 pseudo-labels from an HD2S teacher [O] | full student (3.9 M) | NR |
| ViNet-S / ViNet-A | NR | batch 8 / 6, Adam 1e-4; epochs NR [O] | NR | NR |
| TMFI-Net, THTD-Net, STSANet | NR in the sources consulted | NR | NR | NR |

SURGE-D's **own** training cost is, as far as can be verified, the lowest among the accurate models: under one
GPU-hour per model on one workstation card. The ViT-g seed-0 run took about 49 minutes of wall-clock time, three
holdout evaluations of 6–7 minutes each included, at 10 GB of GPU memory (`runs/vitg_s0/log.jsonl`, `stdout.log`) [M].
Two caveats:
- Most competitors do not report training time, so this is a qualitative comparison.
- SURGE-D inherits V-JEPA 2's pre-training (over 1 M hours of internet video [arXiv:2506.09985]). That cost is
  enormous and hidden; SalFoM likewise inherits UMT's pre-training.

---

## 3. Architectural advantages

| Component | What it does | Evidence that it matters | Novel? |
|---|---|---|---|
| **Frozen V-JEPA 2 encoder** (self-supervised feature prediction, no labels) | supplies all representational capacity; nothing is back-propagated into it | **Strong.** Same decoder: V-JEPA 2 0.578 > V-JEPA 2.1 0.569 > UMT-L 0.546 holdout CC (one seed, T5.2). Frozen UMT-L + SURGE-D decoder scores 0.545 on validation, **below** SalFoM (T5.7), so the gain comes from the encoder, not from freezing. ViT-g beats ViT-L by +0.017 validation CC (T5.6). | First *frozen* use of a JEPA video encoder for this task among peer-reviewed work read. The concurrent PredJSal (NTIRE 2026, arXiv) also uses V-JEPA2 ViT-g with a 3D-conv decoder. Frozen backbone + small readout is DeepGaze II/IIE's paradigm for images. |
| **Multi-frame output with central-frame selection** (16 maps per pass, central 8 sampled positions, interpolation) | amortises the encoder over 16 output frames | **Strong for cost:** measured 22–26× time and ≈9× FLOPs advantage over SalFoM. Accuracy cost: none detectable; offline ≈ causal (T5.8). | TinyHD-M (WACV 2023) already predicts 16 maps per pass; SURGE-D is the first to pair the idea with a foundation encoder in the work read. |
| **Four-tap 3D simple feature pyramid** (blocks 6/12/18/24 or 10/20/30/40, ViTDet-style) | multi-scale maps at 1/4, 1/8 and 1/16 from a plain ViT | **Untested.** No ablation (critical review, M4). The thesis suggests the taps may supply the local detail V-JEPA 2 lacks (p. 44); untested. | Adapts ViTDet / DPT ideas to 3D. |
| **Top-down (2+1)D decoder** (R(2+1)D residual blocks, no attention) | temporal mixing at each scale at low cost | **Untested.** Whether temporal context helps at all is unknown (no frame-shuffle test). | Standard blocks (Tran et al. 2018). |
| **Learned Gaussian centre prior** (41 parameters) | input-independent centre bias | **None measurable:** equivalent to its absence, −0.0010 [−0.0026, +0.0005] (T5.14) | Cornia et al. 2018-style |
| **Deep supervision** (two auxiliary heads) | training signal at 1/16 and 1/8 | **None measurable:** −0.0010 [−0.0023, +0.0003] | standard |
| **Seed ensembling** (decoders on a shared frozen encoder) | variance reduction | +0.007 (ViT-L), +0.0025 (ViT-g) validation CC, slight sAUC loss (TA.8). Potentially near-free at inference (§2.2.3, derived). | Ensembling is standard; near-free *shared-encoder* ensembling is specific to frozen encoders. |
| **Holdout-tuned output blur** | matches map smoothness to the density | Broad optimum: ≤0.003 CC between σ = 0 and 12 (Fig. 5.5). Blur also matters for baselines: SalFoM +0.010 CC at σ = 16 (T5.7). | standard practice |

**Fewer components, lower complexity.** Compared with its closest competitors, SURGE-D's decoder has no attention,
no recurrence, no optical flow, no audio, no diffusion sampling and no distillation:
- SalFoM uses a heterogeneous decoder with a locality-aware spatio-temporal transformer.
- TMFI-Net uses a semantic-guided encoder and an attention decoder.
- DiffSal uses a diffusion U-Net with 4 denoising steps.
- TinyHD uses multiple heterogeneous decoders distilled from a teacher.

With this simpler decoder SURGE-D still reaches higher DHF1K validation accuracy. **But the simplicity is bought with
the largest encoder in the comparison.** Which decoder parts are necessary is not established.

**Spatial features, temporal modelling, attention, fusion and multi-scale structure, specifically.**

| Aspect | What SURGE-D does | Advantage over existing methods |
|---|---|---|
| Spatial features | V-JEPA 2's 16×16 token grid at 256² | Demonstrated, via the encoder comparison |
| Temporal modelling | inherited from the encoder's attention over 8 tubelets (~1 s), plus (2+1)D convolutions | Not demonstrated separately |
| Attention | only in the encoder | none; the decoder is attention-free |
| Feature fusion | concatenation of the two deepest taps plus top-down addition | standard |
| Multi-scale representation | 4 levels from a single-resolution ViT | standard |

---

## 4. Generalisation and robustness

**Transfer without fine-tuning (zero-shot from DHF1K)** [T; T5.9–5.11]:
- **UCF Sports:** ViT-L CC 0.608, against 0.725 after fine-tuning. Task-driven gaze needs adaptation.
- **Six audio-visual datasets** (ViT-L, CC): DIEM 0.675, Coutrot1 0.600, Coutrot2 0.645, ETMD 0.588, AVAD 0.692,
  SumMe 0.551.
  - Zero-shot SURGE-D already exceeds several *fine-tuned* audio-visual models: DIEM against CASP-Net 0.655 and
    DiffSal 0.660; SumMe against CASP-Net 0.499; Coutrot1 against CASP-Net 0.561; AVAD level with CASP-Net 0.691
    [S: DiffSal; O]. Class D caveats apply.
  - Competitors rarely report zero-shot results, so there is no head-to-head comparison.
- **Hollywood-2 (Vig gaze):** 0.632 zero-shot, against 0.680 fine-tuned.

**Across video characteristics.** Strong on varied free-viewing content: DHF1K, DIEM (advertisements, trailers,
documentaries), Coutrot1 (moving objects, landscapes, faces), ETMD (movies), SumMe (user videos). Weak where gaze
follows a speaker (Coutrot2, and the panel-discussion failure case on p. 57). The gain over SalFoM is broad (79 of 100
validation videos), not driven by a few videos.

**Robustness to scene cuts, camera motion, dynamic objects and complex backgrounds: not evaluated.** The thesis
contains no perturbation, cut, camera-motion or background analysis, only qualitative examples. Possible weak points:
- squashing 16:9 frames to 256² (implicit position cues);
- 64×64 output resolution (small faces and objects);
- a fixed one-second window (narrative attention);
- frames-based stride (other frame rates);
- windows that span shot cuts.

**Limits of generalisation.**
- Task-driven datasets need fine-tuning.
- Audio-driven attention is out of reach: the audio fusion tested did not help (T5.16).
- No evaluation on unseen datasets beyond the thesis's nine test sets (for example LEDOV, or the NTIRE 2026 videos).

---

## 5. Training efficiency and data requirements

- **Training data.**
  - SURGE-D uses **540 DHF1K videos** (60 held out for selection), 3 epochs, no static images (SALICON did not help,
    T5.14), no pseudo-labels, no audio.
  - Competitors use the 600 DHF1K training videos, sometimes with more:
    - TinyHD adds unlabeled Kinetics-400 with teacher pseudo-labels [O];
    - DiffSal and UniST pretrain on DHF1K before fine-tuning [O];
    - SalFoM early-stops on the DHF1K validation set [O].
  - **SURGE-D needs no extra annotation or modality.**
- **Dependence on pretrained backbones.** SURGE-D depends **entirely** on V-JEPA 2: self-supervised, no human labels,
  but over 1 M hours of video [arXiv:2506.09985]. The competitors' backbones are:
  - SalFoM: UMT-L, initialised from a checkpoint whose name indicates K710 pre-training and K710/K400 fine-tuning,
    i.e. supervised action labels [SalFoM README];
  - ViNet-A: SlowFast trained for action localisation;
  - TMFI-Net: Video Swin;
  - DiffSal: Kinetics-pretrained MViTv2 plus AudioSet VGGish [O].

  SURGE-D's backbone is the only one in this set trained **without any human labels**, but it is also the largest.
- **Training complexity.** Simple. One stage: decoder only, AdamW, cosine schedule, a 3-epoch schedule fixed from gate
  G1, best epoch selected on the holdout.
- **Sensitivity to hyperparameters.**
  - **Low** sensitivity to the output blur (≤0.003 CC over σ 0–12).
  - Seed standard deviation 0.003 CC.
  - **High** sensitivity to training length: the holdout CC peaks at epoch 2–3 and then declines (Fig. 5.1). Early
    stopping is therefore essential.
  - Learning rates were never tuned.
  - Adapting the encoder (LoRA) hurt, in the one setting tried.
- **Data efficiency.** Validation CC rises with data (holdout CC 0.5645 / 0.5729 / 0.5783 at 135 / 270 / 540
  videos). Whether 540 videos is a limit is open (T5.15).
- **Feature caching (not exploited, derived).** Because the encoder is frozen, its features could be cached once and
  reused across seeds, epochs and hyperparameter sweeps. A fixed clip grid would be needed, plus cached flipped
  features if flip augmentation is kept.

---

## 6. Practical advantages

| Aspect | SURGE-D | Comparison |
|---|---|---|
| Source code | Public repository with model, training, evaluation, splits, holdout lists, decision log and exact commands (App. B) | ViNet-S/A, SalFoM and TMFI-Net release code. TMFI-Net's README says its complete code awaits publication. SalFoM's repository has no test script. |
| Pretrained weights | Encoder public (MIT). **Trained decoders not released**, though only 27–28 MB each. | SalFoM, TMFI-Net (Baidu Netdisk) and ViNet release trained weights. |
| Reproducibility | Training takes 15–31 min per model on one 24 GB GPU; 3-seed results reported; fixed holdout. Main configuration defaults to V-JEPA 2.1 and needs command-line overrides. | Most competitors do not report training time or seeds. |
| Deployment | Offline batch processing on one workstation GPU (2–8 GB VRAM, 97–313 fps per output frame) | Edge or mobile: ViNet-S, UNISAL and TinyHD are far better suited. |
| Limited compute | Training: yes, one GPU and under an hour. Inference: needs a GPU with ≥ 3 GB (ViT-L) or ≥ 7 GB (ViT-g) and 1.2–3.9 GB of weights. | Lightweight models win. |
| Applications | Offline video compression, summarisation, content analysis and dataset annotation, where future frames are available. Not live or streaming use. | Causal models (ViNet family) suit streaming. |

---

## 7. Limitations and trade-offs

**Where competing SOTA models are better:**
- UCF Sports: ViNet-E and ViNet-A on CC, NSS and SIM.
- Coutrot2: ViNet-A, ViNet-E and DiffSal.
- DHF1K test: SalFoM's 0.569 is the best verified result; SURGE-D has no result.
- Hollywood-2: UniST and ViNet-E on the standard ground truth; SURGE-D has no comparable result.
- Model size: every listed competitor except SalFoM is smaller.
- Causal throughput and latency: the ViNet family.
- Compute per frame: TinyHD, UNISAL, HD2S and STAViS.

**Do the gains cost parameters, FLOPs, memory or time?**

| Resource | Cost of SURGE-D's gains |
|---|---|
| Parameters | **Yes, a large cost.** 0.33–1.04 B per model, against 0.04–0.17 B for most SOTA models. |
| On-disk size | 1.2–4.0 GB of weights. |
| Inference memory | 2–8 GB. |
| FLOPs per frame, offline | **No:** lower than most 3D-CNN, transformer and diffusion competitors, and far lower than SalFoM. |
| FLOPs per frame, causal | **Yes:** 911–2,628 GMACs per frame. |
| Inference time, offline | **No:** real-time throughput, though not low-latency. |

**Architectural and methodological limits** (detailed in `critical-review.md`):
- The decoder design is unablated (M4).
- The DHF1K comparison is validation-only with one re-scored baseline (M2, M3).
- Pre-training data may overlap with DHF1K, which is drawn from YouTube (D8).
- The output resolution is 64×64.

**Demonstrated advantages compared with claims the experiments do not test:**

| Statement | Status |
|---|---|
| Higher DHF1K validation accuracy than SalFoM on all six metrics, also against its best blur and under the past-only protocol | **Demonstrated** (class A, paired) |
| Higher DHF1K validation accuracy than all readable peer-reviewed results | **Demonstrated approximately** (class B, cross-evaluator) |
| 22–26× faster per output frame than SalFoM (offline, ViT-g, as released) | **Demonstrated** (measured; implementation-dependent) |
| ≈ 9× less encoder compute per output frame than SalFoM (offline) | **Demonstrated** (counted) |
| Future frames are not the source of the advantage | **Demonstrated** (causal ≈ offline, T5.8) |
| The V-JEPA 2 encoder is the source of the gain | **Demonstrated for L-size against UMT-L** (one seed in G1; frozen UMT-L check on validation) |
| Above the best verified peer-reviewed results on 4 of 6 audio-visual datasets | **Approximate** (class D, ensemble, margins within the cross-paper spread) |
| The decoder design (taps, pyramid, (2+1)D) contributes | **Not tested** |
| The model exploits temporal dynamics | **Not tested** |
| Robustness to cuts, camera motion and clutter | **Not tested** |
| Shared-encoder ensembles at near single-model cost | **Not tested** (derived here; the code does not do it) |
| SOTA on the official DHF1K test set | **Not available** |

---

## 8. Comprehensive SOTA comparison tables

Each table is restricted to one dataset and split. Training cost and FPS columns repeat Table 2.1 and 2.3 where
known. Abbreviations: Ens = ensemble; RT = RTX.

### 8a. DHF1K validation (100 videos): class A for SalFoM, class B for the rest

| Model | Year | Venue | Dataset | AUC-Judd | sAUC | NSS | CC | SIM | KLD | Parameters | FLOPs (per output frame) | FPS | Training cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SURGE-D ViT-g, 3-seed Ens** [T✓] | 2026 | MSc thesis (not peer-reviewed) | DHF1K-val | **0.9314** | 0.7680ᵃ | **3.486** | **0.6029** | **0.4574** | **1.160** | 3 × 1,042 M (3 × 7.39 M trained) | ≈ 985 GFLOPs as run [D]; ≈ 350 with a shared encoder [D] | 32 (batch 1) as run [D] | 3 × ~31 min, 1× RT A5000 |
| **SURGE-D ViT-g, single** [T✓] | 2026 | MSc thesis | DHF1K-val | 0.9308 | **0.7709**ᵃ | 3.457 | 0.6003 | 0.4531 | 1.170 | 1,042 M (7.39 M trained) | 328 GFLOPs (≈ 164 GMACs) [T] | 97 (batch 1) / 130 (batch 8), RT A5000 [M] | ~31 min, 1× RT A5000 [T] |
| SURGE-D ViT-L, 3-seed Ens [T✓] | 2026 | MSc thesis | DHF1K-val | 0.9302 | 0.7614ᵃ | 3.418 | 0.5910 | 0.4486 | 1.187 | 3 × 333 M | ≈ 342 as run; ≈ 135 shared [D] | 60 (batch 1) as run [D] | 3 × ~15 min |
| SURGE-D ViT-L, single [T✓] | 2026 | MSc thesis | DHF1K-val | 0.9283 | 0.7636ᵃ | 3.349 | 0.5836 | 0.4492 | 1.222 | 333 M (6.99 M trained) | 114 GFLOPs (≈ 57 GMACs) | 181 / 313 [M] | ~15 min |
| SalFoM, released, re-scored [T✓][M] | 2024 | ICPR | DHF1K-val | 0.9243 | 0.7406ᵃ | 3.326 | 0.5639 | 0.4130 | 1.297 | 402.9 M [M] | ≥ 2,866 GFLOPs, encoder only [M] | 4.4 / 4.9, RT A5000 [M] | end-to-end, batch 1; NR |
| SalFoM, best blur σ = 16 (upper bound) [T✓] | — | — | DHF1K-val | 0.9277 | 0.7557ᵃ | 3.265 | 0.5741 | 0.4009 | 1.280 | as above | as above | as above | — |
| TMFI-Net [S: ViNet-S] | 2023 | TCSVT | DHF1K-val | 0.924 | NR | 3.201 | 0.554 | 0.428 | NR | 61.34 M [S] | NR | NR | NR |
| TMFI-Net [S: UniST; DiffSal supp.] | 2023 | TCSVT | DHF1K-val | 0.918 | NR | 3.006 | 0.524 | 0.410 | NR | — | — | — | — |
| THTD-Net [S: ViNet-S] | 2024 | VISAPP | DHF1K-val | 0.924 | NR | 3.188 | 0.553 | 0.425 | NR | 57.67 M [S] | NR | NR | NR |
| ViNet-E (Ens of 2) [O] | 2025 | ICASSP | DHF1K-val | 0.922 | NR | 3.134 | 0.549 | 0.409 | NR | 48.19 M [O] | NR | ~90 (batch 1), RT 4090 [O] | NR |
| UniST [O] | 2025 | TCSVT | DHF1K-val | 0.920 | NR | 3.113 | 0.541 | 0.423 | NR | NR | NR | NR | 2× RT 4090, batch 6 [O] |
| STSANet [S: ViNet-S] | 2023 | TMM | DHF1K-val | 0.920 | NR | 3.082 | 0.539 | 0.411 | NR | 168.56 M [S] | NR | NR | NR |
| DiffSal, video-only [O, supp.] | 2024 | CVPR | DHF1K-val | 0.918 | NR | 3.066 | 0.533 | 0.405 | NR | 70.54 M [O] | 161.06 G (likely MACs)ᵇ | NR | 4× RT 4090ᶜ [O] |
| ViNet-S [O] | 2025 | ICASSP | DHF1K-val | 0.919 | NR | 3.008 | 0.529 | 0.399 | NR | 9.5 M [O] | NR | ~200 / 1,070, RT 4090 [O] | NR |
| ViNet-A [O] | 2025 | ICASSP | DHF1K-val | 0.916 | NR | 3.019 | 0.525 | 0.399 | NR | 38.69 M [O] | NR | ~120, RT 4090 [O] | NR |
| ViNet [S: ViNet-S] | 2021 | IROS | DHF1K-val | 0.919 | NR | 2.957 | 0.521 | 0.388 | NR | 31.1–33.97 M [S] | 115.3 GMACs [S] | NR | NR |
| TinyHD-S [S: UniST] | 2023 | WACV | DHF1K-val | 0.907 | NR | 2.873 | 0.492 | 0.388 | NR | 3.94 M [O] | 5.57 GMACs [O] | NR | 200 epochs + pseudo-labels [O] |
| UNISAL [S: ViNet-S] | 2020 | ECCV | DHF1K-val | 0.901 | NR | 2.77 | 0.490 | 0.390 | NR | 3.71–4.06 M [S] | 1.2 GMACs [S] | NR | NR |
| TASED-Net [S: ViNet-S] | 2019 | ICCV | DHF1K-val | 0.894 | NR | 2.706 | 0.481 | 0.362 | NR | 21.26–21.5 M [S] | 91.8 GMACs [S] | NR | NR |

ᵃ The thesis's own sAUC implementation; comparable only within the thesis.
ᵇ See note 3 under Table 2.1.
ᶜ For the audio-visual training stage.

### 8b. DHF1K test (official server): class C, for context only

| Model | Year | Venue | Dataset | AUC-Judd | sAUC | NSS | CC | SIM | KLD | Parameters | FLOPs | FPS | Training cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SURGE-D** | 2026 | MSc thesis | DHF1K-test | **not available (submission pending)** | — | — | — | — | — | — | — | — | — |
| SalFoM [O] | 2024 | ICPR | DHF1K-test | 0.922 | 0.735 | 3.353 | 0.569 | 0.420 | NR | 402.9 M [M] | see 8a | see 8a | NR |
| THTD-Net [S: SalFoM] | 2024 | VISAPP | DHF1K-test | 0.915 | 0.729 | 3.138 | 0.547 | 0.406 | NR | 57.67 M [S] | NR | NR | NR |
| TMFI-Net [S: SalFoM] | 2023 | TCSVT | DHF1K-test | 0.915 | 0.730 | 3.146 | 0.546 | 0.406 | NR | 61.34 M [S] | NR | NR | NR |
| STSANet [S: SalFoM] | 2023 | TMM | DHF1K-test | 0.912 | 0.722 | 3.010 | 0.528 | 0.382 | NR | 168.56 M [S] | NR | NR | NR |
| ViNet [S: SalFoM; TinyHD] | 2021 | IROS | DHF1K-test | 0.908 | 0.729 | 2.872 | 0.511 | 0.381 | NR | see 8a | 115.3 GMACs | NR | NR |
| TinyHD-S [O] | 2023 | WACV | DHF1K-test | 0.909 | 0.714 | 2.921 | 0.505 | 0.396 | NR | 3.94 M | 5.57 GMACs | NR | 200 epochs |
| HD2S [S: SalFoM; TinyHD] | 2021 | IJCV | DHF1K-test | 0.908 | 0.700 | 2.812 | 0.503 | 0.406 | NR | 29.8 M [S] | 11.1 GMACs [S] | NR | NR |
| TinyHD-M (16 maps per pass) [O] | 2023 | WACV | DHF1K-test | 0.905 | 0.707 | 2.819 | 0.493 | 0.387 | NR | 3.92 M | 0.50 GMACs | NR | 200 epochs |
| UNISAL [S: SalFoM] | 2020 | ECCV | DHF1K-test | 0.901 | 0.691 | 2.776 | 0.490 | 0.390 | NR | 3.71–4.06 M | 1.2 GMACs | NR | NR |
| TASED-Net [S: SalFoM] | 2019 | ICCV | DHF1K-test | 0.895 | 0.712 | 2.667 | 0.470 | 0.361 | NR | 21.26–21.5 M | 91.8 GMACs | NR | NR |

### 8c. UCF Sports test (47 videos): class D

| Model | Year | Venue | Dataset | AUC-Judd | sAUC | NSS | CC | SIM | KLD | Parameters | FLOPs | FPS | Training cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ViNet-E (Ens of 2) [O] | 2025 | ICASSP | UCF-test | 0.941 | NR | **4.156** | **0.744** | **0.587** | NR | 48.19 M | NR | ~90 (RT 4090) | NR |
| **SURGE-D ViT-g, 3-seed Ens** [T] | 2026 | MSc thesis | UCF-test | **0.9449** | 0.8173ᵃ | 4.085 | 0.7350 | 0.5857 | 0.736 | 3 × 1,042 M | see 8a | see 8a | + 3 epochs of fine-tuning per seed |
| ViNet-A [O] | 2025 | ICASSP | UCF-test | 0.940 | NR | 4.108 | 0.734 | 0.586 | NR | 38.69 M | NR | ~120 | NR |
| SURGE-D ViT-L, 3-seed Ens [T] | 2026 | MSc thesis | UCF-test | 0.9430 | 0.8153ᵃ | 4.028 | 0.7331 | 0.5833 | 0.749 | 3 × 333 M | — | — | — |
| **SURGE-D ViT-g, single** [T✓] | 2026 | MSc thesis | UCF-test | 0.9442 | 0.8145ᵃ | 4.061 | 0.7305 | 0.5831 | 0.747 | 1,042 M | 328 GFLOPs | 97 / 130 | — |
| SURGE-D ViT-L, single [T] | 2026 | MSc thesis | UCF-test | 0.9406 | 0.8089ᵃ | 3.916 | 0.7252 | 0.5799 | 0.773 | 333 M | 114 GFLOPs | 181 / 313 | — |
| THTD-Net [S: ViNet-S; SalFoM] | 2024 | VISAPP | UCF-test | 0.933 | NR | 3.840 | 0.711 | 0.565 | NR | 57.67 M | NR | NR | NR |
| TMFI-Net [S: ViNet-S; SalFoM; UniST] | 2023 | TCSVT | UCF-test | 0.936 | NR | 3.863 | 0.707 | 0.565 | NR | 61.34 M | NR | NR | NR |
| UniST [O] | 2025 | TCSVT | UCF-test | 0.932 | NR | 3.718 | 0.706 | 0.576 | NR | NR | NR | NR | 2× RT 4090 |
| STSANet [S: SalFoM; UniST] | 2023 | TMM | UCF-test | 0.936 | NR | 3.908 | 0.705 | 0.560 | NR | 168.56 M | NR | NR | NR |
| ViNet-S [O] | 2025 | ICASSP | UCF-test | 0.930 | NR | 3.652 | 0.673 | 0.530 | NR | 9.5 M | NR | ~200 | NR |
| UNISAL [S: SalFoM] | 2020 | ECCV | UCF-test | 0.918 | NR | 3.381 | 0.644 | 0.523 | NR | 3.71–4.06 M | 1.2 GMACs | NR | NR |
| SalFoM [O] | 2024 | ICPR | UCF-test | 0.928 | NR | 3.543 | 0.631 | 0.516 | NR | 402.9 M | ≥ 2,866 GFLOPs | 4.4 / 4.9 | NR |
| TinyHD-S [O] | 2023 | WACV | UCF-test | 0.918 | NR | 3.280 | 0.624 | 0.510 | NR | 3.94 M | 5.57 GMACs | NR | 200 epochs |
| SURGE-D ViT-L, zero-shot (no UCF training) [T] | 2026 | MSc thesis | UCF-test | 0.9223 | 0.7453ᵃ | 3.128 | 0.6077 | 0.5033 | 1.113 | 333 M | — | — | none |

### 8d. Six audio-visual datasets (STAViS folds, fold means): class D

Each cell gives **CC / NSS / SIM / AUC-J**. "AV" = uses audio.

| Model | Year | Venue | Audio | DIEM | Coutrot1 | Coutrot2 | ETMD | AVAD | SumMe |
|---|---|---|---|---|---|---|---|---|---|
| **SURGE-D ViT-g, 3-seed Ens** [T] | 2026 | MSc thesis | no | **0.729** / **2.878** / **0.584** / **0.923** | **0.657** / **3.302** / 0.504 / **0.911** | 0.819 / 6.673 / 0.601 / **0.966** | **0.676** / **3.687** / **0.510** / **0.950** | **0.740** / **4.306** / 0.565 / **0.942** | **0.590** / **3.261** / **0.448** / **0.927** |
| SURGE-D ViT-g, single (CC only) [T] | 2026 | MSc thesis | no | 0.714 | 0.648 | 0.820 | 0.672 | 0.731 | 0.586 |
| SURGE-D ViT-L, zero-shot (CC only) [T] | 2026 | MSc thesis | no | 0.675 | 0.600 | 0.645 | 0.588 | 0.692 | 0.551 |
| ViNet-E (Ens of 2) [O] | 2025 | ICASSP | no | 0.701 / 2.840 / 0.566 / 0.913 | 0.614 / 3.085 / 0.465 / 0.905 | 0.854 / 6.762 / 0.628 / 0.962 | 0.632 / 3.437 / 0.468 / 0.943 | 0.729 / 4.167 / 0.547 / 0.938 | NR |
| ViNet-A [O] | 2025 | ICASSP | no | 0.675 / 2.742 / 0.547 / 0.908 | 0.600 / 3.033 / 0.459 / 0.900 | **0.862** / **6.800** / **0.638** / 0.961 | 0.623 / 3.379 / 0.458 / 0.941 | 0.709 / 4.094 / 0.534 / 0.933 | NR |
| DiffSal [O] | 2024 | CVPR | AV | 0.660 / 2.65 / 0.543 / 0.907 | 0.638 / 3.20 / **0.515** / 0.901 | 0.835 / 6.61 / 0.625 / 0.964 | 0.652 / 3.66 / 0.506 / 0.943 | 0.738 / 4.22 / **0.571** / 0.935 | 0.572 / 3.14 / 0.447 / 0.921 |
| CASP-Net [S: DiffSal] | 2023 | CVPR | AV | 0.655 / 2.61 / 0.543 / 0.906 | 0.561 / 2.65 / 0.456 / 0.889 | 0.788 / 6.34 / 0.585 / 0.963 | 0.620 / 3.34 / 0.478 / 0.940 | 0.691 / 3.81 / 0.528 / 0.933 | 0.499 / 2.60 / 0.387 / 0.907 |
| STAViS [S: DiffSal]ᵈ | 2020 | CVPR | AV | 0.579 / 2.26 / 0.482 / 0.883 | 0.472 / 2.11 / 0.393 / 0.868 | 0.734 / 5.28 / 0.511 / 0.958 | 0.569 / 2.94 / 0.425 / 0.931 | 0.608 / 3.18 / 0.457 / 0.919 | 0.422 / 2.04 / 0.337 / 0.888 |
| AVRSP (CC only) [S: thesis, not verified] | 2026 | TETCI | AV | 0.685 | NR | 0.837 | NR | NR | 0.494 (visual-only variant) |

ᵈ ViNet-S reports STAViS on Coutrot2 at CC 0.652; the two sources disagree by 0.08.

Efficiency of the audio-visual models: DiffSal 76.57 M and 187.31 G; CASP-Net 51.62 M and 283.35 G; STAViS 20.76 M and
15.31 G [O / S: DiffSal]. ViNet-E 48.19 M at ~90 fps; ViNet-A 38.69 M at ~120 fps [O].

### 8e. Hollywood-2: class E, not comparable

SURGE-D ViT-g ensemble on the Vig gaze: 0.9577 / 0.8199ᵃ / 3.907 / 0.6882 / 0.5104 / 0.911
(AUC-J / sAUC / NSS / CC / SIM / KLD) [T].

The literature uses the standard Mathe and Sminchisescu ground truth (UniST 0.777 [O], ViNet-E 0.766 [O], TMFI-Net
0.739 [S], SalFoM 0.709 [O]). The ground truths differ in observers, segmentation and task protocol, so **no row-wise
comparison is valid.**

---

## 9. Final assessment

### 9.1 Everything SURGE-D does better than existing SOTA models (with the evidence class)

1. **DHF1K validation accuracy.**
   - Paired: best on all six metrics against SalFoM's released model, also against SalFoM's best blur and when both
     see only past frames (class A).
   - Approximate: higher than every readable peer-reviewed validation result on CC, NSS, SIM and AUC-J, by +0.046 to
     +0.049 CC (+8 to 9 %; class B).
2. **Free-viewing audio-visual datasets without audio.** Highest CC and NSS among verified peer-reviewed results on
   DIEM, Coutrot1, ETMD and SumMe; AUC-J highest on all six; level on AVAD (class D, approximate).
3. **Offline compute efficiency relative to accuracy.**
   - Lower compute per output frame than every model with reported cost and DHF1K validation CC ≥ 0.52: ≈57 GMACs for
     ViT-L, against 115 for ViNet, 161 for DiffSal (video only) and ≥1,433 for SalFoM. ViT-g (≈164 GMACs) is about
     level with DiffSal while scoring 0.600 against 0.533.
   - 22–26× faster per output frame than SalFoM on the same GPU (measured).
4. **Training cost and simplicity.** Decoder-only training, under 1 GPU-hour per model on a single 24 GB card. No
   audio, optical flow, pseudo-labels, distillation, static images or diffusion sampling.
5. **Label-free backbone.** The only competitive model in this comparison whose encoder used no human annotation.
6. **Zero-shot transfer.** From DHF1K, without fine-tuning, it already exceeds several fine-tuned audio-visual models
   on DIEM, Coutrot1 and SumMe (class D, approximate).
7. **Evaluation rigour,** which most competitors do not offer: a same-evaluator re-scored baseline, paired intervals,
   all seeds, and rules fixed in advance.

### 9.2 Strongest contributions and most significant advantages

**Accuracy and cost.** The strongest result is **accuracy at a moderate offline cost**:
- a frozen self-supervised encoder plus a 7 M-parameter multi-frame decoder outperforms a fully fine-tuned
  foundation-model approach on DHF1K validation;
- it does so while computing roughly an order of magnitude less per output frame.

**The cleanest evidence** is the class-A comparison with SalFoM, including the past-only control (+0.029 CC
[+0.019, +0.040] for the ViT-g ensemble; +0.018 to +0.025 for single models). That control rules out future frames as
the source of the gain.

### 9.3 Where it is weaker than competing models

- **Benchmarks:** UCF Sports (−0.009 CC against ViNet-E), Coutrot2 (−0.043 against ViNet-A), and the DHF1K test
  split and standard Hollywood-2, where there are no results.
- **Model size and memory:** 5–22× the parameters of TMFI-Net or ViNet-E, and 1.2–4 GB of weights.
- **Streaming:** causal throughput of 6–20 fps against the ViNet family's 120–200 fps.
- **Per-frame compute against lightweight models:** TinyHD-M and UNISAL are roughly 50–330× cheaper per frame
  (TinyHD-S about 10–30×), though far less accurate.

### 9.4 Is the performance-to-cost ratio better?

| Measure | Verdict |
|---|---|
| Accuracy per FLOP, offline | **Yes.** On DHF1K validation CC against compute per output frame, SURGE-D ViT-L (0.584 at ≈57 GMACs) and ViT-g (0.600 at ≈164 GMACs) dominate every model with reported cost and CC ≥ 0.52. The only cheaper models (TinyHD, UNISAL, HD2S, STAViS) are far less accurate. Approximate: classes B and F. |
| Accuracy per parameter or per GB | **No.** SURGE-D ViT-L (333 M) is second in size only to SalFoM (403 M), and ViT-g (1,042 M) is the largest model in the comparison. ViNet-E reaches 0.549 with 48 M parameters. |
| Accuracy per training GPU-hour | **Very likely yes**, but competitors rarely report training time, so this is unverified. |
| Accuracy per FLOP, causal | **No for single models.** The causal ViT-g needs 1.83× SalFoM's encoder compute. A shared-encoder ensemble would be cheaper than three single models, but that is not demonstrated. |

### 9.5 Meaningful advancement or incremental improvement?

**A meaningful empirical result, with an incremental architecture.** The architecture recombines known pieces:
- a frozen backbone (DeepGaze-style);
- a ViTDet-style pyramid;
- R(2+1)D blocks;
- TinyHD-style multi-frame output;
- a Gaussian prior;
- seed ensembling.

A concurrent challenge entry (PredJSal) uses the same backbone family.

What is new and useful is **the evidence**:
- frozen feature-prediction video features beat fine-tuned foundation-model saliency on DHF1K validation by a clear,
  statistically resolved margin;
- the multi-frame design makes this cheap offline;
- the encoder, not the decoder bells and whistles (centre prior, deep supervision, audio, static co-training,
  LoRA), carries the accuracy.

The advance is confirmed only on the validation split; the official test score would settle its standing.

### 9.6 Architectural ideas worth adopting or investigating

1. **Frozen video foundation encoders with light decoders.** A strong, cheap-to-train baseline that every new video
   saliency model should report against. Feature-prediction (JEPA) encoders deserve a systematic encoder-family study
   at equal scale (InternVideo2, VideoMAE-g, DINOv2-g).
2. **Multi-frame output to amortise a large encoder.** The cost lever behind SURGE-D's efficiency, and directly
   applicable to any foundation-model saliency system used offline.
3. **Shared-encoder ensembles and feature caching.** A frozen encoder makes decoder ensembles and hyperparameter
   sweeps nearly free at inference and training time. Untested here, and worth measuring.
4. **Dropping components that do not pay off.** The learned centre prior and deep supervision were equivalent to their
   absence, and audio fusion, static co-training and LoRA did not help. Future models can drop them unless shown to
   help with a frozen encoder.
5. **To investigate:**
   - which taps and decoder parts matter (no ablation yet);
   - whether temporal modelling in the decoder matters (offline ≈ causal suggests little);
   - streaming-efficient causal variants (shorter windows, multi-frame forecasting);
   - speaker-aware modules for conversational video, the main remaining failure.

---

## Sources

**Original papers** (numbers read from the open versions of the peer-reviewed papers; camera-ready versions could
differ slightly):

| Model | Paper | Source |
|---|---|---|
| SalFoM | Moradi et al., ICPR 2024 (LNCS 2025) | arXiv:2404.03097, <https://arxiv.org/html/2404.03097v1> (DHF1K test, UCF and Hollywood-2 tables; training details). Repository read locally (`README.md`, `Model.py`). |
| ViNet-S/A/E | Girmaji et al., ICASSP 2025 | arXiv:2502.00397, <https://arxiv.org/html/2502.00397v1> (DHF1K validation, UCF, audio-visual and Hollywood-2 tables; Table I(b) sizes and fps) |
| DiffSal | Xiong et al., CVPR 2024 | arXiv:2403.01226, <https://ar5iv.labs.arxiv.org/html/2403.01226> (audio-visual table; Table 6 parameters and FLOPs; supp. Table 8 DHF1K validation) |
| UniST | Xiong et al., TCSVT 2025 | arXiv:2309.08220, <https://ar5iv.labs.arxiv.org/html/2309.08220> (DHF1K validation, Hollywood-2 and UCF tables; implementation details) |
| TinyHD | Hu et al., WACV 2023 | arXiv:2301.04619, <https://arxiv.org/html/2301.04619> (DHF1K test, UCF and Hollywood-2; parameters and GMACs) |
| TMFI-Net | Zhou et al., TCSVT 2023 | repository README, <https://github.com/wusonghe/TMFI-Net> (weights available; no results or costs listed) |
| V-JEPA 2 | Assran et al. | arXiv:2506.09985 (pre-training data scale) |
| NTIRE 2026 Video Saliency Prediction challenge report | — | arXiv:2604.14816, <https://arxiv.org/html/2604.14816> (PredJSal) |

**Thesis and repository:**
- `Paper/msc-thesis/main.pdf` (commit `b6e499c`) and the run outputs:
  - `runs/vitg/val_*.json`, `runs/final/val_*.json`, `runs/vitg/ucf_test.json`;
  - `runs/review/salfom_val_*.json`, `runs/review/timing_run3.json`;
  - `runs/thesis/cost.json`, `runs/thesis/umt_flops.json`.
- On-disk sizes measured for this report: `SalFoM_DHF1K.pth`, `runs/main_b1_s0/best.pt`, `runs/vitg_s0/best.pt`, and
  the V-JEPA 2 `model.safetensors` files.
