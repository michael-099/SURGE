# SURGE-D: results section (draft)

*Drafted 2026-11-09 from `research/log.md` (sections 1–11). Every number below is copied from a logged run; the log
gives the run folder of each. Items marked **[pending]** wait on things outside this project (the DHF1K test server
and the IMAR Hollywood-2 ground truth).*

## 5 Experiments

### 5.1 Setup

**Datasets.** DHF1K [Wang et al.]: 600 training videos split 540 for training and 60 as an internal holdout (fixed seed);
the 100 official validation videos (601–700) are used as the test set, because test annotations (701–1000) are held
out by the authors. UCF Sports: the standard 103 / 47 split, 10 of the training videos held out. The six STAViS
audio-visual datasets (DIEM, Coutrot1, Coutrot2, ETMD, AVAD, SumMe) with the official STAViS folds (3 folds each, DIEM
one split); 10 % of each fold's training list is held out. Hollywood-2 is evaluated with the open Vig, Dorr & Cox
gaze recordings (4–5 viewers, task-driven viewing) on the official 823 / 884 clip split. These numbers are **not
comparable** to published Hollywood-2 results, which use the Mathe & Sminchisescu ground truth (16–19 viewers, shot
split) **[pending: that ground truth has been requested]**.

**Protocol.** All model selection (epoch, blur σ, gates) uses the holdouts only; each test set is scored once per
reported model. Every decision rule (gates, claim thresholds, which model is submitted) was written in the research log
before the run it governs. Seeds 0–2 for all ensembles; an ensemble averages the members' raw predictions before
blurring.

**Metrics.** CC, NSS, SIM, KLD, AUC-Judd and shuffled AUC, averaged over frames within a video, then over videos
(STAViS: then over folds). The evaluator was validated before any model was scored: on SalFoM's DHF1K-val maps it
reproduces the TMFI-Net evaluation code to |ΔCC| = 0.0003 and |ΔNSS| = 0.0004, and an independent evaluator per video
to < 5·10⁻⁵ on CC, NSS, SIM and KLD. sAUC depends on the negative-sampling protocol and is compared only within this
work. Significance: paired bootstrap over videos (10,000 resamples), 95 % intervals. Published numbers from other
papers come from other evaluation code; we report them side by side and treat margins below 0.01 CC as ties.

**Model.** A frozen V-JEPA 2 video encoder (ViT-L, 0.3B, or ViT-g, 1.0B parameters) feeds four intermediate layers to a
SimpleFPN-3D and a (2+1)D top-down decoder with auxiliary heads and a learned centre prior (7.0M trainable parameters
with ViT-L, 7.4M with ViT-g). Input: 16 frames at stride 2, 256 × 256. Training: 3 epochs, AdamW, cosine schedule;
fine-tuning on the smaller datasets starts from the DHF1K model of the same seed and backbone.

### 5.2 DHF1K

**Table 1.** DHF1K val (100 videos, 60,273 frames). SalFoM's maps were produced with its released weights and scored
by the same evaluator.

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SalFoM | 0.564 | 3.326 | 0.413 | 1.297 | 0.924 | 0.741 |
| SURGE-D ViT-L | 0.584 | 3.349 | 0.449 | 1.222 | 0.928 | 0.764 |
| SURGE-D ViT-L, 3 seeds | 0.591 | 3.418 | 0.449 | 1.187 | 0.930 | 0.761 |
| SURGE-D ViT-g | 0.600 | 3.457 | 0.453 | 1.170 | 0.931 | **0.771** |
| SURGE-D ViT-g, 3 seeds | **0.603** | **3.486** | **0.457** | **1.160** | **0.931** | 0.768 |

SURGE-D with ViT-g beats SalFoM by +0.036 CC [+0.026, +0.048] as a single model and +0.039 [+0.029, +0.050] as an
ensemble, and is better on all six metrics. ViT-g improves on ViT-L by +0.017 CC [+0.012, +0.022] (single) and
+0.012 [+0.008, +0.016] (ensemble), again on all six metrics. Seed ensembling adds +0.007 CC with ViT-L but only
+0.003 with ViT-g. Published DHF1K val references from other evaluators: TMFI-Net 0.554 CC, ViNet-E 0.549.

**DHF1K test.** Maps for videos 701–1000 from the ViT-g ensemble (σ fixed on the holdout) have been generated and
checked (re-scoring val maps written by the same script reproduces the evaluator to |ΔCC| ≤ 8·10⁻⁵). **[pending:
submission to the DHF1K authors; test-server scores to be inserted here.]**

### 5.3 UCF Sports

**Table 2.** UCF Sports test (47 videos), fine-tuned from the DHF1K models. Published rows come from other evaluators.

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L, zero-shot | 0.608 | 3.128 | 0.503 | 1.113 | 0.922 | 0.745 |
| SURGE-D ViT-L | 0.725 | 3.916 | 0.580 | 0.773 | 0.941 | 0.809 |
| SURGE-D ViT-L, 3 seeds | 0.733 | 4.028 | 0.583 | 0.749 | 0.943 | 0.815 |
| SURGE-D ViT-g | 0.731 | 4.061 | 0.583 | 0.747 | 0.944 | 0.815 |
| SURGE-D ViT-g, 3 seeds | **0.735** | 4.085 | 0.586 | **0.736** | **0.945** | **0.817** |
| ViNet-E | **0.744** | **4.156** | **0.587** | — | 0.941 | — |
| ViNet-A | 0.734 | 4.108 | 0.586 | — | 0.940 | — |
| TMFI-Net | 0.707 | 3.863 | 0.565 | — | 0.936 | — |
| SalFoM | 0.631 | 3.543 | 0.516 | — | 0.928 | — |

On UCF, SURGE-D ties ViNet-A and remains below ViNet-E (−0.009 CC). The backbone and ensemble gains are small and
mostly not significant on 47 videos (ViT-g vs ViT-L ensemble: +0.002 CC [−0.007, +0.013]).

### 5.4 STAViS audio-visual datasets

SURGE-D is visual-only here; the audio-visual baselines also use the soundtrack.

**Table 3.** Fold-mean CC; test videos per dataset in parentheses. Best published numbers re-checked in DiffSal
(CVPR 2024), ViNet-S/A/E and DTFSal before any SURGE-D number on these data was seen.

| Model | DIEM (17) | Coutrot1 (60) | Coutrot2 (15) | ETMD (12) | AVAD (45) | SumMe (23) |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L, zero-shot | 0.675 | 0.600 | 0.645 | 0.588 | 0.692 | 0.551 |
| SURGE-D ViT-L | 0.706 | 0.632 | 0.816 | 0.662 | 0.718 | 0.581 |
| SURGE-D ViT-L, 3 seeds | 0.714 | 0.641 | 0.807 | 0.670 | 0.729 | **0.593** |
| SURGE-D ViT-g | 0.714 | 0.648 | 0.820 | 0.672 | 0.731 | 0.586 |
| SURGE-D ViT-g, 3 seeds | **0.729** | 0.657 | 0.819 | **0.676** | 0.740 | 0.590 |
| Best visual-only (published) | 0.701 | 0.623 | **0.862** | 0.653 | 0.742 | 0.561 |
| Best audio-visual (published) | 0.685 | **0.661** | 0.845 | 0.667 | **0.746** | 0.573 |

Against the best published visual-only model, the ViT-g ensemble is clearly higher on four datasets (DIEM +0.027,
Coutrot1 +0.034, ETMD +0.023, SumMe +0.029), ties on AVAD (−0.002) and is clearly lower on Coutrot2 (−0.043). Against
the best audio-visual model it is higher on DIEM (+0.043) and SumMe (+0.017), within 0.01 on Coutrot1, ETMD and
AVAD, and lower on Coutrot2 (−0.026). Within this work, ViT-g's ensemble beats ViT-L's significantly on DIEM,
Coutrot1, ETMD and AVAD (+0.006 to +0.017 CC, pooled test videos), with no significant difference on Coutrot2 and
SumMe. NSS and SIM are below the best published audio-visual model on most datasets even where CC is higher (e.g.
AVAD SIM 0.565 vs 0.595).

Coutrot2 shows four people in a meeting, and viewers look at whoever speaks. A visual model cannot know who that is,
which explains the remaining gap. Section 5.6 tests whether adding the soundtrack closes it.

### 5.5 Hollywood-2 (Vig gaze)

**Table 4.** 884 test clips, 257,733 scored frames. Not comparable to published Hollywood-2 results (see 5.1).

| Model | CC | NSS | SIM | KLD ↓ | AUC-J | sAUC |
|---|---|---|---|---|---|---|
| SURGE-D ViT-L, zero-shot | 0.632 | 3.595 | 0.503 | 1.061 | 0.951 | 0.780 |
| SURGE-D ViT-L | 0.680 | 3.898 | 0.518 | 0.925 | 0.956 | 0.817 |
| SURGE-D ViT-L, 3 seeds | 0.684 | 3.875 | 0.507 | 0.918 | 0.957 | 0.819 |
| SURGE-D ViT-g | 0.684 | **3.933** | **0.521** | 0.917 | 0.957 | 0.818 |
| SURGE-D ViT-g, 3 seeds | **0.688** | 3.907 | 0.510 | **0.911** | **0.958** | **0.820** |

Fine-tuning on the Vig gaze data adds +0.048 CC over zero-shot. ViT-g improves on ViT-L by +0.005 CC [+0.003,
+0.006] as a single model (all six metrics significant) and +0.004 [+0.003, +0.005] as an ensemble. As with ViT-L,
the ensemble raises CC, KLD and AUC but lowers NSS and SIM against its own single model (−0.026, −0.011): with at most
five viewers per frame, averaging blunts the sharp peaks those two metrics reward. **[pending: re-run on the standard Hollywood-2 ground truth once received.]**

### 5.6 Analysis: what helped and what did not

All of these were decided by rules written before the runs; negative outcomes are reported as such.

| Question | Outcome (DHF1K holdout unless noted) |
|---|---|
| Backbone (frozen, 10 epochs, seed 0) | V-JEPA 2 ViT-L 0.578 CC > V-JEPA 2.1 ViT-L 0.569 > UMT-L 0.546. |
| Larger backbone (ViT-g) | +0.008 holdout CC over the ViT-L 3-seed mean; +0.017 on DHF1K val (Table 1). Adopted. |
| Predictive "surprise" as a cue (V-JEPA 2 predictor error) | No: anti-correlated with fixations (NSS −0.19 at 4 fps, −0.50 at 15 fps vs +0.81 for plain frame difference). The surprise module was dropped. |
| LoRA fine-tuning of the backbone | No: lower than the frozen backbone in 3/3 seeds (mean −0.0025 CC). |
| Centre prior, auxiliary heads, SALICON co-training | No measurable effect (|ΔCC| < 0.003, one seed). |
| Frame stride 4 instead of 2 | +0.003 CC, at the threshold; not adopted. |
| 32-frame clips instead of 16 (ViT-g) | 0.5902 holdout CC vs the pre-set bar of 0.5904: not adopted (still improving at the last epoch). |
| All 600 training videos instead of 540 (ViT-g) | DHF1K val ensemble CC 0.604 vs 0.603, ΔCC +0.002 [−0.001, +0.004]: no gain. |
| Seed ensembles | +0.003 to +0.015 CC on most datasets (none on Coutrot2, SumMe with ViT-g); lowers NSS/SIM on Hollywood (Vig) with its ≤ 5 viewers per frame. |
| Audio (frozen AST features, FiLM + audio-visual correlation fusion, STAViS) | No dataset improves (fold-mean ΔCC −0.002 to +0.001). On Coutrot2 the model does use the sound (another video's soundtrack costs −0.018 CC), but the matching soundtrack only matches the visual model. |

Two readings follow. First, almost all of the gain comes from the frozen video representation: a better encoder
(V-JEPA 2 over 2.1 and UMT, ViT-g over ViT-L) moves every benchmark, while decoder additions, backbone fine-tuning,
more data and longer clips do not. Second, frozen self-supervised video features with a light decoder beat the best published
visual-only and audio-visual models on DIEM and SumMe and match them within 0.01 CC on Coutrot1, ETMD and AVAD; the one clear failure
is speaker-driven attention, which our audio module did not solve.

### 5.7 Limitations

- DHF1K results are on the validation set; the test-server number is pending.
- Published baselines on UCF and STAViS come from other evaluators; differences under ~0.01 CC are within the
  cross-paper disagreement we observed (e.g. the same baseline reported 0.734 vs 0.652 on Coutrot2).
- Several STAViS test sets are small (12–23 videos), so per-dataset intervals are wide.
- Hollywood-2 is evaluated on a different, smaller-viewer ground truth than the literature.
- Ablations are single-seed; the stride-4 and 32-frame results sit at their thresholds.
