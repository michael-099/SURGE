# Critical review: *Video Saliency Prediction with a Frozen Self-Supervised Video Encoder* (SURGE-D)

**Document reviewed:** MSc thesis, Addis Ababa University (AAiT), `Paper/msc-thesis/main.pdf` as built at commit
`b6e499c` (114 pages), with its appendices and the accompanying repository (`surge/`, `scripts/`, `configs/`).
**Review standard:** main-track reviewer at CVPR / ICCV / ECCV / NeurIPS.
**Date:** 2026-11-12.

> **Note on independence.** This review was written by the same AI assistant that implemented SURGE-D, ran the
> experiments and drafted the thesis. It is therefore not an independent review. To limit the bias this creates,
> every issue below was checked against the PDF, the code or the cited publications. Anything that could not be
> substantiated is labelled **potential concern**, not flaw. Section 13 lists what was checked and found to be in
> order. A human reviewer who did not take part in the work should read the thesis before relying on this report.

**Conventions.** Page numbers are the printed numbers (front matter in roman numerals). "T5.5" is Table 5.5,
"Eq. 4.7" is Equation 4.7, "§5.5.2" is a section and "Alg. 4.1" is Algorithm 4.1. Register IDs (M = major,
D = moderate, m = minor) refer to the table in Section 10. Each issue carries one of these types: **contradiction**
(two statements or a statement and a result disagree), **unsupported** (the claim goes beyond the evidence),
**missing experiment**, **limitation** (acknowledged or inherent scope), **factual error** (about a cited work) or
**presentation**.

---

## Summary verdict

The thesis tests four claims:
- A frozen V-JEPA 2 encoder with a ~7 M-parameter decoder predicts DHF1K fixations better than SalFoM.
- The error of V-JEPA 2's predictor is not a gaze-attracting "surprise" cue.
- Few design choices matter once the encoder is frozen.
- The model transfers to eight further test sets.

The empirical core holds up better than most papers in this area: one validated evaluator, paired bootstraps over
videos, every seed reported, a baseline-blur control, a causal control, decision rules written before most
experiments, and an unusually candid list of deviations (App. B.2). **No critical flaw was found.**

Five problems keep the work below the bar of a top-tier venue:
1. **Novelty.** Concurrent work (PredJSal, NTIRE 2026) uses almost the same recipe, and frozen-backbone readouts are
   an established paradigm (M1).
2. **The DHF1K evidence is thin.** It rests on the validation split and on a single re-scored baseline, whose
   reproduction is not verified against its authors' pipeline (M2, M3).
3. **The decoder is never ablated,** so the architectural contribution is not isolated (M4).
4. **The headline mixes cost and accuracy.** The accuracy comes from ensembles or a 2.6× larger model, the efficiency
   from single models, and the configuration that wins under the identical (causal) protocol is costlier than
   SalFoM (M5).
5. **The surprise analysis is scoped too narrowly** to carry the weight the conclusion puts on it (D4).

**Recommendation: Weak Reject** for a main-track venue (Section 12). It is a strong MSc thesis.

---

## 1. Contradictions between claims and experimental results

**1.1 Claims that the evidence supports (stated so that the criticism below is read in proportion).**
- *Offline ViT-g beats SalFoM on DHF1K validation* (Abstract p. v; §1.8 p. 5; §5.5.1 pp. 49–51).
  - The ensemble scores CC 0.6029 against 0.5639, ΔCC +0.039 [+0.029, +0.050] (T5.6, p. 50).
  - Every one of the six metrics has an interval excluding zero (TA.1, p. 85).
  - The ensemble has the higher CC on 79 of 100 videos (Fig. 5.2, p. 50).
  - Against SalFoM with its best blur, chosen on the test set and therefore favouring SalFoM, every ViT-g seed and the
    ensemble remain ahead: +0.021 to +0.029 (T5.7, p. 51; TA.3, p. 86).
- *The advantage does not come from future frames* (§5.5.2, pp. 52–53). Causal ensembles beat SalFoM at its best blur
  on the same 6,076 frames (+0.029 for ViT-g, +0.017 for ViT-L). Causal and offline differ by
  +0.0016 [−0.0015, +0.0046] for ViT-g (T5.8, p. 53).
- *The surprise gate fails, whichever mask token is used* (T5.3, p. 46; TA.9–TA.10, p. 90).

**1.2 Where the claims go beyond the results.**

1. **The abstract and introduction mix ensemble accuracy with single-model cost** (M5). The abstract (p. v) reports
   the ensemble's 0.603, then "Offline, the model needs about a ninth of SalFoM's encoder computation per output frame
   and less than a twentieth of its measured time". That ratio holds for one ViT-g model. The three-model ensemble
   needs 3 × 0.328 = 0.98 TFLOPs per output frame, about a third of SalFoM's 2.87, and 23–31 ms against 202–227 ms,
   about 1/7–1/9 of its time (T5.12, T5.13, pp. 57–58).

   The causal comparison goes further. The configuration behind "beat SalFoM on the same frames by 0.029" is the causal
   ViT-g *ensemble*. It needs 3 × 5.26 = 15.8 TFLOPs per output frame (5.5× SalFoM's encoder) and 3 × 123.5–165.5 ms
   (1.8–2.2× SalFoM's measured time). The abstract nevertheless says "the causal variant needs nearly twice the
   computation but less time", which is true only of a *single* causal model.

   A correct summary needs no new experiment: each single causal ViT-g model beats SalFoM at its best blur by
   +0.018 to +0.025 (T5.8). It does so at 1.83× SalFoM's encoder FLOPs and 0.61–0.73× its time.
2. **The abstract states the time ratio without the implementation caveat** (M5). "Less than a twentieth of its
   measured time" (p. v) depends on the released implementations. SalFoM's encoder forms its attention matrix
   explicitly over 3,136 tokens, while SURGE-D uses a fused kernel over 2,048 (§5.5.5, p. 58). The
   implementation-independent ratio is ~8.7× in FLOPs. The introduction keeps the qualifier "as released" (p. 5); the
   abstract dropped it. The claim that the causal variant needs "less time" may reverse under equal implementations,
   because its FLOPs are 1.83× SalFoM's.
3. **"Exceeded the best verified peer-reviewed results on four of six audio-visual datasets, by margins within the
   spread of published evaluations"** (Abstract p. v; §6.1 p. 69). The sentence undercuts itself: a margin inside the
   measurement spread cannot establish "exceeded". The thesis itself states three further limits (§5.5.3, p. 55):
   - The comparison is a three-model ensemble against single published models.
   - Two of the four margins (Coutrot1, ETMD) become ties when non-peer-reviewed results are included.
   - Recent peer-reviewed models (TM2SP, FOSal, Mao et al.) were not checked.

   For the single ViT-g model, the margins over the best verified result shrink to +0.010 to +0.020 on DIEM,
   Coutrot1, ETMD and SumMe (from T5.10's rounded values), at or barely above the 0.01 tie margin (D6).
4. **"Encoder pre-training and scale produced the only large gains"** (Abstract p. v; §5.7 RQ3 p. 64; §6.1 p. 69).
   - The *pre-training* part rests on gate G1: one seed per encoder, a 10-epoch schedule rather than the final
     3-epoch one, UMT-L at 224² against 256² for the others, and V-JEPA 2.1 below its 384² cool-down resolution
     (§5.3, p. 44; D2).
   - The learning curve shows a data effect of the same size as scale. Going from 135 to 540 videos changes holdout
     CC by 0.0138 [0.0048, 0.0238]; ViT-L → ViT-g gains +0.0071 on the same holdout (§5.6, pp. 60–62). RQ3 itself
     lists "the amount of training data" as a factor (§1.4, p. 3), so "the encoder dominates" holds only for
     the factors one can still change at 540 videos (D3).
5. **"Freezing was the outcome of an experiment rather than an assumption"** (§4.4.1, p. 31). The experiment was a
   single LoRA setting (rank 16, last 12 blocks), started from the already-converged frozen model and trained three
   *more* epochs; its best epoch was always the first (§5.6, p. 59). This confounds adaptation with over-training a
   decoder that is known to overfit after 2–3 epochs (Fig. 5.1, p. 45). The evidence supports "this LoRA setting
   did not help", not a decision between freezing and adaptation (D1). The UMT-L check points the other way: frozen
   UMT-L with the SURGE-D decoder scores 0.545, below fine-tuned SalFoM's 0.564 (T5.7). Fine-tuning helped there.
6. **Calling SalFoM "the strongest foundation-model approach"** (§5.7 heading, p. 63) is unsupported. The research
   questions say "closest" (§1.4, p. 2). SalFoM's own follow-up LSTD-Net [moradi2026lstd] claims better DHF1K
   results, but its tables "could not be accessed" (p. 50). NTIRE 2026 entries built on InternVideo2 and V-JEPA2
   backbones are also absent (m5).
7. **"Design choices were decided by rules written in advance"** (Abstract p. v) contradicts §4.1 (p. 24), which
   says the decoder architecture, tapped layers, loss weights, input squashing and learning rates "were not
   governed by a rule" (m2).
8. **"With 7.4 million trained parameters, its three-seed ViT-g ensemble reaches … 0.603"** (§1.8, p. 5). The
   ensemble has 22.2 M trained parameters and 3.1 B in total (m3).
9. **"As accurate as the offline model"** (Abstract p. v; §6.1 p. 68) asserts equivalence from a non-significant
   difference, without an equivalence margin. The causal models are also *significantly better* than the offline
   models on several metrics (SIM, KLD and AUC-J for some seeds and ensembles; TA.6, p. 88), and this is not
   explained (m8).

**1.3 A contradiction about a cited work.** §5.2 (p. 43) and §5.8 (p. 66) say "How the released weights were selected
is not documented". The openly available SalFoM paper states that "We implement early stopping based on the
performance on the target dataset's validation set", which is the set this thesis uses as its test set (m1). The
correction favours the thesis: SalFoM's validation scores are optimistic, so the reported margins are conservative.
It also undercuts the thesis's explanation of the gap to SalFoM's published numbers (see M2).

---

## 2. Methodological inconsistencies

- **Equations against code.** These were checked and agree:
  - the loss (Eqs. 4.14–4.17) against `surge/losses.py` and `scripts/train.py:199-203`;
  - the pyramid and decoder (Eqs. 4.3–4.5) and the (2+1)D block (Eq. 4.4) against `surge/model.py:20-57, 166-195`;
  - the predictor error (Eq. 4.8) against `surge/backbones.py:93-110`;
  - the log-linear model (Eqs. 4.9–4.11) against the docstring of `scripts/g2_incremental.py`;
  - the metrics (§4.8.1) against `surge/metrics.py`.

  Two trivial deviations remain. The NSS loss uses the unbiased standard deviation. The prior's eight means are
  initialised at eight of the nine points of a 3×3 grid (`surge/model.py:92-93`), not "on a grid" as the text
  suggests (p. 32).
- **Context claim of Eq. 4.2** (p. 27): "Each prediction therefore has at least eight original frames (about
  0.27 s) of context on either side." By Alg. 4.1 (p. 39), the last frame of each block (offset b+23) has only seven
  future frames (b+24 … b+30). This off-by-one is minor (m4).
- **The loss and the evaluation are aligned with CC, not with the other metrics.** The output blur σ is tuned on the
  holdout by CC, and one map serves all six metrics (§4.7, p. 38). The thesis acknowledges this (§5.8, p. 66). Under
  this protocol the ensembles' small loss in sAUC (TA.8) is expected and is not a contradiction.
- **The training strategy could encode position implicitly.** Frames are squashed to 256×256 rather than cropped
  (p. 29), so image borders sit at fixed positions. Zero padding in the decoder then lets it represent the centre
  bias. The thesis offers this as the reason the learned prior has no measurable effect (§5.6, p. 60), but tests it
  nowhere (see E8).
- **The surprise readouts and their interpretation** (D4). The masked-tube readout (§4.5.1, p. 34) hides 25 % of the
  tokens *in all frames* and predicts them from the visible tokens of the *whole* clip. It therefore measures how
  hard a region is to inpaint in space and time, not how surprising the future is. Using it to conclude that "The
  form of the query thus does not explain the sign" (p. 48) compares two different constructs.
- **Reproducibility details that are missing.** For the SalFoM baseline, the thesis describes the wrapper's procedure
  (p. 43), but the wrapper itself lives in another project and is not released (§5.8, p. 66; M2). The repository's
  `configs/main_b1.yaml` still selects V-JEPA 2.1 (`backbone: vjepa21_l`); the reported models depend on
  command-line overrides (App. B.3, p. 95; m11).

---

## 3. Experimental design and evaluation weaknesses

1. **No official DHF1K test result** (M3). Every DHF1K claim is on the 100 validation videos (§4.3, p. 28;
   §5.5.1, p. 52). Comparisons with other published models rely on validation numbers from other evaluation code (T5.5
   lower block, p. 49), and published validation numbers disagree by up to 0.03 CC: TMFI-Net is listed at 0.554
   by ViNet-S and at 0.524 by UniST (p. 50). The prepared test-set submission (p. 52) would settle the headline.
2. **One re-scored baseline, whose reproduction is not verified** (M2).
   - SalFoM's maps were produced in bfloat16 by a wrapper from an earlier project (p. 43). SalFoM's repository has
     no test script; it points to TMFI-Net's scripts.
   - The wrapper's protocol matches the SalFoM paper: 16 consecutive frames at 224×224, last frame predicted.
   - The *evaluator* was validated on these maps (CC within 0.0003 of TMFI-Net's code), but the *maps* were never
     checked against the authors' own numbers.
   - The thesis explains the reproduced validation CC (0.5639) being below SalFoM's published *test* CC (0.569)
     with the phrase "because they are computed on the validation set" (p. 43). That explanation is unsupported:
     - SalFoM selected its checkpoint on the validation set, which should make its validation scores optimistic,
       not lower.
     - For TMFI-Net, whose scripts SalFoM used, ViNet-S's table lists validation CC 0.554, while SalFoM's DHF1K
       test table lists 0.546, so validation was the *easier* split for that model.

     The evidence is mixed (UniST lists TMFI-Net's validation CC at 0.524), so this is a **potential concern**. Its
     reach is limited: the ViT-g margins (+0.021 to +0.029 against best blur) would survive a reproduction loss of
     ~0.01 CC, but the ViT-L ensemble's +0.017 [+0.007, +0.027] might not.
3. **The comparison is "identical" in evaluation only** (§5.5.2, p. 52). It equalises videos, frames, evaluator,
   causality and baseline blur. The two systems still differ in:
   - training data (540 videos with an internal holdout, against SalFoM's 600 with early stopping on the validation
     set);
   - input geometry (256² squashed against 224²);
   - temporal span (≈1 s at stride 2 against ≈0.5 s consecutive);
   - post-processing (tuned σ against an 11×11 kernel plus extra blur);
   - map format (float against 8-bit).

   The thesis's definition is explicit, so this is presentation, not error. The **map-format asymmetry was checked
   for this review** and does not favour SURGE-D (Section 13, item 6).
4. **Cross-evaluator comparisons on UCF Sports and the audio-visual datasets** (T5.9, T5.10). Their spread is
   documented: STAViS on Coutrot2 is reported at 0.734 by DiffSal and 0.652 by ViNet-S, both verified for this
   review. That spread exceeds most of the margins claimed (D6). The thesis says so (p. 55), but the abstract still
   uses "exceeded".
5. **The holdout is reused for everything.** The same 60 videos select epochs, σ, gates G1/G2, ablations and the
   learning curve (§4.1, p. 25). Adaptive reuse makes holdout-based decisions slightly optimistic. The DHF1K test
   claims are unaffected, since the validation set was never used for selection (TA.4, p. 87).
6. **Possible overlap with V-JEPA 2's pre-training data** (D8; potential concern).
   - DHF1K comes from YouTube, and V-JEPA 2 was pre-trained on large internet-video collections (acknowledged
     generically, p. 67).
   - The V-JEPA 2 report, which documents those data, is not cited because it is a preprint (p. 31), so the overlap
     cannot be assessed from the thesis.
   - The eleven-clip surprise diagnostic used Kinetics clips (p. 48), and Kinetics is part of V-JEPA 2's
     pre-training data.
7. **Metrics.**
   - The sAUC implementation is non-standard and was not checked against reference code (§4.8, p. 39; §5.2, p. 43).
     It is nonetheless one of the "all six metrics" (m7).
   - "Validated evaluator" (§1.6.4, p. 4) holds for CC and NSS against TMFI-Net's code, and for CC, NSS, SIM and KLD
     against a second evaluator. It does not cover sAUC.

---

## 4. Ablation study criticism

1. **The decoder, which the thesis presents as "a purpose-built decoder"** (p. 2), **is never ablated** (M4). §4.1
   (p. 24) lists the decoder architecture, the tapped layers and the loss weights among the choices "not governed by
   a rule", and T5.14 (p. 59) contains no decoder variant. Several questions are therefore open:
   - whether a linear or 1×1×1 readout of one layer of frozen V-JEPA 2 ViT-g would do as well;
   - whether four taps beat one;
   - whether the (2+1)D temporal convolutions contribute anything.

   The offline and causal models perform alike (T5.8), so temporal mixing beyond the frames the encoder already
   sees may add little; this is untested. RQ3's verdict that "the encoder dominates" (p. 64) is thus reached
   without varying the other major architectural factor.
2. **The component ablations used ViT-L; the reported model is ViT-g** (T5.14, p. 59; D5). Whether the centre prior
   and deep supervision are "equivalent to their absence" for ViT-g is assumed. The equivalence margin (±0.003 CC)
   is about one seed standard deviation (0.0030–0.0034, p. 51). The two-level bootstrap resamples only three seeds
   (§4.8.2, p. 40: "its intervals are approximate"). "Equivalent" is therefore weakly supported.
3. **The LoRA test is confounded** (D1; see 1.2.5). There is no control in which the frozen model is trained three
   further epochs at the LoRA phase's decoder learning rate, only one configuration was tried, and full fine-tuning
   was not tested.
4. **Gate G1 decides the encoder family from one seed with unequal conditions** (D2). The V-JEPA 2 versus 2.1 gap is
   0.009 CC, three times the later seed SD, but n = 1 per encoder, and resolution differs by encoder (p. 44).
5. **No interactions were studied.** Stride × clip length, prior × squashing and ensemble × σ were never crossed.
   Stride 4 was positive in all three seeds (+0.0029 [−0.0009, +0.0068]) at half the cost, yet the main model keeps
   stride 2. That is consistent with the rule but leaves a cheaper configuration untested at the ViT-g scale.
6. **Statistical and practical meaning.** The headline gains (+0.036 to +0.039 CC against SalFoM as released) are
   large by DHF1K standards; TMFI-Net → SalFoM gained +0.023 on test. The component effects (≤0.003) are below what
   the design can resolve. The thesis classifies them correctly as "equivalent" or "inconclusive".

---

## 5. Mathematical and theoretical analysis

- **Equations are correct and dimensionally consistent.**
  - Eq. 4.3 maps 2,048 tokens to D×8×16×16 with levels at 64², 32², 16² and 16².
  - Eq. 4.6 has 8 + 16 + 16 + 1 = 41 parameters, as stated.
  - Eqs. 4.9–4.11 define information gain in bits per fixation, averaged per frame, then per video, then over videos.
  - Eq. 2.1 matches the L1 JEPA objective as described in the cited V-JEPA 2.1 paper.
- **The normalisation in Eq. 4.6 is redundant** under the softmax of Eq. 4.7, but harmless.
- **Eq. 4.2 and the context claim** are off by one for odd frames (Section 2; m4).
- **The predictor error and the concept of surprise** (D4). Eq. 4.8 is an L1 distance between layer-normalised
  predicted and target tokens. Itti and Baldi's Bayesian surprise is a divergence between posterior and prior. §5.8
  (p. 67) acknowledges this, and the conclusion is suitably limited to "one released predictor and two ways of
  reading it out" (p. 64).
- **The log-linear analysis has two weaknesses** (D4):
  1. **The centre prior has a fixed shape.** g enters only through a single fitted weight w₀ (Eq. 4.9). Any
     border-related structure in the error map, such as border tokens having less context, could therefore earn a
     (negative) weight that reflects position rather than content. "Beyond the centre bias as the model of Eq. 4.9
     represents them" (p. 48) is the accurate phrasing. The stronger version in §1.8 (p. 5) and §6.1 (p. 68) is not
     supported.
  2. **The bootstrap holds the fitted weights fixed** (p. 35), so the intervals ignore fitting uncertainty from only
     30 videos per half. They are likely too narrow.
- **The theory does not explain the main empirical surprise.** V-JEPA 2.1 was designed to give denser features and
  should suit a dense task, yet V-JEPA 2 led. The thesis offers two explanations and tests neither (p. 44). That is
  honest, but the frozen-feature theory (Bardes et al.) does not predict the ranking.

---

## 6. Comparison with existing state-of-the-art methods

1. **Numbers checked against the original publications** (all match the thesis):
   - SalFoM: DHF1K *test* CC 0.569, NSS 3.353; end-to-end fine-tuning of UMT; 16 consecutive frames at 224²;
     predicts the last frame; explanation of the weaker UCF result.
   - DiffSal: audio-visual CC 0.638 (Coutrot1), 0.652 (ETMD), 0.738 (AVAD), 0.572 (SumMe), and the NSS/SIM
     statements on p. 55.
   - ViNet-S/A (ICASSP 2025):
     - DHF1K validation: TMFI-Net 0.554, THTD-Net 0.553, ViNet-E 0.549, STSANet 0.539, DiffSal 0.533;
     - UCF: ViNet-E 0.744, ViNet-A 0.734, THTD-Net 0.711, TMFI-Net 0.707;
     - visual-only audio-visual CC: 0.701, 0.614, 0.862, 0.632, 0.729;
     - parameter counts: TMFI-Net 61.34 M, STSANet 168.56 M.
   - Not verified here: the AVRSP numbers, UniST's 0.524/0.541, V-JEPA 2.1's 22.2 mIoU.
2. **Baselines are too few** (M3). TMFI-Net, THTD-Net and ViNet-S/A have public code and weights and could have been
   re-scored with the same evaluator. That would also have calibrated how well the thesis reproduces published
   validation numbers (M2).
3. **The comparison is not matched for size or cost** (M5).
   - SURGE-D ViT-g has 1,042 M parameters against SalFoM's 403 M; its ensembles have 1.0 B (ViT-L) and 3.1 B (ViT-g).
   - At SalFoM's encoder size, no single SURGE-D model is reliably better than SalFoM at its best blur, offline or
     causally (T5.7, T5.8): one ViT-L seed of three is, offline and causally.
   - The thesis reports all this (p. 51) but foregrounds the trained-parameter count (Abstract; §1.8).
   - The compute-matched comparison exists, causal ViT-L single against SalFoM, and it is not significant for two of
     three seeds.
4. **Novelty against prior and concurrent work** (M1).
   - **PredJSal** (team CVSP, NTUA / Athena RC) in the *NTIRE 2026 Challenge on Video Saliency Prediction* report
     (arXiv:2604.14816, April 2026, CVPR 2026 workshop) is "a predictive JEPA-based framework that repurposes the
     rich internal representations of V-JEPA2 for video saliency prediction".
     - It uses a V-JEPA2 Giant (ViT-g/16) backbone, features from three encoder layers, a 3D-convolutional decoder
       and a four-seed ensemble.
     - It is motivated by the idea that saliency "should … correlate with the prediction errors produced by a
       predictive coding model", and it ranked joint second.
     - Whether its backbone was frozen is not stated.
     - The thesis excludes it under its peer-reviewed-only policy. §3.7 (p. 23) therefore says feature-prediction
       encoders "have not been evaluated for this task in the peer-reviewed studies whose full texts could be
       consulted". That is literally true but omits the closest known work.
   - **Frozen backbone with a small trained readout** is the DeepGaze II/IIE design for images
     [kummerer2017understanding; linardos2021deepgaze], both cited on p. 11 without noting that their backbones are
     frozen.
   - **Multi-frame output to amortise cost** was introduced by TinyHD [hu2023tinyhd] (cited, p. 19).
   - **The decoder's parts are standard:** a ViTDet-style simple pyramid, R(2+1)D blocks, a learned Gaussian prior,
     and a KL+CC+NSS loss.

   The most novel element is the **explicit test of the predictor's error against fixations, with incremental
   information**. PredJSal invokes prediction error as motivation but, by the report's description, decodes
   representations.
5. **Results may have been reported selectively.** No evidence of this was found. All seeds, all six metrics, the
   null results and the failure on Coutrot2 are reported, and the qualitative figure was chosen by a fixed rule and
   shows SalFoM marginally closer on four of five frames (p. 57).

---

## 7. Reproducibility and implementation

- **Strengths.** The thesis provides:
  - the full code, configurations, split and holdout lists and commands (App. B, pp. 93–96);
  - a dated decision log;
  - cheap training (15 min ViT-L, 31 min ViT-g per model, p. 42);
  - public encoder checkpoints;
  - evaluation at native resolution.

  The equations match the code (Section 2).
- **Weaknesses.**
  - The SalFoM maps, the basis of RQ1, come from a wrapper that is not in the repository (p. 43; m11).
  - Trained checkpoints (`*.pt`) are not distributed (README).
  - The main configuration defaults to V-JEPA 2.1, and the reported models depend on overrides.
  - The README's "best published" audio-visual numbers include non-peer-reviewed results and differ from T5.10 (for
    example Coutrot1 0.661 against 0.638), which will confuse readers of both.
  - The Hugging Face V-JEPA 2 *predictor* port read an untrained mask token by default (§4.5.1, p. 34). The *encoder*
    port was checked only against itself (visible-token consistency, p. 34), never against Meta's reference code
    (m12; potential concern). The encoder's strong results make a large defect unlikely.
- **External dependencies.** These are disclosed: V-JEPA 2 checkpoints (MIT licence), SalFoM's released weights, the
  STAViS preprocessed releases and the Vig gaze data (CC BY). The AST audio model is used only in the audio
  experiment.

---

## 8. Hidden assumptions and potential failure modes

1. **The holdout and validation sets are assumed to represent the test set.** No official test result exists (M3).
2. **Squashing assumes 16:9 content without letterboxing.** Letterboxed or vertical video moves the borders on which
   the implicit centre bias may depend (§5.6, p. 60). Crop against squash was not ablated (p. 29).
3. **Linear interpolation between sampled frames assumes saliency changes smoothly** (Alg. 4.1). Windows that span
   shot cuts mix scenes; behaviour at cuts is untested.
4. **Thirty frames per second is assumed.** The stride is defined in frames (p. 25), so on 24–25 fps datasets the
   window covers a different duration.
5. **Attention driven by audio or narrative is out of reach.** The model fails where gaze follows the speaker
   (Coutrot2, the panel-discussion video; p. 57), and the one-second window cannot model longer narrative context.
6. **Possible shortcut through position.** The learned prior is "equivalent to its absence" (p. 60), consistent with
   the network encoding the centre bias through borders. A translation-equivariance test would show whether
   predictions follow content or fixed positions (E8).
7. **Does the model use motion at all?** Frame difference is a strong cue (NSS 0.81, T5.3), yet the model's own use
   of temporal information is untested (E4). Without that test, "spatio-temporal decoder" may be describing a model
   that works mostly frame by frame.
8. **Low output resolution.** 256² input, a 16×16 token grid and 64×64 output limit precision on small faces and
   objects. This matters for NSS and AUC at fixation level.

---

## 9. Literature and citation audit

- **The core model is cited through another paper** (m6). §1.3 (p. 2) cites V-JEPA 2.1 [murlabadia2026vjepa21] for "a
  frozen V-JEPA 2 encoder". The footnote on p. 31 explains that the V-JEPA 2 report is a preprint and therefore not
  cited. A top-tier venue would expect the source of the encoder, its pre-training data and objective to be cited
  directly. The omission also hides the data needed for the leakage question (D8).
- **Directly relevant work is omitted** (M1): PredJSal / NTIRE 2026 (above), and the frozen-backbone lineage of
  DeepGaze II/IIE, which is cited but not presented as such.
- **A cited work is misrepresented** (m1): "how the released weights were selected is not documented" (p. 43) is
  contradicted by the SalFoM paper.
- **Cited claims that were verified:**
  - SalFoM's test results and its UCF explanation (pp. 1–2, 20);
  - the cross-paper disagreement on STAViS/Coutrot2 (p. 55);
  - the ViNet-S/A and DiffSal tables (T5.5, T5.9, T5.10);
  - the parameter counts of TMFI-Net and STSANet (p. 19, via ViNet-S).
- **Effect of the restrictive citation policy** (§1.6.1, p. 4). Concurrent preprints (V-JEPA 2, the NTIRE 2026
  report, DTFSal) were excluded although they bear directly on novelty and on the state of the art (D6). The thesis
  is transparent about the policy, but its consequence (a narrower related-work picture) is not stated as a threat to
  validity.

---

## 10. Contradiction and weakness register

| ID | Paper section | Author's claim | Contradictory evidence / problem | Why it matters, and the fix | Severity | Evidence / source |
|---|---|---|---|---|---|---|
| M1 | §3.6–3.7 (pp. 21–23); §1.3 (p. 2) | Feature-prediction encoders "have not been evaluated for this task in the peer-reviewed studies whose full texts could be consulted"; contributions framed around a frozen V-JEPA 2 with a purpose-built decoder | **Unsupported novelty framing.** PredJSal (NTIRE 2026, arXiv:2604.14816): V-JEPA2 ViT-g/16, three encoder layers, 3D-conv decoder, four-seed ensemble, motivated by predictive-coding prediction errors; joint 2nd place. Frozen readouts are the DeepGaze II/IIE paradigm (cited, p. 11, but not as such); multi-frame output is TinyHD's (p. 19). | The architectural contribution is largely a recombination of known parts, and the closest work is not discussed. **Fix:** discuss PredJSal and the DeepGaze lineage. State what is new: the frozen protocol and its cost, the controlled comparison, the explicit predictor-error test. Compare on the NTIRE data if possible. | Major | arXiv:2604.14816, §4.2 and §6.2; thesis T3.1 (p. 22) |
| M2 | §5.2 (p. 43); §5.5.1–5.5.2 | SalFoM's validation scores are "slightly lower than SalFoM's published scores **because** they are computed on the validation set" | **Unsupported; potential concern on baseline fidelity.** The maps come from an external, unreleased bf16 wrapper; SalFoM has no official test script. SalFoM early-stopped on the validation set, which should make its validation score optimistic, not lower. For TMFI-Net, validation 0.554 (ViNet-S's table) is above test 0.546 (SalFoM's Table 1). Contrary evidence: UniST lists 0.524 for TMFI-Net's validation CC. | RQ1 rests on one baseline. The ViT-L ensemble's margin (+0.017 against best blur) is within plausible reproduction error; the ViT-g margins (+0.021 to +0.029) are more robust. **Fix:** run SalFoM through TMFI-Net's official test script; release the wrapper; re-score 2–3 more public baselines and compare each with its published validation number (E2). | Major | SalFoM arXiv v1 ("early stopping … target dataset's validation set"; Table 1); ViNet-S Table I; thesis p. 50 |
| M3 | Abstract; §5.5.1 (pp. 49–52); §6.1 | RQ1 answered "yes" on DHF1K | **Limitation (acknowledged).** No official test-set result. External comparisons cross evaluation code whose disagreement (up to 0.03 CC) exceeds several claimed margins. Only one baseline re-scored. | The main benchmark claim is unconfirmed on the split the field reports. **Fix:** obtain the test scores, which are already prepared (E1). | Major | p. 52; T5.5 lower block; p. 50 (0.554 against 0.524) |
| M4 | §4.1 (p. 24); §4.4 (pp. 30–33); §5.6 | "A purpose-built decoder" (p. 2) whose design "follows from three considerations" (p. 33); RQ3: "the encoder dominates" | **Missing experiment.** Decoder architecture, taps and temporal convolutions were "not governed by a rule" and never ablated. There is no linear-readout baseline. Causal ≈ offline suggests temporal context beyond the clip adds little; the decoder's temporal mixing is untested. | The architectural contribution is not isolated. RQ3's ranking omits the second major factor. **Fix:** decoder ablations at ViT-g with three seeds (E3), plus a temporal test (E4). | Major | p. 24; T5.14 (p. 59) |
| M5 | Abstract (p. v); §1.8 (p. 5); §6.1 (p. 68) | "About a ninth of SalFoM's encoder computation … less than a twentieth of its measured time; the causal variant needs nearly twice the computation but less time"; ensemble 0.603 and causal +0.029 | **Misleading juxtaposition.** The cost ratios are for single models, the accuracy figures for ensembles. The causal ViT-g ensemble behind +0.029 needs 5.5× SalFoM's encoder FLOPs and 1.8–2.2× its time. The time ratio depends on implementation (explicit attention against SDPA); "less time" for the causal model may reverse at equal implementation (1.83× FLOPs). Total parameters: 1.04 B against 0.40 B. | Efficiency is a headline contribution. **Fix:** report accuracy and cost per configuration, e.g. single causal ViT-g: +0.018 to +0.025 at 1.83× FLOPs and 0.61–0.73× time. Time SalFoM with SDPA (E5). Restore "as released" in the abstract. | Major | T5.8 (p. 53); T5.12–5.13 (pp. 57–58); p. 58 |
| D1 | §4.4.1 (p. 31); §5.6 (p. 59) | "Freezing was the outcome of an experiment rather than an assumption" | **Overstated.** One LoRA setting, initialised from the converged frozen model and trained three extra epochs (best epoch always the first); no continued-training control; no full fine-tuning. Frozen UMT-L (0.545) < fine-tuned SalFoM (0.564), so adaptation helped there. | The frozen-versus-adapted conclusion is weak. **Fix:** E6. | Moderate | T5.14; T5.7; Fig. 5.1 |
| D2 | §5.3 (p. 44); Abstract; §5.7 | Encoder pre-training "produced the only large gains" | **Under-powered evidence.** Gate G1 used one seed, a 10-epoch schedule, UMT-L at 224², V-JEPA 2.1 below its 384² resolution. | The family ranking (V-JEPA 2 > 2.1) could be noise or a resolution effect. **Fix:** three seeds at native resolutions with the final schedule; add other foundation encoders at scale (E7). | Moderate | T5.2 (p. 44) |
| D3 | §5.7 RQ3 (p. 64); §6.1 (p. 69) | "The encoder dominates … only gains that were large relative to seed noise" | **Tension with the learning curve.** 135 → 540 videos: Δ = 0.0138 [0.0048, 0.0238], about twice the ViT-L → ViT-g holdout gain (+0.0071); 270 → 540: 0.0054. RQ3 lists data amount as a factor. | The data effect is of the same order as scale. **Fix:** restate RQ3 as "among changes possible at 540 videos". | Moderate | T5.15 (p. 62); p. 61 |
| D4 | §5.4 (p. 48); §1.8 (p. 5); §6.1 (p. 68) | "The form of the query thus does not explain the sign"; the error carries information "beyond the centre bias, motion and atypicality" | **Construct issues.** (a) The masked-tube error is spatial inpainting difficulty, not temporal surprise. (b) Centre bias enters with a fixed shape and one weight, so border artefacts can carry "information". (c) Bootstrap intervals hold the fitted weights fixed. (d) The baseline set excludes semantic cues (faces, people) and SURGE-D itself. | RQ2's positive part ("carries information") is fragile. **Fix:** E10: flexible spatial prior; SURGE-D's own map in the baseline set; wrong-video control; bootstrap the fit; relate the error to texture and edge density. | Moderate | Eqs. 4.8–4.11; T5.4 (p. 47) |
| D5 | §5.6 (pp. 59–60) | Centre prior and deep supervision "equivalent to their absence within 0.003" | **Transfer and power assumed.** Ablations on ViT-L only; margin ≈ seed SD; coarse three-seed bootstrap. | Holds for ViT-L and the holdout, not necessarily for the reported ViT-g. **Fix:** repeat the key ablations at ViT-g. | Moderate | T5.14; p. 40; p. 51 |
| D6 | Abstract (p. v); §5.5.3 (p. 55); §6.1 (p. 69) | "Exceeded the best verified peer-reviewed results on four of six audio-visual datasets" | **Self-undermining claim.** Ensemble against single published models; the thesis says two margins become ties with non-peer-reviewed results; cross-evaluator spread (0.734 against 0.652 for the same model) exceeds the margins; three recent models unchecked; single-model margins are +0.010 to +0.020. | "Exceeded" is not established. **Fix:** report single models against single models and call the margins ties; re-score at least one published audio-visual model. | Moderate | T5.10 (p. 55); DiffSal T1; ViNet-S T III |
| D7 | §4.1 (p. 25); §5.6 | Decisions on the 60-video holdout | **Limitation.** One holdout is reused for epochs, σ, gates, ablations and the learning curve, with adaptive reuse. | Holdout-based conclusions (gates, ablations) are slightly optimistic; test claims are unaffected. **Fix:** a second, untouched holdout for the final gate decisions. | Moderate | T4.1 (p. 26) |
| D8 | §5.8 (p. 67); §4.4.1 footnote (p. 31) | Overlap with pre-training "cannot be ruled out" | **Potential concern, not checked.** DHF1K is from YouTube; V-JEPA 2 pre-training includes large internet-video sets; its report is not cited; the surprise diagnostic used Kinetics, part of V-JEPA 2's pre-training data. | Could inflate the encoder's apparent fit to DHF1K and bias the surprise diagnostic. **Fix:** E11. | Moderate | p. 48; p. 67 |
| m1 | §5.2 (p. 43); §5.8 (p. 66) | "How the released weights were selected is not documented" | **Factual error.** SalFoM's paper states early stopping on the target dataset's validation set. | Favours the thesis (conservative margins) but misstates a source. **Fix:** cite it and state the bias direction. | Minor | SalFoM arXiv v1 |
| m2 | Abstract (p. v) | "Design choices were decided by rules written in advance" | **Contradiction** with §4.1 (p. 24): many choices "were not governed by a rule". | Overstates the protocol. **Fix:** "Each tested component was decided by a rule written in advance." | Minor | p. 24 |
| m3 | §1.8 (p. 5) | "With 7.4 million trained parameters, its three-seed ViT-g ensemble …" | The ensemble has 22.2 M trained and 3.1 B total parameters. | Inaccurate. **Fix:** state per-model counts. | Minor | T5.12 |
| m4 | §4.2, Eq. 4.2 (p. 27) | "At least eight original frames … of context on either side" | Alg. 4.1 gives 7 future frames for the last frame of each block. | Minor inaccuracy. **Fix:** "at least seven". | Minor | Alg. 4.1 (p. 39) |
| m5 | §5.7 heading (p. 63) | "The strongest foundation-model approach" | **Unsupported.** LSTD-Net's tables were not accessed (p. 50); NTIRE 2026 InternVideo2 and V-JEPA2 systems are absent. | Overclaims the baseline's standing. **Fix:** "closest", as in RQ1. | Minor | p. 50 |
| m6 | §1.3 (p. 2) | "A frozen V-JEPA 2 encoder [murlabadia2026vjepa21]" | Cites the V-JEPA 2.1 paper for V-JEPA 2. | Misattribution; hides the pre-training data (D8). **Fix:** cite the V-JEPA 2 report. | Minor | p. 31 footnote |
| m7 | §4.8 (p. 39); §5.2 (p. 43) | "Validated evaluator"; "better on all six metrics" | sAUC is non-standard and not checked against reference code. | One of the six metrics is unvalidated. **Fix:** validate sAUC or report it separately. | Minor | p. 43 |
| m8 | Abstract (p. v); §6.1 (p. 68) | Causal "as accurate as the offline model" | Equivalence without a margin. Causal is significantly better on some metrics (TA.6, p. 88). | Unexplained result; wording overstates. **Fix:** "no measurable difference in CC"; investigate why causal is better on SIM, KLD and AUC-J. | Minor | TA.6 |
| m9 | §1.8 (p. 6) | "A new, openly reproducible Hollywood-2 ground truth" | ≤5 observers per frame; reliability and inter-observer ceiling not measured. | Weak as a contribution. **Fix:** E12. | Minor | §4.3.2 (p. 29); T5.11 |
| m10 | §4.1 (p. 24) | "Pre-registered decision rules" | Internal, editable log; deviations occurred (App. B.2). | Protection depends on commit history. **Fix:** cite the commit hash and time of each rule in T4.1. | Minor | App. B.2 (pp. 93–95) |
| m11 | App. B (pp. 93–96); repository | Reproducible with the given commands | SalFoM wrapper not released; checkpoints not distributed; the main configuration defaults to V-JEPA 2.1; README audio-visual "best published" numbers differ from T5.10. | Reproduction friction and an inconsistent public record. **Fix:** release the wrapper and checkpoints, fix the default, align the README. | Minor | `configs/main_b1.yaml`; README |
| m12 | §4.4.1; §4.5.1 (p. 34) | Encoder features as released | **Potential concern.** The HF predictor port had a mask-token defect; the encoder port was never compared with Meta's reference code. | Low risk given the results. **Fix:** compare features on a few clips. | Minor | p. 34 |

**Critical issues: none found.** Section 13 lists the checks that came out in the thesis's favour.

---

## 11. Experiments that could falsify the main claims (in priority order)

Each entry gives the purpose, the protocol and the outcome that would count against the thesis. Costs assume the
single RTX A5000 used in the thesis.

| # | Purpose | Protocol | Falsifying outcome | Cost |
|---|---|---|---|---|
| E1 | Confirm RQ1 on the benchmark | Obtain the DHF1K test scores of the already-prepared ViT-g ensemble submission; compare with SalFoM's published test CC 0.569 | Test CC within 0.01 of 0.569 | none (submission) |
| E2 | Baseline fidelity | Re-generate SalFoM's validation maps with TMFI-Net's official test script (fp32), compare with the wrapper's maps; re-score TMFI-Net and ViNet-A/S with their public weights, compare each with its published validation number | Wrapper maps score ≥0.01 CC lower than the official script, or the re-scored baselines systematically underperform their published numbers | 2–4 GPU-h |
| E3 | Is the decoder needed? | ViT-g, three seeds: (a) 1×1×1 readout of the last tap + bilinear upsampling; (b) last tap only with the full decoder; (c) temporal kernels set to 1 (2D decoder); (d) C = 64 | (a) or (b) within 0.003 CC of SURGE-D on holdout and validation | ~12 × 31 min |
| E4 | Does the model use time? | At test time: (a) one frame repeated 16×; (b) frames shuffled within the window; (c) reversed order | CC drops < 0.005: the model is effectively frame-wise | < 1 GPU-h |
| E5 | Is the efficiency real? | Time SalFoM with fused attention (SDPA) and end to end (decode, resize, blur, I/O); plot accuracy against FLOPs and time for all SURGE-D configurations (single, ensemble, causal, stride 4) | At equal implementation, SalFoM is faster than the causal ViT-g single model, and no SURGE-D configuration at or below SalFoM's cost beats it significantly | 2 GPU-h |
| E6 | Freezing against adaptation | LoRA (rank and learning-rate grid on the holdout) and full fine-tuning of the top blocks, trained jointly from scratch for the same schedule; control: frozen model trained for the same extra epochs | An adapted model beats frozen by ≥ 0.003 CC with three seeds | 10–20 GPU-h |
| E7 | Does V-JEPA 2's objective matter? | Same frozen protocol with InternVideo2-1B, VideoMAE-g and DINOv2-g (per frame), three seeds | Another encoder of similar size matches or beats V-JEPA 2 ViT-g | 10 GPU-h |
| E8 | Shortcut through position | Translate content by crop and pad (maps should move with it); crop against squash; letterboxing; centre-prior-only and blank-input baselines on validation | Predictions stay at fixed positions under translation, or blank or centre-only input recovers a large share of CC | 1–2 GPU-h |
| E9 | Generalisation without fine-tuning | Zero-shot on LEDOV and on the NTIRE 2026 VSP videos (openly licensed and post-dating pre-training), against SalFoM zero-shot | SURGE-D's advantage disappears off DHF1K | 2–4 GPU-h |
| E10 | Robustness of RQ2 | Add SURGE-D's own prediction and a flexible spatial prior (per-cell bias or distance to border) to the baseline set; wrong-video error map as control; bootstrap including the weight fit; regress the error on texture, edge density and face/person masks | Incremental information vanishes once SURGE-D or a flexible prior is in the baseline set, or the control also shows it | CPU, hours |
| E11 | Pre-training leakage | Match DHF1K YouTube IDs against the public lists of V-JEPA 2's sources (Kinetics, HowTo100M, YT-Temporal-1B); compare scores on overlapping and non-overlapping videos | A significantly higher score on overlapping videos | CPU |
| E12 | Quality of the Hollywood-2 (Vig) ground truth | Leave-one-observer-out CC and NSS; compare model scores with this ceiling | Ceiling close to model scores, so the ground truth cannot rank models | CPU |

E1, E2, E3 and E5 decide whether the main contribution stands. E4 and E8 test for shortcuts. E10 decides how much
RQ2's "carries information" is worth.

---

## 12. Final reviewer assessment

1. **Main contribution, independent of the claims.**
   - A frozen V-JEPA 2 ViT-g encoder with a small multi-scale decoder, trained in about 30 minutes on 540 DHF1K videos,
     scores clearly above SalFoM's released model on DHF1K validation under one validated evaluator, including when
     both are restricted to past frames.
   - A scoped negative result: the error of V-JEPA 2's predictor is anti-correlated with fixations, and its small
     incremental information carries a negative weight.
   - A transparent record of null results: centre prior, deep supervision, audio fusion, more data and static images.
2. **The strongest evidence.**
   - RQ1 for ViT-g: all seeds, the ensemble, the best-blur baseline (chosen on the test set in SalFoM's favour), the
     causal control, 79 of 100 videos, and all six metrics.
   - The pre-registered timing result: 22–26× faster per output frame in three runs, criterion 2×.
   - The robustness of the G2 failure to the mask token and the readout.
   - The audio null result with a rotated-audio control.
   - Published numbers that, where checked, match their sources.
3. **The five most serious issues.**
   - M1: novelty against PredJSal and the frozen-readout lineage.
   - M2: unverified reproduction of the single re-scored baseline.
   - M3: no official test-set result.
   - M4: decoder never ablated.
   - M5: cost and accuracy mixed in the headline; the causal configuration that wins is costlier than SalFoM.
4. **Methodological weaknesses most likely to undermine results.**
   - The SalFoM reproduction (M2).
   - The ViT-L-only, three-seed ablations with a margin about equal to seed noise (D5).
   - The confounded LoRA test (D1).
   - The single-seed encoder gate (D2).
   - The fixed-shape centre prior and fixed-weight bootstrap in RQ2 (D4).
5. **Missing experiments.** E1–E5 above: official test scores, baseline fidelity and more re-scored baselines,
   decoder and temporal ablations, and a compute-matched comparison with an efficient SalFoM. Then E6–E10.
6. **Novelty.** Incremental. The system combines known parts on a frozen V-JEPA 2 backbone, and the concurrent PredJSal
   is close in backbone, decoder and ensembling. The explicit predictor-error analysis and the evaluation discipline
   are the original parts; the latter is good practice rather than a scientific contribution.
7. **Reproducibility.** High for SURGE-D: code, configurations, splits, commands, logs and cheap training. Weaker for
   the baseline (wrapper not released) and for exact numbers (checkpoints not distributed).
8. **Scientific validity.** The body's conclusions mostly follow from the evidence and are carefully hedged
   (§5.5.3, §5.8). The abstract and introduction overstate in places:
   - efficiency stated next to ensemble accuracy (M5);
   - "exceeded" on the audio-visual datasets (D6);
   - "the encoder dominates" (D2, D3);
   - "freezing was the outcome of an experiment" (D1);
   - "design choices were decided by rules written in advance" (m2).

   None of these reverses a main conclusion.
9. **Recommendation: Weak Reject** (main track, CVPR / ICCV / ECCV / NeurIPS).

   The rigour of the evaluation is above the field's average, and the main empirical finding is probably real. The
   paper still falls short of the bar on three counts:
   - (i) its methodological novelty is limited and the closest concurrent work is not discussed;
   - (ii) the primary benchmark result is unconfirmed on the official test split and rests on one re-scored baseline
     whose reproduction is not verified;
   - (iii) the architectural contribution is not isolated, because the decoder is never ablated and the advantage at
     matched encoder size needs an ensemble.

   **What would move it to Weak Accept:** E1 (test scores), E2 (baseline fidelity plus two more re-scored baselines),
   E3 and E4 (decoder and temporal ablations), a cost-matched presentation (M5), and a related-work section that
   positions the work against PredJSal and the DeepGaze lineage. As an MSc thesis the work is sound, and the
   recommendation would be to accept with minor revisions (M5, D6, m1–m8).

---

## 13. Checks that came out in the thesis's favour

These were tested because they could have undermined the work; none did.

1. **SalFoM's published numbers and protocol.**
   - DHF1K *test* CC 0.569 and NSS 3.353 (curator-scored).
   - End-to-end fine-tuning of UMT ("encoder parameters are fine-tuned at training time").
   - 16 consecutive 224² frames; the last frame is predicted.
   - The UCF-weakness explanation (task-driven annotations and overfitting).

   This confirms the thesis's description and its characterisation of SalFoM as "per-frame" and "fine-tuned"
   (pp. 1–2, 20, 43). Source: SalFoM, arXiv:2404.03097v1 (HTML version).
2. **The ViNet-S/A tables** (ICASSP 2025, arXiv:2502.00397): every validation, UCF and visual-only audio-visual
   number used in T5.5, T5.9 and T5.10; STAViS on Coutrot2 at 0.652; TMFI-Net 61.34 M and STSANet 168.56 M
   parameters.
3. **DiffSal** (CVPR 2024, arXiv:2403.01226), Table 1: CC 0.660 / 0.638 / 0.835 / 0.738 / 0.652 / 0.572 and the NSS
   and SIM values used on p. 55. DiffSal's own STAViS/Coutrot2 entry is 0.734, which confirms the cross-paper
   disagreement the thesis cites.
4. **TinyHD's multi-output design** (WACV 2023) is correctly credited (p. 19). **V-JEPA 2** is still preprint-only
   (arXiv:2506.09985); the thesis's footnote (p. 31) is accurate.
5. **Code against equations:** Eqs. 4.3–4.8 and 4.14–4.17, Alg. 4.1's hop and interpolation, and the auxiliary-loss
   weighting all match the implementation (Section 2).
6. **The map-format asymmetry does not favour SURGE-D.**
   - **Concern:** SalFoM is scored from min-max-scaled 8-bit PNGs (`scripts/evaluate.py:123-128`) and SURGE-D from
     float maps (`:147`). With KLD's machine-epsilon floor (`surge/metrics.py:45`), exact zeros in SalFoM's maps
     could have inflated its KLD.
   - **Test:** every 10th frame of the 100 validation videos (6,076 frames).
   - **Result:**
     - Fewer than 0.005 % of SalFoM's pixels are zero, and they carry no ground-truth mass.
     - Adding a one-grey-level floor *worsens* SalFoM's KLD (1.302 → 1.358) and SIM (0.413 → 0.386); the
       corresponding values at σ = 16 are 1.283 → 1.349 and 0.401 → 0.373.
   - **Conclusion:** if anything, the 8-bit min-max format favours SalFoM on SIM.
   - Script and output: `report/evidence/zero_pixel_kld.py`, `report/evidence/zero_pixel_kld_output.txt`.
7. **Internal arithmetic**, re-derived from the tables, is consistent throughout:
   - ablation deltas and gate thresholds (0.5833, 0.5904);
   - learning-curve differences;
   - timing ratios across the three runs, and FLOP ratios (25×, 8.7×, 0.64×, 1.83×);
   - published-margin statements (0.046–0.049 above the best published validation CC).

**Not verified in this review:** the AVRSP numbers (T5.10), UniST's validation numbers (p. 50), V-JEPA 2.1's 22.2 mIoU
(p. 31), the remaining entries of the 132-item bibliography, whether PredJSal's backbone was frozen, LSTD-Net's DHF1K
numbers, and anything that would require re-training.

## Sources consulted

- SalFoM: M. Moradi et al., *SalFoM: Dynamic Saliency Prediction with Video Foundation Models*, ICPR 2024 (LNCS
  2025); arXiv:2404.03097 — <https://arxiv.org/html/2404.03097v1>. Official repository read locally
  (`D:/Research/baselines/SalFoM`: `README.md`, `Model.py`).
- DiffSal: J. Xiong et al., CVPR 2024; arXiv:2403.01226 — <https://ar5iv.labs.arxiv.org/html/2403.01226>.
- ViNet-S/A: R. Girmaji et al., *Minimalistic Video Saliency Prediction via Efficient Decoder & Spatio Temporal
  Action Cues*, ICASSP 2025; arXiv:2502.00397 — <https://arxiv.org/html/2502.00397v1>.
- TinyHD: F. Hu et al., WACV 2023 —
  <https://openaccess.thecvf.com/content/WACV2023/html/Hu_TinyHD_Efficient_Video_Saliency_Prediction_With_Heterogeneous_Decoders_Using_Hierarchical_WACV_2023_paper.html>.
- NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results, arXiv:2604.14816 —
  <https://arxiv.org/html/2604.14816>.
- V-JEPA 2: M. Assran et al., arXiv:2506.09985 — <https://arxiv.org/abs/2506.09985>.
- LSTD-Net (SalFoM follow-up), CVIU 2026, DOI 10.1016/j.cviu.2026.104740 (tables not accessed).
