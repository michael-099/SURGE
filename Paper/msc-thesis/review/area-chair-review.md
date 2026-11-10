# Area-chair review and revision plan

**Thesis:** *Video Saliency Prediction with a Frozen Self-Supervised Video Encoder* (SURGE-D), as built at
commit `bb197a7` (`Paper/msc-thesis/main.pdf`, 89 pp.).

**Inputs used.**
- **A (the paper):** the thesis text. It was audited against the repository: code (`surge/`, `scripts/`), splits
  (`configs/splits/`), run outputs (`runs/`) and the decision log (`research/log.md`). Chapter 5 states that the
  log is part of the material accompanying the thesis.
- **B (new research questions): not supplied.** The request contained only the `{PASTE: …}` placeholder.
  - Part 1 does not depend on B.
  - Part 2 therefore uses the thesis's own RQ1–RQ4 **verbatim** as a stand-in for B. They are neither loosened nor
    reworded; Section 7 only adds operational definitions to them.
  - If B differs, Sections 6–10 must be redone against it.

**Conventions.**
- "§5.5.1" is a thesis section; "T5.7" is Table 5.7; "Fig 5.2" is Figure 5.2; "log §11A" is a section of
  `research/log.md`.
- "(derived)" marks arithmetic I did on numbers that are in the input.
- "Unverified, check this" marks outside knowledge that I could not check here.

---

# Part 1: Review

## Table 1: Claims, evidence and status

| # | Claim (where it is made) | Evidence offered | Status |
|---|---|---|---|
| C1 | On DHF1K val, SURGE-D ViT-g beats SalFoM on all six metrics: CC 0.600 single and 0.603 ensemble, against 0.564; paired intervals exclude 0 (Abstract ¶3; §1.8; §6.1 RQ1) | T5.4, T5.5, TA.1, Fig 5.2: one re-scored baseline, 100 videos | **Supported**, as an end-to-end comparison of two systems |
| C2 | The comparison is "under an identical protocol" (RQ1 text; §5.5.1; §6.1 RQ1; §6.2) | Same evaluator, videos and ground truth (§5.2). But SURGE-D is non-causal (§4.2), the output blur was tuned only for SURGE-D (§4.7), and SalFoM's inference settings are undocumented | **Overstated**: the evaluation is identical, the protocol is not |
| C3 | A frozen V-JEPA 2 encoder is "sufficient … without adapting it"; it is open whether "adaptation is necessary" (Abstract ¶1; §1.3; §6.1 RQ1) | C1, plus one LoRA configuration on ViT-L (T5.10). Full fine-tuning was not tested (§1.7) | **Overstated** |
| C4 | The gain "speaks to the quality of the pretrained V-JEPA representations rather than to a training technique", and G1 supports this (§5.7 RQ1) | G1 used one seed per encoder, and UMT-L was run at 224² against 256² (T5.2); LoRA (T5.10) | **Overstated**: attribution not isolated |
| C5 | About 1/6 (ViT-g) and 1/17 (ViT-L) of SalFoM's encoder compute per output frame; "at a fraction of its computational cost" (Abstract ¶3; §1.8; §5.5.4; §6.1 RQ1; §6.2) | T5.9 for SURGE-D. For SalFoM: UMT-L FLOPs counted on a re-implementation (`scripts/count_umt_flops.py`) and the assumption "once per output frame". SalFoM was not timed | **Overstated**: offline-only, and an estimate |
| C6 | The predictor error is a weaker cue than frame difference and atypicality; the raw error is anti-correlated with fixations (Abstract ¶3; §5.4) | T5.3: 60 holdout videos, two frame rates, no intervals | **Supported** for this predictor, this readout, used standalone |
| C7 | The error "mainly reflects the atypicality of regions rather than violated expectations" (§1.8; §5.4; §6.1) | Diagnostic on 11 clips without gaze data; r ≈ 0.60, 0.62 and −0.11 (§5.4) | **Overstated**: a plausible mechanism, not tested |
| C8 | "the predictor error of a self-supervised video model is not a fixation cue" (§6.2) | One model, one family of readouts | **Unsupported** as a general claim |
| C9 | No information "beyond that carried by simple appearance and motion cues" (RQ2 text; §5.7 RQ2) | Only standalone comparisons. Information gain was computed for a fixed 0.5/0.5 mixture with the centre prior (`scripts/g2_surprise.py` l.8, 57–64), not for a fitted combination with the reference cues | **Unsupported**: the "beyond" test was not run |
| C10a | Encoder scale gives a large gain (§5.6; §5.7 RQ3) | Holdout: higher for all 3 seeds. Val: +0.0167 [+0.0116, +0.0222] (TA.2) | **Supported** |
| C10b | Encoder pre-training gives a large gain: V-JEPA 2 > V-JEPA 2.1 > UMT-L (Abstract; §5.7; §6.1) | T5.2: one seed each; V-JEPA 2.1 run below its native resolution, UMT-L at 224² | **Overstated** |
| C11 | LoRA, centre prior, deep supervision, SALICON, longer clips, more videos and audio produce no gain "that could be measured"; "the results separate the factors cleanly" (Abstract; §1.8; §5.7; §6.1) | T5.10: one seed for the components; one LoRA configuration; the 32-frame run is one unconverged seed; train-600 was evaluated on val. T5.11: ViT-L, seed 0 | **Overstated** |
| C12 | With a strong encoder, dataset size "rather than the capacity of the decoder limits further progress" (§5.7, hedged as "a consistent explanation"; §6.1, hedged as "suggests") | Fig 5.1 curves only. No data-scaling run; 11 % more data gave no CC gain | **Unsupported** (not shown, not shown false) |
| C13 | Seed ensembles add "a small but reliable gain" (§1.8; §5.7) | T5.4 (no interval for ensemble vs single on DHF1K); UCF +0.0080 [+0.0039, +0.0123]; on Hollywood the ensembles were significantly lower on NSS and SIM (§5.5.2) | **Overstated**: reliable for CC only |
| C14 | Competitive with the best published results on UCF Sports (Abstract; §6.1) | T5.6: 0.735 against ViNet-E 0.744 (−0.009); lower on CC, NSS and SIM | **Supported** as "competitive, not ahead" |
| C15 | "exceeded the best peer-reviewed results that could be verified on five of the six audio-visual datasets" (Abstract); "a higher correlation … on five of the six" (§6.1) | T5.7: against the best AV result, AVAD is +0.002, a tie under the thesis's own rule (T4.1) | **Overstated** |
| C16 | Fails on Coutrot2 because gaze follows the speaker (Abstract; §5.7 RQ4) | T5.7 (−0.043 / −0.018); Fig 5.3 row 5; the rotated-audio drop on Coutrot2 (T5.11) | Result **supported**; cause plausible, untested |
| C17 | Audio fusion did not improve any dataset (Abstract; §5.6; §6.1) | T5.11: ViT-L, seed 0, one module, rotated-audio control | **Supported** for this module and setting |
| C18 | "Every design choice was tested under a rule written before the experiment"; "pre-registered" (Abstract ¶2; §1.3; §1.6; §6.1; §6.2) | T4.1 lists 11 rules, one of them set after the fact (§4.1). Most architectural and training choices were never tested | **Overstated** |
| C19 | DHF1K val "was never used for selection"; every test set "scored once per reported model" (§4.1; Ch. 5 intro) | The log shows val chose the model for the official test submission (§8) and decided the 600-video question (§11A). Each reported model was indeed scored once | **Overstated**: first half incorrect as worded |
| C20 | The evaluator is validated against reference code (Abstract; §5.2) | CC and NSS agree within 0.0004 of the TMFI-Net code. SIM, KLD and AUC-J were checked against "a second, independently written evaluator" of unstated origin. sAUC has no external check | **Supported** for CC and NSS; partial for the rest |
| C21 | "a new, openly reproducible Hollywood-2 ground truth" (§1.8; §6.2) | Recipe in §4.3; data CC BY; at most five observers; no reliability analysis | **Supported** as a recipe; overstated as a benchmark |
| C22 | Novelty: first frozen feature-prediction encoder for video saliency, and first test of a learned latent predictor's error against fixations, "in the peer-reviewed studies whose full texts could be consulted" (§3.6, §3.7) | Chapters 2–3 | **Supported** as qualified; completeness unverified |

---

## Section 2: Attacks, grouped by claim

### 2.0 Implementation audit: what was built, run and measured

| Item | Thesis | What the repository shows | Consequence |
|---|---|---|---|
| **Splits** | DHF1K 540/60/100; UCF 93/10/47; Hollywood (Vig) 741/82/884; STAViS folds with a "holdout = 10 % of each training fold" (T4.2) | All match `configs/splits/`. The AV holdouts contain **1 video per fold for Coutrot2 and ETMD**, 2 for SumMe, 3 for AVAD, 4 for Coutrot1 and 6 for DIEM | In 6 of 16 folds, epoch and σ are chosen on a single video. This is selection noise, not leakage, and it is unstated |
| **Counts** | SumMe "25 videos"; DIEM "84 videos", split 64/17 (T4.2) | The bootstrap pools 23 SumMe test videos (TA.4); the log notes "SumMe 23/25 videos with audio" (log §11D). DIEM 64+17 = 81 | Unexplained in the thesis |
| **Baseline (SalFoM)** | "produced on the same machine with the authors' released weights" (§5.2) | Not retrained. The maps were generated in a **separate repository**, before this project started: `D:/Research/baselines/salfom_val/summary.json` records `scripts\baselines\salfom_dhf1k.py` at commit `9a50089` of the GRAFT-A project, bf16, batch 2. Neither the wrapper nor its settings (clip length, sampling, resizing, post-processing) are in the thesis or this repository | The main claim's baseline cannot be regenerated from the thesis code (A9) |
| **Other baselines** | Copied from papers (T5.4, T5.6, T5.7) | No other baseline was re-scored | Every other comparison crosses evaluation code |
| **Tuning budget** | — | **SURGE-D:** best epoch of 3 and σ from a 6-value grid, both on the holdout; 3 seeds; architecture fixed in advance. **SalFoM:** no tuning, and no σ. **LoRA:** one configuration (r = α = 16, learning rate 1e-4); "its best epoch was always the first" (§5.6). **Audio:** one module, ViT-L, seed 0. **32-frame:** one seed, "still improving at its last epoch" (§5.6) | The baseline and the adaptation arm received the least tuning |
| **Seeds** | Seeds 0, 1, 2; "single" = seed 0 | Fixed in advance (log §4, l.77). Only `val_single_s0.json` exists in `runs/final`, `runs/vitg` and `runs/vitg600`, so seeds 1 and 2 were never scored singly on val. Components, G1, audio and 32-frame used one seed | Seed variance on the test set is unknown |
| **Metrics** | Six metrics at native resolution; one CC-tuned map for all metrics (§4.7); sAUC sampling specific to this thesis | As stated | Disclosed (§5.8) |
| **Compute** | One RTX A5000; about 15 min (ViT-L) and 31 min (ViT-g) per 3-epoch run (§5.1.1) | FLOPs from the PyTorch counter; UMT-L FLOPs on a CPU re-implementation | New runs are cheap (Section 9) |
| **Ablation design** | One factor at a time, from ViT-L seed 0; \|ΔCC\| ≥ 0.003 with one seed (T4.1) | As stated; the conclusions were applied to ViT-g without testing | Low power (A5) |
| **Leakage / test reuse** | Val "never used for selection" (§4.1) | Val was used for two decisions (log §8 and §11A). The G2 centre prior comes from training videos (`g2_surprise.py` l.32, 82), so that part is fine. V-JEPA 2 pre-training may overlap DHF1K's YouTube sources (acknowledged in §5.8; unverified) | Reported headline numbers are unaffected; the protocol statement is wrong (A8) |
| **Stored outputs** | — | G2 kept per-video scores (`runs/g2/g2_results.json`) but not the maps | Intervals for T5.3 need only analysis; an incremental test needs the maps recomputed |

**Missing information, treated as weaknesses:**
- SalFoM's inference settings and the wrapper that produced its maps.
- Seed-level validation results.
- Intervals for G2.
- The SumMe and DIEM counts.
- The origin of the "second, independently written evaluator" (§5.2).
- Whether SalFoM's released DHF1K checkpoint saw the validation videos in training. If it did, the comparison favours SalFoM. Unverified, check the release notes.

### 2.1 Attacks

**A1. C1, C2: temporal access is a confound in the headline comparison.**
*Severity:* MAJOR. *Fix:* new experiment and changed claim.
- *(b) Confound.*
  - SURGE-D sees "at least eight original frames (about 0.27 s)" after the predicted frame (§4.2). SalFoM "predicts the last frame of its input clip" (§5.5.1).
  - The thesis names this confound (§5.5.1 "Two qualifications"; §5.8) but never measures it.
  - The abstract and §6.2 still say "under an identical protocol". §1.7 puts causal prediction out of scope, yet RQ1 demands an identical protocol. The thesis therefore scoped out the one experiment its own RQ requires.
- *(c) Fairness.*
  - SURGE-D's output blur is tuned on the holdout over six values (T4.3); SalFoM's maps are scored as produced.
  - Fig 5.4 shows σ moves SURGE-D's CC by at most 0.003, so the asymmetry is unlikely to explain +0.036. It is still an asymmetry that one analysis removes.
- *(d) Statistics.*
  - The "single" rows are seed 0 only. On the holdout, the seed SD is about 0.0016 (ViT-L) and 0.0022 (ViT-g) (derived from the values in §5.6), an order of magnitude below the +0.036 gain, so seed noise is not a credible rival.
  - It should nonetheless be shown on val, where the seed spread was never measured.
- *(f) Simplest rival.* "Bidirectional context, not the frozen V-JEPA 2 encoder, produces most of the gain." This predicts that a causal SURGE-D loses most of ΔCC.
- *Settles it:* F2 and F5.

**A2. C3, C4: "Frozen is sufficient" and "the gain reflects representation quality" are not isolated.**
*Severity:* MAJOR. *Fix:* changed claim, plus new experiments.
- *(b) What else differs.* SURGE-D and SalFoM differ in:
  - encoder objective and pre-training data;
  - encoder size (ViT-g 1,012M parameters);
  - decoder;
  - temporal formulation;
  - loss and schedule;
  - preprocessing (a 256² squash).

  The ViT-L variant, at 303.9M parameters, removes the size confound and still wins: +0.020 single, +0.027 ensemble (T5.5). That is the strongest evidence the thesis has, and it should lead.
- *(g) Under-tuned adaptation arm.*
  - The only adaptation arm is one LoRA configuration on ViT-L. "Its best epoch was always the first" (§5.6) is the signature of too high a learning rate or too many steps.
  - The per-seed deficits (−0.0029, −0.0010, −0.0037) give a paired t ≈ −3.2, df = 2, two-sided p ≈ 0.09 (derived). The data show no gain from this setting; they do not show harm.
  - Full fine-tuning was not run (§1.7).
- *(f) G1 cannot carry the attribution.* It uses one seed, UMT-L at 224² against 256², and a *frozen* UMT-L, which is not SalFoM's fine-tuned UMT-L.
- *Supportable claim:* "A frozen V-JEPA 2 encoder with this decoder was enough to exceed SalFoM's released model, and one LoRA setting did not improve it."
- *Settles it:* F4, F10.

**A3. C5: the efficiency comparison crosses temporal regimes and is an estimate.**
*Severity:* MAJOR. *Fix:* changed claim, plus a cheap measurement.
- *(b) The amortisation depends on non-causal operation.*
  - The 16× amortisation (Alg. 4.1) depends on the bidirectional multi-frame output.
  - With this bidirectional encoder, a strictly causal SURGE-D can serve only the last frame of each pass. Its per-output-frame cost is then the per-clip cost: 1.656 + 0.166 = 1.822 TFLOPs (ViT-L) and 5.088 + 0.168 = 5.256 TFLOPs (ViT-g) (derived from T5.9).
  - That is comparable to SalFoM's 1.90 TFLOPs for ViT-L, and about 2.8× more for ViT-g. The advantage therefore exists only offline.
- *(i) Reproducibility.*
  - The SalFoM side is a FLOP count of a UMT-L re-implementation plus the assumption "once per output frame" (`count_umt_flops.py` docstring).
  - SalFoM's decoder and its wall-clock time on the same GPU were not measured, although its code and weights are on the machine.
- *Note for Part 2:* no RQ concerns efficiency, so the claim leaves the abstract.
- *Settles it:* F6.

**A4. C6–C9: RQ2 measures standalone accuracy, not information "beyond" simple cues, and the readout is out of distribution.**
*Severity:* MAJOR for the RQ2 answer; MINOR for the standalone result. *Fix:* analysis, a cheap zero-shot experiment, and a changed claim.
- *(a) Construct validity.*
  - RQ2 asks about information *beyond* appearance and motion cues. The gate (T4.1) tests whether surprise *beats* them standalone.
  - A cue can be weaker alone and still add information, or be strong alone and redundant.
  - The only combination computed is a fixed 0.5/0.5 mixture with the centre prior.
- *(a, f) Readout.*
  - The predictor was trained with masks spanning the whole clip in time (footnote in §5.4) but is queried to predict the second half from the first.
  - Its error therefore mixes "surprise" with out-of-distribution query error.
  - The thesis offers this as the explanation (§5.4) and then generalises beyond it (§6.2, C8).
- *(d) Statistics.*
  - T5.3 has no intervals over the 60 videos, although the per-video scores exist (`runs/g2/g2_results.json`).
  - The contextual variant has positive NSS at 4 fps (+0.116); "anti-correlated" is accurate only for the raw error at both rates and the contextual error at 15 fps.
- *(f) Mechanism.* The "because atypicality" account rests on 11 clips without gaze data.
- *Settles it:* F1a, F1b, F9.

**A5. C11–C13: the null results are underpowered and partly misreported.**
*Severity:* MAJOR. *Fix:* cheap new experiments, plus changed claims.
- *(d) Power of the single-seed rule.*
  - The rule is \|ΔCC\| ≥ 0.003 with one seed per arm (T4.1). The SD of a difference between two single-seed runs is √2 × 0.0016–0.0022 ≈ 0.0022–0.0031 (derived), so the threshold sits at about 1.0–1.4 SD.
  - With no true effect, the rule flags a "difference" roughly 17–33 % of the time. With a true effect of 0.003, it detects it about half the time (derived; the SD itself comes from only three seeds).
  - §5.8 concedes that "effects of a few thousandths of CC cannot be excluded". The abstract, §1.8 ("add nothing measurable"), §5.7 ("separate the factors cleanly") and §6.1 nevertheless report the nulls as findings.
- *(g) Inconclusive arms reported as nulls.*
  - The 32-frame run missed its gate by 0.0002 while "still improving at its last epoch" (§5.6; log §11C). That is inconclusive, not null.
  - LoRA was run in one configuration; audio with one seed on ViT-L only.
  - All component conclusions were transferred to ViT-g without testing.
- *(d) Selective reporting, 600 vs 540 videos.*
  - T5.10 and §5.6 report only the ensemble's ΔCC (+0.0015 [−0.0006, +0.0037]) and say "within noise".
  - Log §11A also records, for seed 0:
    - significant NSS gain (+0.039);
    - significant SIM gain (+0.006);
    - significant sAUC loss (−0.006).
  - For the ensemble, it records a significant SIM loss (−0.0017 [−0.0032, −0.0001]).
  - The decision rule was based on CC, so the decision stands. The sentence "neither … made a measurable difference" does not.
- *(f) C12 is untested.* No data-scaling run was made. The flat +11 % result is, if anything, weak evidence against a steep dependence on data size.
- *C13.* On Hollywood (Vig gaze), ensembles significantly lowered NSS and SIM (§5.5.2); no ensemble-vs-single interval is given for DHF1K.
- *Settles it:* F7, F8, F11.

**A6. C10b: the G1 ranking is one seed and confounded by input settings.**
*Severity:* MINOR as a gate decision; MAJOR if used as evidence for the "pre-training" claim. *Fix:* new experiment and changed claim.
- V-JEPA 2 led V-JEPA 2.1 by 0.009, about 3–4 SD of a single-seed difference (derived), so it is probably not noise. But 2.1 ran at 256² instead of its 384² cool-down (§5.3), and UMT-L ran at 224² with a different token layout.
- "Pre-training produced large gains" conflates objective, data, distillation and resolution. Supportable: "the V-JEPA 2 checkpoint was better under these settings (one seed)."
- *Settles it:* F12.

**A7. C15: the audio-visual "five of six" contradicts the thesis's own rule and depends on the comparator set.**
*Severity:* MAJOR. *Fix:* changed claim (analysis only).
- *(d) The thesis's own rule.*
  - Under T4.1's tie rule (below 0.01 CC is a tie), the last row of T5.7 against the best audio-visual result gives:
    - higher: DIEM +0.044, Coutrot1 +0.019, ETMD +0.024, SumMe +0.018;
    - tie: AVAD +0.002;
    - lower: Coutrot2 −0.018.
  - "Exceeded … on five of six" (Abstract) is false by the thesis's own rule. §5.5.2 and §5.7 ("matched or exceeded") state it correctly.
- *(c) The margin is not calibrated.*
  - The tie margin was set after the first AV results (§4.1). It is smaller than the cross-paper disagreement the project recorded for the same models on the same data (log §7): STAViS on Coutrot2 is reported as 0.734 in one paper and 0.652 in another; ViNet as 0.754 and 0.724.
  - Margins of 0.018–0.024 lie inside that noise.
- *(c) The comparator set changes the conclusion.*
  - On 2026-11-06, before any SURGE-D AV number existed, the log recorded DTFSal and DTFSal(V) as the strongest published results on Coutrot1, Coutrot2, ETMD, AVAD and SumMe (log §7, then a preprint, arXiv 2504.10070).
  - Against those, the ViT-g ensemble (log §11E):
    - ties on Coutrot1 (−0.004), ETMD (+0.009) and AVAD (−0.006);
    - is clearly higher only on DIEM and SumMe;
    - is lower on Coutrot2.
  - The thesis excludes non-peer-reviewed results by policy and says so in general terms (§5.5.2). It does not say that the exclusion removes *stronger* results, so a reader cannot see that the conclusion depends on the comparator set.
- *(c) Unequal comparison.* A three-seed ensemble is compared with single published models; the single ViT-g model's margins are smaller (T5.7).
- *Author decision required:*
  - either name the non-peer-reviewed comparator, which needs an exception to the citation policy;
  - or state the sensitivity without naming it, and drop the superiority wording.

**A8. C18, C19: the protocol claims are stronger than the protocol.**
*Severity:* MAJOR for credibility, because rigour is the thesis's distinctive selling point. *Fix:* text only.
- *"Every design choice was tested under a rule written before the experiment" (Abstract).*
  - Never tested: decoder architecture, tap layers, loss weights (T4.3), input squashing ("its effect on accuracy was not ablated", §4.3), learning rates, number of prior components, output positions.
  - One rule was written afterwards (§4.1).
- *"Pre-registered."* The rules are dated entries in the author's own log in the same repository, not an external, time-stamped registration. "Specified in advance in a dated log" is accurate.
- *"The validation set … was never used for selection" (§4.1).* This is false as worded:
  - log §8 chose the official-test submission as "the configuration that scored best on DHF1K val", and a later archive was preferred for its higher val CC (log §11B);
  - log §11A decided on val whether the 600-video models would replace the 540-video ones.

  The reported models were fixed on the holdout, so no reported number is biased. The sentence must still be corrected, and a ledger of test-set uses added.

**A9. C20, C21: reproducibility and measurement validity.**
*Severity:* MINOR to MAJOR. *Fix:* documentation and analysis.
- *(i) Baseline provenance.* The SalFoM maps were generated outside the thesis repository (2.0), so a third party cannot regenerate the baseline of the main claim from the thesis code. This is the most important reproducibility gap.
- *(a) Evaluator validation.* The "second, independently written evaluator" (§5.2) is not identified, and sAUC has no external check.
- *(a) Hollywood-2 (Vig gaze) ground truth.*
  - Each "fixation" is the median of one observer's non-saccade samples within a frame (§4.3). That is closer to a gaze position than to a detected fixation.
  - With at most five observers, the ceiling is unknown, so the +0.004 CC model differences on these data (T5.8) cannot be related to it.

**A10. Generalisation and multiple comparisons.**
*Severity:* MINOR. *Fix:* analysis and changed claim.
- *(e)* Every design-factor conclusion comes from the DHF1K holdout with ViT-L. Transfer was measured after per-dataset fine-tuning, with selection on 1–6 videos per AV fold. Zero-shot transfer is much weaker (T5.6: 0.608 against 0.735; zero-shot rows of T5.7).
- *(d)* Six metrics and many comparisons with no adjustment.
  - This is harmless for the SalFoM comparison (TA.1 intervals are far from zero), except for ViT-L single NSS.
  - It matters for the borderline secondary findings (Hollywood ensembles, train-600 metrics).
  - Fix: declare CC primary, as the rules already do, and Holm-adjust the rest.

**A11. C22: novelty.**
*Severity:* MINOR. *Fix:* changed claim.
- *(h)* SURGE-D is a composition of known parts:
  - a simple feature pyramid from a plain ViT [`li2022exploring`];
  - R(2+1)D blocks [`tran2018closer`];
  - deep supervision [`lee2015deeply`];
  - a learned centre-bias prior.

  Multi-frame output may also exist in prior work: T3.1 calls TinyHD "multi-output" (unverified whether this means multiple frames per pass; check this).
- What is new: a frozen feature-prediction encoder evaluated for this task, the predictor-error test, and the evaluation discipline. The contribution list should lead with these rather than with SURGE-D as a model.
- The completeness of the novelty search, including frozen non-JEPA foundation encoders for saliency, is unverified; check this.

**Severity summary.**
- FATAL: none.
- MAJOR:
  - A1, A2, A3, A4, A5, A7 and A8.
  - A9 only for the baseline provenance.
- MINOR: A6 (unless used for C10b), A10 and A11.

---

## Section 3: Falsification experiments, ranked by information gained per cost

Cost basis: §5.1.1 gives about 15 min (ViT-L) and 31 min (ViT-g) per 3-epoch DHF1K run; log §11C gives about 30 min per epoch for 32-frame clips. GPU-hour figures below are estimates.

1. **F1a: intervals for Gate G2** (C6, C9).
   - *Method:* video-bootstrap intervals for T5.3 and for each surprise − reference difference, from the stored per-video scores. *Cost:* minutes (CPU).
   - *Refutes "anti-correlated":* the raw-surprise NSS interval includes 0 at a given frame rate.
   - *Refutes "weaker than frame difference":* the paired interval for surprise − frame difference includes 0.
2. **F2: SalFoM at its best output blur** (C1, C2).
   - *Method:* apply the same σ grid {0, 4, 8, 12, 16, 24} to SalFoM's val maps and report SalFoM at its best σ *on val*. This upper bound favours the baseline. *Cost:* minutes (CPU).
   - *Refutes C1:* the ViT-g single or ensemble ΔCC interval against best-σ SalFoM includes 0.
3. **F3: seeds 1 and 2 scored singly on val** (C1).
   - *Method:* inference only. *Cost:* about 1–2 GPU-h.
   - *Refutes "single model better":* any seed's ΔCC interval against SalFoM includes 0, or the seed SD on val is comparable to the gain.
4. **F4: frozen UMT-L + SURGE-D decoder on val** (C4).
   - *Method:* score the existing G1 checkpoint (`runs/g1_umt_l/best.pt`); inference only. *Cost:* < 1 GPU-h.
   - *Refutes C4's attribution:* the frozen UMT-L system's interval against SalFoM lies at or above 0. The gain would then not be specific to V-JEPA 2's representations.
   - *Supports the "encoder objective" reading:* it lies clearly below SalFoM and below SURGE-D ViT-L.
5. **F5: causal SURGE-D** (C1, C2; the blocker for RQ1).
   - *Method:* the same recipe and seeds, but the target is the last sampled frame of each window, so no frame after the predicted one enters the input. ViT-L and ViT-g, 3 seeds each; σ on the holdout; scored once on val.
   - *Cost:* about 3–4 GPU-h of training. Causal inference costs one pass per output frame, so scoring on a pre-declared every-5th-frame subset, for both models on the same frames, keeps it to a few GPU-h.
   - *Refutes "better under an identical protocol":* the causal ensemble's ΔCC interval against SalFoM includes 0 or lies below it.
6. **F6: measured cost** (C5).
   - *Method:* time SalFoM (released code) and SURGE-D, offline and causal, on the A5000 with the same frame loading. *Cost:* < 1 GPU-h.
   - *Refutes the offline efficiency claim:* SalFoM's measured throughput is within a factor of 2 of SURGE-D ViT-g offline.
   - *Expected to refute any causal efficiency claim:* on the FLOPs alone (A3).
7. **F7: three seeds per component** (C11).
   - *Method:* seeds 1 and 2 for no prior, no auxiliary heads, SALICON and stride 4 (8 ViT-L runs); paired seed-level differences with a two-level bootstrap; TOST at ±0.003. *Cost:* about 2–4 GPU-h.
   - *Refutes "no effect" for a factor:* its 95 % interval excludes 0 with \|Δ\| ≥ 0.003.
   - *Supports "no effect larger than 0.003":* the 90 % interval lies within ±0.003. Anything else is "inconclusive".
8. **F8: learning curve** (C12).
   - *Method:* 135 / 270 / 540 training videos, 3 seeds each, ViT-L. *Cost:* about 2–3 GPU-h.
   - *Refutes C12:* holdout CC is flat (within ±0.003) from 270 to 540 videos.
9. **F1b and F9: incremental information, and an in-distribution readout** (C9, C8).
   - *Method:* recompute the G2 maps (zero-shot GPU job). Fit a log-linear combination of {centre prior, frame difference, atypicality} with and without surprise; cross-fit over two halves of the 60 videos; report the incremental information gain (bits per fixation) with a video-bootstrap interval. Repeat with a readout that matches pre-training (spatial tube masks over the whole clip).
   - *Cost:* about 1–2 GPU-h plus CPU.
   - *Refutes the RQ2 answer "no":* the incremental gain's interval lies above 0 for a pre-specified readout.
10. **F10: adaptation with a fair budget** (C3, C11).
    - *Method:* a pre-specified LoRA grid (learning rate {1e-5, 3e-5, 1e-4} × rank {4, 16}), selected on the holdout, then 3 seeds of the best. Optionally full fine-tuning of ViT-L.
    - *Cost:* medium for the grid (about 9 runs); high for full fine-tuning (memory and time).
    - *Refutes "adaptation does not help":* the best adapted arm beats the frozen encoder by ≥ 0.003 with its interval above 0.
11. **F11: 32-frame clips with a converged schedule** (C11).
    - *Method:* ViT-g, up to a pre-specified maximum number of epochs, 3 seeds. *Cost:* medium, about 0.5 GPU-h per epoch and seed.
    - *Refutes "longer clips do not help":* the gain over 16 frames is ≥ 0.003 with its interval above 0.
12. **F12: G1 with 3 seeds and native resolutions** (C10b).
    - *Method:* V-JEPA 2.1 at 384²; V-JEPA 2 and UMT-L both at 224². *Cost:* medium.
    - *Refutes the ranking:* V-JEPA 2.1 at 384², or UMT-L at matched resolution, is within ±0.003 of V-JEPA 2 or above it.
13. **F13: reliability of the Hollywood (Vig) ground truth** (C21).
    - *Method:* leave-one-observer-out CC on the test clips (the inter-observer ceiling), and re-scoring the models
      against maps built from different subsets of observers. *Cost:* CPU.
    - *Refutes its usefulness as a benchmark:* the ViT-g vs ViT-L ranking, or the sign of the ensemble effects, changes
      between observer subsets; the 0.004 CC model differences are then within the ground truth's own variation.
14. **F14: re-scored AV baselines** (C15).
    - *Method:* run any AV baseline whose weights are released through the thesis evaluator. Weight availability is unverified; check this. *Cost:* medium to high.
    - *Refutes C15:* any re-scored baseline is within the tie margin or above SURGE-D on the datasets claimed as wins.

---

## Section 4: Steelman

The headline comparison is more careful than most in this field: the baseline was re-run from its released weights
and scored on the same 100 videos by an evaluator that reproduces the reference CC to within 0.0003, and the ViT-g
single model's +0.036 CC gain is an order of magnitude above the measured seed spread, holds on all six metrics, and
recurs with the ViT-L encoder, which is the size of SalFoM's (the ensemble is ahead on 79 of 100 videos). The
headline models were fixed on a holdout before the validation set was scored, and the remaining confounds (future
frames, one baseline, pending test-set scores) are disclosed in §5.5.1 and §5.8. The negative results are reported
against rules written before the runs, the thesis never claims a state of the art on the audio-visual data, and §5.7
scopes the surprise result to one predictor. This defence does not answer A1 (the future-frame advantage is
disclosed but not measured), A3 (the cost claim crosses temporal regimes), A5 (the nulls lack power), A7 (the
audio-visual conclusion depends on the comparator set) or A8 (the misstatements about pre-registration and the use
of the validation set).

---

## Section 5: Verdict and top-3 changes

**Claims that survive as written:**
- C1 (its "identical protocol" wording is C2, below);
- C10a; C14; C16 (the result); C17;
- C20 for CC and NSS;
- C22 as qualified.

**Claims that must be weakened:**
- C2: "identical evaluation".
- C3, C4: "was enough to exceed SalFoM; adaptation beyond one LoRA setting untested; source of the gain not isolated".
- C5: "offline only; estimate", and out of the abstract.
- C6, C7: scoped to this predictor and readout; the mechanism is offered as a hypothesis.
- C10b: "this checkpoint under these settings, one seed".
- C11: effect, equivalent or inconclusive per factor; 32-frame inconclusive; train-600 mixed.
- C13: "CC gain; lowers NSS and SIM on sparse ground truth".
- C15: "four of six, a tie on AVAD, sensitive to the comparator set and within cross-paper noise".
- C18, C19: accurate protocol wording, plus a ledger.
- C21: "recipe; reliability to be reported".

**Claims that must be dropped:**
- C8, the general form.
- C9, until F1b is run.
- C12 as a finding; keep it as a hypothesis.
- "The results separate the factors cleanly."

**Top-3 changes.**
1. **Measure the confounds of the main comparison instead of only disclosing them.** Run:
   - the causal SURGE-D (F5);
   - SalFoM at its best σ (F2);
   - seeds 1 and 2 singly on val (F3);
   - frozen UMT-L on val (F4).

   Answer RQ1 from these, keeping the offline result as secondary. Also document the SalFoM inference in the repository.
2. **Turn the nulls into estimates.**
   - Use three or more seeds per arm, with an effect / equivalent / inconclusive outcome against a pre-set margin (F7, F8, F11).
   - Report 32-frame as inconclusive and the train-600 result on all metrics.
   - Drop the data-limitation explanation unless F8 supports it.
3. **Correct every claim that the thesis's own evidence contradicts.**
   - The AV "five of six".
   - "Identical protocol".
   - "Every design choice … before the experiment".
   - "Never used for selection".
   - The general surprise claim.

   Also run the cheap "beyond simple cues" analysis that RQ2 actually asks for (F1a, F1b).

---

# Part 2: Align and revise

> **Stand-in for B.** The RQ text below is the thesis's RQ1–RQ4, verbatim (§1.4). Section 7 adds, for each RQ, a
> primary metric, hypothesis, control and falsification condition without changing the question. These are
> stricter than the thesis's current operationalisation, never looser. If you supply a different B, redo
> Sections 6–10 against it.

## Section 6: RQ gap table

| RQ | Existing coverage | Missing | Action |
|---|---|---|---|
| **RQ1** (accuracy against SalFoM under an identical protocol) | **Partly.** T5.4, T5.5, TA.1, Fig 5.2: identical *evaluation*, offline only; seed 0 and the ensemble; one baseline | The identical *protocol*: causal access and the same post-processing. Seed spread on val. A documented SalFoM run. A control separating encoder from training method (frozen UMT-L on val) | F2, F3, F4, **F5 (blocker)**; add the SalFoM wrapper and its settings; restate the answer with the causal result as primary |
| **RQ2** (surprise *beyond* simple cues) | **Partly.** T5.3: standalone zero-shot scores on 60 holdout videos without intervals; diagnostic on 11 clips | The "beyond" test (incremental information); intervals; a readout that matches pre-training | **F1a, F1b (blocker for the RQ2 answer)**, F9; scope the conclusion to one predictor and its readouts |
| **RQ3** (which factors matter) | **Partly.** Scale: adequate (3 seeds plus val). Pre-training: 1 seed, confounded. LoRA: 3 seeds, 1 configuration. Components: 1 seed each. Temporal sampling: stride 4 at the threshold (1 seed); 32 frames unconverged (1 seed). Data amount: one point (+11 %), measured on val. Ensembling: CC only | Power and equivalence testing; seeds; an adaptation grid; a converged 32-frame run; a learning curve; an encoder comparison at native settings; ensemble-vs-single intervals | **F7 (blocker for RQ3 claims)**, F8, F10, F11, F12; report each factor as effect, equivalent or inconclusive |
| **RQ4** (transfer; does audio help) | **Mostly.** T5.6–T5.8 zero-shot and fine-tuned; T5.11 audio with rotated control | Comparisons under the thesis's own tie rule; a margin calibrated to cross-paper disagreement; a statement of comparator-set sensitivity; single vs single comparisons; a note that AV folds select on 1–6 videos; Hollywood (Vig) reliability; audio with more than one seed | Rewrite (analysis only); F13; optional F14 and audio with 3 seeds |

**Material that answers no RQ:**
- §5.5.4 computational cost, and the cost sentences in the Abstract, §1.2, §1.8, §5.5.4, §6.1 and §6.2.
  - No RQ asks about cost.
  - Keep it as a clearly labelled *secondary outcome* in Chapter 5, limited to offline use and with SalFoM measured (F6).
  - Remove it from the abstract and the contribution list. If B adds an efficiency RQ, it returns.
- §5.5.3 qualitative examples: illustrative only. Move them to Appendix A, keeping the selection rule.
- §4.3, the paragraph on building the Hollywood-2 (Vig gaze) maps: move to an appendix. The results stay under RQ4.
- §5.2, evaluator validation: move into the protocol section, since it supports every RQ.
- Chapter 2's broader background (eye movements, attentional synchrony, world models, transfer learning): the thesis
  format needs it, but it should be trimmed to what RQ1–RQ4 use. A paper version would cut it.

---

## Section 7: Revised research-question section (LaTeX)

```latex
% Replaces Section 1.4. RQ wording unchanged; operational definitions added.
% Requires enumitem (already used by the thesis).
\section{Research Questions}\label{sec:questions}

This study answers four research questions. For each, the primary metric, the
hypothesis, the control and the condition that rejects the hypothesis are stated
below. They were written into the research log on \RESULT{date}, after the first
round of experiments and before the experiments that answer them in this form
(\cref{sec:protocol}); earlier results are used only where they meet these
definitions. A rejected hypothesis is reported as a result.

\begin{enumerate}[label=\textbf{RQ\arabic*.},leftmargin=*]
  \item Can a frozen self-supervised video encoder of the \acrshort{V-JEPA} family, combined
    with a lightweight spatio-temporal decoder, predict human fixations in video
    more accurately than the closest foundation-model approach, SalFoM, when both
    are evaluated under an identical protocol?
    \begin{description}[leftmargin=1.2em,font=\normalfont\itshape]
      \item[Primary metric.] Per-video \acrshort{CC} on the 100 DHF1K validation videos.
        The difference SURGE-D $-$ SalFoM is estimated by a paired bootstrap over
        videos (10{,}000 resamples, 95\,\% percentile interval). Secondary: the
        other five metrics, with Holm-adjusted intervals.
      \item[Hypothesis.] H1: the difference in \acrshort{CC} is positive when neither model
        sees any frame after the predicted one (\emph{causal} setting, as in
        SalFoM). The difference in the \emph{offline} setting, in which SURGE-D
        also sees later frames, is reported as secondary.
      \item[Control.] SalFoM's released model, with its maps passed through the
        same blur grid as SURGE-D and reported at its best value (an upper bound
        for the baseline); every SURGE-D seed scored singly; a frozen \acrshort{UMT}-L
        encoder with the SURGE-D decoder, which separates the encoder from the
        training method.
      \item[Rejected if.] The 95\,\% interval of the causal difference between the
        three-seed ensemble and SalFoM at its best blur includes zero or lies
        below it. RQ1 is then answered \enquote{not shown under an identical
        protocol}, whatever the offline result.
    \end{description}
  \item Does the prediction error of the encoder's pretrained latent predictor
    (\enquote{predictive surprise}) carry information about where people look,
    beyond that carried by simple appearance and motion cues?
    \begin{description}[leftmargin=1.2em,font=\normalfont\itshape]
      \item[Primary metric.] Incremental information gain, in bits per fixation,
        of a log-linear combination of the centre prior, frame difference, token
        atypicality and predictive surprise over the same combination without
        surprise, on the 60 holdout videos. The weights are fitted on one half of
        the videos and evaluated on the other, both ways round; the 95\,\%
        interval comes from a bootstrap over videos.
      \item[Hypothesis.] H2: the incremental gain is positive for at least one
        pre-specified readout: the raw and contextual error with future targets
        (\cref{sec:surprise}), and the raw error with spatial masks spanning the
        whole clip, as in pre-training.
      \item[Control.] The same fit with the surprise map of another video
        (expected gain zero); the standalone scores of every map, with intervals.
      \item[Rejected if.] The interval includes zero or lies below it for every
        readout. The answer is then that no information beyond these cues was
        detected for this predictor and these readouts; it does not extend to
        other predictors.
    \end{description}
  \item Which design factors determine accuracy when the encoder is frozen: the
    pre-training and scale of the encoder, parameter-efficient adaptation, the
    learned centre prior, deep supervision, static-image co-training, temporal
    sampling, the amount of training data and ensembling?
    \begin{description}[leftmargin=1.2em,font=\normalfont\itshape]
      \item[Primary metric.] Holdout \acrshort{CC} (60 videos). For each factor, the mean
        over three seeds of the paired per-video difference to the reference
        configuration trained with the same seeds, with 90\,\% and 95\,\% intervals
        from a bootstrap over seeds and videos.
      \item[Hypothesis.] For each factor separately, either an \emph{effect}
        ($|\Delta|\geq0.003$ \acrshort{CC} with the 95\,\% interval excluding zero) or
        \emph{equivalence} (the 90\,\% interval within $\pm0.003$, two one-sided
        tests). A direction is assumed only for encoder scale (positive).
      \item[Control.] The reference configuration with the same seeds, schedule
        and data. For adaptation, the best of a pre-specified grid selected on the
        holdout, which favours adaptation. For clip length, training continued
        until holdout \acrshort{CC} stops rising, up to a pre-specified maximum. For the
        amount of data, a learning curve over 135, 270 and 540 videos.
      \item[Rejected if.] A factor stated to have no effect shows an effect, or a
        factor stated to have an effect is equivalent. A factor that is neither
        is reported as inconclusive. The statement that the encoder dominates is
        rejected if any other factor's effect is at least as large as that of
        encoder scale.
    \end{description}
  \item How well does the approach transfer to eye-tracking datasets with other
    content and viewing tasks, and does adding the soundtrack improve it on
    audio-visual datasets?
    \begin{description}[leftmargin=1.2em,font=\normalfont\itshape]
      \item[Primary metric.] Fold-mean \acrshort{CC} per dataset, zero-shot and after
        fine-tuning; internal comparisons by a paired bootstrap over the pooled
        test videos. Published numbers are descriptive references only.
      \item[Hypothesis.] H4a: after fine-tuning, a single SURGE-D model is within
        the tie margin of, or above, the best verified published result on each
        dataset. H4b: audio fusion raises fold-mean \acrshort{CC} by at least 0.010, and
        the matched soundtrack beats a rotated one by at least 0.005.
      \item[Control.] The zero-shot model; single models compared with single
        published models, with ensembles reported separately; rotated audio. The
        tie margin is set from the documented disagreement between papers on the
        same models and data, before the comparison is read.
      \item[Rejected if.] H4a: on a dataset, the single model is below the best
        verified result by more than the margin (reported per dataset). H4b:
        either audio criterion fails on a dataset (the rule of \cref{tab:rules}).
    \end{description}
\end{enumerate}
```

---

## Section 8: Revised text (LaTeX)

Add this once to `preamble.tex`. It typesets every pending result as a red `[RESULT: …]`, which can be found with
`grep RESULT`.

```latex
\newcommand{\RESULT}[1]{\textcolor{red}{[RESULT: #1]}}
```

### 8a. Abstract and contribution list

```latex
% frontmatter/abstract.tex. Each sentence is tagged with its RQ and evidence.
\prelimchapter{Abstract}

% Context for RQ1.
Video saliency models draw much of their accuracy from large pretrained encoders,
which they adapt to eye-tracking datasets of a few hundred videos.
% RQ1--RQ4.
This thesis asks whether a frozen self-supervised video encoder trained by feature
prediction, \acrshort{V-JEPA}~2, combined with a decoder of 7.4 million parameters,
predicts human fixations more accurately than SalFoM, the closest foundation-model
approach, under an identical protocol (RQ1); whether the error of the encoder's
pretrained predictor carries information about fixations beyond simple appearance
and motion cues (RQ2); which design factors determine accuracy once the encoder is
frozen (RQ3); and how the approach transfers to other eye-tracking datasets and
whether the soundtrack helps (RQ4).
% RQ1. Evidence: Tables 5.4, 5.5. Planned: F2, F3, F5.
On the 100 videos of the DHF1K validation set, scored by one evaluator validated
against reference code, the model with the \acrshort{ViT}-g encoder reached a correlation
coefficient of 0.600 as a single model and 0.603 as a three-seed ensemble, against
0.564 for SalFoM's released model, and the paired bootstrap intervals excluded zero
on all six metrics. In that comparison the model sees frames after the predicted
frame and SalFoM does not; when neither sees later frames, the difference in
correlation is \RESULT{causal \acrshort{ViT}-g ensemble $-$ SalFoM at its best blur, mean
and 95\,\% interval over 100 videos}.
% RQ2. Evidence: Table 5.3. Planned: F1a, F1b, F9.
The predictor's error, read out by predicting the second half of a clip from the
first, was a weaker cue than frame difference and token atypicality on a 60-video
holdout, and its raw form had a negative normalised scanpath saliency at both
sampling rates tested; its information beyond these cues was
\RESULT{cross-fitted incremental information gain, bits per fixation, 95\,\%
interval, per readout}.
% RQ3. Evidence: Tables 5.5, 5.10, A.2. Planned: F7, F8, F10--F12.
A larger encoder (\acrshort{ViT}-g instead of \acrshort{ViT}-L) raised validation correlation by 0.017
for single models; for the centre prior, deep supervision, static-image
co-training, temporal stride, clip length, low-rank adaptation and the amount of
training data, \RESULT{three-seed effect and interval per factor, each classified
as an effect, equivalent within $\pm0.003$, or inconclusive}.
% RQ4. Evidence: Tables 5.6, 5.7, 5.11. Planned: single-model comparisons (Table rq4).
After fine-tuning, the three-seed ensemble was 0.009 below the best published
correlation on UCF Sports; on six audio-visual datasets its correlation was above
the best peer-reviewed results that could be verified on four, tied on one and
lower on recordings of conversations, by margins of the same order as the
disagreements between published evaluations of the same models. For single
models, \RESULT{outcome per dataset under the pre-fixed tie margin}. A module that
fused the soundtrack did not meet the pre-specified criterion for improvement on
any of these datasets.

\vspace{\baselineskip}
\noindent\textbf{Keywords:} video saliency prediction, eye fixations,
self-supervised learning, joint-embedding predictive architecture, frozen
encoder, predictive surprise
```

```latex
% Section 1.8. Replaces the four bullets; Section 6.2 should mirror it.
\section{Contributions}\label{sec:contributions}

The contributions of this thesis, each tied to one research question, are:
\begin{itemize}
  \item \textbf{A controlled comparison (RQ1).} SURGE-D, a decoder of 7.4 million
    parameters on a frozen \acrshort{V-JEPA}~2 encoder, compared with SalFoM's released model
    on the DHF1K validation set by one validated evaluator with paired bootstrap
    intervals, in the offline setting and \RESULT{in a causal setting matched to
    SalFoM}.
  \item \textbf{A scoped negative result (RQ2).} The error of the released
    \acrshort{V-JEPA}~2 predictor, read out as in \cref{sec:surprise}, is a weaker fixation
    cue than frame difference and token atypicality; \RESULT{whether it adds
    information beyond them}. The result concerns this predictor and these
    readouts only.
  \item \textbf{Effect estimates (RQ3).} Three-seed estimates, with pre-specified
    equivalence margins, of the effect of encoder scale and of seven decoder,
    training and data factors on a frozen encoder \RESULT{outcome per factor}.
  \item \textbf{Transfer results (RQ4).} Results on eight further test sets
    against verified published numbers, an audio-fusion test with a rotated-audio
    control, and a Hollywood-2 ground truth built from openly licensed gaze
    recordings, released with its inter-observer ceiling
    \RESULT{leave-one-observer-out \acrshort{CC}}.
\end{itemize}
The evaluator, the dated decision log, the splits and the code support all four
contributions and are released with the thesis.
```

### 8b. Introduction: motivation and gap

```latex
% Section 1.2. Replaces the three motivation paragraphs.
\section{Motivation}\label{sec:motivation}

Three considerations motivated this work. The first is practical.
SalFoM~\cite{moradi2024salfom}, the first video saliency model built on a video
foundation encoder, fine-tunes a 300-million-parameter encoder on a few hundred
training videos and runs it once for every output frame. Fine-tuning so large a
model on so little data risks overfitting, which its authors offer as an
explanation for its weaker results on a small dataset. Whether a frozen encoder
with a small decoder is as accurate is therefore worth testing (RQ1), together
with which other design choices still matter when the encoder is not trained
(RQ3). The cost of the two designs is reported as a secondary outcome
(\cref{sec:cost}).

The second consideration is scientific. The idea that attention is drawn to the
unexpected has support from computational models built on hand-crafted
features~\cite{itti2009bayesian}, but, in the literature reviewed for this thesis,
it had not been tested with the learned predictor of a modern video model. A
self-supervised encoder that contains such a predictor makes the test possible,
provided the test asks whether the predictor's error adds to what simpler cues
already explain (RQ2).

The third consideration is methodological. Published comparisons in video
saliency rely mostly on numbers copied from other papers, computed with different
evaluation code and sometimes on different data splits, and they rarely report
uncertainty~\cite{girmaji2025minimalistic,xiong2025unist}. A study that re-scores
its main baseline with one validated evaluator, controls what else differs between
the compared systems, and fixes its decision rules in advance can give more
reliable answers, including negative ones, and can show how far they hold beyond
the main benchmark (RQ4).
```

```latex
% Section 1.3. The gap paragraph, leading to RQ1--RQ4.
\section{Statement of the Problem}\label{sec:problem}

Recent video saliency models adapt large pretrained encoders to eye-tracking
datasets of a few hundred videos. Whether the adaptation is needed is not known.
Encoders trained by feature prediction are reported to transfer well when
frozen~\cite{bardes2024revisiting}, but no study has compared such an encoder,
frozen, with an adapted foundation-model approach for fixation prediction under a
common protocol, and such a comparison is only informative if it controls what
else differs between the two systems, above all their access to frames after the
predicted one (\cref{sec:formulation}). It is also not known whether the error of
such an encoder's predictor, a candidate learned measure of surprise, adds anything
to the simple cues that already predict fixations. Because published comparisons
seldom use re-scored baselines or report uncertainty, it is further unclear which
design choices beyond the encoder still matter, and how far results obtained on
one benchmark transfer to other content, viewing tasks and audio-visual input.
This thesis addresses these four questions (\cref{sec:questions}).
```

### 8c. Experimental protocol

```latex
% New Section 4.8 (before the metric definitions). It absorbs Section 5.2,
% the evaluator validation.
\section{Evaluation Protocol}\label{sec:protocol}

\subsection{Data and Splits}
\begin{itemize}
  \item \textbf{DHF1K}~\cite{wang2018revisiting,wang2021revisiting}: 540 training
    videos and a 60-video holdout drawn once, with a fixed seed, from the 600
    official training videos; the 100 official validation videos are the test
    set. The 300 official test videos are not scored in this thesis.
  \item \textbf{UCF Sports}~\cite{rodriguez2008action,mathe2015actions}:
    93 / 10 / 47 videos (training / holdout / test).
  \item \textbf{Hollywood-2 with the gaze of Vig et al.}~\cite{vig2012space}:
    741 / 82 / 884 clips.
  \item \textbf{Audio-visual datasets}~\cite{tsiami2020stavis}: the released folds.
    The holdout is 10\,\% of each training fold: one video per fold for Coutrot2
    and ETMD, two for SumMe, three for AVAD, four for Coutrot1 and six for DIEM.
    \RESULT{why 23 of the 25 SumMe videos and 81 of the 84 DIEM videos are used}.
\end{itemize}

\subsection{Baselines}
SalFoM~\cite{moradi2024salfom} is the only baseline scored by the evaluator of
this thesis. It was not retrained: the authors' released DHF1K weights were run
on the 100 validation videos with the authors' model code
\RESULT{clip length, frame sampling, input size, post-processing and numerical
precision of this run; the wrapper script, added to the repository}.
\RESULT{whether the released weights were trained on the official training
videos only}. Because SalFoM was trained on all 600 official training videos,
including the holdout of this thesis, its output blur cannot be tuned on that
holdout; it is therefore reported both as released and at the best value of the
blur grid on the test set, an upper bound for the baseline. All other results in
\cref{tab:dhf1k,tab:ucf,tab:av} are copied from the cited papers, were computed
with other evaluation code and enter no statistical test. The comparison set is
restricted to peer-reviewed publications
\RESULT{author decision: whether to state that results outside this set are
higher on several audio-visual datasets}.

\subsection{Seeds and Tuning Budget}
\Cref{tab:budget} lists what each compared method received. Every SURGE-D
configuration was trained with seeds 0, 1 and 2; the single model of each table
is seed 0, fixed before any scoring, and \RESULT{seeds 1 and 2 are also scored
singly}. The architecture, the tapped layers, the loss weights and the learning
rates of SURGE-D were fixed in advance and not tuned (\cref{tab:hyper}).

\begin{table}[htbp]
  \centering
  \caption{Training, selection data and tuning budget of each compared
  method}\label{tab:budget}
  \footnotesize
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}p{2.5cm}>{\raggedright}p{1.7cm}>{\raggedright}X>{\raggedright\arraybackslash}p{3.0cm}@{}}
    \toprule
    \textbf{Method} & \textbf{Trained here} & \textbf{Settings chosen, and on which data} & \textbf{Scored on the test set} \\
    \midrule
    SURGE-D, offline & yes & best of three epochs and output blur (six values),
      both on the holdout & seed 0 \RESULT{and seeds 1, 2}; ensemble of seeds 0--2 \\
    SURGE-D, causal & \RESULT{yes} & as above & \RESULT{seeds 0--2; ensemble} \\
    SalFoM & no (released weights) & none; \RESULT{blur grid on the test set, as an
      upper bound} & one released model \\
    \acrshort{LoRA} variant & yes & one setting ($r=\alpha=16$, learning rate $10^{-4}$)
      \RESULT{pre-specified grid, selected on the holdout} & not scored (holdout only) \\
    Published models & no & as in their papers; other evaluation code & -- \\
    \bottomrule
  \end{tabularx}
\end{table}

\subsection{Metrics}
The six metrics of \cref{sec:evaluation-metrics} are computed at the native
resolution of the ground truth. \acrshort{CC} is the primary metric of every decision and
research question; the other five are secondary, and their intervals are
Holm-adjusted within each comparison. One map per model, with the blur chosen by
\acrshort{CC}, is used for all metrics~\cite{kummerer2018saliency}. The evaluator was
validated against reference code before any model was scored
(\cref{sec:metric-validation}) \RESULT{identify the second evaluator used for
\acrshort{SIM}, \acrshort{KLD} and \acrshort{AUC-J}; check \acrshort{sAUC} against reference code}.

\subsection{Statistical Analysis}
Two models scored on the same test videos are compared by a paired bootstrap over
videos~\cite{efron1979bootstrap} (10{,}000 resamples, 95\,\% percentile interval).
Configurations trained with several seeds are compared by a two-level bootstrap
over seeds and videos. Each ablation has one of three outcomes, with the margin
$\delta=0.003$ \acrshort{CC} fixed in advance: an \emph{effect} (the 95\,\% interval
excludes zero and $|\Delta|\geq\delta$), \emph{equivalence} (the 90\,\% interval
lies within $\pm\delta$) or \emph{inconclusive}. Published results are
descriptive; differences smaller than \RESULT{tie margin derived from the
documented disagreement between papers on the same models and data, fixed before
the comparison is read} are treated as ties.

\subsection{Use of the Test Sets}
Each test set was scored once per reported model, after all of that model's
settings had been fixed. No reported model or setting was selected on the DHF1K
validation set. \Cref{tab:ledger} lists every scoring of that set, including the
two decisions about the official test submission that it informed.

\begin{table}[htbp]
  \centering
  \caption{Every scoring of the DHF1K validation set, the test set of this
  thesis}\label{tab:ledger}
  \footnotesize
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}p{4.4cm}>{\raggedright}X>{\raggedright\arraybackslash}p{4.4cm}@{}}
    \toprule
    \textbf{Model} & \textbf{Purpose} & \textbf{Decision informed} \\
    \midrule
    SalFoM, released & evaluator validation; baseline & none \\
    SURGE-D \acrshort{ViT}-L, seed 0 and ensemble & RQ1 & maps prepared for the official
      test set (later superseded) \\
    SURGE-D \acrshort{ViT}-g, seed 0 and ensemble & RQ1; RQ3 (scale) & maps prepared for the
      official test set \\
    SURGE-D \acrshort{ViT}-g on 600 videos, seed 0 and ensemble & RQ3 (amount of data) &
      whether they replace the 540-video models for the official test set (they
      did not) \\
    \RESULT{seeds 1 and 2 singly; causal models; SalFoM blur grid; frozen \acrshort{UMT}-L}
      & RQ1 & none \\
    \bottomrule
  \end{tabularx}
\end{table}
```

### 8d. Results skeleton, one subsection per RQ

```latex
\section{Results}\label{sec:results}

%----------------------------------------------------------------------------
\subsection{RQ1: Frozen Encoder and SalFoM under an Identical Protocol}\label{sec:rq1}
% Comparison shown: per-video paired differences, SURGE-D minus SalFoM, on the
% 100 DHF1K validation videos. Primary: causal setting, CC, three-seed ViT-g
% ensemble against SalFoM at its best blur. Secondary: offline setting; seeds
% singly; frozen UMT-L control; the other five metrics in Appendix A (Holm).

\begin{table}[htbp]
  \centering
  \caption{RQ1: paired bootstrap differences to SalFoM on DHF1K validation (100
  videos, 10{,}000 resamples, 95\,\% intervals). \enquote{Causal}: no frame after the
  predicted one is used, as in SalFoM. \enquote{Best $\sigma$}: SalFoM's maps blurred
  with the grid value that is best for SalFoM on this set (an upper bound for the
  baseline)}\label{tab:rq1}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}X>{\raggedright}p{2.8cm}cc@{}}
    \toprule
    \textbf{SURGE-D model} & \textbf{Setting / SalFoM} & $\Delta$\textbf{\acrshort{CC}} [95\,\% \acrshort{CI}] & $\Delta$\textbf{\acrshort{NSS}} [95\,\% \acrshort{CI}] \\
    \midrule
    \acrshort{ViT}-L, seed 0   & offline / released & $+0.020$ [$+0.010$, $+0.031$] & $+0.023$ [$-0.050$, $+0.097$] \\
    \acrshort{ViT}-g, seed 0   & offline / released & $+0.036$ [$+0.026$, $+0.048$] & $+0.130$ [$+0.060$, $+0.205$] \\
    \acrshort{ViT}-g, ensemble & offline / released & $+0.039$ [$+0.029$, $+0.050$] & $+0.160$ [$+0.089$, $+0.233$] \\
    \acrshort{ViT}-g, seeds 1 and 2 & offline / released & \RESULT{per seed} & \RESULT{per seed} \\
    \acrshort{ViT}-g, ensemble & offline / best $\sigma$ & \RESULT{} & \RESULT{} \\
    \acrshort{ViT}-L, seeds 0--2 and ensemble & causal / best $\sigma$ & \RESULT{} & \RESULT{} \\
    \acrshort{ViT}-g, seeds 0--2 & causal / best $\sigma$ & \RESULT{} & \RESULT{} \\
    \textbf{\acrshort{ViT}-g, ensemble (primary)} & \textbf{causal / best} $\boldsymbol{\sigma}$ & \RESULT{} & \RESULT{} \\
    Frozen \acrshort{UMT}-L + SURGE-D decoder, seed 0 & offline / released & \RESULT{} & \RESULT{} \\
    \bottomrule
  \end{tabularx}
\end{table}

\begin{figure}[htbp]
  \centering
  % \includegraphics[width=0.6\textwidth]{rq1_causal_vs_offline.pdf}
  \fbox{\parbox{0.6\textwidth}{\centering\RESULT{scatter of per-video $\Delta$\acrshort{CC}
  to SalFoM, offline (x) against causal (y); the diagonal marks no loss from
  removing later frames}}}
  \caption{RQ1: per-video \acrshort{CC} difference to SalFoM in the offline and causal
  settings (\acrshort{ViT}-g ensemble, 100 videos)}\label{fig:rq1-causal}
\end{figure}

\paragraph{Answer to RQ1.} \RESULT{H1 accepted or rejected under the rule of
\cref{sec:questions}; use the matching sentence pre-written in \cref{sec:conclusion}}.

\paragraph{Secondary outcome: cost (no research question).}\label{sec:cost}
% Comparison shown: TFLOPs and measured throughput per output frame, same GPU and
% frame loading, offline and causal. Existing numbers from Table 5.9.
\begin{table}[htbp]
  \centering
  \caption{Cost per output frame on one RTX A5000, network forward pass
  only}\label{tab:cost2}
  \footnotesize
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}X>{\raggedright}p{1.6cm}>{\centering}p{2.6cm}>{\centering\arraybackslash}p{3.4cm}@{}}
    \toprule
    \textbf{Model} & \textbf{Setting} & \textbf{\acrshort{TFLOPs} per output frame} & \textbf{Output frames/s (batch 4)} \\
    \midrule
    SURGE-D \acrshort{ViT}-L & offline & 0.114 & 301 \\
    SURGE-D \acrshort{ViT}-g & offline & 0.328 & 129 \\
    SURGE-D \acrshort{ViT}-L / \acrshort{ViT}-g & causal & \RESULT{} & \RESULT{} \\
    SalFoM (encoder only, estimate) & causal & 1.90 & \RESULT{measured, full model} \\
    \bottomrule
  \end{tabularx}
\end{table}

%----------------------------------------------------------------------------
\subsection{RQ2: Predictive Surprise beyond Simple Cues}\label{sec:rq2}
% Comparisons shown: (1) standalone NSS and CC of each map with 95% intervals
% over the 60 holdout videos; (2) incremental information gain of each surprise
% readout over {centre prior, frame difference, token atypicality},
% cross-fitted over two halves of the videos, with a wrong-video control.

\begin{table}[htbp]
  \centering
  \caption{RQ2: standalone scores on the DHF1K holdout (60 videos), with 95\,\%
  intervals over videos}\label{tab:rq2-standalone}
  \footnotesize
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}Xcccc@{}}
    \toprule
    & \multicolumn{2}{c}{\textbf{\acrshort{NSS}}} & \multicolumn{2}{c}{\textbf{\acrshort{CC}}} \\
    \cmidrule(lr){2-3}\cmidrule(l){4-5}
    \textbf{Map} & 4\,fps & 15\,fps & 4\,fps & 15\,fps \\
    \midrule
    Raw surprise, future targets & \shortstack{$-0.190$\\\RESULT{CI}} & \shortstack{$-0.502$\\\RESULT{CI}}
      & \shortstack{$-0.043$\\\RESULT{CI}} & \shortstack{$-0.099$\\\RESULT{CI}} \\
    Contextual surprise & \shortstack{$0.116$\\\RESULT{CI}} & \shortstack{$-0.366$\\\RESULT{CI}}
      & \shortstack{$0.018$\\\RESULT{CI}} & \shortstack{$-0.072$\\\RESULT{CI}} \\
    Raw surprise, whole-clip spatial masks & \RESULT{} & \RESULT{} & \RESULT{} & \RESULT{} \\
    Token atypicality & \shortstack{$0.454$\\\RESULT{CI}} & \shortstack{$0.650$\\\RESULT{CI}}
      & \shortstack{$0.081$\\\RESULT{CI}} & \shortstack{$0.123$\\\RESULT{CI}} \\
    Frame difference & \shortstack{$0.810$\\\RESULT{CI}} & \shortstack{$0.803$\\\RESULT{CI}}
      & \shortstack{$0.149$\\\RESULT{CI}} & \shortstack{$0.148$\\\RESULT{CI}} \\
    Empirical centre prior & \shortstack{$1.464$\\\RESULT{CI}} & \shortstack{$1.479$\\\RESULT{CI}}
      & \shortstack{$0.289$\\\RESULT{CI}} & \shortstack{$0.292$\\\RESULT{CI}} \\
    \bottomrule
  \end{tabularx}
\end{table}

\begin{table}[htbp]
  \centering
  \caption{RQ2: incremental information gain (bits per fixation) of each surprise
  readout over a log-linear combination of the centre prior, frame difference and
  token atypicality, cross-fitted over two halves of the 60 holdout videos; 95\,\%
  intervals over videos}\label{tab:rq2-incremental}
  \footnotesize
  \begin{tabular}{@{}lcc@{}}
    \toprule
    \textbf{Added map} & \textbf{4\,fps} & \textbf{15\,fps} \\
    \midrule
    Raw surprise, future targets & \RESULT{} & \RESULT{} \\
    Contextual surprise & \RESULT{} & \RESULT{} \\
    Raw surprise, whole-clip spatial masks & \RESULT{} & \RESULT{} \\
    Control: surprise map of another video & \RESULT{} & \RESULT{} \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Answer to RQ2.} \RESULT{H2 accepted or rejected; scope: one released
predictor, these readouts}.

%----------------------------------------------------------------------------
\subsection{RQ3: Design Factors with a Frozen Encoder}\label{sec:rq3}
% Comparison shown: per factor, the three-seed mean paired difference to the
% reference trained with the same seeds, on the 60-video holdout; outcome under
% the three-way rule of Section 4.8.

\begin{table}[htbp]
  \centering
  \caption{RQ3: effect of each factor on holdout \acrshort{CC} (60 videos), three seeds per
  arm; outcome under the rule of \cref{sec:protocol}}\label{tab:rq3}
  \footnotesize
  \begin{tabularx}{\textwidth}{@{}>{\raggedright}X>{\raggedright}p{2.7cm}ccc@{}}
    \toprule
    \textbf{Factor} & \textbf{Reference} & $\Delta$\textbf{\acrshort{CC}} & \textbf{90\,\% / 95\,\% interval} & \textbf{Outcome} \\
    \midrule
    Encoder scale (\acrshort{ViT}-g) & \acrshort{ViT}-L & \RESULT{} & \RESULT{} & \RESULT{} \\
    Pre-training: \acrshort{V-JEPA}~2.1 at $384^2$; \acrshort{UMT}-L at $224^2$ & \acrshort{V-JEPA}~2 at the same input size & \RESULT{} & \RESULT{} & \RESULT{} \\
    \acrshort{LoRA}, best of the grid & frozen & \RESULT{} & \RESULT{} & \RESULT{} \\
    Without centre prior & with & \RESULT{} & \RESULT{} & \RESULT{} \\
    Without auxiliary heads & with & \RESULT{} & \RESULT{} & \RESULT{} \\
    With SALICON co-training & without & \RESULT{} & \RESULT{} & \RESULT{} \\
    Stride 4 & stride 2 & \RESULT{} & \RESULT{} & \RESULT{} \\
    32 frames, converged schedule & 16 frames & \RESULT{} & \RESULT{} & \RESULT{} \\
    135 and 270 training videos & 540 & \RESULT{} & \RESULT{} & \RESULT{} \\
    Three-seed ensemble & single seeds & \RESULT{} & \RESULT{} & \RESULT{} \\
    \bottomrule
  \end{tabularx}
\end{table}

% Training on 600 instead of 540 videos uses the validation set (no holdout is
% left); report all six metrics, single and ensemble, from research/log.md 11A:
% ensemble dCC +0.0015 [-0.0006, +0.0037], dSIM -0.0017 [-0.0032, -0.0001];
% seed 0: dCC -0.0001 (n.s.), dNSS +0.039 and dSIM +0.006 (significant),
% dsAUC -0.006 (significant).

\begin{figure}[htbp]
  \centering
  \fbox{\parbox{0.6\textwidth}{\centering\RESULT{holdout \acrshort{CC} against the number of
  training videos (135, 270, 540), three seeds, with intervals}}}
  \caption{RQ3: learning curve of SURGE-D \acrshort{ViT}-L on the DHF1K
  holdout}\label{fig:learning-curve}
\end{figure}

\paragraph{Answer to RQ3.} \RESULT{list factors by outcome; inconclusive factors
are named as such}.

%----------------------------------------------------------------------------
\subsection{RQ4: Transfer and Audio}\label{sec:rq4}
% Comparisons shown: (1) fine-tuned single model against zero-shot (paired
% bootstrap over pooled test videos); (2) single model against the best verified
% published single model (descriptive, pre-fixed tie margin); (3) ensembles
% separately; (4) audio: matched minus visual and matched minus rotated.

\begin{table}[htbp]
  \centering
  \caption{RQ4: \acrshort{CC} on the transfer test sets (\acrshort{ViT}-g unless marked). Published values
  were computed with other evaluation code and are descriptive; the comparison set
  is restricted to peer-reviewed results}\label{tab:rq4}
  \footnotesize
  \setlength{\tabcolsep}{2pt}
  \begin{tabular}{@{}lcccccl@{}}
    \toprule
    \textbf{Dataset} & \textbf{Zero-shot} & \textbf{Single} & \textbf{Seeds 0--2} & \textbf{Ensemble} & \textbf{Best published} & \textbf{Outcome} \\
    \midrule
    UCF Sports & \RESULT{} & 0.7305 & \RESULT{} & 0.7350 & 0.744 (ViNet-E) & \RESULT{} \\
    DIEM       & \RESULT{} & 0.714 & \RESULT{} & 0.729 & 0.701 (ViNet-E) & \RESULT{} \\
    Coutrot1   & \RESULT{} & 0.648 & \RESULT{} & 0.657 & 0.638 (DiffSal) & \RESULT{} \\
    Coutrot2   & \RESULT{} & 0.820 & \RESULT{} & 0.819 & 0.862 (ViNet-A) & \RESULT{} \\
    ETMD       & \RESULT{} & 0.672 & \RESULT{} & 0.676 & 0.652 (DiffSal) & \RESULT{} \\
    AVAD       & \RESULT{} & 0.731 & \RESULT{} & 0.740 & 0.738 (DiffSal) & \RESULT{} \\
    SumMe      & \RESULT{} & 0.586 & \RESULT{} & 0.590 & 0.572 (DiffSal) & \RESULT{} \\
    Hollywood-2 (Vig gaze) & \RESULT{} & 0.6844 & \RESULT{} & 0.6882 & not comparable & -- \\
    \bottomrule
  \end{tabular}
\end{table}
% Existing zero-shot values are for ViT-L (Tables 5.6-5.8); keep them in a separate
% column or replace them with ViT-g zero-shot runs.

% Audio: keep Table 5.11 (ViT-L, seed 0) and add \RESULT{seeds 1 and 2; ViT-g}.
% Hollywood-2 (Vig gaze): add \RESULT{leave-one-observer-out CC per clip, mean
% and interval} as the reference ceiling for Table 5.8.

\paragraph{Answer to RQ4.} \RESULT{per dataset: above, tie or below under the
pre-fixed margin; audio: rule met or not per dataset}.
```

### 8e. Limitations and conclusion

```latex
% Section 5.8. Replaces Threats to Validity.
\section{Threats to Validity}\label{sec:threats}

\paragraph{Protocol.} The decision rules were written in a dated log in the thesis
repository before the experiments they govern. The exception is the tie margin for
published results, which was set after the first audio-visual results had been
seen (\cref{sec:design}). This is pre-specification in an internal log, not an
external registration. Many choices were fixed in advance and never tested: the
decoder architecture, the tapped layers, the loss weights, the squashing of the
input, the learning rates and the output positions. No reported model or setting
was selected on the DHF1K validation set, but that set informed two decisions about
the official test submission (\cref{tab:ledger}).

\paragraph{Internal validity.} \RESULT{if the causal comparison has been run,
state its outcome here; otherwise keep:} In the comparison with SalFoM, SURGE-D
sees at least 0.27\,s of video after the predicted frame and SalFoM does not, and
the size of this advantage was not measured. Only SalFoM was re-scored, and the
program that produced its maps \RESULT{is documented in Appendix B, or: is not
part of the thesis code}. \RESULT{if three-seed ablations have not been run,
keep:} The component ablations used one seed, and their threshold of 0.003 \acrshort{CC}
is close to the seed-to-seed spread, so differences of that size could not be
detected reliably; their outcomes are inconclusive rather than null. On the
audio-visual datasets, the best epoch and the output blur were selected on one to
six holdout videos per fold.

\paragraph{Construct validity.} The predictor-error readout asks the predictor to
predict the second half of a clip from the first, a task it was not trained on,
so its error mixes surprise with the cost of an unfamiliar query. One map per
model, with the blur chosen by \acrshort{CC}, is used for all metrics. The Hollywood-2
(Vig gaze) maps are built from per-frame gaze positions of at most five observers
\RESULT{inter-observer ceiling}. The shuffled \acrshort{AUC} uses a sampling scheme
specific to this thesis.

\paragraph{External validity and comparability.} The DHF1K results are
validation-set results; the official test-set scores were not available.
Comparisons with published numbers cross evaluation code, whose disagreements on
the same models and data reach a few hundredths of \acrshort{CC}, and are restricted to
peer-reviewed results. On the audio-visual datasets the conclusion depends on that
restriction \RESULT{author decision: state that results outside the set are higher
on several datasets, or name them}. The design-factor results come from DHF1K and,
except for scale, from the \acrshort{ViT}-L encoder; they may not hold for \acrshort{ViT}-g or for other
data. The pre-training data of \acrshort{V-JEPA}~2 may overlap the sources of DHF1K, as for
every method built on a large pretrained video encoder.
```

```latex
% Section 6.1. Replaces the conclusion. Both outcomes of each falsification test
% are pre-written as comments; keep the one that the result supports.
\section{Conclusion}\label{sec:conclusion}

This thesis asked four questions about a frozen self-supervised video encoder for
video saliency prediction. It developed SURGE-D, a decoder of 7.4 million
parameters on a frozen \acrshort{V-JEPA}~2 encoder, wrote its decision rules in a dated log
before the experiments they govern, scored every test set once per reported model
with an evaluator validated against reference code, and assessed differences with
paired statistics.

\paragraph{RQ1.} In the offline setting, SURGE-D was more accurate than SalFoM's
released model on the DHF1K validation set under one evaluator: \acrshort{CC} 0.600 for a
single \acrshort{ViT}-g model and 0.603 for a three-seed ensemble, against 0.564, with paired
intervals above zero on all six metrics; the \acrshort{ViT}-L model, whose encoder is the
size of SalFoM's, was also better ($+0.020$ \acrshort{CC}). With causal access for both models,
the difference was \RESULT{causal ensemble $-$ SalFoM at its best blur, 95\,\%
interval}.
% Interval includes zero or lies below it:
%   "Under an identical protocol the advantage was not shown; it depends on access
%    to frames after the predicted one."
% Interval above zero:
%   "The advantage held under an identical protocol."
\RESULT{frozen \acrshort{UMT}-L control: whether the gain is specific to the \acrshort{V-JEPA}~2
encoder}. These are validation-set results against one re-scored baseline; the
official test-set scores were not available.

\paragraph{RQ2.} The error of the released \acrshort{V-JEPA}~2 predictor, read out by
predicting the second half of a clip from the first, was a weaker fixation cue
than frame difference and token atypicality, and its raw form had a negative \acrshort{NSS}
at both sampling rates tested. Beyond these cues, it added \RESULT{incremental
information gain and interval, per readout}.
% Every interval includes zero:
%   "No information beyond these cues was detected for this predictor and these
%    readouts."
% One interval above zero:
%   "It added information beyond these cues although it was weaker on its own."
The finding concerns one released predictor, whose pre-training masks did not
require forecasting, and the readouts tested; it is not a test of surprise-based
accounts of attention.

\paragraph{RQ3.} Encoder scale mattered: \acrshort{ViT}-g raised validation \acrshort{CC} by 0.017 for
single models and raised holdout \acrshort{CC} for all three seeds. \RESULT{outcome per
factor: effect, equivalent within $\pm0.003$ \acrshort{CC}, or inconclusive}. Training on
600 instead of 540 videos did not change validation \acrshort{CC} ($+0.0015$
[$-0.0006$, $+0.0037$] for ensembles) but moved other metrics in both directions.
\RESULT{learning curve: whether accuracy still rises with the number of training
videos}.

\paragraph{RQ4.} After three epochs of fine-tuning, the three-seed ensemble was
0.009 \acrshort{CC} below the best published result on UCF Sports. On the six audio-visual
datasets, its \acrshort{CC} was above the best peer-reviewed results that could be verified
on four (by 0.018 to 0.028), tied on AVAD and lower on Coutrot2, recordings of
conversations in which observers look at the person who speaks. These margins are
of the order of the disagreements between published evaluations of the same
models, and the comparison excludes results that had not been peer reviewed. For
single models, \RESULT{outcome per dataset under the pre-fixed tie margin}. A module that fused
the soundtrack did not meet the pre-specified criterion on any dataset, although
on Coutrot2 the model scored lower with another video's soundtrack than with its
own.
```

---

## Section 9: Experiment and analysis TODO list (in priority order)

Cost uses the run times in §5.1.1 and log §11C; GPU-hour figures are estimates.
- **Blocker** means the item is required before the main claim (RQ1) can be stated as in Section 8.
- **RQ blocker** means it is required before the stated RQ's answer can be given.

| Priority | Item | Serves | Cost | Blocker |
|---|---|---|---|---|
| 1 | **T1.** Document the SalFoM run: put a wrapper in this repository (or regenerate the maps with it) and record clip length, sampling, input size, post-processing and precision. Check whether the released weights saw DHF1K val | RQ1 | low | **Blocker** (reproducibility of the main comparison) |
| 2 | **T2 = F2.** SalFoM over the blur grid on val; report it as released and at its best σ | RQ1 | low (CPU, minutes) | **Blocker** |
| 3 | **T3 = F1a.** Video-bootstrap intervals for T5.3 and for the surprise − reference differences, from `runs/g2/g2_results.json` | RQ2 | low (CPU, minutes) | RQ2 blocker |
| 4 | **T4 = F3.** Score ViT-L and ViT-g seeds 1 and 2 singly on val | RQ1 | low (about 1–2 GPU-h) | no; strengthens RQ1 |
| 5 | **T5 = F4.** Score the frozen UMT-L G1 checkpoint (`runs/g1_umt_l/best.pt`) on val | RQ1 (attribution) | low (< 1 GPU-h) | no; needed for any "representation quality" wording |
| 6 | **T6 = F5.** Causal SURGE-D: ViT-L and ViT-g, 3 seeds each, target = last sampled frame; σ on the holdout; scored once on val. A pre-declared every-5th-frame subset for both models keeps causal inference affordable | RQ1 | medium (about 3–4 GPU-h training plus a few GPU-h scoring) | **Blocker** |
| 7 | **T7 = F7.** Seeds 1 and 2 for no prior, no auxiliary heads, SALICON and stride 4 (8 ViT-L runs); two-level bootstrap; TOST at ±0.003 | RQ3 | low (about 2–4 GPU-h) | RQ3 blocker |
| 8 | **T8 = F1b + F9.** Recompute the G2 maps; cross-fitted incremental information gain; in-distribution readout; wrong-video control | RQ2 | low (about 1–2 GPU-h plus CPU) | RQ2 blocker |
| 9 | **T9 = F8.** Learning curve: 135 / 270 / 540 videos × 3 seeds (ViT-L) | RQ3 | low (about 2–3 GPU-h) | only if the data-limitation statement is kept |
| 10 | **T10 = F6.** Time SalFoM and SURGE-D on the A5000 with the same frame loading, offline and causal | secondary outcome (cost) | low | only if any cost statement is kept |
| 11 | **T11.** Calibrate the tie margin from the cross-paper disagreements recorded in log §7; fix it before re-reading T5.6 and T5.7 | RQ4 | low (analysis) | RQ4 blocker for H4a |
| 12 | **T12.** Holm adjustment of secondary metrics; ensemble-vs-single intervals on DHF1K val; all six metrics for train-600 (already in log §11A) | RQ1, RQ3 | low (analysis) | no |
| 13 | **T13 = F13.** Hollywood (Vig) leave-one-observer-out CC | RQ4 | low (CPU) | no |
| 14 | **T14.** Explain the SumMe (23/25) and DIEM (81/84) counts; identify the second evaluator and check sAUC against reference code | RQ4, all | low | no |
| 15 | **T15 = F10.** LoRA grid (learning rate {1e-5, 3e-5, 1e-4} × rank {4, 16}) on ViT-L, best × 3 seeds; full fine-tuning of ViT-L if memory allows | RQ1, RQ3 | medium (grid); high (full fine-tuning) | no; required before any "adaptation is unnecessary" wording |
| 16 | **T16 = F11.** 32-frame ViT-g with an extended, pre-specified schedule × 3 seeds | RQ3 | medium (about 0.5 GPU-h per epoch and seed) | no |
| 17 | **T17 = F12.** G1 with 3 seeds; V-JEPA 2.1 at 384²; UMT-L and V-JEPA 2 both at 224² | RQ3 | medium | no; required before any "pre-training" claim |
| 18 | **T18.** Audio fusion with seeds 1 and 2, and with ViT-g | RQ4 | medium (48 short fine-tunes per encoder) | no |
| 19 | **T19 = F14.** Re-score released AV baselines with the thesis evaluator (weight availability unverified) | RQ4 | medium to high | no |
| 20 | **T20.** DHF1K official test scores: send the prepared archive. This is your action; Claude does not send it | RQ1 | external | no |

The GPU jobs are T4–T10 and T15–T19; run each under the safety monitor, one at a time. T1–T3 and T11–T14 are
documentation or CPU analysis. Every new rule (margins, grids, maximum epochs, the causal target) goes into
`research/log.md` before the run it governs.

---

## Section 10: Change log

| Section | Old text or claim | New text or claim | Reason |
|---|---|---|---|
| Abstract ¶1 | "asks whether … V-JEPA 2, is sufficient for the task without adapting it, and whether the error …, a learned measure of surprise, indicates where people look" | States RQ1–RQ4 as worded in §1.4; "candidate" measure of surprise | A2, A4; every sentence maps to an RQ |
| Abstract ¶2 | "Every design choice was tested under a rule written before the experiment, all models were scored once per test set …" | Removed from the abstract; the protocol section lists rules, exceptions and a test-set ledger | A8 (C18, C19) |
| Abstract ¶3 | "better on all six metrics … while needing about one sixth of SalFoM's encoder computation per output frame" | Accuracy kept and scoped as offline; causal result as `[RESULT]`; cost sentence removed (no RQ) | A1, A3 (C2, C5) |
| Abstract ¶3 | "exceeded the best peer-reviewed results that could be verified on five of the six audio-visual datasets" | "above … on four, tied on one and lower on recordings of conversations, by margins of the same order as the disagreements between published evaluations" | A7 (C15) |
| Abstract ¶3 | "The predictor's error … was anti-correlated with fixations and was beaten by simple frame differencing" | "a weaker cue than frame difference and token atypicality …; its raw form had a negative NSS at both sampling rates …; information beyond these cues `[RESULT]`" | A4 (C6, C9) |
| Abstract ¶3 | "Encoder pre-training and scale produced the only large gains, whereas … produced none that could be measured" | Scale: +0.017. Other factors: `[RESULT]`, each an effect, equivalent or inconclusive | A5, A6 (C10b, C11) |
| Abstract ¶4 | "an efficient frozen-encoder saliency model, a documented negative result on predictive surprise, controlled evidence on which design choices matter, and a transparent evaluation protocol" | Moved to §1.8 as four RQ-tied contributions | A2, A3, A5; RQ mapping |
| §1.2 | "A frozen encoder with a decoder that predicts many frames at once would avoid both problems, if its features are good enough." | The adaptation question leads to RQ1 and RQ3; cost is a secondary outcome | A3 |
| §1.2 | "it had not been tested with the learned predictor of a modern video model" | Adds "in the literature reviewed for this thesis" | A11 (C22) |
| §1.3 | "… a purpose-built decoder, and a pre-registered evaluation on nine eye-tracking test sets." | A gap paragraph that names the temporal-access control and leads to the four RQs | A1, A8 |
| §1.4 | RQ1–RQ4 | Same wording, plus primary metric, hypothesis, control and falsification condition | Part 2 step 10 |
| §1.6 Evaluation | "every design decision was governed by a rule written before the corresponding experiment" | "each tested component was governed by a rule written in a dated log before the experiment, with one exception" | A8 |
| §1.7 | "causal (streaming) prediction were outside the scope of the study" | Causal *evaluation* is in scope as RQ1's control; streaming deployment stays out of scope | A1 (RQ1 requires it) |
| §1.8 | Four bullets, including "about six times more computation", "does not indicate where people look", "produce none that could be measured", "pre-registered decision rules" | Four RQ-tied bullets with `[RESULT]` placeholders | A2–A8 |
| §4.1 | "the 100-video validation set serves as the test set and was never used for selection" | "No reported model or setting was selected on the validation set; Table `tab:ledger` lists its every use, including two decisions about the official test submission" | A8 (C19) |
| T4.1 | Single-seed ablations: \|ΔCC\| ≥ 0.003 | Three seeds per arm, with an effect / equivalent / inconclusive outcome at ±0.003 | A5 |
| §4.3 | "holdout = 10 % of each training fold" | Adds the sizes (1–6 videos) and explains the SumMe and DIEM counts | 2.0 audit |
| New §4.8 | — | Evaluation protocol: splits, baselines (retrained, released or copied), tuning-budget table, metrics, statistics, test-set ledger | A1, A8, A9 |
| §5.2 | "against a second, independently written evaluator" | Identify it, or validate against the MIT reference code; check sAUC | A9 (C20) |
| §5.3 | "V-JEPA 2 led on all four metrics … adopted" | Kept as the gate decision; labelled indicative for "pre-training" (one seed; inputs differ) | A6 |
| §5.4 | "The outcome was unambiguous" | Intervals added; "for standalone use"; incremental test in §5.x RQ2 | A4 |
| §5.5.1 | "a comparison under an identical protocol" | "under one evaluator"; causal comparison as `[RESULT]` | A1 (C2) |
| §5.5.2 (AV) | "has the higher CC on five of the six datasets" | "higher by at least 0.01 on four, a tie on AVAD, lower on Coutrot2; margins of the order of cross-paper disagreement; sensitive to the comparator set" | A7 (C15) |
| §5.5.3 | Qualitative examples in the main text | Moved to Appendix A (illustrative; the selection rule is kept) | No RQ |
| §5.5.4 | "SalFoM … about 6 times … The multi-frame output thus allows an encoder of three times the size at a fraction of the cost of a per-frame design." | Secondary outcome; "in offline use, estimated encoder FLOPs"; measured times `[RESULT]`; causal cost stated | A3 (C5) |
| §5.6 | "Neither more context nor 11 % more videos made a measurable difference." | 32 frames: inconclusive (unconverged, one seed). 600 videos: CC unchanged; for seed 0, NSS and SIM up and sAUC down (log §11A) | A5 (C11) |
| §5.7 RQ1 | "the gain does not come from adapting the encoder … speaks to the quality of the pretrained V-JEPA representations rather than to a training technique" | "the frozen encoder was enough to exceed SalFoM; which part of the difference is responsible `[RESULT: frozen UMT-L, causal]`" | A2 (C4) |
| §5.7 RQ3 | "The results separate the factors cleanly." … "the limiting factor is the small number of training videos" | Removed; the data-limitation statement becomes a hypothesis tested by the learning curve | A5 (C12) |
| §5.7 RQ4 | "matched or exceeded … on five of the six" | As §5.5.2 | A7 |
| T5.7 caption | — | Adds that the comparison set is restricted to peer-reviewed results and that the conclusion is sensitive to this | A7 |
| T5.10 | Ensemble ΔCC only for 600 vs 540 | All six metrics, single and ensemble | A5 |
| §5.8 | Threats to validity | Adds the internal (not external) pre-specification, untested design choices, the val ledger, temporal access, baseline provenance, AV holdout sizes, comparator-set sensitivity and the readout mismatch | A1, A4, A7, A8, A9 |
| §6.1 | "Every design decision was tested under a rule written before the corresponding experiment" | "wrote its decision rules in a dated log before the experiments they govern" | A8 |
| §6.1 RQ1 | "at about one sixth of SalFoM's encoder computation per output frame" | Removed from the RQ1 answer; offline result plus causal `[RESULT]` with pre-written outcomes | A1, A3 |
| §6.1 RQ3 | "made no difference that could be measured … the size of the saliency datasets … limits further progress" | Effect, equivalent or inconclusive per factor; learning curve `[RESULT]` | A5 |
| §6.1 RQ4 | "a higher correlation than the best peer-reviewed results … on five of the six" | "four (by 0.018 to 0.028), tied on AVAD, lower on Coutrot2; within cross-paper noise; excludes non-peer-reviewed results" | A7 |
| §6.2 | "outperformed a fine-tuned foundation-model approach … under an identical protocol at a fraction of its computational cost" | "under one evaluator, offline; causal `[RESULT]`" | A1, A3 |
| §6.2 | "the predictor error of a self-supervised video model is not a fixation cue" | "the error of the released V-JEPA 2 predictor, read out as tested, was a weaker cue than simple ones; `[RESULT]` beyond them" | A4 (C8) |
| §6.3 | "Causal prediction … would allow a stricter comparison" | Moved into RQ1 (F5); future work keeps only streaming deployment | A1 |
| Appendix B | — | Adds the SalFoM inference wrapper and settings, plus a table of test-set uses | A9 |
