# Claims ledger: what the SURGE-D research supports, and how strongly

**Purpose.** A single list of every claim the research can make, sorted by the strength of its evidence, with
wording that can be reused in the thesis, the defence and any paper.
**Sources.**
- The thesis: `Paper/msc-thesis/main.pdf` at `14d3caf`; table and page numbers refer to it. The decoder-state update
  (2026-11-18) added about four pages from §4.5.9 on, so later pages are now one to four higher; Q19, W15 and X26
  cite tables and figures of the updated thesis.
- The run outputs in `runs/`.
- `report/critical-review.md` (codes M, D, m) and `report/sota-advantage-analysis.md`.

**Tiers.**

| Tier | Meaning | How to word it |
|---|---|---|
| 1. Strong | Paired or measured evidence, rules fixed in advance, survives the controls | State plainly, with the number |
| 2. Qualified | True, but only with a stated condition such as cross-paper comparison, one seed, an ensemble or a specific setting | Always attach the condition |
| 3. Weak | Suggestive evidence, or a plausible explanation that was not tested | Only as "suggests", "is consistent with" or "a possible explanation is" |
| 4. Do not claim | Untested, unsupported, or contradicted by evidence | Use the "say instead" wording |

Effects are differences in CC unless stated, with 95 % paired bootstrap intervals over videos. "Val" is the 100-video
DHF1K validation set, the thesis's test set.

---

## Tier 1: strong claims

### Accuracy against SalFoM (same evaluator, same videos, paired)

| # | Claim (suggested wording) | Evidence | Condition to keep |
|---|---|---|---|
| S1 | On DHF1K val, SURGE-D with the frozen V-JEPA 2 ViT-g encoder is more accurate than SalFoM's released model on all six metrics, as a single model and as a three-seed ensemble. | CC +0.036 [+0.026, +0.048] single; +0.039 [+0.029, +0.050] ensemble. Every interval on the six metrics excludes 0 (T5.6, TA.1). | "on DHF1K validation" |
| S2 | Every ViT-g seed is better than SalFoM on its own, so the result is not a lucky seed. | Seeds: +0.036, +0.031, +0.032, all intervals above 0 (T5.7). The reported seed 0 is the best of three; the mean is 0.5969 against 0.5639. | — |
| S3 | The gain is broad, not driven by a few videos. | The ensemble has the higher CC on 79 of 100 videos (Fig. 5.2). | — |
| S4 | The advantage survives the baseline's best blur, even though that blur was chosen on the test set in SalFoM's favour. | Ensemble +0.029 [+0.019, +0.039]; seeds +0.026, +0.021 and +0.022. ViT-g single and ensemble better on all six metrics (T5.7, TA.3). | — |
| S5 | The advantage does not come from seeing future frames. Restricted to past frames like SalFoM, SURGE-D is still more accurate on the same frames. | Causal ViT-g ensemble +0.029 [+0.019, +0.040] against SalFoM at its best blur on 6,076 frames, better on all six metrics. Each causal seed +0.018 to +0.025 (T5.8, TA.5). | "on every tenth frame of DHF1K val" |
| S6 | The comparison is conservative with respect to checkpoint selection. | SalFoM's paper states early stopping on the target dataset's validation set, which is our test set, so its validation scores are optimistic. **Correct the thesis on this point**: it currently says "not documented" (§5.2, p. 47). | Do not add "the comparison is conservative overall": the reproduction of SalFoM is not independently verified (see Q11). |

### What drives the result

| # | Claim | Evidence | Condition |
|---|---|---|---|
| S7 | Freezing the encoder does not by itself explain the gain. With the same decoder and frozen protocol, SalFoM's own encoder family falls below SalFoM. | Frozen UMT-L + SURGE-D decoder: 0.545 on val, −0.019 [−0.029, −0.010] against SalFoM, and 0.039 below frozen V-JEPA 2 ViT-L (T5.7). | UMT-L: one seed, 224² input |
| S8 | A larger encoder helps: ViT-g is better than ViT-L. | Val: single +0.017 [+0.012, +0.022] CC, better on CC, NSS, KLD, AUC-J and sAUC (SIM borderline, [+0.0000, +0.0077]). Ensembles better on all six (TA.2). Holdout: better in all three seeds. Hollywood-2 (Vig gaze): better on all six metrics (+0.0046 [+0.0032, +0.0060]). | — |
| S9 | At SalFoM's encoder size, single models are **not** reliably better; only the ensemble is. | ViT-L seeds against SalFoM at best blur: +0.010 [−0.001, +0.020], +0.006 [−0.003, +0.016], +0.013 [+0.003, +0.024]. Ensemble +0.017 [+0.007, +0.027] offline and +0.017 [+0.007, +0.028] causal (T5.7, T5.8). | A strong statement of a limitation; keep it. |
| S10 | Three-seed ensembles raise CC but slightly lower sAUC. | ViT-L +0.0074 [+0.0050, +0.0099], ViT-g +0.0025 [+0.0002, +0.0049]; sAUC −0.0022 and −0.0029, intervals below 0 (TA.8). | — |

### Efficiency

| # | Claim | Evidence | Condition |
|---|---|---|---|
| S11 | Offline SURGE-D needs far less time per output frame than SalFoM on the same GPU. | 22–26× less (ViT-g) and 41–64× less (ViT-L), in three runs at batch 1 and 8. Criterion fixed in advance: ≥2× (T5.13, p. 60). | "offline", "as released" (the time ratio depends on implementation) |
| S12 | Offline SURGE-D needs about 9× (ViT-g) to 25× (ViT-L) less encoder computation per output frame than SalFoM. | 0.328 and 0.114 TFLOPs per output frame against 2.866 for SalFoM's encoder alone (T5.12; `runs/thesis/umt_flops.json`). | "offline" |
| S13 | The saving comes from predicting 16 frames per forward pass. | One pass yields maps for 16 original frames (Alg. 4.1); the per-frame cost is the clip cost / 16. | — |
| S14 | Offline SURGE-D runs faster than real time on one workstation GPU. | 97 fps (ViT-g) and 181 fps (ViT-L) per output frame at batch 1 on an RTX A5000; model forward pass only (T5.13). | Throughput, not latency. The offline mode needs up to ~0.5 s of future frames. |
| S15 | Training is cheap: only a ~7 M-parameter decoder is trained. A model trains in under an hour on one 24 GB GPU. | ~15 min (ViT-L) and ~31 min (ViT-g) of training; ~49 min wall clock for ViT-g including evaluation; 10 GB at batch 8 (p. 45; `runs/vitg_s0/log.jsonl`). | Excludes V-JEPA 2's own (very large) pre-training |

### Predictive surprise (RQ2)

| # | Claim | Evidence | Condition |
|---|---|---|---|
| S16 | The prediction error of the released V-JEPA 2 predictor is not a useful fixation cue on its own: it is anti-correlated with fixations, and the gate fails. | Trained token, raw error NSS −0.665 [−0.763, −0.566] at 4 fps and −0.877 at 15 fps. The masked-tube readout scores −0.958. Every paired difference against frame difference and atypicality is below 0, with both mask tokens and at both rates (T5.3, TA.9, TA.10). | "the released V-JEPA 2 predictor", "these readouts" |
| S17 | There is no evidence that higher predictor error attracts gaze. Where the error carries information, its weight is **negative**: observers look away from high-error regions. | All interval estimates of U above 0 have negative weights in both halves of the cross-fitting (T5.4). | — |
| S18 | The negative result does not depend on the mask-token bug. | The untrained token (first run) and the trained token both fail the gate; the trained token is *more* negative (T5.3). | — |

### Negative and null results (design factors)

| # | Claim | Evidence | Condition |
|---|---|---|---|
| S19 | With a frozen encoder, the learned centre prior and deep supervision are dispensable: removing either is equivalent to keeping it within ±0.003 CC. | −0.0010 [−0.0026, +0.0005] and −0.0010 [−0.0023, +0.0003]; both 90 % intervals within ±0.003, three seeds (T5.14). | "ViT-L, DHF1K holdout" |
| S20 | The LoRA setting we tested lost to the frozen encoder in all three seeds. | −0.0029, −0.0010, −0.0037; mean −0.0025 (T5.14). | "the setting we tested" (rank 16, last 12 blocks), **not** adaptation in general (see X6) |
| S21 | The audio fusion module we tested improved no audio-visual dataset. On Coutrot2 the model did use the soundtrack, but it gained nothing over the visual model. | Largest change +0.001 against a threshold of +0.010. Rotated audio cost 0.018 on Coutrot2 (T5.16). | "this module", ViT-L, one seed |
| S22 | Halving the training set from 270 to 135 videos lowers accuracy. | −0.0084 [−0.0164, −0.0004]; 135 against 540: −0.0138 [−0.0238, −0.0048] (T5.15). | Holdout, ViT-L, three seeds |

### Results on other datasets

| # | Claim | Evidence | Condition |
|---|---|---|---|
| S23 | On UCF Sports, SURGE-D clearly exceeds SalFoM's published result. | CC 0.725–0.735 against 0.631 [SalFoM paper], a margin of 0.09–0.10, above the largest documented disagreement between papers (0.08). | "published result" |
| S24 | On Hollywood-2 with the Vig gaze, three-seed ensembles raise CC and KLD but significantly lower NSS and SIM, with both encoders. | T5.11 | Our ground truth only |

### Method and reproducibility facts

| # | Claim | Evidence |
|---|---|---|
| S25 | The Hugging Face port of the V-JEPA 2 predictor reads an untrained mask token by default. | Tokens 1–9 are zero vectors (norm 0.0); only token 0 is trained (norm 0.7692); the library default is `mask_index = 1` (§4.5.1, p. 35). |
| S26 | Our evaluator reproduces reference code. | CC within 0.0003 and NSS within 0.0004 of TMFI-Net's code on SalFoM's maps; per-video agreement within 5×10⁻⁵ with a second evaluator for CC, NSS, SIM and KLD (§5.2). **sAUC excluded** (see X17). |
| S27 | Decision rules for each tested component were written in a dated, version-controlled log before the experiment, and every deviation is disclosed. | T4.1–4.2; App. B.2; git history. Word it as an internal log, not external registration. |
| S28 | The whole pipeline is reproducible from public code and public encoder weights with the commands given. | App. B; repository. Decoder checkpoints are not yet released (27–28 MB each). |

---

## Tier 2: claims that need a stated condition

| # | Claim (suggested wording) | Evidence | Condition you must state |
|---|---|---|---|
| Q1 | SURGE-D achieves the highest DHF1K validation scores among the peer-reviewed results we could verify. | CC 0.600 / 0.603 against 0.554 for the best published value (TMFI-Net, as reported by ViNet-S); also ahead on NSS (+8 %), SIM (+6 %) and AUC-J. | Different evaluation code; published validation numbers for the same model differ by up to 0.03 (TMFI-Net 0.524–0.554); our margin (+0.046) exceeds that spread. **Validation set, not the official test.** |
| Q2 | Without using audio, the ViT-g ensemble reaches higher CC than the best verified peer-reviewed results on DIEM, Coutrot1, ETMD and SumMe, ties on AVAD, and is lower on Coutrot2. | T5.10; DiffSal and ViNet tables verified. NSS is higher on all six against DiffSal; AUC-J is higher everywhere. | Ensemble against mostly single models; margins (0.02–0.03) are within cross-paper disagreement (up to 0.08); non-peer-reviewed results excluded (they would make Coutrot1 and ETMD ties); recent models unchecked. Word it as **"competitive with or above"**, not "state of the art". |
| Q3 | On UCF Sports, SURGE-D is competitive with the best published models. | Ensemble 0.735: level with ViNet-A (0.734) and 0.009 below ViNet-E (0.744), a tie under our margin. AUC-J is higher, NSS lower. | "competitive, not ahead"; ViNet-E is better on CC, NSS and SIM |
| Q4 | Offline, SURGE-D needs less computation per output frame than published 3D-CNN, transformer and diffusion models whose cost is reported. | ≈57 GMACs (ViT-L) and ≈164 (ViT-g), against TASED-Net 92, ViNet 115, DiffSal 161–187 and CASP-Net 283. | Counting conventions differ (MACs against FLOPs; some counters omit attention); competitor costs come from secondary tables; lightweight models (TinyHD, UNISAL) are far cheaper. |
| Q5 | Combined with the centre prior, frame difference and token atypicality, the predictor error adds a small amount of information about fixations, with a negative weight. | U = +0.069 [+0.035, +0.107] and +0.062 [+0.032, +0.093] bits per fixation (primary comparisons); about a fifth of what the two simple cues add (T5.4). | "as represented by our log-linear model". The centre prior enters with a fixed shape, and the intervals hold the fitted weights fixed. |
| Q6 | The form of the query does not explain the negative sign. | Forecasting the second half of a clip and filling in masked tubes (the pre-training mask shape) are both anti-correlated with fixations. | The masked-tube error measures spatial in-painting difficulty, not temporal surprise. |
| Q7 | V-JEPA 2 features suit saliency better than V-JEPA 2.1 and UMT-L at the ViT-L size. | Holdout CC 0.578 against 0.569 against 0.546, same decoder (T5.2). | One seed per encoder, 10-epoch schedule, UMT-L at 224², V-JEPA 2.1 below its 384² training resolution. The UMT-L gap (0.032) is robust; the 2 against 2.1 gap (0.009) is not. |
| Q8 | Future frames add no measurable accuracy. | Causal against offline: +0.0016 [−0.0015, +0.0046] (ViT-g) and +0.0012 [−0.0022, +0.0047] (ViT-L) CC (T5.8). | Say "no measurable difference in CC", not "equally accurate": no equivalence test, and the causal model is significantly better on some metrics (TA.6). |
| Q9 | Static-image co-training, 32-frame clips and 11 % more training videos brought no gain that could be established. | SALICON +0.0009 [−0.0039, +0.0058]; 32 frames 0.0002 below its gate; 600 against 540 +0.0015 [−0.0006, +0.0037] (T5.14). | "inconclusive", not "harmful" or "useless". The 32-frame model was still improving; one seed. |
| Q10 | A temporal stride of four halves the cost per output frame without a measurable loss. | +0.0029 [−0.0009, +0.0068], positive in all three seeds; the 90 % interval excludes a loss of 0.003 (p. 63). | Outside the rule's three outcomes; ViT-L, holdout |
| Q11 | Our SalFoM baseline follows the authors' published protocol. | 16 consecutive frames, 224², last frame predicted, released weights (matches SalFoM's paper); bf16 against fp32 CC ≥ 0.9999 on 16 clips. | Maps were produced by a wrapper that is not released; no official SalFoM test script exists; our validation CC (0.564) is below SalFoM's published *test* CC (0.569). Reproduction fidelity is not independently verified. |
| Q12 | SURGE-D beats a fully fine-tuned foundation-model approach while training only 7 M parameters. | 6.99 M (ViT-L) and 7.39 M (ViT-g) trained parameters. | Always add "with a frozen encoder of 0.30 B (ViT-L) or 1.01 B (ViT-g) parameters". The deployed model is 0.33–1.04 B. |
| Q13 | From DHF1K, without any fine-tuning, SURGE-D already transfers well to free-viewing audio-visual datasets. | Zero-shot ViT-L: DIEM 0.675 and SumMe 0.551, above some fine-tuned published models (CASP-Net 0.655 / 0.499) (T5.10). | Cross-paper comparison; competitors rarely report zero-shot; task- and speaker-driven data (UCF, Coutrot2) still need fine-tuning. |
| Q14 | Among accurate models, SURGE-D has one of the lowest training costs. | Under 1 h on one A5000. DiffSal uses 4× RTX 4090; UniST 2× RTX 4090; TinyHD 200 epochs. | Most competitors do not report training time. Excludes encoder pre-training. |
| Q15 | The encoder used no human labels. | V-JEPA 2 is self-supervised; competitors use Kinetics-, AVA- or CLIP-supervised backbones. | Pre-trained on over 1 M hours of internet video, whose possible overlap with DHF1K was not checked. |
| Q16 | A second frozen encoder (UMT-L) raised holdout CC in all six seeds, by +0.0038 on average, but the gain could not be established; it was not adopted. | Six seeds: +0.0040, +0.0070, +0.0033, +0.0038, +0.0017, +0.0032; 97.5 % interval [−0.0009, +0.0093] (second look, T5.17–5.18). First look: DINOv2-L +0.0041 [−0.0007, +0.0094], UMT-L +0.0047 [−0.0001, +0.0103], both inconclusive. | Say "consistent in sign but not established"; report both looks. |
| Q17 | A decoder given the predictor's error for every clip learned to ignore it. | Seed 0: holdout CC 0.5857 with matched, 0.5856 with 180°-rotated and 0.5856 with zero error maps; best epoch 0.5857 < screen 0.5896 and 0.0009 below SURGE-D (T5.17). | One seed; "learned to ignore it on DHF1K". |
| Q18 | Sound-source maps (EZ-VSL) did not help on Coutrot2 and hardly depended on the sound. | Fold-mean CC 0.8203 visual, 0.8220 matched, 0.8213 rotated soundtrack; maps − visual +0.0017 [+0.0001, +0.0036], matched − rotated +0.0007 [−0.0002, +0.0021] (T5.19). Matched and rotated maps correlate 0.96 per frame. | One seed, 15 videos, one dataset; other datasets not run. |
| Q19 | A decoder that carries its state across windows (a stream buffer in its temporal convolutions) halves the encoder computation per output frame and, within the same training runs, was more accurate than the standard decoder with overlapping windows. | Holdout, seeds 0–2: +0.0051 [+0.0022, +0.0083] against the standard decoder of the same runs (+0.0050, +0.0051, +0.0052 per seed); 0.032 against 0.063 encoder passes per output frame (T5.20, Fig. 5.6). Against the separately trained SURGE-D: +0.0044 [−0.0009, +0.0098], inconclusive; not adopted. | Say "within the same training runs"; the registered primary comparison, with separately trained models, was inconclusive. |

---

## Tier 3: weak claims (only as "suggests", "is consistent with" or "a possible explanation")

| # | Statement | Why it is only weak | How to word it |
|---|---|---|---|
| W1 | Accuracy may still be limited by the amount of training data at 540 videos. | 270 → 540: +0.0054 [−0.0011, +0.0112], inconclusive; train–holdout gap widens as data shrink (descriptive). | "The learning curve is consistent with, but does not establish, a data limit." |
| W2 | The predictor error mainly reflects how atypical a region is. | Eleven clips, no gaze data, untrained token: r ≈ 0.60 with atypicality and 0.62 with the wrong-past error map. | "A small diagnostic suggests …" |
| W3 | The learned centre prior adds nothing because the network already encodes position through the borders of the squashed input. | Not tested. | "A possible explanation is …" |
| W4 | V-JEPA 2 beat V-JEPA 2.1 because the four-tap decoder supplies local detail, or because 2.1 ran below its training resolution. | Two explanations, neither tested. | "Two explanations are compatible with the data." |
| W5 | LoRA failed because 540 videos add more variance than signal, or because adaptation degrades good features. | The LoRA test is confounded with extra training epochs; one setting. | "Possible reasons include …" |
| W6 | The Coutrot2 failure arises because gaze follows the speaker, which an image-only model cannot determine. | Consistent with the literature and the panel-discussion example; not tested. | "is consistent with" |
| W7 | Ensembles help ViT-L more than ViT-g because the stronger encoder leaves less variance. | An interpretation of two numbers. | "possibly because" |
| W8 | Ensembles lower NSS and SIM on Hollywood-2 (Vig) because the sparse ground truth (≤ 5 observers) rewards sharp peaks. | Plausible; not tested. | "likely reflects" |
| W9 | Longer training might let 32-frame clips pass their gate. | The model was still improving at its last epoch; one seed. | "might" |
| W10 | A shared-encoder ensemble would cost about as much as a single model; the causal ensemble would then need ~0.7–0.8× SalFoM's time. | Derived from measured encoder and decoder times; not implemented (the code runs each member's encoder). | "would, by our estimate" |
| W11 | Part of the 22–26× time advantage reflects implementation (SalFoM's explicit attention against fused attention). | The share was not measured. The design-level figure is ≈ 9× in FLOPs. | "The measured ratio also reflects the released implementations." |
| W12 | SURGE-D's maps are qualitatively similar to SalFoM's on single frames. The gain shows over whole videos. | Five frames chosen by a fixed rule; on four of them SalFoM is marginally closer (Fig. 5.3). | Descriptive only |
| W13 | The decoder ignores the predictor's error because its information is already in the encoder's features. | Equally consistent with too little training (3 epochs, 540 videos) for the zero-initialised modulation to learn it. | "One reading is …; another is …" |
| W14 | Sound-source maps fail on conversations because at 7×7 they mark people rather than the speaker. | Matched and rotated maps correlate 0.96, and the peak moves in about half the frames; whether the moved peak is the speaker was not measured. | "in these scenes they marked the people more than the speaker" |
| W15 | The decoder state closes the positional gap at the boundaries between windows; what remains lies in each video's first window. | Post hoc and descriptive: without each video's first window, positions 0–1 trail 8–9 by 0.003 (0.041 and 0.066 for the standard decoders); with it, by 0.018 (Fig. 5.6). | "An analysis added after the result suggests …" |

---

## Tier 4: do not claim

| # | Do not claim | Why | Say instead |
|---|---|---|---|
| X1 | "State of the art on DHF1K." | No official test-set score; the submission is pending. | "Best among verifiable peer-reviewed results on the DHF1K validation set." |
| X2 | "State of the art on UCF Sports / the audio-visual datasets." | UCF: ViNet-E is ahead. Audio-visual: margins within cross-paper spread; ensemble against single models. | Q2, Q3 wording |
| X3 | Any comparison with published Hollywood-2 results. | Different ground truth (Vig gaze, ≤ 5 observers, whole clips). | "Results on Hollywood-2 with the Vig gaze; not comparable with published numbers." |
| X4 | "The decoder design (four taps, 3D pyramid, (2+1)D blocks) is responsible for the gains" or "is better than simpler decoders". | The decoder was never ablated; there is no linear-readout baseline. | "The decoder design was fixed without a test." |
| X5 | "The model exploits temporal dynamics or motion." | No frame-shuffle or static-input test; offline ≈ causal. | Say nothing, or "the contribution of temporal modelling was not measured". |
| X6 | "A frozen encoder is better than an adapted one" or "freezing was decided by experiment". | One LoRA setting, confounded with extra epochs; full fine-tuning untested. For UMT-L, fine-tuning (SalFoM) beat freezing. | S20 wording: "the LoRA setting tested did not help". |
| X7 | "The gain comes from V-JEPA's feature-prediction objective." | No equal-scale comparison with other foundation encoders (InternVideo2, VideoMAE-g). | "comes from the V-JEPA 2 encoder rather than from freezing as such" (S7) |
| X8 | "Surprise does not drive human attention" or "Bayesian surprise is refuted". | One predictor and two readouts. The error is a feature-space distance, not a change of belief. | "The error of the released V-JEPA 2 predictor does not behave as a gaze-attracting signal." |
| X9 | "SURGE-D is more efficient than SOTA models" (in general). | 0.33–1.04 B parameters and 1.2–3.9 GB of weights. Lightweight models are 10–330× cheaper per frame. The causal mode is slower than the ViNet family. | "Offline, it needs less computation per output frame than SalFoM and most published 3D-CNN, transformer and diffusion models." |
| X10 | "Real-time streaming saliency" or "low latency". | Causal mode 6–20 fps on an A5000. The offline mode needs future frames (~0.5 s). | "Faster than real time in offline throughput." |
| X11 | "A lightweight 7 M-parameter model" (without qualification). | The 7 M parameters are only the trained decoder. | Q12 wording |
| X12 | "All design choices were decided by rules written in advance." | The decoder architecture, taps, loss weights, squashing and learning rates were fixed without a rule (§4.1). The current abstract says "Design choices were decided by rules written in advance". | "Every tested component was decided by a rule written in advance." |
| X13 | "First to use V-JEPA (2) for video saliency." | PredJSal (NTIRE 2026 challenge report, arXiv) used V-JEPA2 ViT-g with a 3D-conv decoder. | Make no priority claim. Describe what is distinct: the frozen protocol, multi-frame output, the controlled comparison, and the explicit test of the predictor error. |
| X14 | "Robust to scene cuts, camera motion, clutter or dynamic backgrounds." | Not evaluated. | Do not mention, or list as future work. |
| X15 | "Generalises to unseen datasets." | Only the nine test sets; task-driven data need fine-tuning. | Q13 wording |
| X16 | "The causal model is as accurate as the offline model." | No equivalence test. | Q8 wording |
| X17 | "Validated evaluator" for **all** six metrics. | sAUC was not checked against reference code and uses a non-standard negative sampling. The abstract and Chapters 1, 3 and 6 use the phrase without this qualification. | "validated for CC, NSS, SIM and KLD; sAUC is comparable only within this work" |
| X18 | "A new, reliable Hollywood-2 ground truth." | Reliability (inter-observer ceiling) not measured. | "An openly reproducible ground truth built from the Vig et al. recordings." |
| X19 | Any number for the official DHF1K test set. | Not yet scored. | "Predictions for the official test set have been prepared." |
| X20 | "How SalFoM's weights were selected is not documented." | Contradicted: SalFoM's paper states early stopping on the validation set. | S6 wording |
| X21 | "Less time than SalFoM" for the **causal ensemble** as implemented. | As run (three encoder passes) it needs 1.8–2.2× SalFoM's time and 5.5× its encoder FLOPs. | "A single causal ViT-g model beats SalFoM (+0.018 to +0.025) in 0.61–0.73× of its time, at 1.8× its encoder computation." |
| X22 | "Exceeded" published audio-visual results while also saying "by margins within the spread of published evaluations" (the current abstract). | The second half undercuts the first. | Q2 wording |
| X23 | "A second encoder improves SURGE-D" or "an ensemble of encoders is better". | Six seeds positive, but the 97.5 % interval includes zero; not adopted. | Q16 wording |
| X24 | "Speaker-aware" or "audio-aware" SURGE-D. | The speaker maps were dropped before training (faces 12–15 px); neither audio module helped. | "Neither a global audio embedding nor sound-source maps improved the visual model; speaker maps could not be built from the released frames." |
| X25 | "A novel architecture" for SURGE-D or its extensions. | The novelty assessment rates SURGE-D incremental, and none of the five extensions was adopted. | Describe the frozen-encoder model, its controlled evaluation and the pre-registered negative results. |
| X26 | "The decoder state improves SURGE-D" or "the same accuracy at half the cost". | Its registered comparison with the separately trained SURGE-D was inconclusive (+0.0044 [−0.0009, +0.0098]), and at this spread between seeds equivalence could not be shown. | Q19 wording |

---

## How to upgrade a claim

| Claim now | Experiment that would upgrade it | Effort |
|---|---|---|
| X1, X19: DHF1K state of the art | Official test-set scores of the prepared submission | none on our side (waiting) |
| Q11: baseline fidelity | Run SalFoM through TMFI-Net's official test script; re-score TMFI-Net and ViNet-A/S with the same evaluator | 2–4 GPU-h |
| X4: decoder design | ViT-g with 3 seeds: linear readout; last tap only; 2D decoder; narrower decoder | ~6 GPU-h |
| X5: temporal modelling | Static (repeated frame) and shuffled-frame inputs at test time | < 1 GPU-h |
| W10, X21: cheap ensembles | Implement shared-encoder ensembles and time them | < 1 h of coding, plus timing |
| W11: implementation share of the time ratio | Time SalFoM with fused attention (SDPA) | 1 GPU-h |
| X6, X7: freezing against adaptation, and the objective | LoRA or partial fine-tuning trained jointly from scratch; other foundation encoders at ViT-g scale | 10–20 GPU-h |
| Q5: surprise information | Flexible spatial prior and SURGE-D's own map in the baseline set; bootstrap including the weight fit | CPU, hours |
| X18: Hollywood-2 ground truth | Leave-one-observer-out ceiling | CPU |
| X14, X15: robustness and generalisation | Cut, translation and crop-against-squash tests; zero-shot on LEDOV or the NTIRE 2026 videos | 2–4 GPU-h |
