# Architectural novelty of SURGE-D: how its components are combined, arranged and connected

**Question.** Have the existing components been organised, and made to interact, in a way that previous research
has not explored? Does that arrangement constitute a meaningful research contribution?
**Scope.** This report assesses *architectural* novelty: the arrangement and information flow. The novelty of
individual components is not the question.
**Date.** 2026-11-13.

**Architectures assessed.**

| ID | Architecture | Status | Source |
|---|---|---|---|
| A | **SURGE-D** | The thesis model | `Paper/msc-thesis/main.pdf` (commit `b6e499c`), §4.4, pp. 30–33; code in `surge/model.py` |
| B | **SURGE-D + gated second frozen expert** | Under test (research log §14) | `surge/model.py` (`Expert`, `ExpertFusion`) |
| C | **The predictor-error path** | Exists in the code (`SurpriseEncoder`, `SCM`), never trained in the final model | thesis §4.5.1 |

Arrangement C is included because it is the one connection in the project whose interaction appears unexplored.

> **Note on independence.** This assessment was written by the same AI assistant that designed SURGE-D, so it is
> not independent. Every precedent below is cited with its source. Where a check could not be made, the text says
> so.

---

## Verdict

| Architecture | Architectural novelty | Defensibility as a contribution |
|---|---|---|
| A. SURGE-D (thesis) | **Incremental** | Low as architecture. Its value is empirical: a frozen video world-model encoder with a light multi-frame decoder is enough for video saliency, at low offline cost. |
| B. SURGE-D + gated second frozen expert | **Weak to incremental** | Low. The arrangement closely parallels TM2SP (TCSVT 2025), and its benefit is not established (14A inconclusive). |
| C. Predictor-error-gated decoder (not built) | **Potentially moderate** | Untested. It is the only arrangement here with no precedent found, and the thesis's RQ2 result gives it a rationale. |

**Answer to the central question, for the thesis architecture (A): no.** SURGE-D arranges established components in
established ways, in a combination that, as far as was checked, nobody has assembled for video saliency before:
- a frozen backbone with a light readout, a log centre prior and a pixel softmax (the DeepGaze II/IIE design);
- a plain-ViT multi-depth pyramid (DPT, ViTDet);
- a (2+1)D top-down decoder;
- multi-frame output from overlapping windows (TinyHD-M; Video Depth Anything).

The components interact in a standard feed-forward way. No connection between them is new, and the thesis's own
evidence attributes the accuracy gain to the encoder, not to the arrangement (Section 4).

---

## 1. Decomposition: components, arrangement and information flow

### 1.1 Components of SURGE-D (A)

| # | Component | Role | Origin of the component |
|---|---|---|---|
| C1 | Frozen V-JEPA 2 ViT (L or g); 16 frames, stride 2, 256² (squashed); 2×16×16 tubelets → 8×16×16 tokens | Representation; receives no gradient | V-JEPA / V-JEPA 2 (Bardes et al. TMLR 2024; Assran et al. 2025 arXiv) |
| C2 | Four taps at equal depth fractions (6/12/18/24 or 10/20/30/40) | Multi-level features | DPT (Ranftl et al., ICCV 2021) |
| C3 | 3D *simple feature pyramid*: per-tap 1×1×1 projection, GN+GELU, transposed-convolution upsampling for shallow taps (P4, P8); the two deepest taps both stay at 1/16 (P16a, P16b) | Multi-scale maps; time kept at 8 | ViTDet (Li et al., ECCV 2022), with DPT's taps; extended to 3D |
| C4 | Top-down decoder of R(2+1)D residual blocks: concatenate P16a and P16b, then upsample and add FPN-style, 1/16 → 1/8 → 1/4 | Spatial refinement, temporal mixing with 3×1×1 kernels | FPN (Lin et al., CVPR 2017); R(2+1)D (Tran et al., CVPR 2018) |
| C5 | Auxiliary heads on D16 and D8 (training only) | Deep supervision | Lee et al., AISTATS 2015 |
| C6 | Trilinear upsampling in time (8 → 16 frames), plus a learned log Gaussian-mixture centre prior, then a softmax over pixels | Output is a per-frame probability distribution | Pixel softmax with a log centre bias: DeepGaze II / IIE (Kümmerer et al., ICCV 2017; Linardos et al., ICCV 2021). Learned Gaussian priors: Cornia et al. (TIP 2018) |
| C7 | Multi-frame output: maps for all 16 sampled frames; only the central half is kept; windows hop by 16 original frames and tile the video; frames in between are linearly interpolated | One encoder pass serves 16 output frames | Multi-output: TinyHD-M (WACV 2023). Overlapping-window central selection: standard in video dense prediction, e.g. Video Depth Anything (CVPR 2025) |
| C8 | Loss KL + (1 − CC) − 0.1·NSS | Distribution matching | Common in video saliency (TASED-Net, ViNet, SalFoM) |
| C9 | Seed ensembles (decoders averaged) | Variance reduction | Deep ensembles (Lakshminarayanan et al., NeurIPS 2017) |

### 1.2 Information flow

```
frames ──► [C1 frozen V-JEPA 2] ──taps 4×(D,8,16,16)──► [C3 per-tap projection + upsampling]
                (no gradient)                                     │ P4 (8×64²) P8 (8×32²) P16a, P16b (8×16²)
                                                                  ▼
                                       [C4 top-down: concat(P16a,P16b) → R(2+1)D² → ↑+P8 → R(2+1)D² → ↑+P4 → R(2+1)D²]
                                                                  │ logits 8×64²   (C5 aux heads on D16, D8: training only)
                                                                  ▼
                                       [C6 trilinear 8→16 in time] + log G_θ (learned prior) → softmax over pixels
                                                                  │ 16 maps per pass
                                                                  ▼
                                       [C7 keep central 8 sampled positions → 16 original frames; hop 16; interpolate]
```

**Properties of the flow:**
- It is strictly feed-forward from frozen features to output.
- There is no feedback, no conditioning across components, no attention in the decoder, and no use of the encoder's
  predictor.
- The only learned interaction between "parts" is the FPN-style addition of the top-down path to the lateral pyramid
  levels.

### 1.3 Arrangement B: added interaction

```
frames ─► resize 224² ─► [frozen expert: DINOv2-L per frame, or UMT-L] ─taps─► resample to (8,16,16) ─► [own simple pyramid] ─► P^e_l
for every pyramid level l:   P_l ← P_l + γ_l ⊙ σ(W_l [P_l ; P^e_l]) ⊙ P^e_l      (γ_l initialised to 0)
```

The new connections are:
- a second frozen encoder whose taps are aligned to the first encoder's token grid;
- per-level, content-dependent gating of the second pyramid into the first, starting at zero.

### 1.4 Arrangement C: the unused predictor path

The V-JEPA 2 *predictor* (the second half of the world model) produces a per-token prediction-error map (Eq. 4.8).
The code can inject this map into all four pyramid levels through zero-initialised feature-wise modulation
(`SurpriseEncoder` and `SCM`).

The thesis never trained it, because the error failed as a stand-alone cue (gate G2). It did, however, find that the
error carries information about fixations with a *negative* weight (T5.4: +0.069 and +0.062 bits per fixation
beyond the centre prior, frame difference and atypicality).

---

## 2. Closest existing architectures (peer-reviewed unless marked)

| Work | What it shares with SURGE-D's *arrangement* | Key difference |
|---|---|---|
| **DeepGaze II** (Kümmerer et al., ICCV 2017); **DeepGaze IIE** (Linardos et al., ICCV 2021) | **Frozen pretrained backbone → small trained readout → blur → plus centre bias → softmax over pixels.** "Only the readout network, blur size and centre-bias weight are trained; the feature extractor's weights stay fixed" [DeepGaze IIE, arXiv 2105.12441]. IIE also ensembles several backbones. | Images, not video. CNN backbones, 1×1-convolution readout, no temporal modelling, no multi-scale top-down decoder. |
| **Video Depth Anything** (Chen et al., CVPR 2025) | **Frozen encoder + lightweight spatio-temporal head + windowed multi-frame output.** It "leverages Depth Anything V2 as a frozen encoder", replaces the DPT head with a lightweight spatio-temporal head, and argues that keeping temporal modelling in the head "prevents the learned representation from being corrupted by the limited video data". Long videos are processed in overlapping segments [arXiv 2501.12375]. | Depth, not saliency. Image encoder (DINOv2-based) applied per frame; temporal attention in the head, not (2+1)D convolutions. |
| **DPT** (Ranftl et al., ICCV 2021); **ViTDet** (Li et al., ECCV 2022) | Multi-depth taps of a plain ViT, reassembled into a multi-resolution pyramid and fused top-down (DPT); a simple pyramid by up/down-sampling from a plain ViT (ViTDet). | 2D, and the backbones are fine-tuned. SURGE-D's C2–C4 is DPT's tap-to-pyramid mapping with ViTDet-style resampling, extended to 3D. |
| **TinyHD-M** (Hu et al., WACV 2023) | **Multi-frame output in video saliency:** the multi-input/multi-output variant predicts one map per input frame, 16 per 16-frame clip [arXiv 2301.04619]. | Small distilled CNN student (3.9 M), trained end to end; no frozen foundation encoder. |
| **SalFoM** (Moradi et al., ICPR 2024) | Video foundation encoder (UMT-L, a ViT) with a multi-branch decoder, for video saliency. | Encoder **fine-tuned end to end**; **one map per pass** (the last frame); heterogeneous decoder with a locality-aware transformer [arXiv 2404.03097]. |
| **UniST** (Xiong et al., TCSVT 2025) | Multi-level encoder features, top-down additive fusion, 16-frame clips with a non-causal (middle-frame) target. | Image MViT encoder per frame; saliency-aware transformer; the decoder collapses time to **one map per clip** (the middle frame) [arXiv 2309.08220]. |
| **TM2SP** (Li & Liu, TCSVT 2025) | **Two encoders fused through a multi-level pyramid with gated fusion.** A Video Swin encoder plus "a pre-trained image saliency encoder" supplying per-frame salient features that guide fusion and decoding, with "Inception Gate-Controlled Fusion" [abstract via Semantic Scholar, DOI 10.1109/TCSVT.2025.3529473]. | Trained end to end; the second encoder is a saliency model, not a generic frozen SSL encoder. Full text not read here. |
| **Cambrian-1** (Tong et al., NeurIPS 2024) | **Aggregating several frozen vision encoders:** its Spatial Vision Aggregator lets queries cross-attend to the features of multiple vision encoders [NeurIPS 2024 proceedings]. | Multimodal LLMs, not dense saliency; cross-attention aggregation rather than per-level gating. |
| **Flamingo** (Alayrac et al., NeurIPS 2022); **ControlNet** (Zhang et al., ICCV 2023) | Zero-initialised injection of a new pathway into a frozen model: tanh gates initialised at zero (Flamingo); zero convolutions (ControlNet). The same idea as B's γ = 0 gates and C's zero-initialised modulation. | Other tasks; generic mechanism. |
| **PredJSal** (CVSP team, NTIRE 2026 challenge report, arXiv 2604.14816; **not peer-reviewed**) | V-JEPA2 ViT-g backbone, features from three encoder layers, a 3D-convolutional decoder with multi-scale skips and temporal self-attention, and a four-seed ensemble. Motivated by prediction errors in predictive coding. | Whether its backbone is frozen is not stated; **one output frame per pass**; uses encoder *representations*, not the predictor's error. |
| **Itti & Baldi** (Vision Research 2009) | Prediction error ("Bayesian surprise") as an attention signal. | Hand-crafted features and an explicit Bayesian model; no learned decoder. |

**Searches that returned no matching architecture.** Absence of evidence is not evidence of absence.
- A frozen *video* foundation encoder used for video saliency in peer-reviewed work: SalFoM fine-tunes; PredJSal's
  freezing is unknown.
- A saliency network that takes a learned video predictor's error map as an input or gate. The closest work,
  PredJSal, uses only the representations.

---

## 3. Arrangement-level comparison

| Feature of the arrangement | SURGE-D (A) | DeepGaze IIE | Video Depth Anything | TinyHD-M | SalFoM | UniST | TM2SP | PredJSal (arXiv) |
|---|---|---|---|---|---|---|---|---|
| Frozen pretrained encoder | **yes** (video SSL ViT) | **yes** (image CNNs) | **yes** (image ViT) | no | no | not stated | no | not stated |
| Multi-depth taps → multi-scale pyramid | yes (4 taps, 3D) | multi-layer readout | DPT-style | hierarchical | yes | yes | yes | yes (3 layers) |
| Temporal modelling in the decoder | (2+1)D convolutions | none | temporal attention | 3D / multi-decoder | transformer | transformer | transformer | temporal attention |
| Outputs per forward pass | **16** (central half kept) | 1 (image) | **multi-frame** (segments) | **16** | 1 | 1 | 1 | 1 |
| Log centre prior + pixel softmax | **yes** | **yes** | n/a | no | no | no | no | no |
| Second encoder fused into the pyramid | (B: yes, gated) | output ensemble | no | teacher only | no | no | **yes (gated)** | no |
| Uses the encoder's own predictor | (C: possible) | — | — | — | — | — | — | motivation only |

Every row in which SURGE-D says "yes" also has a "yes" in some precedent. What no single precedent has is the
**co-occurrence** of the following:
- a frozen *video* foundation encoder (the JEPA family);
- a 3D DPT/ViTDet-style pyramid;
- a (2+1)D decoder;
- a DeepGaze-style log-prior softmax;
- multi-frame central-window output;
- application to video saliency.

---

## 4. Is the SURGE-D arrangement a meaningful contribution?

The test applied: novelty of arrangement requires (i) a connection or interaction that previous work did not
explore, and (ii) evidence that the arrangement, not merely its parts, produces an advantage.

### 4.1 What is genuinely new, and how much

1. **A frozen large *video* SSL encoder with a multi-frame central-window decoder, for video saliency.** As far as
   was checked, this co-occurrence is new in peer-reviewed video saliency work. Its consequence is real and
   measured: one pass of a 1 B-parameter frozen encoder serves 16 output frames, making offline SURGE-D 22–26× faster
   per output frame than SalFoM's fine-tuned per-frame design on the same GPU (T5.13).
2. **But the coupling is the obvious composition of two known ideas.**
   - Multi-output per pass (TinyHD-M) amortises any encoder.
   - Freezing (DeepGaze; Video Depth Anything) protects a pretrained encoder from small datasets.
   - Video Depth Anything already pairs a frozen encoder, a light temporal head and windowed multi-frame output, with
     the same stated rationale.

   SURGE-D transfers that pattern to saliency with a video (rather than image) encoder. That is a domain transfer of
   an established arrangement, not a new interaction.
3. **The internal arrangement (taps → 3D simple pyramid → (2+1)D top-down) is standard dense-prediction practice**
   (DPT + ViTDet + FPN + R(2+1)D) extended to 3D. The output head (log prior + softmax) is DeepGaze's.

### 4.2 Does the arrangement produce the advantage?

The thesis's own evidence says the accuracy advantage comes from the **encoder**, not the arrangement:
- **Freezing alone doesn't help.** The same decoder on frozen UMT-L scores 0.545, below SalFoM's 0.564 (T5.7).
- **The encoder's family and scale move accuracy:** V-JEPA 2 > V-JEPA 2.1 > UMT-L (T5.2), and ViT-g > ViT-L (T5.6).
- **The decoder parts tested matter little or not at all.** The centre prior and deep supervision are equivalent to
  their absence (T5.14); future frames add nothing measurable (T5.8).
- **The defining parts were never ablated.** The taps, the 3D pyramid and the (2+1)D blocks have no ablation
  (critical review, M4), so no claim that this arrangement beats a simpler one is supported.

The **efficiency** advantage comes from the arrangement, through multi-frame output, but its mechanism is TinyHD-M's.

**Conclusion for A: incremental.** The combination is sensible and effective but not unexplored in kind. Its
defensible contribution is the empirical finding, not the architecture.

### 4.3 One arrangement-level property the thesis does not exploit

Because every ensemble member and every fine-tuned head share **identical frozen encoder weights**, one encoder pass
can feed many decoders.

| What it enables | Estimated cost |
|---|---|
| Three-seed ensemble | About a single model's cost: +6 % compute (`report/sota-advantage-analysis.md` §2.2, derived) |
| Dataset- or task-specific heads served together | Not estimated |
| Cached features for cheap decoder training | Not estimated |

This is a real arrangement-level property of "frozen encoder + light heads", but it is general to frozen-backbone
systems (multi-task heads on frozen DINOv2 features are common). It is not specific to saliency. At most an
engineering contribution, and not yet implemented or measured.

---

## 5. Arrangement B: the gated second frozen expert

**What already exists.**
- *A video encoder plus a second, per-frame image encoder fused into a multi-level pyramid through gated fusion* is
  TM2SP's arrangement (TCSVT 2025).
- *Combining several frozen vision encoders* is established (Cambrian-1, NeurIPS 2024; DeepGaze IIE's backbone
  ensemble at the output level).
- *Zero-initialised gated injection* into a frozen pathway is Flamingo's and ControlNet's mechanism.

**What is new.**
- Both encoders are frozen.
- The pairing is a feature-prediction world model (V-JEPA 2) with a generic SSL or action expert, rather than with a
  saliency-trained encoder.
- The second encoder's taps are aligned to the first's spatio-temporal token grid before per-level gating.

These are variations of TM2SP's idea, not a new kind of interaction.

**Evidence of advantage.** None that meets the pre-registered bar so far:
- three-seed holdout gain +0.0047 CC [−0.0001, +0.0103] for UMT-L and +0.0041 [−0.0007, +0.0094] for DINOv2-L
  (research log 14A);
- a six-seed follow-up (14D) is running.

Even if adopted, two controls would be needed to call the *arrangement* a contribution:
- **Output-level ensemble:** average SURGE-D's map with a decoder trained on the expert alone.
- **Ungated fusion:** concatenation or plain addition of the two pyramids.

Without those controls, any gain could come from having more features, not from the gated arrangement.

**Conclusion for B: weak to incremental,** and currently unsupported by evidence.

---

## 6. Arrangement C: the predictor-error path

**Why it is the strongest candidate.**
- **No precedent was found** for a saliency decoder that takes the error map of a learned *latent* video predictor
  as an input or gate:
  - Itti and Baldi's surprise is hand-crafted;
  - CASP-Net uses predictive coding for audio-visual consistency, not as a saliency input;
  - PredJSal invokes prediction error only as motivation and decodes encoder representations.
- **It uses both halves of the world model.** The encoder supplies "what is there"; the predictor's error supplies
  "what was not predictable". Feeding the error into the decoder through a learned (sign-free) gate creates an
  interaction between the two halves of a JEPA model that previous saliency architectures do not have.
- **It has a scientific rationale from the thesis itself.** The error is anti-correlated with fixations yet carries
  incremental information (+0.06–0.07 bits per fixation beyond simple cues). A learned gate could exploit that
  information as an *inhibitory* signal, which a hand-designed "surprise attracts gaze" module could not.

**Why it is not yet a contribution.**
- It has not been trained or tested.
- The incremental information was measured against simple cues, not against SURGE-D's own features, which may
  already contain it.
- Its cost is real: the masked-tube readout needs about four extra encoder passes on visible tokens.

**Conclusion for C: potentially moderate novelty, conditional on a demonstrated gain.** It needs:
- a pre-registered test against SURGE-D;
- an ablation that replaces the error map with a control (atypicality, or another video's error map), to show that
  the *predictor interaction*, not extra input, causes any gain.

---

## 7. Final assessment

| Question | Answer |
|---|---|
| Are SURGE-D's components organised and made to interact in a way previous research has not explored? | **No.** The co-occurrence is new for video saliency, but each connection follows an established pattern: frozen backbone + readout + log-prior softmax (DeepGaze), taps → 3D pyramid → top-down (DPT/ViTDet/FPN), multi-frame windowed output (TinyHD-M; Video Depth Anything). The flow is plain feed-forward, with no new interaction. |
| Does the arrangement constitute a meaningful research contribution? | **As architecture, only incrementally.** The arrangement's measurable advantage is offline efficiency, through a known mechanism. The accuracy advantage is attributable to the encoder. The defensible contribution is empirical (a frozen JEPA video encoder suffices and runs cheaply offline) plus the evaluation methodology. |
| Novelty rating | **A: incremental. B: weak to incremental (unproven). C: potentially moderate (untested).** |
| Most defensible framing for the thesis or a paper | "We show that a frozen self-supervised video world-model encoder, read out by a light multi-frame decoder in the manner of DeepGaze and DPT, outperforms a fine-tuned foundation-model approach at a fraction of the offline cost", rather than "we propose a novel architecture". If arrangement C is built and shown to help under a pre-registered rule with the controls above, it would be the project's strongest architectural claim: *an interaction between a world model's encoder and its predictor inside a saliency decoder*. |
| What would not survive review | Claiming the multi-frame decoder, the frozen-encoder readout, the pyramid or the gated expert fusion as architectural novelty. Each has a direct precedent (Sections 2–5). Priority claims about V-JEPA for video saliency are also undercut by PredJSal (non-peer-reviewed) and should be avoided. |

---

## Sources

| Work | Source |
|---|---|
| DeepGaze IIE | Linardos et al., ICCV 2021, <https://openaccess.thecvf.com/content/ICCV2021/html/Linardos_DeepGaze_IIE_Calibrated_Prediction_in_and_Out-of-Domain_for_State-of-the-Art_Saliency_ICCV_2021_paper.html>; arXiv:2105.12441 |
| Video Depth Anything | Chen et al., CVPR 2025, <https://cvpr.thecvf.com/virtual/2025/poster/33862>; arXiv:2501.12375 |
| TinyHD | Hu et al., WACV 2023, arXiv:2301.04619 |
| SalFoM | Moradi et al., ICPR 2024, arXiv:2404.03097 |
| UniST | Xiong et al., TCSVT 2025, arXiv:2309.08220 |
| TM2SP | Li & Liu, TCSVT 2025, DOI 10.1109/TCSVT.2025.3529473; abstract via <https://api.semanticscholar.org> |
| Cambrian-1 | Tong et al., NeurIPS 2024, <https://proceedings.neurips.cc/paper_files/paper/2024/hash/9ee3a664ccfeabc0da16ac6f1f1cfe59-Abstract.html> |
| NTIRE 2026 Video Saliency Prediction challenge report (PredJSal) | arXiv:2604.14816 |
| CASP-Net | Xiong et al., CVPR 2023 |

Cited from the thesis bibliography (verified there against publisher records), not re-fetched for this report:
DPT (ICCV 2021), ViTDet (ECCV 2022), FPN (CVPR 2017), R(2+1)D (CVPR 2018), deep supervision (AISTATS 2015),
learned Gaussian priors (Cornia et al., TIP 2018), Itti & Baldi (Vision Research 2009), DeepGaze II (ICCV 2017).

Cited from general knowledge, not in the thesis bibliography and not re-fetched for this report: Flamingo (Alayrac
et al., NeurIPS 2022) and ControlNet (Zhang et al., ICCV 2023), for the zero-initialised gating mechanism.
