# SURGE: Surprise-Gated Video Saliency Prediction with a Self-Supervised World Model

**Research design report**. Prepared 2026-11-04. Literature searched through early November 2026.

> **How to read this document.** Statements are tagged when the distinction matters:
> **[Established]** = reported in a published or preprint source (cited);
> **[Hypothesis]** = a claim this project would test;
> **[Expected]** = my reasoned expectation, not a measured result;
> **[Estimate]** = a back-of-envelope number (method stated) that must be measured.
> No experimental results in this document are real. All numbers attributed to other methods are copied from the cited sources.

---

## 0. Executive summary

| Item | Outcome |
|---|---|
| **Selected architecture** | **SURGE** (SURprise-Gated Encoder–decoder; a working name). A frozen V-JEPA 2 video world model provides two things: encoder features (*what is there*) and the error of its own pretrained latent predictor (*what was not expected*). The error map (1) modulates every level of a multi-level 3D saliency decoder through zero-initialized spatial FiLM, and (2) through its global magnitude, gates a learned center prior over time, to model the documented post-cut return of gaze to the center. |
| **Core hypothesis** | Prediction errors from a world model pretrained on more than 1M hours of video carry fixation-relevant information (expectation violations) that encoder features plus small saliency datasets do not provide. |
| **Novelty status** | **Potentially novel. No direct precedent found.** The closest work, PredJSal (2nd place, NTIRE 2026), is explicitly *motivated* by predictive coding but uses only V-JEPA2 encoder features, not predictor error. Surprise-as-saliency exists in low-level, non-learned form (Itti & Baldi, 2005–2009). Latent-prediction-error "surprise" has been used for robotic event triage (DINO-Explorer, 2026) and intuitive physics (Garrido et al., 2025), but not for human fixation prediction. The center-prior gate has **partial precedent** (Gorji & Clark, CVPR 2018; time-varying center bias). |
| **Feasibility** | About 7M trainable parameters on a frozen ViT-L (300M) plus predictor (about 22M). [Estimate] about 2.7 TFLOPs per 32-frame chunk that yields 16 frames of output, and about 8–15 T4 GPU-hours per training run. Fits Kaggle/Colab. |
| **Performance target** | Competitive: ≥ TMFI-Net on the DHF1K validation set (CC 0.554, NSS 3.201). Stretch: SalFoM level (DHF1K **test** CC 0.569). **Not guaranteed.** The primary scientific test is SURGE vs. an identical encoder-only baseline, with capacity and motion controls. |
| **Main risks** | Surprise may be redundant with motion or encoder features. The predictor was not pretrained on causal (future-frame) masks. A frozen ViT-L may trail LoRA-tuned or billion-scale models. Default HuggingFace usage leaks target frames into the context (see §10.4). |

---

## 1. Literature review and architecture comparison

### 1.1 Scope and method

I prioritized CVPR/ICCV/ECCV/NeurIPS/TPAMI/TIP/TCSVT/IJCV papers, the official DHF1K leaderboard, and the two recent video-saliency challenge reports (AIM 2024, NTIRE 2026), plus arXiv preprints for 2024–2026 work. Searches were targeted, not exhaustive. Novelty statements below therefore mean *"not found by targeted search as of 2026-11-04"*, never *"verified absent"*.

### 1.2 How the field evolved

| Generation | Representative models | Spatial features | Temporal modeling | Output per forward pass |
|---|---|---|---|---|
| 2D CNN + recurrence (2017–2020) | ACLNet, DeepVS (OM-CNN), SalEMA, UNISAL, STRA-Net, SG-FCN, Gorji & Clark | ImageNet CNNs (VGG, MobileNetV2), static-saliency attention | ConvLSTM, ConvGRU, EMA; optical flow in two-stream variants | 1 map per frame (recurrent) |
| 3D CNN encoders (2019–2025) | TASED-Net, ViNet, HD2S, TSFP-Net, STSANet, TinyHD, ViNet-S | Kinetics-pretrained S3D, or 3D-inflated MobileNetV2 | 3D convolutions over a 16–32-frame clip; self-attention in STSANet | 1 map per clip (usually the last frame); TinyHD-M outputs 16 maps |
| Video transformers (2022–2024) | VSFT, TMFI-Net, THTD-Net, UniST, DiffSal | Video Swin (Kinetics) | Shifted-window spatio-temporal attention; temporal reduction in the decoder | 1 map per clip; DiffSal by iterative denoising |
| Video foundation models (2024–2026) | SalFoM (UMT-L), ViNet-A (AVA SlowFast), NTIRE 2026 entries (InternVideo2-6B/1B, V-JEPA2-g) | Large video encoders, often LoRA-tuned | Encoder attention plus light temporal attention or gating in the decoder | 1 map per clip; ensembles common |

### 1.3 Architecture-by-architecture analysis

DHF1K CC values are **test-set** values from the official leaderboard unless marked (val). Size and time come from the leaderboard: per-frame times were measured by different authors on different GPUs, so they are only roughly comparable.

| Model (venue) | Information flow and key components | Spatial / temporal / output | Strengths | Limitations | Cost | DHF1K CC |
|---|---|---|---|---|---|---|
| **ACLNet** (CVPR'18 / TPAMI'19) | VGG-16 features → a *supervised* attention module (trained on static SALICON fixations) reweights features → ConvLSTM → per-frame map | 2D CNN / ConvLSTM / per frame | Introduced DHF1K. Uses static data: [Established] training without static images drops AUC-J from 0.890 to 0.847 in the authors' analysis | No explicit motion; ConvLSTM memory is weak | 250 MB, 0.02 s | 0.434 |
| **DeepVS / OM-CNN+2C-LSTM** (ECCV'18) | Objectness subnet + motion subnet → 2-layer ConvLSTM with center-bias dropout; introduced LEDOV | 2D CNN + motion / ConvLSTM / per frame | Explicit object and motion priors | Heavy; pre-3D era | 344 MB, 0.05 s | 0.344 |
| **SalEMA** (BMVC'19) | SalGAN encoder–decoder + exponential moving average on bottleneck features | 2D / EMA / per frame | Showed simple recurrence competes with ConvLSTM. Highest SIM on the leaderboard (0.466) despite a low CC, a reminder that SIM rewards different map properties | Weak temporal reasoning | 364 MB, 0.01 s | 0.449 |
| **UNISAL** (ECCV'20) | MobileNetV2 → Bypass-RNN (ConvGRU, skipped for images) → domain-adaptive priors (learned Gaussians), fusion, smoothing, and BN; joint image and video training | 2D / ConvGRU / per frame | About 4M params; fast; multi-domain; learned priors match dataset center bias | Light backbone caps accuracy | 15.5 MB, 0.009 s | 0.490 |
| **TASED-Net** (ICCV'19) | S3D (Kinetics) on 32 frames → decoder with auxiliary pooling (temporal reduction) and transposed convolutions → last-frame map | 3D CNN / within clip / 1 per clip | First strong 3D-CNN VSP model | One 32-frame forward pass per output frame | 82 MB, 0.06 s | 0.470 |
| **ViNet** (IROS'21) | S3D → U-Net-style 3D decoder with skip connections and trilinear upsampling (audio variant AViNet) | 3D CNN / within clip / 1 per clip | Simple and strong; became a template | Clip-bounded | 124 MB, 0.016 s | 0.511 |
| **HD2S** (IJCV'21) | S3D → hierarchical decoder emitting maps at several abstraction levels (+ gradient-reversal domain adaptation) → fused | 3D CNN / within clip / 1 per clip | Multi-level supervision; cross-domain robustness | Clip-bounded | 116 MB, 0.03 s | 0.503 |
| **TSFP-Net** (2021) | S3D → temporal–spatial feature pyramid with top-down fusion | 3D CNN / within clip | Efficient (15.3M params) | Clip-bounded | 58.4 MB, 0.011 s | 0.517 |
| **STSANet** (TMM) | S3D → spatio-temporal self-attention at several levels → attentional multi-scale fusion | 3D CNN + attention / long range *within* clip | Long-range relations inside the clip | 168.6M params | 643 MB, 0.035 s | 0.529 |
| **TinyHD** (WACV'23) | 3D-inflated MobileNetV2 → heterogeneous light decoders; hierarchical-map distillation from HD2S; **MIMO** variant outputs 16 maps per 16-frame pass | 3D / within clip / **multi-frame** | 3.9M params; multi-output efficiency | Accuracy below SOTA | — | 0.493 (TinyHD-M) |
| **VSFT** (TCSVT'22) | Video saliency *forecasting* transformer | Transformer | Anticipation formulation | — | 71.4 MB, 0.04 s | 0.519 |
| **TMFI-Net** (TCSVT'23) | Video Swin (4 scales) → semantic-guided encoder (top-down propagation of high-level semantics) → hierarchical decoder with channel/temporal/spatial attention → several maps fused by 3D conv | Video Swin / within clip / 1 per clip | Strong across DHF1K, Hollywood-2, UCF; a common base (used by the 4th-place NTIRE 2026 team) | 61.3M params; clip-bounded | 234 MB, 0.033 s | 0.546 (val 0.554) |
| **THTD-Net** (VISAPP'24) | Video Swin → a single decoder that keeps a high temporal dimension and reduces it gradually | Video Swin / within clip | Matches multi-branch decoders with one branch | Clip-bounded | 220 MB, 0.08 s | 0.548 (val 0.553) |
| **DiffSal** (CVPR'24) | Video (+ audio) encoders → Saliency-UNet with multimodal attention modulation → conditional diffusion denoising | Transformer / within clip / iterative | Strong on Hollywood-2 (CC 0.765) and audio-visual sets | Iterative sampling; 70.5M params | — | val 0.533 (video only) |
| **SalFoM** (arXiv'24) | **UMT-L** (fine-tuned), 16×224×224 → heterogeneous decoder (TCFE transformer branch, dynamic DFD branch, static SFD branch) → concatenation → 2D conv; KL+CC loss | Foundation ViT / within clip | **Best DHF1K test** among published leaderboard entries | Weaker on Hollywood-2 (0.709) and UCF (0.631) than TMFI. Heavy: 1574 MB, 0.6 s/frame | 1574 MB, 0.6 s | **0.569** |
| **ViNet-S / ViNet-A / ViNet-E** (ICASSP'25) | ViNet-S: S3D + grouped-conv/channel-shuffle decoder (9.5M). ViNet-A: SlowFast pretrained for AVA action *localization* (38.7M). E = average of both | 3D CNN / within clip (ViNet-A samples 32 of 64 frames) | Very fast (ViNet-S about 1000 fps batched on an RTX 4090). ViNet-E best UCF (0.744) and Hollywood-2 CC (0.766) | DHF1K val 0.525–0.549 | 36–184 MB | val 0.529 / 0.525 / 0.549 |
| **NTIRE 2026 top entries** (2026) | ViSAGE: InternVideo2-6B + LoRA + two expert decoders (temporal gating, FiLM from CLS token, learned center prior). PredJSal: **V-JEPA2-g encoder** + temporal self-attention + 3D conv decoder, 4-seed ensemble. ARK: InternVideo2-1B + temporal attention | Billion-scale encoders | Top accuracy on a 2000-video mouse-tracking dataset | 2–7B params | — | n/a (different dataset) |
| **Memory / past-saliency models** | SG-FCN (TCYB'19): previous-frame saliency + motion-boundary map + frame. Gorji & Clark (CVPR'18): multi-stream ConvLSTM with "attentional push" pathways (actor gaze, attentional bounce, **abrupt scene changes**). STM-SalNet (ICONIP'25): ViT + memory bank with "active forgetting". Wulff et al. (arXiv'25): dual-stream transformer fed **ground-truth** past saliency maps (DAVSOD, not fixations) | Various | Explicit history | Weak backbones, no fixation benchmarks (Wulff), or not SOTA-level | — | — |

### 1.4 What drives performance (evidence)

1. **Backbone and pretraining dominate.** [Established] DHF1K test CC rises with each backbone generation: VGG/MobileNet 0.43–0.49, S3D 0.47–0.53, Video Swin 0.546–0.548, UMT-L 0.569. All three top NTIRE 2026 entries use video foundation models with at least 1B parameters, and ViSAGE's ablation found that removing LoRA adaptation causes its largest drop.
2. **Multi-level decoding with intermediate maps** (HD2S, TMFI, TinyHD, ViSAGE expert 2) is near-universal in strong models. [Established as practice]
3. **Clip length and sampling.** Most models use 16–32 frames (about 0.5–1 s); ViNet-A samples 32 frames from a 64-frame window. [Established as practice]
4. **Priors.** Center bias is strong: on NTIRE 2026 a Gaussian center prior alone reaches CC 0.41 (mouse-tracking data). Learned priors appear in UNISAL and ViSAGE. [Established]
5. **Ensembling** gives small, consistent gains (ViNet-E, PredJSal seeds, ViSAGE experts). [Established]
6. **Data scarcity versus model size.** SalFoM leads on DHF1K but trails TMFI on Hollywood-2 and UCF. Its authors note that overfitting hurts generalization on the small UCF set. [Established observation]

### 1.5 Benchmarks and protocol caveats (important for planning)

- **DHF1K**: 1000 YouTube videos at 640×360 and 30 fps (582,605 frames, 19,420 s), 17 observers, SMI RED 250 eye tracker at 250 Hz, split 600/100/300. **Test annotations are held out** (results are emailed to the dataset curators). Recent papers (DiffSal, ViNet-S/A) report **validation only**, so many "SOTA" claims are val numbers, and early stopping on val inflates them.
- **Hollywood-2**: 1,707 videos from 69 movies, 19 observers, mostly *task-driven* (12 action recognition, 4 context recognition, 3 free viewing). The benchmark uses **single-shot split clips** (823/884 videos, about 3,100/3,559 shot clips; some clips are only a few frames long), so **standard Hollywood-2 evaluation contains no within-clip cuts**.
- **UCF-Sports**: 150 videos (103/47), task-driven.
- **DIEM**: 84 videos with about 50 free-viewing observers, including edited material (trailers, ads), so it contains real cuts. It is used in audio-visual saliency work.
- **LEDOV**: 538 videos, 32 observers.
- **AViMoS (AIM 2024)** and **NTIRE 2026**: 1,500 and 2,000 videos with crowdsourced *mouse tracking* (≥70 viewers per video). They are not interchangeable with eye tracking.
- **Metrics**: implementations differ (MATLAB vs Python ports, sAUC negative sampling). The DHF1K leaderboard reports AUC-J, SIM, sAUC, CC and NSS but **not KL**. Ground-truth density estimation matters: a 2026 study re-estimated the LEDOV inter-observer ceiling from 2.00 to 3.05 bits/fixation, suggesting much explainable structure remains unmodeled.

---

## 2. Research opportunities and limitations

### 2.1 Limitations of current models

| ID | Limitation | Evidence |
|---|---|---|
| **L1** | **Clip-bounded, recomputed context.** Most SOTA models see about 0.5–1 s and predict one frame per forward pass. Persistence beyond the window is not modeled, and windows are recomputed per frame. | Model designs in §1.3. Partial remedies: RNNs (weak backbones), STM-SalNet (memory bank), TinyHD-M (multi-output). |
| **L2** | **No explicit expectation-violation cue.** Encoders describe *what is present*. Cognitive science shows gaze is drawn to *what is unexpected* (Bayesian surprise; abrupt onsets). Deep VSP models can only learn this implicitly from about 600 training videos. | Itti & Baldi showed surprise predicts gaze shifts better than other tested metrics. PredJSal (NTIRE 2026) *argues* saliency should track predictive-coding errors but uses only encoder features. |
| **L3** | **Static output calibration.** Map spread is fixed, while inter-observer agreement (attentional synchrony) varies over time. | IOVC literature (Le Meur 2011; Yue et al. 2021); deep saliency models are often overconfident (arXiv 2105.12441). |
| **L4** | **Event structure ignored by current SOTA.** Center bias is strongest at scene onset. After jump cuts, memory-guided attention collapses and recovers over about 2.5 s. | Tseng et al. 2009; Carmi & Itti 2006. Modeled by Gorji & Clark 2018 (scene-change pathway) and in non-deep time-varying center-bias models (Zhao & Koch 2011; Coutrot et al. 2017), but absent from TMFI/SalFoM/NTIRE-era designs. |
| **L5** | **Data scarcity versus capacity.** Large fine-tuned models overfit small datasets. | SalFoM on UCF/Hollywood-2 (§1.4). |
| **L6** | **Evaluation inconsistency.** Test labels are withheld, metric ports differ, and Hollywood-2 is shot-split. | §1.5. |

### 2.2 Candidate combinations and comparison matrix

| # | Component / principle | Original source(s) | Existing use in (video) saliency | Proposed new integration | Precedent status |
|---|---|---|---|---|---|
| 1 | Pretrained **latent predictor** of a self-supervised video world model | V-JEPA (Bardes et al. 2024); V-JEPA 2 (Assran et al. 2025) | Encoder features only (PredJSal, NTIRE 2026) | Predictor **error** as a spatial cue for fixation prediction (C2) | **No direct precedent found** |
| 2 | Surprise / prediction error as an attention signal | Itti & Baldi 2005/2009 (low-level Bayesian surprise); Garrido et al. 2025 (V-JEPA surprise, physics); DINO-Explorer 2026 (robot event triage) | Low-level, non-learned models (2005–2009) | Semantic latent surprise inside a supervised deep VSP model (C2) | Conceptual precedent; implementation new |
| 3 | Spatial feature modulation (FiLM / SPADE) | Perez et al. 2018; Park et al. 2019 | ViSAGE: FiLM from the *global* CLS token | Spatially varying, surprise-conditioned, zero-initialized modulation (C2) | Mechanism known; conditioning signal new |
| 4 | Learned center prior | SAM-ResNet (Cornia et al. 2018); UNISAL 2020; ViSAGE 2026 | Static or per-domain | Time-varying, gated by global surprise (C2) or by memory continuity (C1) | **Partial precedent** (Gorji & Clark 2018; Zhao & Koch 2011; Coutrot et al. 2017) |
| 5 | Mask-conditioned key–value memory (appearance keys; appearance+mask values) | STCN 2021; XMem 2022; SAM 2 2024 (mask summed into memory, occlusion embedding) | SAM 2 adaptations for *video salient object detection*; STM-SalNet (feature memory bank) | Fixation-density-weighted values plus familiarity read-outs (C1) | **Partial precedent** |
| 6 | Past saliency as input or prior | Rudoy et al. 2013; SG-FCN 2018; Wulff et al. 2025 | Previous-frame map or ground-truth past maps | Through memory values (C1) | **Substantial precedent** |
| 7 | Scene-change / continuity handling | Carmi & Itti 2006 (cognitive); Gorji & Clark 2018 (pathway) | Dedicated ConvLSTM pathway | Continuity derived from memory familiarity (C1) or from global surprise (C2) | **Partial precedent** |
| 8 | Inter-observer congruency (IOC) prediction | Le Meur et al. 2011; Yue et al. 2021; Bruckert & Christie 2023 | Standalone IOC predictors | Predicted IOC sets the output temperature and prior mixture (C3) | **Partial precedent** |
| 9 | Calibration / temperature | Guo et al. 2017; saliency calibration (arXiv 2105.12441); noise-aware VSP (BMVC'21) | Post-hoc analysis; noise-aware training | Learned per-frame temperature (C3) | **Partial precedent** |
| 10 | Multi-frame (MIMO) decoding | TinyHD 2023 | Efficiency | Amortizing a frozen foundation encoder (C2) | **Substantial precedent** (not claimed as novel) |

### 2.3 Directions considered and rejected

- **Mamba / state-space temporal modules.** Substantial precedent: SSF-DiffNet (NTIRE 2026) uses Mamba-style blocks, Samba covers video SOD, and SUM covers image saliency. Mamba also offers efficiency rather than a new information flow.
- **Object-centric tracking tokens** (SAM 2 tracks + object transformer). Costly mask preprocessing, sensitivity to mask errors, and partial precedent (OM-CNN objectness, salient-object ranking). The claim that objects predict fixations better than early saliency is itself contested (Einhäuser et al. 2008 vs. Borji et al. 2013).
- **Vision-language priors** (prompted "face/text/person" maps). Strong backbones already capture these, and global semantic conditioning exists (ViSAGE FiLM; AAM used CLIP text in NTIRE 2026).
- **Diffusion decoders.** Precedent exists (DiffSal, SSF-DiffNet), and iterative sampling conflicts with the compute budget.

---

## 3. Three candidate architectures

### C1. ECHO: Episodic, Continuity-gated History-Of-attention memory (streaming)

**Idea.** Write *what was attended* into a key–value memory, and read it back for the current chunk. The memory's **familiarity field** (best match of each current token to memory) is reused twice: locally as a **novelty** map (new content attracts gaze), and globally as a **continuity** gate that controls both memory injection and the center-prior strength. When the memory no longer matches, as after a cut, the model falls back to the center prior and bottom-up cues, mirroring Carmi & Itti (2006).

```
stream ─► chunk k: 16 frames, 224×384, hop H = 8
              │
      Video Swin-T (Kinetics-400) ─► F1 96×8×56×96 · F2 192×8×28×48 · F3 384×8×14×24 · F4 768×8×7×12
              │                              │
              │                  Q = W_q F3 ─┴─► A = softmax(Q Kᵀ/√d)  ◄── memory keys K (N_m×336 tokens)
              │                                   │            │
              │                        readout R = A·V   familiarity φ_i = max_j cos(q_i, k_j)
              │                                   │            ├─ novelty   ν = 1 − φ        (where)
              │                                   │            └─ continuity c = MLP(stats of φ) (when)
              ▼                                   ▼
      F3' = F3 + c·Z(R) + Z(ν)                (Z = zero-initialized 1×1×1 conv)
              │
      top-down decoder F4 → F3' → F2 → F1 → logits L (8×56×96) → keep last 8 frames
              │
      L + (β0 + β1·(1 − c))·log G_θ  →  softmax  →  S
              │
      write: K_new = W_k F̄3 ;  V_new = W_v[ F̄3 ⊙ (1 + γ·Ŝ↓) ; ψ(Ŝ↓) ]  → FIFO memory (N_m = 16 entries ≈ 4.3 s)
```

| Stage | Input → output | Purpose |
|---|---|---|
| Short-term encoder | 16×3×224×384 → F1–F4 (shapes above) | Appearance and motion within about 0.5 s |
| Memory read (at 1/16 scale) | Q: 2688×64; K: (16·336)×64; V: (16·336)×128 → R: 2688×128, φ: 8×14×24, c: 8 | Attention persistence, novelty, continuity |
| Fusion | F3 + c·Z(R) + Z(ν) | Inject history only when the scene is continuous |
| Decoder + prior | Multi-level top-down 3D → 8 maps at 56×96; prior exponent β0 + β1(1−c) | Saliency, with a prior that strengthens after cuts |
| Memory write | Predicted maps, downsampled to 14×24 → saliency-weighted values | Store *attended* content, not all content |

**Math.** $A=\mathrm{softmax}(QK^\top/\sqrt{d})$, $R=AV$, $\varphi_i=\max_j \langle \hat q_i,\hat k_j\rangle$, $\nu_i=1-\varphi_i$, $c=\sigma(\mathrm{MLP}[\bar\varphi,\mathrm{sd}(\varphi),q_{10}(\varphi),q_{90}(\varphi)])$, $S_t=\mathrm{softmax}(L_t+(\beta_0+\beta_1(1-c))\log G_\theta)$.

**Complexity.** [Estimate] About 35M parameters (Swin-T 28M, decoder about 6M, memory about 1M). torchvision lists 43.9 GFLOPS for Swin3D-T at its reference input. Each chunk yields 8 frames, so the cost per output frame is a few GFLOPs. Memory attention: 2688 × 5376 scores per head, which is negligible.

**Why it might work.** Persistence of attended objects beyond 1 s; principled reset at cuts; streaming efficiency.

**Key risks.** Exposure bias, because memory is written from the model's own predictions (mitigate with teacher forcing annealed to 0). Standard benchmarks contain few within-clip cuts (Hollywood-2 is shot-split). Novelty is diluted by STM-SalNet, SG-FCN, Wulff et al. and Gorji & Clark (§4).

### C2. SURGE: SURprise-Gated Encoder–decoder (selected; full specification in §5–6)

**Idea.** Use *both halves* of a JEPA world model. The frozen V-JEPA 2 encoder supplies multi-level features. Its frozen **pretrained predictor** forecasts the latent representation of the last 8 sampled frames from the first 8. The **per-token prediction error** is a semantic surprise map. Its spatially normalized form modulates the decoder features (*where*). Its horizon-normalized global magnitude gates a learned center prior (*when*). Parameters: about 7M trainable, about 325M frozen. [Estimate] Cost: about 2.7 TFLOPs per 32-frame chunk producing 16 output frames.

### C3. SYNC: Synchrony-adaptive density decoding

**Idea.** Predict, per frame, how strongly observers agree (dispersion of the fixation distribution). Use that prediction to set the output temperature and the mixture weight between a focused content map and a diffuse prior.

```
clip (32 frames, 224×384) → Video Swin-S → TMFI-style hierarchical decoder → content logits L_t (56×96)
                                 │
                            F4 → spatial GAP → temporal conv → MLP → d̂_t   (predicted dispersion)
                                                                    │
                       τ_t = τ_min + softplus(a·d̂_t + b) ,  π_t = σ(c·d̂_t + e)
                                                                    │
                   S_t = (1 − π_t)·softmax(L_t / τ_t) + π_t·G_θ
```

| Stage | Input → output | Purpose |
|---|---|---|
| Encoder + decoder | 32×3×224×384 → L_t: 56×96 | Content saliency (TMFI-like) |
| Synchrony head | F4 (768×16×7×12) → d̂_t (scalar per frame) | Predict attentional synchrony; target $d_t = H(Y_t)/\log\lvert\Omega\rvert$ (normalized entropy of the ground-truth density) or observer-dispersion statistics |
| Density shaping | (L_t, d̂_t) → S_t | Calibrated spread for KL/SIM/CC |

**Loss.** $\mathcal L=\mathcal L_{sal}+\lambda_d\,|\hat d_t-d_t|$.

**Complexity.** Base model (about 50M with Swin-S) plus about 0.2M. **Why it might work:** KL, SIM and CC penalize spread mismatch, and synchrony varies strongly over time (Dorr et al. 2010; Mital et al. 2011; Smith & Mital 2013). **Risks:** small gains; the entropy target depends on observer count and the blur used to build ground-truth maps; possible gaming of distribution metrics without better localization.

---

## 4. Novelty analysis and selection

### 4.1 Per-candidate novelty

| Candidate | Closest prior work found | What remains new | Status |
|---|---|---|---|
| **C1 ECHO** | STM-SalNet (memory bank with forgetting, DHF1K/H2/UCF); SG-FCN and Rudoy et al. (conditioning on the previous saliency map); Wulff et al. 2025 (past-saliency stream, long context helps; GT priors, VSOD data); SAM 2 / STCN (mask-conditioned memory, occlusion embedding); Gorji & Clark 2018 (scene-change pathway); time-varying center bias (Zhao & Koch 2011; Coutrot et al. 2017) | Reusing one familiarity field for novelty (local) *and* continuity (global), where continuity gates memory injection and the center-prior exponent | Memory and history: **substantial precedent**. Familiarity-gated integration: **potentially novel but incremental** |
| **C2 SURGE** | PredJSal (V-JEPA2 encoder only; predictive-coding motivation); Itti & Baldi (low-level Bayesian surprise); Garrido et al. 2025 (V-JEPA surprise for physics); DINO-Explorer 2026 (DINOv3 latent surprise for robot event triage, ego-motion compensated); Cambrian-S (surprise for memory and event segmentation in video MLLMs); Gorji & Clark (scene-change push) | A pretrained video world model's **predictor error**, computed leak-free under causal masking, as a spatially resolved cue that **modulates** a fixation-prediction decoder, with global error **gating** the center prior | Surprise cue: **potentially novel (no direct precedent found)**. Prior gate: **partial precedent** |
| **C3 SYNC** | IOC prediction (Le Meur 2011; Yue et al. 2021; Bruckert & Christie 2023); noise-aware VSP (Prashnani et al. 2021); saliency calibration (arXiv 2105.12441); mixture-with-center-bias density models (DeepGaze line; 2026 empirical-density mixtures) | Coupling predicted synchrony to the temperature and mixture weight inside a VSP model | **Partial precedent; incremental** |

### 4.2 Decision matrix

Scores are my judgments on a 1–5 scale, shown so the reasoning is auditable.

| Criterion (weight) | C1 ECHO | **C2 SURGE** | C3 SYNC |
|---|---|---|---|
| Novelty of the interaction (0.30) | 2.5 | **4.0** | 2.0 |
| Expected performance potential (0.25) | 3.0 | **3.5** | 2.5 |
| Feasibility on Colab/Kaggle (0.20) | 4.5 | 3.0 | **5.0** |
| Testability and falsifiability (0.15) | 3.5 | **4.5** | 3.5 |
| Simplicity and coherence (0.10) | 3.0 | 4.0 | **5.0** |
| **Weighted total** | 3.23 | **3.75** | 3.25 |

**Selection: C2 (SURGE).** It has the clearest, best-evidenced gap: a top 2026 entry was explicitly motivated by predictive coding yet never used prediction error. It adds one principled interaction rather than new modules, because encoder and predictor are two halves of one pretrained system. It also admits clean falsification controls (§9). The best idea from C1, resetting toward the center after cuts, survives in SURGE as a near-free global-surprise gate. C1 and C3 remain compatible follow-ups.

---

## 5. Selected architecture: SURGE, complete pipeline

**Defaults** (Colab/Kaggle configuration):
- Clip: T = 16 sampled frames at stride s = 2, spanning 32 original frames (about 1.07 s at 30 fps, the same span as TMFI's 32-frame window).
- Input resolution: 256×256. Frames are squashed from 16:9 to a square, as SalFoM does at 224×224, because the HuggingFace RoPE implementation assumes a square token grid (§10.4).
- Context = first 8 sampled frames (tubelets 1–4). Targets = last 8 sampled frames (tubelets 5–8).
- Each clip outputs the 16 original frames of the target span. Consecutive clips overlap by 16 original frames (hop = 16).

| # | Stage | Input → output (B = batch) | Trainable? | Purpose |
|---|---|---|---|---|
| 1 | **Full-clip encoding** | X: B×16×3×256×256 → tokens B×2048×1024 (8 tubelets × 16×16), taps at layers 6/12/18/24 | Frozen | Multi-level "what" features; latent targets z for the last 4 tubelets |
| 2 | **Context-only encoding** | X[:, :8] → c: B×1024×1024 | Frozen | Encoder sees **only the past**, so there is no leakage |
| 3 | **Latent prediction** | (c, positions of the 1024 target tokens) → ẑ: B×1024×1024 | Frozen | World-model expectation for the next 4 tubelets |
| 4 | **Surprise** | ẑ, z_target → s: B×4×16×16; relative r: B×4×16×16; global g: B×4 | — | Per-token unpredictability ("where"); per-tubelet event magnitude ("when") |
| 5 | **SimpleFPN-3D** | taps → P4: B×256×8×64×64, P8: B×256×8×32×32, P16a/P16b: B×256×8×16×16 | Yes | Multi-scale pyramid from a plain ViT (ViTDet-style) |
| 6 | **Surprise encoder Φ** | [r zero-padded on context tubelets; availability mask]: B×2×8×16×16 → U: B×64×8×16×16 (resized per level) | Yes | Temporal receptive field of 5 tubelets (about 0.67 s), covering saccade latency (about 0.2 s) |
| 7 | **SCM** (surprise-conditioned modulation) | F_ℓ, U_ℓ → F̃_ℓ (same shapes) | Yes (zero-init) | Surprise acts as a **gain** on features. At initialization the model equals the encoder-only baseline |
| 8 | **Top-down decoder** | F̃ → D16 → D8 → D4 with (2+1)D residual blocks → logits B×1×8×64×64 | Yes | Multi-level fusion and temporal integration |
| 9 | **Temporal upsampling + selection** | 8 tubelets → 16 sampled frames → keep last 8 → interpolate to the 16 original frames | — | Multi-frame output (MIMO) amortizes the encoder |
| 10 | **Gated center prior** | logits + π_t·log G_θ, where π_t = f(g, onset flag) | Yes (about 45 params) | Time-varying center bias: stronger after cuts and onsets |
| 11 | **Output** | softmax over pixels → B×16×64×64 → bilinear upsample to native resolution (+ Gaussian blur, σ tuned) | — | Probability map per frame |

**Video start.** For the first clip, the context is the first frame replicated. A binary onset flag ρ is given to the gate. This also represents the known initial central fixation at stimulus onset.

**Strictly causal (streaming) mode.** Set the context to T−2 frames, predict only the final tubelet, and use hop = 2 sampled frames. This costs more per frame and is reported separately.

---

## 6. Architectural diagram and mathematical formulation

### 6.1 Diagram

```
 Input clip X: 16 sampled frames (stride s), 256×256
      │
      ├──────────── full clip (frames 1–16) ──────────────┐
      │                                                    │
      │                                     context prefix only (frames 1–8)
      ▼                                                    ▼
┌──────────────────────────────────────┐   ┌──────────────────────────────────┐
│ V-JEPA 2 encoder E (ViT-L/16)        │   │ V-JEPA 2 encoder E (same, frozen)│
│ frozen · 2048 tokens (8 × 16 × 16)   │   │ 1024 tokens (4 × 16 × 16)        │
└──┬────────┬────────┬────────┬────────┘   └────────────────┬─────────────────┘
   │h6      │h12     │h18     │h24 = z                      │ c
   │        │        │        │                             ▼
   │        │        │        │             ┌──────────────────────────────────┐
   │        │        │        │             │ V-JEPA 2 predictor P (frozen)    │
   │        │        │        │             │ predicts tubelets 5–8 → ẑ        │
   │        │        │        │             └────────────────┬─────────────────┘
   │        │        │        └───── z[tubelets 5–8] ────────► s = mean_D |LN(ẑ) − LN(z)|
   │        │        │                                       │                    │
   │        │        │                          relative r ("where")     global g ("when")
   │        │        │                                       │                    │
   ▼        ▼        ▼        ▼                              ▼                    │
 SimpleFPN-3D: P4 (1/4) · P8 (1/8) · P16a, P16b (1/16)   Surprise encoder Φ       │
   │                                                     (3×3×3 convs, 2→32→64)   │
   │                                                         │ U                  │
   └──► SCM_ℓ : F̃ = F ⊙ (1 + γ_ℓ(U)) + β_ℓ(U)  ◄─────────────┘ (γ, β zero-init)  │
                 │                                                                │
   top-down (2+1)D residual decoder: 1/16 → 1/8 → 1/4                             │
                 │                                                                │
   logits L: 8 tubelets × 64×64 → temporal ×2 → last 8 frames                     │
                 │                                                                │
   L_final = L + π_t · log G_θ ,  π_t = softplus(β0) + softplus(β1)·σ(a·g + b + e·ρ) ◄┘
                 │
   S_t = softmax_xy(L_final) → upsample to native resolution (+ blur) → 16 original frames per chunk
```

### 6.2 Formulation

**Notation.** Clip $X\in\mathbb R^{T\times3\times H\times W}$ with $T=16$, $H=W=256$. Tubelets $\tau\in\{1,\dots,8\}$, grid $16\times16$, token $n=(\tau,i,j)$, $D=1024$. Context tubelets $\mathcal C=\{\tau\le4\}$; target tubelets $\mathcal T=\{\tau>4\}$.

**(1) Encoding.** $\{h^{(\ell)}\}_{\ell\in\{6,12,18,24\}}=E(X)$, and $z=\mathrm{LN}_0(h^{(24)})$, where $\mathrm{LN}_0$ is parameter-free layer normalization.

**(2) Leak-free context.** $c=E(X_{1:8})$. The encoder never attends to target frames.

**(3) Prediction.** $\hat z=\mathrm{LN}_0\big(P(c;\ \mathrm{pos}(\mathcal T))\big)$.

**(4) Latent surprise** (L1, matching the JEPA training loss):
$$s_n=\frac1D\big\lVert \hat z_n-z_n\big\rVert_1,\qquad n\in\mathcal T.$$

**(5) Relative ("where") and global ("when") surprise.** Let $\mu_\tau,\sigma_\tau$ be the spatial mean and standard deviation of $s$ at tubelet $\tau$, and $k=\tau-4$ the prediction horizon:
$$r_{\tau ij}=\frac{s_{\tau ij}-\mu_\tau}{\sigma_\tau+\epsilon},\qquad g_\tau=\frac{\mu_\tau-m_k}{d_k+\epsilon},$$
where $m_k$ and $d_k$ are the mean and standard deviation of $\mu$ for horizon $k$ over training clips. This removes the trivial growth of error with horizon.

**(6) Surprise embedding.** $U=\Phi([\tilde r;\ a])$, with $\tilde r$ equal to $r$ zero-padded on $\mathcal C$, $a$ an availability mask, and $\Phi$ two $3{\times}3{\times}3$ conv + GELU layers (2 → 32 → 64 channels). $U_\ell$ is $U$ resized to pyramid level $\ell$.

**(7) Surprise-conditioned modulation (SCM).**
$$\tilde F_\ell=F_\ell\odot\big(1+\gamma_\ell(U_\ell)\big)+\beta_\ell(U_\ell),\qquad \gamma_\ell,\beta_\ell:\ 1{\times}1{\times}1\ \text{conv},\ \text{zero-initialized}.$$
Multiplicative gain lets surprise *amplify or suppress* feature evidence, rather than paste a fixed map. Zero initialization makes SURGE start exactly at the encoder-only baseline, so any gain is attributable to the surprise path.

**(8) Decoder.** $D_{16}=\mathcal R(\mathrm{Conv}[\tilde F_{16a};\tilde F_{16b}])$, $D_8=\mathcal R(\tilde F_8+\uparrow D_{16})$, $D_4=\mathcal R(\tilde F_4+\uparrow D_8)$, $L=\mathrm{Head}(D_4)$, where $\mathcal R$ is a (2+1)D residual block.

**(9) Gated center prior** (product of experts):
$$G_\theta(p)=\sum_{k=1}^{8}\mathrm{softmax}(w)_k\,\mathcal N\big(p;\mu_k,\mathrm{diag}(\sigma_k^2)\big)\ \text{(normalized over pixels)},$$
$$\pi_t=\mathrm{softplus}(\beta_0)+\mathrm{softplus}(\beta_1)\,\sigma(a\,g_{\tau(t)}+b+e\,\rho),$$
$$S_t(p)=\frac{\exp\big(L_t(p)+\pi_t\log G_\theta(p)\big)}{\sum_q\exp\big(L_t(q)+\pi_t\log G_\theta(q)\big)}.$$
Here $\pi_t$ is an interpretable time course of center-bias strength.

**(10) Loss** (over output frames $\mathcal O$; $Y_t$ = ground-truth density, $F_t$ = binary fixation map):
$$\mathcal L=\frac1{|\mathcal O|}\sum_{t\in\mathcal O}\Big[\mathrm{KL}(Y_t\Vert S_t)+\lambda_{cc}\big(1-\mathrm{CC}(Y_t,S_t)\big)-\lambda_{nss}\,\mathrm{NSS}(S_t,F_t)\Big]+\lambda_{aux}\mathcal L_{aux},$$
with $\mathrm{KL}(Y\Vert S)=\sum_pY(p)\log\!\big(\epsilon+\tfrac{Y(p)}{\epsilon+S(p)}\big)$. Defaults: $\lambda_{cc}=1$, $\lambda_{nss}=0.1$, $\lambda_{aux}=0.25$ (deep supervision at $D_{16}$ and $D_8$), all tuned on an internal validation split.

**(11) Zero-shot surprise saliency** (an analysis baseline, not part of the model): $S^{zs}_t=\mathrm{Norm}\big(\mathrm{Blur}_\sigma(\uparrow\mathrm{ReLU}(r_{\tau(t)}))\big)$.

### 6.3 Complexity and parameters

[Estimate] Method: dense layers ≈ 2·N·P FLOPs; attention ≈ 4·N²·D per layer. ViT-L: P ≈ 303M, L = 24, D = 1024.

| Component | Tokens N | FLOPs per clip | Params |
|---|---|---|---|
| Full-clip encoder | 2048 | ≈ 1.24 T (dense) + 0.41 T (attention) ≈ **1.65 T** | 303M (frozen) |
| Context encoder | 1024 | ≈ 0.62 T + 0.10 T ≈ **0.72 T** | shared |
| Predictor (384-d, 12 layers) | 2048 | ≈ **0.17 T** | ≈ 22M (frozen) |
| FPN + decoder + SCM + Φ | — | ≈ **0.1–0.2 T** | ≈ **7M trainable** |
| **Total** | | **≈ 2.7 TFLOPs per clip → 16 original frames** | |

Per output frame this is ≈ 0.17 TFLOPs (≈ 0.34 per sampled frame). A one-frame-per-clip ViT-L design on 16×224×224 input (the SalFoM setting) needs ≈ 1.2 TFLOPs per output frame for the encoder alone, so SURGE should be **about 3.5–7× cheaper per output frame** despite two encoder passes. All models must be re-measured with the same tool (§9.7).

---

## 7. Comparison with the closest existing methods

| Method | Encoder | Uses predictor error? | Explicit surprise cue | Event/cut-aware prior | Output frames per pass | Fixation benchmarks |
|---|---|---|---|---|---|---|
| **PredJSal** (NTIRE 2026, 2nd) | V-JEPA2-g | **No** (encoder only; predictive coding is the *motivation*) | No | No | 1 | NTIRE (mouse tracking) |
| **SalFoM** (2024) | UMT-L, fine-tuned | n/a | No | No | 1 | DHF1K, H2, UCF |
| **TMFI-Net** (2023) | Video Swin | n/a | No | No | 1 | DHF1K, H2, UCF |
| **ViNet-A** (2025) | SlowFast (AVA) | n/a | No (action cues) | No | 1 | DHF1K, H2, UCF, AV sets |
| **Itti & Baldi** (2005–2009) | Low-level feature channels | Bayesian surprise (prior → posterior KL) | **Yes (low-level, not learned)** | No | per frame | Own video gaze data |
| **Gorji & Clark** (CVPR 2018) | Static saliency + ConvLSTM | No | No | **Yes (scene-change pathway)** | per frame | Video gaze datasets |
| **DINO-Explorer** (2026) | DINOv3 + recurrent predictor | **Yes (latent)** | Yes | Ego-motion compensation | — | **None** (robot event triage) |
| **Wulff et al.** (2025) | ViViT | No | No | No | 1 | None (DAVSOD; GT past maps) |
| **STM-SalNet** (2025) | ViT + memory | No | No | No | — | DHF1K, H2, UCF |
| **SURGE (proposed)** | V-JEPA 2-L, frozen | **Yes (pretrained, leak-free)** | **Yes (semantic, spatial)** | **Yes (global surprise gate)** | **16** | DHF1K, H2, UCF, DIEM |

### 7.1 The strongest argument that SURGE is not novel, and how the experiments answer it

> *"Predictor error is just a temporal-change or motion channel. Optical flow has been injected into VSP since two-stream networks (Bak et al.; STRA-Net; OM-CNN), every 3D encoder already sees motion, and Itti & Baldi did surprise-based saliency twenty years ago. PredJSal's V-JEPA2 features already encode predictability. SURGE re-parameterizes known cues."*

This is a serious objection, and the plan treats it as the main threat:

1. **Motion ≠ surprise, in principle.** A smoothly pursued object with constant velocity is predictable, so it gives low surprise. An abrupt onset or an unexpected change of motion gives high surprise. The stratified analysis in §9.5 (frames with high flow but low surprise, and the reverse) tests whether SURGE exploits this difference.
2. **Matched controls.** Feature differencing, optical-flow magnitude, pixel-space prediction error, a randomly initialized predictor, and a from-scratch trainable predictor are each injected through the *same* SCM path (A1–A2).
3. **Redundancy test.** SURGE + flow vs. encoder-only + flow (A2e).
4. **What would concede the point.** If the controls match SURGE, I would drop the claim that learned expectations matter and reframe the contribution as an empirical finding that "world-model surprise ≈ motion saliency for human gaze in these benchmarks".

---

## 8. Expected advantages, weaknesses, and performance targets

### 8.1 Expected advantages

- **[Expected] Orthogonal cue.** Expectation violation is not directly encoded by a single feature map. The predictor carries priors learned from more than 1M hours of video, which 600 saliency-labeled videos cannot teach.
- **[Expected] Less overfitting.** Only about 7M trainable parameters, which addresses L5 (SalFoM's weaker cross-dataset results).
- **[Expected] Efficiency.** Multi-frame output on a frozen encoder: about 3.5–7× fewer FLOPs per output frame than a one-frame-per-clip ViT-L design (§6.3).
- **Interpretability.** Surprise maps and the center-prior time course $\pi_t$ are directly inspectable.
- **Plug-in recipe.** Applies to any JEPA-style model with a released predictor.

### 8.2 Potential weaknesses

1. **Redundancy.** V-JEPA features are trained for predictability, so explicit error may add little (the B1 vs. SURGE test).
2. **Predictor mismatch.** V-JEPA pretraining uses multi-block masks, not future-frame masks, so causal prediction is out of distribution. Garrido et al. show future-prediction surprise still works for physics. Mitigation: optional Stage-0 self-supervised causal adaptation of the predictor on unlabeled training videos (A5).
3. **Confounds.** Camera motion reveals unpredictable content at frame borders, and texture, appearance and prediction difficulty also inflate error (Hristov et al., NeurIPS 2026 workshop). Mitigations: spatial z-scoring, the center prior, and an ego-motion-compensated ablation in the style of DINO-Explorer.
4. **Coarse grid.** Surprise has 16×16 tokens per frame; each token covers about 40×22 pixels of the original 640×360 frame. This is adequate for blurred ground-truth maps but limits small targets.
5. **Square resize distorts aspect ratio.** SalFoM uses the same practice; a patched RoPE allows rectangular grids (§10.4).
6. **A frozen ViT-L may trail SOTA.** LoRA-tuned or billion-scale encoders lead NTIRE 2026. Reaching the stretch target may need Stage-2 LoRA.
7. **Pretraining overlap.** V-JEPA 2 data includes YouTube-derived corpora, and DHF1K videos come from YouTube. This is shared by all foundation-model baselines but should be disclosed.
8. **The prior gate may be weak on standard benchmarks**, which contain few within-clip cuts. Test it on DIEM and unsplit material.

### 8.3 Published reference results (targets, not predictions)

**DHF1K test** (official leaderboard):

| Model | AUC-J | SIM | sAUC | CC | NSS |
|---|---|---|---|---|---|
| SalFoM | **0.922** | 0.421 | **0.735** | **0.569** | **3.354** |
| THTD-Net | 0.915 | 0.406 | 0.730 | 0.548 | 3.139 |
| TMFI-Net | 0.915 | 0.407 | 0.731 | 0.546 | 3.146 |
| STSANet | 0.912 | 0.383 | 0.723 | 0.529 | 3.010 |
| ViNet | 0.908 | 0.381 | 0.729 | 0.511 | 2.872 |
| UNISAL | 0.901 | 0.390 | 0.691 | 0.490 | 2.776 |
| SalEMA | 0.890 | **0.466** | 0.667 | 0.449 | 2.574 |

**DHF1K validation** (as reported in the ViNet-S/A paper):

| Model | CC | NSS | AUC-J | SIM |
|---|---|---|---|---|
| TMFI-Net | **0.554** | **3.201** | **0.924** | **0.428** |
| THTD-Net | 0.553 | 3.188 | 0.924 | 0.425 |
| ViNet-E | 0.549 | 3.134 | 0.922 | 0.409 |
| STSANet | 0.539 | 3.082 | 0.920 | 0.411 |
| DiffSal (video only) | 0.533 | 3.066 | 0.918 | 0.405 |
| ViNet-S | 0.529 | 3.008 | 0.919 | 0.399 |
| ViNet-A | 0.525 | 3.019 | 0.916 | 0.399 |

**Hollywood-2 and UCF-Sports test** (CC / NSS / AUC-J / SIM; from the ViNet-S/A and SalFoM papers):

| Model | Hollywood-2 | UCF-Sports |
|---|---|---|
| ViNet-E | **0.766** / **4.168** / 0.947 / 0.609 | **0.744** / **4.156** / **0.941** / **0.587** |
| DiffSal | 0.765 / 3.955 / **0.951** / **0.610** | 0.685 / 3.483 / 0.928 / 0.543 |
| ViNet-A | 0.756 / 4.119 / 0.945 / 0.604 | 0.734 / 4.108 / 0.940 / 0.586 |
| TMFI-Net | 0.739 / 4.095 / 0.940 / 0.607 | 0.707 / 3.863 / 0.936 / 0.565 |
| THTD-Net | 0.726 / 3.965 / 0.939 / 0.585 | 0.711 / 3.840 / 0.933 / 0.565 |
| SalFoM | 0.709 / 3.902 / 0.935 / 0.583 | 0.631 / 3.543 / 0.928 / 0.516 |

*Data caveat:* STSANet's Hollywood-2 and UCF numbers appear swapped between the leaderboard and the ViNet-S/A paper (0.721 vs. 0.705 CC). This is one more reason to re-evaluate baselines with a single evaluation code base.

**Targets** (not guarantees):

| Level | DHF1K val | Hollywood-2 | UCF-Sports |
|---|---|---|---|
| Competitive | CC ≥ 0.554, NSS ≥ 3.20 (≥ TMFI) | CC ≥ 0.74 | CC ≥ 0.71 |
| Stretch | Approach SalFoM's test CC 0.569 (submit to the test server if available) | CC ≥ 0.76 | CC ≥ 0.73 |
| **Scientific** (primary) | SURGE − B1 > seed noise, and SURGE − B1 > every control (§9) | same | same |

KL is not reported on these leaderboards. Report it with the standard definition so future work can compare.

### 8.4 Which mechanism should move which metric, and the trade-offs

| Mechanism | Expected effect | Metrics most likely helped | Likely cost |
|---|---|---|---|
| Frozen V-JEPA 2 features + multi-level decoder | Overall localization | All | Underfits relative to LoRA tuning |
| Surprise SCM (relative surprise) | Up-weights unexpected or new events | NSS, AUC-J (fixation-location metrics), CC on event frames; sAUC when events are off-center | Border false positives under camera motion, which hurt SIM and KL |
| Global-surprise-gated center prior | Stronger center bias after cuts and onsets | CC, KL, NSS at onsets | **sAUC decreases** whenever the prior strengthens, because sAUC gives no credit for center bias |
| MIMO decoding + temporal interpolation | Temporal smoothness | SIM, KL | Slight NSS loss at abrupt changes |
| Output blur / temperature (tuned) | Spread calibration | SIM, KL, CC | NSS and AUC-J largely insensitive |

These metrics do not move together. SalEMA has the highest SIM with a low CC, and sharper maps raise NSS while lowering SIM and KL. Following Kümmerer et al. (ECCV 2018), report **one map for all metrics** in the main comparison (the fair protocol), and give metric-specific maps (for example, dividing by the center prior for sAUC) only as a clearly separated supplement.

---

## 9. Experimental design and ablation strategy

### 9.1 Hypotheses and falsification criteria

| ID | Hypothesis | Supported if… | Falsified if… |
|---|---|---|---|
| **H1** (core) | Pretrained-predictor surprise adds fixation-relevant information beyond encoder features | SURGE > B1 on DHF1K-val and Hollywood-2 (CC and NSS; paired test over videos, p < 0.05, consistent over 3 seeds), **and** SURGE > every control in A1–A2 | SURGE ≈ B1, **or** any control (random or trainable-from-scratch predictor, feature difference, flow) reaches SURGE's gain, **or** shuffled surprise (A7) matches SURGE |
| **H2** (specificity) | Gains concentrate where human attention shifts | ΔNSS(SURGE − B1) is larger in the top quintile of human attention-shift frames than in the bottom quintile. Shift is defined from ground truth as $\mathrm{KL}(Y_t\Vert Y_{t-8})$, independent of the model, so the test is not circular | Gains are uniform or concentrated in low-shift frames |
| **H3** (gate) | Global surprise can stand in for scene-continuity information | On DIEM and unsplit Hollywood-2 (cuts detected by TransNetV2), the gated prior improves CC and KL in the 0–2.5 s after a cut over a constant prior, and π_t peaks after cuts *without cut supervision* | No time-since-cut dependence, or π_t is unrelated to cuts |
| **H4** (zero-shot) | Surprise alone predicts fixations | Surprise-only map has positive information gain over the center prior and beats flow magnitude and feature difference on NSS | Surprise-only ≤ flow magnitude |

Pre-register these thresholds (for example in an OSF note or a dated repository commit) before running the main comparisons.

### 9.2 Data and protocol

- **DHF1K.** Train on 540 of the 600 training videos, use a fixed, released list of **60 internal validation videos** for model selection, early stopping and blur tuning, and **report on the official 100-video val set as the test set**. Submit to the official test set if the curators respond.
- **Hollywood-2.** Standard shot-split protocol (823/884 videos); 10% of training as internal val. Short shots: pad context by replicating the first frame and discard padded outputs.
- **UCF-Sports.** 103/47; 10% of training as internal val.
- **Cross-dataset (zero-shot).** DHF1K → Hollywood-2, UCF, **DIEM** (cut-rich; use the DIEM split adopted in audio-visual saliency work for comparability), and optionally LEDOV.
- **Fine-tuning protocol.** Initialize from the DHF1K model and fine-tune separately on Hollywood-2 and UCF, a common protocol in this literature. Baselines differ in the details, so check each baseline's exact protocol and match it.
- **Optional scale study.** Pretrain on NTIRE 2026 and AViMoS (mouse tracking), fine-tune on DHF1K. Report separately because of the modality shift.

### 9.3 Evaluation

- Use the official DHF1K MATLAB metric code, or a Python port validated to within ±0.001 against it on a subset. Validate the pipeline by re-scoring a released baseline (for example TMFI-Net, if its checkpoint or predicted maps are available) and matching its published numbers.
- Metrics: AUC-J, sAUC (same negative-sampling protocol as the leaderboard), NSS, CC, SIM, KL; also information gain over the center prior.
- Evaluate at native resolution (DHF1K 360×640).
- Statistics: 3 seeds for B1 and SURGE; mean ± std; paired Wilcoxon signed-rank over videos; bootstrap 95% CIs; Holm correction across ablations.

### 9.4 Baselines

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Learned center prior only | Floor |
| **B1** | **Same frozen V-JEPA 2 + identical decoder and prior, no surprise** (PredJSal-style) | The key comparison for H1 |
| B1+ | B1 + LoRA (Stage 2) | Isolates the effect of backbone adaptation |
| B2 | TMFI-Net, THTD-Net, ViNet-S (re-evaluated from released checkpoints where available); otherwise, and for SalFoM, DiffSal and ViNet-A, published numbers with the protocol noted | SOTA context |
| B3 | Itti–Baldi-style low-level surprise; SURGE zero-shot surprise | Classic and zero-shot reference |

### 9.5 Ablations and novel-interaction tests

| ID | Variant | Question answered |
|---|---|---|
| A1a/b/c | No surprise (B1) / **randomly initialized frozen predictor** / **predictor trained from scratch end-to-end** (same architecture) | Do *pretrained expectations* matter, or only extra capacity? |
| A2a/b/c/d | Inject through the same SCM: feature difference $\lVert z_\tau-z_{\tau-1}\rVert_1$ / RAFT flow magnitude / pixel-space error (VideoMAE-B reconstruction, causal mask) / Itti–Baldi surprise | Is surprise just motion or change? Does latent beat pixel prediction? |
| A2e | SURGE + flow vs. B1 + flow | Is surprise redundant once motion is explicit? |
| A3 | Fusion: SCM (gain) vs. concatenation vs. attention-logit bias vs. late map fusion | Does *gain modulation* matter? |
| A4 | Relative vs. raw surprise; global gate on/off; constant prior; no prior | Contribution of where vs. when |
| A5 | Context/target split (4/12, 8/8, 12/4 frames); stride s ∈ {1, 2, 4}; Stage-0 causal predictor adaptation on/off | Horizon and temporal-scale sensitivity; mismatch fix |
| A6 | **Leakage sanity check**: surprise computed the default HuggingFace way (context taken from the full-clip encoding) | Verifies the leak-free implementation; leaky surprise should be near-uninformative or behave differently |
| A7 | **Shuffled surprise** (from another video) | Does the decoder actually use spatial surprise? |
| A8 | Ego-motion-compensated surprise (subtract a global-flow-predicted component) | The camera-motion confound |
| A9 | MIMO hop (16 vs. 4 original frames) and strictly causal mode | Accuracy cost of amortization; streaming viability |
| A10 | Surprise from other layers (ẑ vs. intermediate predictor states) | Which representation level carries the signal |

**Stratified analyses** (H2/H3):
- ΔNSS by attention-shift quintile.
- ΔNSS in high-flow/low-surprise vs. low-flow/high-surprise frames.
- Time-since-cut curves of CC, KL and π_t on DIEM.
- Task-driven (Hollywood-2/UCF) vs. free-viewing (DHF1K/DIEM) effect sizes.

### 9.6 Generalization

Train on DHF1K, then test zero-shot on Hollywood-2, UCF, DIEM and LEDOV. Compare the cross-dataset drop of SURGE vs. B1 vs. SalFoM-style fine-tuned encoders. **[Hypothesis]** Freezing reduces the drop.

### 9.7 Computational reporting

Report parameters (total and trainable) and GMACs per **output frame**, measured with fvcore using identical counting for all models. Also report latency (ms per output frame at batch 1) and throughput (frames/s at batch 8) on a T4, plus an A100 if available, and peak memory.

---

## 10. Implementation roadmap and computational requirements

### 10.1 Components

| Module | Implementation |
|---|---|
| Backbone | `facebook/vjepa2-vitl-fpc64-256` via HuggingFace `transformers` (`VJEPA2Model`: encoder + predictor; predictor is 384-d, 12 layers, 12 heads). Check the model license before redistributing weights or derived code. |
| Feature taps | `output_hidden_states=True`; layers 6/12/18/24 (verify whether index 0 is the embedding output) |
| Surprise extractor | Leak-free two-pass procedure (§10.4); cache **surprise maps** to disk (tiny: 4×16×16 floats per clip) so later runs need only the full pass |
| SimpleFPN-3D | 1×1×1 projections to 256 channels + ConvTranspose3d (1,2,2) ×2 (layer 6) and ×1 (layer 12) |
| Decoder | (2+1)D residual blocks (spatial 1×3×3 + temporal 3×1×1), GELU, GroupNorm |
| Prior | 8 learnable diagonal Gaussians in normalized coordinates; gate scalars (β0, β1, a, b, e) |

### 10.2 Training configuration

| Setting | Default | Rationale |
|---|---|---|
| Input | 16 frames, stride s = 2 (choose s ∈ {1, 2, 4} by zero-shot surprise NSS on the *internal* val split), 256×256 | Matches the 32-frame span of the SOTA; square grid |
| Stage 0 (optional) | Fine-tune predictor only, causal masks, L1, lr 1e-5, 1–2 epochs on unlabeled DHF1K **training** videos | Fixes the mask mismatch; no labels used |
| Stage 1 (main) | Encoder and predictor frozen; train FPN + decoder + SCM + prior | Clean attribution; cheap |
| Optimizer | AdamW, lr 3e-4 (decoder), 1e-3 (prior and gate scalars), weight decay 0.01, cosine schedule, 1-epoch warm-up | Standard |
| Schedule | 30 epochs × 4,800 clips (8 random clips per training video per epoch) | Diversity without full enumeration |
| Batch | 8 clips (gradient accumulation if needed) | Fits 16 GB |
| Augmentation | Horizontal flip (flip ground truth too), mild color jitter; **no spatial crops** | Crops corrupt center-bias statistics |
| Precision | fp16 autocast + GradScaler on T4 (no bf16 support); bf16 on L4/A100 | Hardware |
| Stage 2 (optional) | LoRA r = 16 on qkv/proj of the last 12 blocks, used *only* for the feature pass. The surprise pass uses adapter-disabled weights so the predictor sees its native features. lr 1e-4 (LoRA), 1e-5 (decoder); batch 2 with accumulation 4; gradient checkpointing | NTIRE evidence that adaptation matters (ViSAGE) |
| Inference | Chunked (hop 16 original frames), temporal interpolation to every frame, bilinear upsample, Gaussian blur σ tuned on internal val; flip test-time augmentation reported separately | Fair reporting |

### 10.3 Compute budget

[Estimate] Assumes 15–25 effective TFLOPS fp16 on a T4. Measure before committing.

| Job | Approximate cost |
|---|---|
| Frozen forward (encoder ×1.5 + predictor + decoder) | ≈ 2.7 TFLOPs per clip → ≈ 0.15–0.3 s per clip on a T4 |
| One Stage-1 epoch (4,800 clips) | ≈ 15–30 min |
| One Stage-1 run (30 epochs) | **≈ 8–15 GPU-h** |
| DHF1K-val inference (about 58k frames ≈ 3.6k clips) | ≈ 15–20 min |
| Stage-2 LoRA run (15 epochs) | ≈ 20–30 GPU-h (optional) |
| Whole program (B1 and SURGE × 3 seeds, about 12 ablations, H2/UCF fine-tuning, zero-shot tests) | **≈ 120–180 GPU-h**: roughly 4–6 weeks at about 30 GPU-hours per week. It can be faster if a 2×T4 session runs two configurations in parallel (check how your quota counts multi-GPU sessions) or with Colab Pro (L4/A100) |
| Disk | DHF1K frames at 256×256 JPEG ≈ 6 GB (train + val); surprise cache under 1 GB |

GPU memory: the frozen ViT-L in fp16 is about 0.6 GB, and decoder activations at batch 8 are about 3 GB, so Stage 1 fits easily on 16 GB.

### 10.4 Implementation pitfalls in the HuggingFace V-JEPA 2 code (verified by reading the source)

1. **Target leakage.** `VJEPA2Model.forward` encodes the *full* clip and only then selects context tokens with `context_mask`. Context tokens have therefore already attended to the target frames, which makes surprise meaningless. **Encode the context prefix separately** and pass those features to `model.predictor`.
2. **Square-grid RoPE.** Position IDs use `grid_size = crop_size // patch_size` for both height and width. Non-square inputs get wrong positions unless `_get_frame_pos` and `_get_height_pos` are patched to use (h, w). Hence the 256×256 default.
3. **Normalization.** The encoder's final LayerNorm is affine. Compare predictor output and targets under parameter-free LayerNorm in fp32.

Untested sketch, written against the HF source as read on 2026-11-04. Verify `hidden_states` indexing and output shapes on a GPU before relying on it:

```python
import torch, torch.nn.functional as F
from transformers import AutoModel

vj = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16).cuda().eval()
for p in vj.parameters():
    p.requires_grad_(False)
TAPS = (6, 12, 18, 24)

@torch.no_grad()
def encode_and_surprise(x, n_ctx_frames=8, tubelet=2):
    """x: (B, 16, 3, 256, 256), normalized; square input (HF RoPE assumes a square grid)."""
    B, T = x.shape[:2]
    full = vj(pixel_values_videos=x, skip_predictor=True, output_hidden_states=True)
    z = full.last_hidden_state                          # (B, 2048, 1024)
    taps = [full.hidden_states[l] for l in TAPS]        # verify: is hidden_states[0] the embedding output?
    N = z.shape[1]
    per_tub = N // (T // tubelet)                       # 256 tokens per tubelet
    n_ctx = (n_ctx_frames // tubelet) * per_tub         # 1024 context tokens
    # Leak-free context: the encoder never sees the target frames.
    c = vj(pixel_values_videos=x[:, :n_ctx_frames], skip_predictor=True).last_hidden_state
    ctx_ids = torch.arange(n_ctx, device=x.device).unsqueeze(0).repeat(B, 1)
    tgt_ids = torch.arange(n_ctx, N, device=x.device).unsqueeze(0).repeat(B, 1)
    z_hat = vj.predictor(encoder_hidden_states=c, context_mask=[ctx_ids],
                         target_mask=[tgt_ids]).last_hidden_state   # (B, 1024, 1024)
    ln = lambda t: F.layer_norm(t.float(), t.shape[-1:])
    s = (ln(z_hat) - ln(z[:, n_ctx:])).abs().mean(-1)               # (B, 1024)
    return taps, s.view(B, (T - n_ctx_frames) // tubelet, 16, 16)   # surprise per target tubelet
```

### 10.5 Timeline (about 10–12 weeks)

| Weeks | Milestone | Exit criterion |
|---|---|---|
| 1 | Data pipeline (DHF1K, Hollywood-2 shots, UCF); evaluation code validated against a released baseline | Published baseline numbers reproduced within ±0.005 CC |
| 2 | Leak-free surprise extractor; A6 leakage check; visualizations; **H4 zero-shot study**; choose stride s | Surprise maps qualitatively sensible; H4 result recorded (either outcome) |
| 3 | B1 baseline (3 seeds) | Stable training; competitive val numbers |
| 4 | SURGE Stage 1 (3 seeds) | **H1 go/no-go decision point** |
| 5–6 | Controls A1–A2 and ablations A3–A5, A7–A8 | Controls complete |
| 7 | Hollywood-2/UCF fine-tuning; DIEM zero-shot; H2/H3 analyses | Stratified results |
| 8 | Efficiency measurements; optional Stage 0 and Stage 2 | Compute table |
| 9–12 | Writing; test-server submission; code and prediction release | Paper draft |

**If H1 fails at week 4:** the controls and the zero-shot study still make a publishable analysis ("Does world-model surprise explain human gaze in video?"). Alternatively, pivot to C1 or C3, which reuse the same data and evaluation code.

### 10.6 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Surprise is redundant (H1 fails) | Medium | High | Pre-registered controls turn a negative result into a finding; C1/C3 fallbacks |
| Predictor is unreliable under causal masks | Medium | Medium | Stage-0 adaptation; A5 horizon sweep |
| Silent leakage or position bugs | Medium (if using HF defaults) | High | §10.4; A6 sanity check; unit test that leaky and leak-free surprise differ |
| Frozen ViT-L below the competitive target | Medium | Medium | Stage-2 LoRA; keep the H1 claim (relative to B1) separate from SOTA claims |
| Compute overrun | Low–medium | Medium | Cache surprise; stride s = 4; fewer epochs; run on 2×T4 in parallel |
| DHF1K test server unavailable | High | Low | Report val with internal-val model selection, as recent papers do |
| Benchmark protocol mismatch with baselines | Medium | Medium | Re-score released checkpoints with one evaluation code base |

---

## Appendix A. Sources

**Benchmarks and SOTA**
- Wang et al., *Revisiting Video Saliency* (ACLNet / DHF1K), CVPR 2018 / TPAMI 2019: https://arxiv.org/abs/1801.07424 · leaderboard: https://mmcheng.net/videosal/ · data: https://github.com/wenguanwang/DHF1K
- SalFoM (Moradi et al., 2024): https://arxiv.org/abs/2404.03097
- ViNet-S/A (Girmaji et al., ICASSP 2025): https://arxiv.org/abs/2502.00397
- DiffSal (Xiong et al., CVPR 2024): https://arxiv.org/abs/2403.01226
- TMFI-Net (Zhou et al., TCSVT 2023), DOI 10.1109/TCSVT.2023.3278410; code: https://github.com/wusonghe/TMFI-Net
- THTD-Net (Moradi et al., VISAPP 2024): https://arxiv.org/abs/2401.07942
- TinyHD (Hu et al., WACV 2023): https://arxiv.org/abs/2301.04619
- UNISAL (Droste et al., ECCV 2020): https://arxiv.org/abs/2003.05477
- SalEMA (Linardos et al., BMVC 2019): https://arxiv.org/abs/1907.01869
- STSANet: https://arxiv.org/abs/2108.10696 · TSFP-Net: https://arxiv.org/abs/2105.04213
- Two-stream spatio-temporal saliency networks (Bak et al.): https://arxiv.org/abs/1607.04730
- ViDaS (Hollywood-2 clip counts): https://arxiv.org/abs/2305.11729
- NTIRE 2026 Video Saliency Prediction challenge report: https://arxiv.org/abs/2604.14816 · ViSAGE: https://arxiv.org/abs/2604.08613
- AIM 2024 Video Saliency Prediction challenge: https://arxiv.org/abs/2409.14827
- DIEM: https://qualinet.github.io/databases/video/dynamic_images_and_eye_movements_diem

**Memory, history and priors**
- SG-FCN (Sun et al., TCYB): https://arxiv.org/abs/1809.07988
- Wulff et al., *Balancing long- and short-term dynamics…* (2025): https://arxiv.org/abs/2504.05913
- STM-SalNet (Xu et al., ICONIP 2025; DOI 10.1007/978-981-95-4097-6_23): https://pure.ecnu.edu.cn/en/publications/stm-salnet-a-biologically-inspired-spatial-temporal-memory-networ/
- SAM 2 (Ravi et al., 2024): https://arxiv.org/abs/2408.00714 · SAM-DAQ (AAAI 2026): https://arxiv.org/abs/2511.09870 · M⁴-SAM (2026): https://arxiv.org/abs/2605.11760
- Gorji & Clark, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/papers/Gorji_Going_From_Image_CVPR_2018_paper.pdf
- Rudoy et al., CVPR 2013: https://openaccess.thecvf.com/content_cvpr_2013/html/Rudoy_Learning_Video_Saliency_2013_CVPR_paper.html
- Coutrot et al., time-dependent master saliency map: https://arxiv.org/abs/1702.00714
- Zhao & Koch, JoV 2011: https://www-users.cse.umn.edu/~qzhao/publications/pdf/learning_saliency_jov11.pdf

**Cognitive evidence**
- Itti & Baldi, Bayesian surprise: https://ilab.usc.edu/publications/Itti_Baldi06cvnms.html (also Vision Research 2009)
- Carmi & Itti, JoV 2006 (memory after jump cuts): https://ilab.usc.edu/publications/Carmi_Itti06jov.html
- Tseng et al., JoV 2009 (center bias): https://ilab.usc.edu/publications/Tseng_etal09jov.html

**World models and surprise**
- V-JEPA 2 (Assran et al., 2025): https://arxiv.org/abs/2506.09985 · HF docs: https://huggingface.co/docs/transformers/model_doc/vjepa2 · code: https://github.com/facebookresearch/vjepa2
- V-JEPA 2.1 (2026; distilled ViT-B/L predict the ViT-G teacher's space): https://arxiv.org/abs/2603.14482
- Garrido et al., intuitive physics from V-JEPA surprise (2025): https://arxiv.org/abs/2502.11831
- Hristov et al., *When does latent prediction error support physical surprise claims?* (NeurIPS 2026 workshop): https://neurips.cc/virtual/2026/175860
- DINO-Explorer (2026): https://arxiv.org/abs/2604.12933 · Cambrian-S: https://arxiv.org/abs/2511.04670
- Self-information saliency from deep CNN representations: https://arxiv.org/abs/1511.02872

**Calibration and inter-observer consistency**
- Noise-aware VSP (Prashnani et al., BMVC 2021): https://arxiv.org/abs/2104.08038
- Calibration of deep saliency models: https://arxiv.org/abs/2105.12441
- *Raising the Ceiling* (empirical fixation densities, 2026): https://arxiv.org/abs/2605.03885
- IOVC in video: https://pure.ecnu.edu.cn/en/publications/inter-observer-visual-congruency-in-video-viewing/ · Bruckert & Christie 2023: https://project.inria.fr/wicedxcinemotions2023/files/2023/06/WICED_x_Cinemotions_2023_paper_8.pdf · Le Meur et al. 2011: https://halshs.archives-ouvertes.fr/halshs-00644506

**Rejected directions**
- SUM (Mamba, image saliency): https://arxiv.org/abs/2406.17815 · Samba+: https://arxiv.org/abs/2602.01593
- torchvision Video Swin: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.swin3d_t.html

Standard components cited by name without a link: FiLM (Perez et al. 2018), SPADE (Park et al. 2019), ViTDet SimpleFPN (Li et al. 2022), R(2+1)D (Tran et al. 2018), STCN/XMem (Cheng et al. 2021/2022), TransNetV2 (Souček & Lokoč 2020), RAFT (Teed & Deng 2020), VideoMAE (Tong et al. 2022), LoRA (Hu et al. 2022), metric analyses by Kümmerer et al. (ECCV 2018) and Bylinskii et al. (TPAMI 2019).
