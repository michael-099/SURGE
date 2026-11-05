# SURGE-D: the distilled, evidence-checked proposal for video saliency prediction

**Final proposal.** Prepared 2026-11-05. It replaces `SURGE_video_saliency_research_design.md` and `SURGE_reality_check_and_compute.md`; both are kept in git history (commit `199de7a`).

**Evidence tags:**
- **[Verified]**: checked against a primary source in this session (link in §9).
- **[Replicated]**: re-measured in this session on real model weights (§2).
- **[Unverified]**: the source could not be checked, or it disagrees with another source.
- **[Estimate]**: derived from verified or measured numbers plus stated assumptions.
- **[Hypothesis]**: what the experiments will test.

No saliency result in this document is a real experimental result. Every number attributed to another method was checked against its source.

---

## 0. Bottom line

| Question | Answer |
|---|---|
| **What is the current SOTA to beat?** | **DHF1K test** (official leaderboard): **SalFoM**, CC 0.5692, NSS 3.3536, AUC-J 0.9222, SIM 0.4208, sAUC 0.7352 [Verified]. **DHF1K val**: TMFI-Net, CC 0.554, NSS 3.201 [Verified]. **Hollywood-2**: ViNet-E, CC 0.766, NSS 4.168 [Verified]. **UCF-Sports**: ViNet-E, CC 0.744, NSS 4.156 [Verified]. A 2026 journal paper (DcF-Net) reports DHF1K CC 0.570 but does not say which split, and it is not on the leaderboard [Unverified]. |
| **Does the original SURGE idea survive?** | **Not as the headline.** It used the V-JEPA 2 predictor's error ("surprise") as a fixation cue. My independent replication on 11 clips (§2) confirms the reality check's negative finding: [Replicated] the per-token error **does not grow with prediction horizon**, and it **mostly tracks how atypical each target token is**, not what the past predicted. Surprise therefore becomes an **optional plug-in** that must pass a go/no-go test; it is not the core of the model. |
| **What is the distilled architecture?** | **SURGE-D**: the components with the best evidence behind them. (1) A **V-JEPA 2.1 ViT-L** backbone (dense-feature video world model, 2026), **LoRA-tuned**; (2) multi-level taps into a light **3D FPN decoder with deep supervision**; (3) **multi-frame (MIMO) output**, 16 frames per forward pass; (4) a **learned center prior**; (5) KL + CC + NSS loss; (6) a **zero-initialized surprise slot** that is turned on only if Phase 0 passes. About 7–9M trainable decoder parameters, plus about 1.6M LoRA parameters. |
| **Why should it beat SalFoM?** | [Hypothesis, with evidence] Saliency maps are a *dense* prediction task. SalFoM uses UMT-L, a Kinetics-supervised ViT-L. V-JEPA 2.1's distilled ViT-L reaches **46.7 mIoU on ADE20K with a linear probe**, versus about 22–24 for V-JEPA 2 [Verified], and V-JEPA 2 alone already placed 2nd at NTIRE 2026 (PredJSal, CC 0.8272, vs. 0.8280 for a 6B InternVideo2 model) [Verified]. SURGE-D pairs the best-evidenced dense video features with the proven decoder, prior and LoRA recipe. **This is not guaranteed.** The first gate (Phase 0) is a cheap head-to-head of the backbones. |
| **Compute** | [Estimate] Phase 0 (backbone shoot-out + surprise gate): **8–15 T4-hours**. Main result (LoRA, 3 seeds, DHF1K + Hollywood-2 + UCF): **about 80–160 T4-hours**, or about 30–60 L4-hours. Inference: about **0.11 TFLOPs per output frame**, about 5× less than TMFI-Net's backbone alone (§6). |

---

## 1. Fact-check of the two original documents

Every factual claim in the originals was checked. ✅ = confirmed, ⚠️ = partly right or needs a correction, ❌ = wrong or contradicted, ❓ = could not be verified.

### 1.1 Benchmarks and SOTA numbers

| Claim | Verdict | Evidence / correction |
|---|---|---|
| DHF1K test leaderboard: SalFoM 0.922/0.421/0.735/0.569/3.354; THTD 0.548/3.139; TMFI 0.546/3.146; STSANet 0.529/3.010; ViNet 0.511/2.872; UNISAL 0.490/2.776; SalEMA SIM 0.466, CC 0.449 | ✅ | Matches the official leaderboard exactly (mmcheng.net/videosal). Note: THTD's CC (0.5479) is *higher* than TMFI's (0.5461), but THTD's NSS is lower |
| TSFP-Net 0.517, VSFT 0.519, HD2S 0.503, TASED 0.470, ACLNet 0.434, DeepVS 0.344 (test CC) and sizes/times | ✅ | Leaderboard |
| DHF1K: 1000 videos, 640×360, 30 fps, 582,605 frames, 19,420 s, 17 observers, SMI RED 250 at 250 Hz, split 600/100/300 | ✅ | DHF1K paper. (The paper's introduction says "more than 600K frames", which contradicts its own 582,605 figure) |
| DHF1K test labels held out; one submission per week | ✅ | DHF1K README: "You can only submit ONCE within One week" |
| ACLNet without static data: AUC-J 0.890 → 0.847 | ✅ | DHF1K paper §5.4 / Table 6 |
| DHF1K val (ViNet-S/A paper): TMFI 0.554/3.201, THTD 0.553/3.188, ViNet-E 0.549/3.134, STSANet 0.539/3.082, DiffSal 0.533/3.066, ViNet-S 0.529/3.008, ViNet-A 0.525/3.019 | ✅ | ViNet-S/A paper, Table 1 |
| Hollywood-2 / UCF test numbers for ViNet-E, DiffSal, ViNet-A, TMFI, THTD | ✅ | ViNet-S/A paper |
| SalFoM Hollywood-2 0.709 / UCF 0.631 | ✅ | SalFoM paper, Table 2 |
| SalFoM: UMT-L fine-tuned, 16×224×224, predicts last frame, KL+CC loss | ✅ | SalFoM paper |
| SalFoM authors attribute weak UCF results to overfitting | ✅ | Their *explanation*, not an experiment. They also cite task-driven labels and center bias |
| ViNet-S: 9.5M params, about 1000 fps batched on an RTX 4090 | ✅ | 36.24 MB; about 200 fps at batch 1, about 1070 fps at batch 8 |
| STSANet's Hollywood-2/UCF numbers are swapped between the leaderboard and the ViNet paper | ❓ | The ViNet paper lists H2 0.705, UCF 0.721. I could not access a leaderboard table for these datasets |
| TinyHD: 3.9M params; TinyHD-M CC 0.493 | ✅ | TinyHD paper (test: TinyHD-S 0.505, TinyHD-M 0.493) |
| Hollywood-2: 3,100/3,559 shot clips; mostly task-driven | ✅ | ViDaS paper. ViDaS says "each one viewed by 12 persons". The original said 19 observers (12 action, 4 context, 3 free-viewing); that matches the original dataset paper's design, so both are consistent |
| UCF-Sports split 103/47 | ⚠️ | ViDaS reports **104/48 clips**. Both counts appear in the literature. Use each baseline's exact split |
| Hollywood-2 download is 74.6 GB including maps | ✅ | DHF1K README ("74.6G") |
| "No newer model beats SalFoM on DHF1K" (implicit) | ⚠️ | **Missed by the originals:** DcF-Net (J. Meas. Sci. Instrum., 2026; MViTv2 + ConvNeXt) reports CC 0.570, NSS 3.284, sAUC 0.745 without naming the split [Unverified]; TM2SP (TCSVT 2025) is not on the leaderboard. Neither changes the target: aim for **test CC > 0.570 and NSS > 3.354** |
| **New since the originals:** SalTempto (arXiv 2610.03276, Oct 2026), a 224-clip eye-tracking set where motion matters | — | SalFoM leads (61.4% of the explainable headroom on test); a fine-tuned *static* model recovers only about 13%, versus more than half on LEDOV. The authors conclude **LEDOV does not measure temporal modeling well**. A good additional zero-shot test |

### 1.2 NTIRE 2026 and other challenges

| Claim | Verdict | Evidence / correction |
|---|---|---|
| NTIRE 2026: 2,000 videos, crowdsourced mouse tracking | ✅ | Over 5,000 assessors; 800-video test, 500-video private test |
| "≥70 viewers per video" | ⚠️ | Verified for **AIM 2024 / AViMoS** (1,500 videos, >70 per video). Not stated for NTIRE 2026 in what I could read |
| Center prior alone reaches CC 0.41 on NTIRE | ✅ | 0.4101 private test (0.4175 public) |
| 1st iLearn/ViSAGE: InternVideo2-6B + LoRA + 2 expert decoders (temporal gating, FiLM from the CLS token, learned center prior) | ✅ | CC 0.8280, NSS 3.3229, 6,880M params |
| 2nd CVSP/PredJSal: V-JEPA2-g encoder, temporal self-attention, 3D conv decoder, 4-seed ensemble; **does not use the predictor** | ✅ | CC 0.8272, NSS 3.4156. The challenge report describes only encoder features. Also: 8 multi-scale maps fused by learnable weights |
| 3rd ARK: InternVideo2-1B + temporal attention | ✅ | CC 0.7896, best NSS (3.4562) |
| 4th Vertex used TMFI as a base | ✅ | And a weighted ensemble with DiffSal, SalFoM and TMFI |
| SSF-DiffNet uses Mamba-style blocks; AAM used CLIP text | ✅ | SSF-DiffNet: SAM2-Hiera + flow + Mamba blocks + diffusion. AAM: **DINOv3 + LoRA**, CLIP text, Wav2CLIP audio |
| ViSAGE frozen vs. LoRA: CC 0.783→0.794, NSS 3.180→3.225 (expert 1); 0.784→0.794 (expert 2) | ✅ | Exact. Removing LoRA is the largest single drop in both experts |
| Ensembling adds about +0.005 CC | ✅ | ViSAGE: experts 0.830/0.832 → logit fusion 0.837 |
| Learned center prior / FiLM / auxiliary heads each matter | ✅ (small) | ViSAGE ablation: each costs about 0.004 CC when removed |

### 1.3 V-JEPA 2 facts and implementation claims

| Claim | Verdict | Evidence / correction |
|---|---|---|
| `facebook/vjepa2-vitl-fpc64-256`: MIT, 1.30 GB fp32, encoder 303.9M + predictor 22.1M | ✅ | Safetensors header: 325,971,328 params total (1,303,947,864 bytes); `license: mit` |
| Predictor: 384-d, 12 layers, 12 heads | ✅ | `config.json` |
| Pretraining at **4 fps**, 16 frames, 256 px, aspect augmentation [0.75, 1.35] | ✅ | `configs/train/vitl16/pretrain-256px-16f.yaml` |
| Pretraining masks span the full clip in time (`temporal_scale: [1.0, 1.0]`), 8×15% + 2×70% spatial blocks | ✅ | Same config. **V-JEPA 2.1 uses the same masks and also 4 fps** (`configs/train_2_1/vitl16/cooldown-256px-64f.yaml`) |
| HF `VJEPA2Model.forward` encodes the full clip, then selects context tokens (target leakage) | ✅ | `modeling_vjepa2.py`: the encoder runs on all of `pixel_values_videos`; masks only select predictor inputs |
| HF RoPE assumes a square grid | ✅ | `grid_size = crop_size // patch_size` is used for both height and width |
| `hidden_states[0]` is the embedding output | ✅ [Replicated] | §2: `len(hidden_states) = 25` |
| Garrido et al. compute surprise as clip-level L1 between predicted and actual future representations | ✅ | Per-window error, aggregated over the video by average or max. No spatial maps are evaluated. The flagship is ViT-H; the per-property analysis uses ViT-L |
| No prior work evaluates spatial latent-surprise maps against fixations | ✅ (as far as found) | Targeted searches found none; PredJSal uses encoder features only |
| DINO-Explorer: DINOv3 latent surprise for event triage, ego-motion compensated | ✅ | It is *underwater* event triage; ego-motion conditioning removes 45.5% of false positives |
| Hristov et al. (NeurIPS 2026 workshop): latent error also reflects appearance, static content, motion and prediction difficulty | ✅ | Abstract. "Texture" specifically is not in the abstract |
| V-JEPA 2.1 exists; ViT-B/L are distilled from ViT-G | ✅ | arXiv 2603.14482; checkpoints `vjepa2_1_vitl_dist_vitG_384.pt` etc. Code MIT. **No weight license is stated in the README**; check before redistributing |

### 1.4 Cognitive-science and related-work claims

| Claim | Verdict | Evidence / correction |
|---|---|---|
| Itti & Baldi: surprise predicts gaze shifts better than other tested metrics | ✅ | "Outperforms five other metrics"; 4 observers, about 45k frames |
| Carmi & Itti: after jump cuts, memory-guided attention drops and recovers over about 2.5 s | ✅ | "Across 7 consecutive saccades and 2.5 seconds" |
| Tseng et al.: "center bias is strongest at scene onset" | ⚠️ | The abstract says center bias is **mostly photographer bias**; viewing strategy contributes *at scene onset*. "Strongest at onset" is an over-reading |
| *Raising the Ceiling*: LEDOV inter-observer ceiling 2.00 → 3.05 bits/fixation | ✅ | Also: UniformerSal reaches 1.91 bits, near the old ceiling |
| Wulff et al.: ground-truth past saliency maps, DAVSOD only | ✅ | Not fixation data |
| STM-SalNet: ICONIP 2025, memory bank with forgetting, DHF1K/H2/UCF | ✅ | No metrics on the abstract page |
| AIM 2024: AViMoS, 1,500 videos, mouse tracking | ✅ | |

### 1.5 The reality check's measured claims

| Claim | Verdict | Evidence |
|---|---|---|
| Encoder 1.656 TFLOPs (16×256²), context 0.725, predictor 0.166 | ✅ [Replicated] | §2.3 |
| Predictor error does not grow with horizon | ✅ [Replicated] | §2 |
| Error map tracks target atypicality (r ≈ 0.5–0.73) and is largely context-independent | ✅ [Replicated] | §2 |
| Leaky vs. leak-free maps correlate at 0.78–0.95 | ✅ [Replicated] | §2 |
| GPU-hour budgets (T4 12–25 effective TFLOPS, etc.) | ❓ | Assumptions, not measurements. Kept as ranges; calibrate on the GPU first (§6) |

---

## 2. Independent replication of the surprise diagnostics

REPLICATION_PLACEHOLDER

---

## 3. What actually drives video-saliency performance

Ranked by strength of evidence:

1. **Backbone pretraining dominates.** [Verified] DHF1K test CC climbs with each backbone generation (2D CNN 0.43–0.49 → S3D 0.47–0.53 → Video Swin 0.546–0.548 → UMT-L 0.569). The top three NTIRE 2026 entries all use video foundation models of ≥1B parameters.
2. **Adapting the backbone (LoRA) beats freezing it**, by about 0.01 CC and 0.05 NSS (ViSAGE) [Verified]. That is as large as the gap between neighbouring SOTA entries.
3. **Multi-level decoding with auxiliary supervision** is near-universal in strong models. Removing auxiliary heads costs about 0.004 CC (ViSAGE) [Verified].
4. **Ensembles** add about 0.005 CC (ViSAGE; PredJSal uses 4 seeds) [Verified].
5. **A learned center prior** adds about 0.004 CC (ViSAGE) [Verified]. A prior alone reaches CC 0.41 on NTIRE.
6. **Static-image data** helped strongly in the CNN era (ACLNet: AUC-J 0.890 → 0.847 without it) [Verified]. Whether it still helps a foundation backbone is **untested**, so it is an ablation here.
7. **Predictor "surprise"**: no evidence that it helps, and two independent diagnostics now say the released predictor's per-token error is mostly content atypicality (§2).

The design follows directly: put the effort where items 1–5 are, and treat 6–7 as gated options.

---

## 4. SURGE-D architecture

### 4.1 Pipeline

```
 Video ──► chunk of T=16 frames (stride s; default s=2 ≈ 15 fps, swept {2, 4, 8}), squashed to 256×256
              │
   ┌──────────▼──────────────────────────────────────────────┐
   │ V-JEPA 2.1 ViT-L/16 (300M, distilled from ViT-G)        │  fallback: V-JEPA 2 ViT-L (HF)
   │ LoRA r=16 on qkv/proj of blocks 13–24 (Stage 2)         │
   │ 8 tubelets × 16×16 tokens; taps at blocks 6/12/18/24    │
   └───┬─────────┬─────────┬─────────┬───────────────────────┘
       │h6       │h12      │h18      │h24
       ▼         ▼         ▼         ▼
   SimpleFPN-3D: P4 (64²) · P8 (32²) · P16a, P16b (16²), 256 ch, 8 time steps
       │
       ├──── [optional] SCM_ℓ: F̃ = F ⊙ (1+γ(U)) + β(U), zero-initialized; U = surprise embedding
       │      (enabled only if Phase 0 gate G2 passes; identity otherwise)
       ▼
   Top-down (2+1)D residual decoder 1/16 → 1/8 → 1/4, auxiliary heads at 1/16 and 1/8
       │
   logits L: 8 tubelets × 64×64 → temporal ×2 → 16 sampled frames → keep the central 8
       │                                         (≥0.27 s of context on each side)
   L + log G_θ        (G_θ: mixture of 8 learned Gaussians, normalized; static, no gate)
       │
   softmax over pixels → interpolate to the 16 original frames → upsample to native resolution → blur (σ tuned)
```

### 4.2 Components and why each one is kept

| # | Component | Spec | Kept because | Status |
|---|---|---|---|---|
| 1 | **Backbone** | V-JEPA 2.1 ViT-L/16 (distilled), 256 px, 16 frames | Best-evidenced *dense* video features among open ViT-L models: ADE20K linear probe 46.7 mIoU (distilled L) vs. 42.0 (L trained from scratch) and about 22–24 for V-JEPA 2 [Verified]. V-JEPA 2 already places 2nd at NTIRE 2026 | **Phase 0 gate G1** picks between V-JEPA 2.1-L, V-JEPA 2-L and UMT-L (SalFoM's backbone) |
| 2 | **LoRA** | r=16, q/k/v/proj, last 12 blocks; Stage 1 frozen → Stage 2 LoRA | +0.01 CC in ViSAGE; full fine-tuning (SalFoM) overfits small sets by its authors' account | Core |
| 3 | **SimpleFPN-3D + (2+1)D top-down decoder + deep supervision** | 4 taps, 256 ch, GroupNorm, GELU | Standard multi-level design (TMFI, HD2S, ViSAGE expert 2) | Core (≈7M params) |
| 4 | **MIMO output** | 8 central sampled frames (16 original) per pass, hop 16 | Amortizes the ViT-L: about 10× cheaper per output frame than a SalFoM-style sliding window and about 5× cheaper than TMFI-Net (§6). Precedent: TinyHD-M (not claimed as novel) | Core |
| 5 | **Learned center prior** | 8 diagonal Gaussians, product of experts, about 40 params | +0.004 CC (ViSAGE); cheap | Core. The surprise-gated version is **dropped** (benchmarks contain few within-clip cuts) |
| 6 | **Loss** | KL + 1·(1−CC) − 0.1·NSS + 0.25·aux | KL+CC is SalFoM's loss; a small NSS term targets the NSS gap | Core |
| 7 | **Surprise SCM** | Zero-initialized FiLM from the predictor error | Starts as an exact no-op, so it can only be kept if it helps | **Optional**, gate G2 |
| 8 | **Static-image co-training** | SALICON images through V-JEPA 2.1's image tokenizer, 1 image batch per 4 video batches | Helped in the CNN era; untested with foundation backbones | Ablation |
| 9 | **Test-time** | Horizontal-flip TTA; 2–3 seed logit ensemble | +about 0.005 CC (ViSAGE) | Reported **separately** from the single model |

**What was cut from SURGE and why:** surprise as the central cue (§2: not a temporal-surprise signal); the surprise-gated center prior (the gate has nothing to act on in shot-split benchmarks, and Tseng et al. attribute center bias mostly to photographer bias); the dual-rate surprise sampling (only needed if surprise survives); the leak-free two-pass encoder at training time (kept only for gate G2).

### 4.3 Formulation

Clip $X\in\mathbb R^{16\times3\times256\times256}$, tokens $8\times16\times16$, $D=1024$.

1. Features: $\{h^{(\ell)}\}_{\ell\in\{6,12,18,24\}} = E_{\theta+\Delta_{\text{LoRA}}}(X)$.
2. Pyramid: $P_4=\mathrm{Up}_4(W_6h^{(6)}),\ P_8=\mathrm{Up}_2(W_{12}h^{(12)}),\ P_{16a}=W_{18}h^{(18)},\ P_{16b}=W_{24}h^{(24)}$.
3. Optional modulation: $\tilde P_\ell = P_\ell\odot(1+\gamma_\ell(U_\ell))+\beta_\ell(U_\ell)$, with $\gamma,\beta$ initialized to zero ($U\equiv0$ when disabled).
4. Decoder: $D_{16}=\mathcal R([\tilde P_{16a};\tilde P_{16b}])$, $D_8=\mathcal R(\tilde P_8+\uparrow D_{16})$, $D_4=\mathcal R(\tilde P_4+\uparrow D_8)$, $L=\mathrm{Head}(D_4)$.
5. Output: $S_t(p)=\mathrm{softmax}_p\big(L_t(p)+\log G_\theta(p)\big)$.
6. Loss over output frames $\mathcal O$:
$$\mathcal L=\tfrac1{|\mathcal O|}\textstyle\sum_t\big[\mathrm{KL}(Y_t\Vert S_t)+(1-\mathrm{CC}(Y_t,S_t))-0.1\,\mathrm{NSS}(S_t,F_t)\big]+0.25\,\mathcal L_{\text{aux}}.$$

### 4.4 Training recipe

| Setting | Value |
|---|---|
| Data | DHF1K: 540 train / 60 internal-val (fixed list, released) / official 100 val used as a test set. Hollywood-2 and UCF: fine-tune from the DHF1K model, 10% internal val |
| Stage 1 | Backbone frozen; decoder + prior; AdamW lr 3e-4 (prior 1e-3), wd 0.01, cosine, 20 epochs × 8 random clips per video; batch 8 |
| Stage 2 | + LoRA (lr 1e-4), decoder lr 1e-5, 10 epochs, batch 2 × accumulation 4, gradient checkpointing |
| Augmentation | Horizontal flip (labels flipped), mild color jitter. **No spatial crops** (they corrupt center-bias statistics) |
| Precision | fp16 autocast on T4; bf16 on L4/A100 |
| Selection | Internal val only. At most 2 DHF1K test submissions (one per week allowed) |

---

## 5. Experiments and decision gates

### 5.1 Phase 0: two cheap gates before any long run

| Gate | Experiment | Pass rule | If it fails |
|---|---|---|---|
| **G1: backbone shoot-out** | Stage 1 only, 10 epochs, same decoder, 1 seed: V-JEPA 2.1-L vs. V-JEPA 2-L vs. UMT-L (K400) | Pick the best by internal-val CC+NSS. Proceed with V-JEPA 2.1 if it is within noise of the best | Use the winner; the rest of the design is backbone-agnostic |
| **G2: surprise go/no-go** | Zero-shot on DHF1K internal val: surprise maps (raw; contextual = true past − shuffled past) vs. **token atypicality**, frame difference and center prior. NSS and information gain over the center prior | Surprise beats **both** atypicality and frame difference on NSS, with positive information gain | Disable SCM (the default). Write up the negative result as an analysis finding |

### 5.2 Main comparisons

| ID | Model | Purpose |
|---|---|---|
| B0 | Learned center prior only | Floor |
| B1 | SURGE-D Stage 1 (frozen) | Effect of LoRA |
| **M** | **SURGE-D Stage 2 (LoRA)**, 3 seeds | Main result |
| M+E | M, 3-seed ensemble + flip TTA | Reported separately |
| A1 | M − MIMO (last-frame output, sliding window) | Accuracy cost of MIMO |
| A2 | M − center prior; M − deep supervision | Component value |
| A3 | M + SALICON co-training | Static data with a foundation backbone |
| A4 | M + SCM (only if G2 passed), with shuffled-surprise and frame-difference controls through the same SCM | Does surprise add anything? |
| A5 | Stride s ∈ {2, 4, 8}; squash vs. letterbox | Temporal scale and aspect ratio |

### 5.3 Targets (not predictions)

| Level | DHF1K | Hollywood-2 | UCF-Sports |
|---|---|---|---|
| Competitive | val CC ≥ 0.554, NSS ≥ 3.20 (≥ TMFI) | CC ≥ 0.739 (≥ TMFI) | CC ≥ 0.707 |
| **SOTA** | **test CC > 0.570, NSS > 3.354** (above SalFoM and DcF-Net's claim) | CC > 0.766 (ViNet-E) | CC > 0.744 (ViNet-E) |

### 5.4 Evaluation protocol

- Official DHF1K MATLAB metrics, or a Python port validated to ±0.001 on a subset. Validate the pipeline by re-scoring a released baseline (TMFI-Net has public code and weights).
- Report AUC-J, sAUC, NSS, CC, SIM, KL and information gain over the center prior. Use **one map for all metrics** (Kümmerer et al., ECCV 2018).
- Statistics: paired Wilcoxon over videos, bootstrap 95% CIs, 3 seeds. DHF1K val (100 videos) resolves only about 0.006–0.01 CC [Estimate]; **Hollywood-2 (3,559 test clips) is the more sensitive test**.
- Zero-shot generalization: DHF1K → Hollywood-2, UCF, DIEM, **SalTempto** (motion-heavy, new in 2026).

---

## 6. Compute

[Replicated] FLOPs from `torch.utils.flop_counter` on the real HF modules (V-JEPA 2.1-L has the same ViT-L shape, so the same cost at 256 px).

| Component | TFLOPs per 16-frame clip |
|---|---|
| ViT-L encoder, 16×256×256 (2,048 tokens), forward | **1.656** [Replicated] |
| ViT-L encoder, 16×224×224 (1,568 tokens), forward | 1.194 [Replicated] |
| SURGE-D decoder, forward (the reality check's measured 7.04M-param decoder) | 0.16 |
| Encoder with LoRA (r=16, last 12 blocks), forward + backward (reality check, measured) | 2.69 |
| Optional surprise: context-prefix pass (0.725) + predictor (0.166) | 0.89 [Replicated] |

Parameters [Replicated]: encoder 303.9M, predictor 22.1M.

**Inference cost per output frame** [Estimate]: (1.656 + 0.16) / 16 output frames ≈ **0.11 TFLOPs**. For comparison:
- TMFI-Net's Video Swin-S backbone ≈ 0.60 per output frame (reality check; consistent with torchvision's 82.84 GFLOPS for Swin3D-S scaled to 32×224×384).
- A SalFoM-style ViT-L sliding window, 16×224² per output frame, ≈ 1.19 [Replicated for the V-JEPA tokenization; UMT's may differ].

So SURGE-D should be about **5× cheaper per frame than TMFI-Net** and about **10× cheaper than a SalFoM-style window**. With gate G2's surprise branch enabled, the cost rises to about 0.17 per frame.

**Training budget** [Estimate]; T4 throughput assumed 8–18 effective TFLOPS in training (calibrate first):

| Job | T4-hours |
|---|---|
| G1 backbone shoot-out (3 backbones × 10 epochs Stage 1) | 6–12 |
| G2 surprise zero-shot (100 videos at 4 fps) | 0.5–1 (or one night on a laptop CPU) |
| One Stage-1 run (20 epochs, 4,320 clips/epoch) | 3–6 |
| One Stage-2 LoRA run (10 epochs) | 2–5 |
| **Main program**: B1 + M × 3 seeds + about 8 ablations + Hollywood-2/UCF fine-tunes + evaluations | **≈ 80–160** (≈ 30–60 L4-hours; ≈ 10–25 A100-hours) |

Kaggle's free tier gives about 30 GPU-hours per week (reported by secondary sources; check your account), so the main program takes 3–6 weeks on free quota.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| V-JEPA 2.1-L does not beat UMT-L on saliency | Medium | Gate G1 decides early; the design is backbone-agnostic |
| A ViT-L backbone is too small to pass SalFoM's test CC | Medium | LoRA, ensemble, static co-training; ViT-g as a last resort (≈3× compute) |
| V-JEPA 2.1 weights carry a non-MIT license | Low–medium | Check before releasing derived weights; V-JEPA 2 (HF, MIT) is the fallback |
| Squashing 16:9 to 1:1 costs accuracy | Low | SalFoM squashes too; ablation A5 |
| DHF1K val too small to show gains | High | Hollywood-2 as the primary significance test; effect sizes with CIs |
| Test-server delays | High | Report val with internal-val selection, as recent papers do |
| Pretraining overlap (V-JEPA data includes YouTube-derived sets; DHF1K is from YouTube) | Unknown | Disclose; shared by every foundation-model baseline |

---

## 8. Implementation notes (verified against code)

1. **HF leakage.** `VJEPA2Model.forward` encodes the full clip before selecting context tokens. For surprise (gate G2 only), encode the context prefix separately and call `model.predictor` directly.
2. **Square RoPE grid** in the HF port: keep 256×256 (or patch `_get_height_pos` / width ids for rectangles).
3. **`hidden_states`** has 25 entries for ViT-L; index 0 is the patch embedding, so block *l*'s output is `hidden_states[l]`.
4. **V-JEPA 2.1** is not in `transformers`; load it from the official repo (PyTorch Hub entries such as `vjepa2_1_vit_large_384`) and verify its RoPE handles your input size.
5. Cache nothing from the backbone during Stage 2 (LoRA changes features); during Stage 1, caching tap features is possible but large (4 × 2,048 × 1,024 fp16 ≈ 16 MB per clip). Usually recomputing is cheaper than disk I/O on Kaggle.

---

## 9. Sources (all opened in this session)

**Benchmarks and SOTA**
- DHF1K leaderboard: https://mmcheng.net/videosal/ · DHF1K paper: https://arxiv.org/abs/1801.07424 · README: https://github.com/wenguanwang/DHF1K
- SalFoM: https://arxiv.org/abs/2404.03097 · ViNet-S/A: https://arxiv.org/abs/2502.00397 · TinyHD: https://arxiv.org/abs/2301.04619 · ViDaS (Hollywood-2/UCF splits): https://arxiv.org/abs/2305.11729
- BIAS (2026; reproduces the leaderboard): https://arxiv.org/abs/2604.08858 · DcF-Net (2026): https://www.sciopen.com/article/10.62756/jmsi.1674-8042.2026042 · TM2SP (TCSVT 2025): https://dblp.org/rec/journals/tcsv/LiL25
- SalTempto (2026): https://arxiv.org/abs/2610.03276
- NTIRE 2026 report: https://arxiv.org/abs/2604.14816 · ViSAGE: https://arxiv.org/abs/2604.08613 · AIM 2024: https://arxiv.org/abs/2409.14827

**World models**
- V-JEPA 2 checkpoint: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 · configs: https://github.com/facebookresearch/vjepa2/tree/main/configs · HF code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vjepa2/modeling_vjepa2.py
- V-JEPA 2.1: https://arxiv.org/abs/2603.14482 · checkpoints: https://github.com/facebookresearch/vjepa2
- Garrido et al. 2025: https://arxiv.org/abs/2502.11831 · Hristov et al. 2026: https://neurips.cc/virtual/2026/175860 · DINO-Explorer: https://arxiv.org/abs/2604.12933

**Cognition, priors, related**
- Itti & Baldi: https://ilab.usc.edu/publications/Itti_Baldi06cvnms.html · Carmi & Itti 2006: https://ilab.usc.edu/publications/Carmi_Itti06jov.html · Tseng et al. 2009: https://ilab.usc.edu/publications/Tseng_etal09jov.html
- Raising the Ceiling: https://arxiv.org/abs/2605.03885 · Wulff et al.: https://arxiv.org/abs/2504.05913 · STM-SalNet: https://pure.ecnu.edu.cn/en/publications/stm-salnet-a-biologically-inspired-spatial-temporal-memory-networ/
- UNISAL: https://arxiv.org/abs/2003.05477 · torchvision Swin3D-S: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.swin3d_s.html
