# SURGE: reality check and compute estimate

Companion to `SURGE_video_saliency_research_design.md`. Prepared 2026-11-05.

**Evidence tags used below:**
- **[Measured]**: counted or timed in this session on the real model code (FLOPs with `torch.utils.flop_counter`, FLOPs = 2 × MACs).
- **[Verified]**: checked in a primary source, config file, or source code.
- **[Assumption]**: a planning input that you should replace with your own measurement.
- **[Estimate]**: derived from measured numbers combined with assumptions.

---

## 0. Bottom line

| Question | Answer |
|---|---|
| **Can it be built from public parts?** | **Yes.** [Verified] The `facebook/vjepa2-vitl-fpc64-256` checkpoint (MIT license, 1.30 GB, fp32) contains the encoder (303.9M params) **and the predictor (22.1M params)**. HuggingFace `VJEPA2Model` exposes both. [Measured] The SURGE decoder as specified runs end to end with 7.04M trainable params. |
| **Is the core signal supported by evidence?** | **No, and early evidence is negative.** Latent "surprise" had only been validated as a *clip-level number* (Garrido et al.). **[Measured] On real weights (3 clips × 2 frame rates, §2), the released predictor's per-token error does not grow with prediction horizon, and its spatial map mostly reflects how atypical each target token is (r ≈ 0.5–0.73). It highlights background texture rather than people or actions.** Only a small part of it depends on the clip's past. This is anecdotal (no fixation data), but consistent in all 6 cases. **A de-risking phase (3–6 T4-hours) must come before any decoder training** (§6). |
| **Do the design assumptions hold?** | **Three need changing.** (1) V-JEPA 2 was pretrained at **4 fps**, not the ~15 fps the design used. (2) Squashing 16:9 frames to a square is outside its aspect-ratio augmentation range. (3) A frozen backbone costs about 0.011 CC in the closest published ablation, so LoRA tuning belongs in the main plan. |
| **Compute** | [Estimate] **Minimum viable experiment (MVE): 14–29 T4-hours**, about one week of Kaggle quota. **Paper-grade program: about 150–325 T4-hours**: 5–11 weeks of free Kaggle quota, or about 50–130 L4-hours / 18–54 A100-hours. Inference costs about **0.16 TFLOPs per output frame**, *less* than TMFI-Net's backbone alone (0.60). |
| **Can a real gain be detected?** | [Estimate] DHF1K-val (100 videos) can only resolve differences of about **0.006–0.01 CC**. Gains from an added cue on a strong backbone are plausibly about that size. Hollywood-2 (884 test videos) resolves about 3× finer, so make it the primary test set for H1. |

---

## 1. Assumption audit

| # | Assumption in the design report | What I checked | Status |
|---|---|---|---|
| A1 | The predictor is released for an affordable model | Read the safetensors header remotely: 199 `predictor.*` tensors, 22.1M params, next to a 303.9M encoder. License: MIT | ✅ Holds |
| A2 | Surprise can be computed without leakage through the HF API | Read `modeling_vjepa2.py`. The default `forward` encodes the *full* clip and only then selects context tokens (leaky). Encoding the context prefix separately and calling `model.predictor` works (§2) | ✅ Holds, but **only with the leak-free procedure** |
| A3 | `hidden_states[l]` is layer *l*'s output | Tested on a small model: L+1 entries, index 0 is the patch embedding, final output = LayerNorm(`hidden_states[-1]`) | ✅ Holds |
| A4 | Sampling at stride 2 (~15 fps) suits the predictor | Official `configs/train/vitl16/pretrain-256px-16f.yaml` and `cooldown-256px-64f.yaml`: **`fps: 4`** (16-frame clips span 4 s; cooldown 64 frames = 16 s) | ❌ **Fails.** At 15 fps the predictor's dynamics prior is about 4× off its training time scale. Redesign in §3.4 |
| A5 | The predictor can extrapolate to future frames | Pretraining masks: 8×15% and 2×70% spatial blocks, all with **`temporal_scale: [1.0, 1.0]`**, i.e. tubes spanning the whole clip. The predictor never saw "predict the future" masks. Garrido et al. used causal masks with their own internal ViT-H checkpoint and clip-averaged error | ❌ **Weak in practice** [Measured, §2]. The past is used (error drops by about 0.09 relative to a wrong past), but error is flat over horizons of 0.5–2 s |
| A6 | Per-token error is a meaningful *spatial* map | No prior work evaluates spatial latent-surprise maps against fixations | ❌ **Doubtful** [Measured, §2]. The maps track target atypicality (r 0.50–0.73) and are largely context-independent. Fixation value is untested: run H4 first |
| A7 | Squashing 16:9 → 256×256 is harmless | Pretraining aspect augmentation: `random_resize_aspect_ratio: [0.75, 1.35]`; DHF1K frames are 1.78. HF RoPE assumes a square grid; rectangular inputs need a patch and extrapolate positions beyond the 16-token training grid | ⚠️ Mildly out of distribution either way. Settle it with a 2-run comparison (squash vs. 224×400 rectangular, which costs +49% FLOPs) |
| A8 | A frozen backbone gets close to SOTA | ViSAGE (NTIRE 2026) ablation, frozen vs. LoRA: CC 0.783→0.794, NSS 3.180→3.225 (expert 1); 0.784→0.794 (expert 2) | ⚠️ About 0.01 CC penalty, the size of SOTA gaps. **Promote Stage-2 LoRA to the main plan.** It costs only about 1.5× a Stage-1 step (§3) |
| A9 | The center-prior gate has events to act on | Hollywood-2 is evaluated as single-shot clips; DHF1K cut frequency is unknown | ⚠️ Likely little effect on standard benchmarks. Keep it as an ablation on DIEM, not a headline component |
| A10 | Data is easy to stage on Kaggle/Colab | DHF1K comes from a Google Drive folder (videos + annotations, several GB). Hollywood-2 is **74.6 GB** including maps. Kaggle working disk is reportedly about 20 GB; this machine has 31 GB free | ⚠️ Hollywood-2 needs a streaming preprocessing pass (download a part → 256×256 frames + 64×64 maps → delete); final size is a few GB. Budget a day for it |
| A11 | DHF1K test-set results are attainable | README: test annotations are held out; **one submission per week** | ⚠️ Plan at most 2–3 submissions; select models on an internal split |

---

## 2. Real-weights sanity check (V-JEPA 2 ViT-L, CPU)

### Setup

- **Model:** released ViT-L weights, fp32, on this laptop's CPU.
- **Clips:** three 10-s public Kinetics clips (bowling, high jump, archery). One 16-frame window per clip, ending at t = 6 s, squashed to 256×256.
- **Frame rates:** 4 fps (the pretraining rate; a 4-s window) and 15 fps (a 1.07-s window).
- **Split:** the first 8 frames are context; the last 8 frames (4 tubelets) are predicted.
- **Error:** L1 between the raw predictor output and layer-normed targets, as in the official surprise code (`jepa-intuitive-physics`).

**No fixation data were used.** This checks how the signal *behaves*, not whether it predicts gaze.

The scripts are in the session scratchpad (`bench/demo_surprise.py`, `diag_surprise.py`, `contextual_surprise.py`). Figures: `bench/surprise_demo.png` and `bench/contextual_surprise.png`.

### Results

[Measured] Mean over the 3 clips, with the range in brackets:

| Diagnostic | 4 fps | 15 fps | What a genuine expectation-violation signal should show |
|---|---|---|---|
| Predictor error, given the clip's true past | 0.576 [0.566–0.591] | 0.566 [0.559–0.573] | — |
| … given **another video's** past | 0.667 [0.632–0.696] | 0.667 [0.646–0.682] | Higher. ✅ It is: the predictor uses the past |
| … predicting zero (the mean) | 0.658 [0.650–0.669] | 0.662 [0.655–0.670] | Higher. ✅ |
| … copying the last context tubelet forward | 0.730 | 0.684 | Higher. ✅ |
| Change in error from horizon 1 to horizon 4 | ≈ 0 (within ±0.008 in every clip) | ≈ 0 | **Should grow with horizon.** ❌ |
| Spatial correlation of the map with *target atypicality* (how unusual each token is within its own frame) | 0.68–0.73 | 0.50–0.70 | **Low.** ❌ The map mostly reflects target content |
| Spatial correlation with the map computed from the *wrong* video's past | 0.47–0.78 | 0.53–0.71 | **Low.** ❌ The map is largely context-independent |
| Spatial correlation with pixel motion | −0.15 to 0.15 | −0.16 to 0.09 | — (not a motion detector, which is fine) |
| "Contextual" surprise (true-past error minus wrong-past error): correlation with the raw map / with motion | 0.59–0.84 / −0.12 to 0.14 | 0.65–0.85 / −0.17 to 0.06 | The fix did not separate out an event-related component ❌ |
| Leaky (HF default) vs. leak-free map correlation | 0.78–0.94 | 0.90–0.95 | — |

### Interpretation

1. **The predictor is a functioning forecaster in aggregate.** It beats every trivial baseline, and the true past lowers its error by about 0.09–0.10 compared with a wrong past.
2. **Its per-token error is not a temporal-surprise map.** It is flat across horizons (0.5–2 s ahead at 4 fps) and is dominated by how atypical each target token is within its own frame. In the figures, it highlights background foliage and wall texture rather than the people and actions viewers would fixate.
   - **Likely cause:** pretraining used only full-duration tube masks, so the predictor learned to fill in *scene content*, not to extrapolate *dynamics*.
3. **The leakage bug matters less than the design report claimed.** Leaky and leak-free maps correlate at 0.78–0.95, because both are dominated by target atypicality. Keep the leak-free procedure, which is conceptually required, but problem 2 is the real one.
4. **Subtracting the error under a wrong past does not fix it on these clips.**

**Caveat.** This is 3 clips, one window each, with Kinetics content rather than DHF1K and no fixations. Treat it as a **warning light, not a verdict**. The verdict is H4 on DHF1K-val (§6).

**What it means for SURGE.** With the released predictor, SURGE's input behaves more like a *token-atypicality* (self-information-like) map than like *temporal surprise*.
- That could still help saliency, since self-information is a classic saliency principle (Bruce & Tsotsos).
- But it no longer supports the "expectation violation" story.
- Its value beyond encoder features, which already encode appearance, is doubtful.

**The proposal now needs a de-risking phase before any decoder training** (§6).

---

## 3. Measured compute

### 3.1 Per-component cost

[Measured] FLOPs for one 16-frame clip at 256×256 (2,048 tokens), counted by `torch.utils.flop_counter` (eager attention) on the real HF modules with random weights. FLOP counts do not depend on the weights.

| Component | Input | TFLOPs per call | Params |
|---|---|---|---|
| V-JEPA 2 encoder, full clip | 16×256×256 | **1.656** | 303.9M (frozen) |
| V-JEPA 2 encoder, context prefix | 8×256×256 | **0.725** | shared |
| V-JEPA 2 predictor | 1,024 context + 1,024 target tokens | **0.166** | 22.1M (frozen) |
| SURGE decoder (FPN + Φ + SCM + (2+1)D decoder + prior), forward | 4 taps + surprise | **0.160** | **7.04M (trainable)** |
| SURGE decoder, forward + backward | | **0.476** | |
| Encoder with LoRA r = 16 on q/k/v/proj of the last 12 blocks, forward + backward | 16×256×256 | **2.693** | +1.57M (trainable) |
| *Encoder at other resolutions* | 16×224×224 / 16×224×400 / 16×256×448 | 1.194 / 2.471 / 3.439 | |

### 3.2 Cost per output frame versus published architectures

[Measured] Backbone forward FLOPs only, for the published input configuration; decoders would add to the baselines.

| Model | Input per forward | Output frames per forward | **TFLOPs per output frame** |
|---|---|---|---|
| SURGE S1 (single rate, ~15 fps) | 32 frames at stride 2 | 16 | **0.169** |
| **SURGE S2 (dual rate, recommended)** | features at ~15 fps + surprise at 4 fps | 16 / 60 | **0.156** |
| SURGE S3 (all at 4 fps) | 4-s clips | 60 | 0.045 (too coarse in time; see §3.4) |
| TMFI-Net backbone (Video Swin-S) | 32×224×384 | 1 | **0.601** |
| ViNet-S backbone (S3D) | 32×224×384 | 1 | 0.123 |
| SalFoM-like ViT-L (UMT-L proxy, tubelet 2) | 16×224×224 | 1 | 1.194 (≥; UMT's actual tokenization may cost more) |

**Takeaway.** Multi-frame output on a frozen encoder makes SURGE cheaper per frame than TMFI-Net and about 7× cheaper than a SalFoM-style ViT-L sliding window. The heavy backbone is not an inference problem.

### 3.3 Memory (for a 16 GB T4)

[Measured] Bytes of non-parameter tensors saved for backward, fp32; fp16 autocast roughly halves them.

| Training mode | Saved activations per clip | Implication on a T4 (16 GB) |
|---|---|---|
| Stage 1 (decoder only; encoder under `no_grad`) | 0.79 GB fp32 (≈0.4 GB fp16) | Batch 8 ≈ 3–6 GB. Weights in fp16 ≈ 0.65 GB. Easy fit |
| Stage 2 LoRA, no gradient checkpointing | 1.27 GB fp32 per clip (encoder) + decoder | Batch 4 ≈ 6–9 GB. Fits |
| Stage 2 LoRA, with gradient checkpointing | 0.23 GB fp32 per clip (encoder) + decoder | Batch 8 fits; about 30% slower |

[Measured on this laptop's CPU, i7-1165G7, fp32] Full encoder 7.3 s, context 3.2 s, predictor 0.9 s per clip, i.e. about 0.23 TFLOPS achieved. **The H4 zero-shot test on all 100 DHF1K-val videos at 4 fps is about 1,100 clips, roughly 3.5 h on this laptop overnight**, without any GPU.

### 3.4 Sampling redesign forced by the 4 fps finding

| Variant | Encoder sampling | Surprise sampling | Cost per frame | Problem |
|---|---|---|---|---|
| S1 (original report) | ~15 fps | ~15 fps (same pass) | 0.169 | Predictor about 4× off its training time scale |
| S3 | 4 fps | 4 fps | 0.045 | Features at 0.5-s tubelet resolution; saliency must be interpolated across about 15 frames, which blurs fast attention shifts |
| **S2 (recommended)** | ~15 fps (feature pass, full clip only) | **4 fps** (separate full + context + predictor pass over 4-s windows, hop 2 s) | **0.156** | Two sampling rates to align: interpolate the 4-fps surprise maps to feature-tubelet times |

S2 is the principled choice: the world model's error is computed **in distribution**, and saliency keeps its temporal resolution. It even costs slightly less than S1, because one 4-fps surprise window covers 60 frames.

During training, the surprise maps are **cached once**: about 4×16×16 values per window, under 1 GB for all datasets. Each training step then needs only the frozen feature pass plus the decoder: 2.13 TFLOPs per clip.

Make the fps choice empirical: in H4, compare surprise computed at 4, 8 and 15 fps.

---

## 4. GPU-hour budget

### 4.1 Throughput assumptions

These are the biggest uncertainty (±2×). Replace them with the calibration script's numbers (§4.4).

| GPU | Effective TFLOPS, frozen forward (fp16/bf16) | Effective TFLOPS, training step | Basis |
|---|---|---|---|
| T4 (Kaggle/Colab free) | 12–25 | 8–18 | 65 TF fp16 tensor peak, power-capped at 70 W; typical transformer utilization 20–40% |
| P100 (Kaggle) | 6–10 | 4–8 | No tensor cores (18.7 TF fp16 peak). **Avoid for this model** |
| L4 (Colab Pro) | 40–60 | 25–45 | 121 TF dense fp16 peak |
| A100 (Colab Pro+) | 120–180 | 80–140 | 312 TF dense; FlashAttention-2 available |

### 4.2 Job costs

[Estimate] Measured FLOPs × throughput range. The program totals in §4.3 add 20% for data loading and restarts.

| Job | PFLOPs | T4 | L4 | A100 | P100 |
|---|---|---|---|---|---|
| Cache 4-fps surprise, DHF1K train+val (with flips) | 38 | 0.4–0.9 h | 0.2–0.3 h | ~0.1 h | 1.1–1.8 h |
| Cache surprise, Hollywood-2 train+test | 27 | 0.3–0.6 h | 0.1–0.2 h | <0.1 h | 0.8–1.3 h |
| **Stage-1 run** (30 epochs × 4,320 clips, surprise cached) | 276 | **4.3–9.6 h** | 1.7–3.1 h | 0.5–1.0 h | 9.6–19 h |
| Ablation run (15 epochs) | 138 | 2.1–4.8 h | 0.9–1.5 h | 0.3–0.5 h | 4.8–9.6 h |
| **Stage-2 LoRA run** (15 epochs) | 205 | **3.2–7.1 h** | 1.3–2.3 h | 0.4–0.7 h | 7.1–14 h |
| Trainable-predictor control A1c (15 epochs) | 217 | 3.4–7.5 h | 1.3–2.4 h | 0.4–0.8 h | 7.5–15 h |
| Hollywood-2 fine-tune (10 epochs, 2 clips per shot) | 132 | 2.0–4.6 h | 0.8–1.5 h | 0.3–0.5 h | 4.6–9.2 h |
| Evaluate DHF1K-val (S2) | 9 | 0.1–0.2 h | <0.1 h | <0.1 h | 0.3–0.4 h |
| Evaluate Hollywood-2 test (S2, short shots padded) | 42 | 0.5–1.0 h | 0.2–0.3 h | ~0.1 h | 1.2–1.9 h |

Dataset sizes: DHF1K is exact (582,605 frames / 1,000 videos). Hollywood-2 is approximate (about 220k test frames, 3,559 test and 3,100 train shots).

### 4.3 Programs

| Program | Contents | T4-hours | Calendar time (free tiers) |
|---|---|---|---|
| **Phase 0** (signal de-risking, §6) | Causal predictor adaptation (1.2–2.7 h) + H4 zero-shot for raw, contextual and adapted surprise × 3 frame rates (under 1 h) + pipeline checks | **3–6** | 1–2 days |
| **MVE** (decision point; includes Phase 0) | Pipeline + calibration; Phase 0; B1 vs. SURGE vs. two controls (random predictor, feature difference) at 15 epochs, 1 seed | **16–32** | About 1 week of Kaggle quota |
| **Paper-grade** | Above, plus B1/SURGE × 3 seeds × 30 epochs; about 18 ablation runs; A1c; LoRA B1+/SURGE+ × 2 seeds; Hollywood-2/UCF fine-tunes and evaluations; DIEM zero-shot and cut analyses; efficiency benchmarks | **≈150–325** | 5–11 weeks at about 30 GPU-h/week; **about 50–130 h on L4 or 18–54 h on A100** |

Kaggle as reported in 2026 secondary sources: about 30 GPU-h/week (floating), 12-h GPU sessions, dual T4 or P100, 4 CPU cores and 29 GB RAM. Check your account. Data loading is not the bottleneck: decoding about 16 JPEGs at 256×256 per clip on 4 cores is far faster than the GPU step.

### 4.4 Calibrate before committing

`tools/calibrate_surge_compute.py` takes about 5 minutes on Kaggle or Colab. It times the frozen passes, the predictor, and a decoder training step under fp16 autocast + SDPA. It prints achieved TFLOPS, peak memory, and hours for a Stage-1 run, surprise caching, and per-frame inference. Its decoder reproduces the measured 7.04M params and 0.476 TFLOPs, checked on CPU. Bf16 is used only on sm80+ GPUs, because a T4 would emulate it slowly.

---

## 5. Can a positive result be detected?

For a paired comparison over N videos, the smallest difference detectable at α = 0.05 with 80% power is about 2.8 · σ_d / √N, where σ_d is the standard deviation of per-video metric differences between the two models.

| σ_d (CC) [Assumption] | DHF1K-val (N = 100) | Hollywood-2 test (N = 884) |
|---|---|---|
| 0.01 | 0.0028 | 0.0009 |
| 0.02 | 0.0056 | 0.0019 |
| 0.03 | 0.0084 | 0.0028 |

Training randomness adds variance on top: with 3 seeds per model, seed-level noise of about 0.003 CC alone makes differences below about 0.007 hard to claim.

Plausible effect sizes for an added cue on a strong backbone are of the same order: LoRA vs. frozen gave +0.011 CC in ViSAGE, and its expert ensemble added about +0.005. So:

1. Measure σ_d from the first B1 seed pair and redo this table.
2. Use Hollywood-2 for the primary H1 test.
3. Report effect sizes with confidence intervals.
4. Lean on the more sensitive stratified tests (attention-shift frames) and the zero-shot H4 test.

---

## 6. Revised plan and decision rules

1. **Day 0–1: calibrate and stage data.** Run the calibration script; download DHF1K; set up streaming Hollywood-2 preprocessing.
2. **Phase 0, de-risk the signal (about 3–6 T4-hours in total).**
   - **(a) Causal predictor adaptation.** Fine-tune *only* the predictor, self-supervised, with future-frame masks on unlabeled DHF1K *training* videos at 4 fps (5 epochs ≈ 27k windows ≈ 78 PFLOPs ≈ 1.2–2.7 T4-hours). Then repeat the §2 diagnostics. You want error that grows with horizon and maps that are less atypicality-dominated.
   - **(b) H4 zero-shot on DHF1K-val** for raw surprise, contextual surprise (true past vs. 4 shuffled pasts), and adapted surprise, at 4, 8 and 15 fps. Score against fixations (NSS, CC, information gain over the center prior) and against pixel motion, latent feature change, and token atypicality. Under 1 T4-hour, or one night on the laptop CPU.
   - **Go:** some surprise variant beats *both* motion and atypicality on NSS, with positive information gain.
   - **Kill:** all variants are ≤ atypicality or motion. SURGE's premise then fails. Write the negative result up as an analysis finding, and pivot to ECHO (memory) or SYNC, reusing the same frozen V-JEPA 2 + decoder baseline (B1), which is useful either way.
3. **Step 2, MVE training (16–32 T4-hours including Phase 0).** B1 vs. SURGE (S2 sampling, best surprise variant) vs. a random-predictor control vs. a feature-difference control, 15 epochs, 1 seed.
   - **Go:** SURGE − B1 ≥ +0.005 CC on DHF1K-val, larger than both controls, *and* the same sign on a 200-video Hollywood-2 subset.
4. **Step 3, paper program (about 150–325 T4-hours).** Seeds, the full ablation table, LoRA, cross-dataset tests. Put the DHF1K test submission last (one per week).

## 7. Changes to the design report

- **Signal:** the claim that the released predictor's error is an expectation-violation map is **not supported** by the real-weights check (§2). Phase 0 (predictor adaptation + H4) becomes a hard gate. Correction: the HF leakage issue changes the map less than claimed (r 0.78–0.95 between leaky and leak-free); the procedure still matters conceptually, but it is not the main threat.
- **Sampling:** single-rate stride-2 (S1) becomes dual-rate (S2), with surprise at 4 fps (in distribution) and features at about 15 fps.
- **Training:** LoRA (Stage 2) moves into the main plan. It costs 3.17 TFLOPs per clip vs. 2.13 for Stage 1, with 1.57M extra params.
- **Center-prior gate:** demoted from a headline component to a DIEM ablation.
- **Aspect ratio:** add a squash vs. rectangular-RoPE comparison (2 short runs).
- **First experiment:** H4 is now the go/no-go step that precedes all training.
- **Compute:** the report's "8–15 GPU-h per run" becomes **4.3–9.6 T4-h** (surprise cached), and "120–180 GPU-h" becomes **150–325 T4-h**, because the program now counts LoRA, fine-tunes and more ablations.

## Sources

- V-JEPA 2 checkpoint and license: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
- V-JEPA 2 pretraining configs (fps, masks): https://github.com/facebookresearch/vjepa2/tree/main/configs/train/vitl16
- HF implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vjepa2/modeling_vjepa2.py
- Garrido et al. surprise code (causal masks, clip-level error): https://github.com/facebookresearch/jepa-intuitive-physics
- ViSAGE frozen vs. LoRA ablation: https://arxiv.org/abs/2604.08613
- DHF1K README (splits, test submission limit, Hollywood-2 size): https://github.com/wenguanwang/DHF1K
- Kaggle limits (secondary, 2026): https://gpuperhour.com/blog/free-cloud-gpus-and-credits · https://lilys.ai/en/notes/google-tpu-20251128/free-kaggle-gpu-29gb-4-cpu-update
