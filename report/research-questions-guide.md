# The five research questions of the SURGE-D thesis: a study guide

This guide teaches the reasoning behind each research question (RQ) of the thesis: what each question means, how the
thesis tries to answer it, how each experiment produces evidence, and how strong that evidence is. It is based only on
the thesis, `Paper/msc-thesis/main.pdf` (127 pages, built at commit `5c4f9bd`). Section (§), table, figure, equation
(Eq.) and algorithm numbers refer to that PDF. Where the thesis does not address something, the guide says so.

**How to read it.** Part 0 explains the task, the model, the data, the metrics and the decision rules. Every question
depends on them, so read it first. Parts 1 to 5 take one research question each, in a fixed order: the question, the
approach, the experiments, the results in plain language, and a critical examination. Part 6 shows how the questions
connect. Part 7 lets you test yourself.

**Labels for the strength of evidence** (used in every critical examination):

| Label | Meaning |
|---|---|
| **Demonstrated** | Shown by a test whose rule was written before the experiment, with an interval that excludes zero or that establishes equivalence, or shown by a direct measurement |
| **Suggested** | Consistent evidence that did not meet its rule, or that is descriptive, rests on one seed, or was added after the result |
| **Claimed** | An interpretation the thesis offers to explain its results; the guide judges how well the evidence supports it |
| **Unproven** | Not tested, or explicitly left open by the thesis |

Passages marked *Hypothetical example* illustrate a concept. They are not results of the thesis.

---

## Contents

- [Part 0. The toolkit you need before the questions](#part-0-the-toolkit-you-need-before-the-questions)
- [Part 1. RQ1: Is the frozen-encoder model more accurate than SalFoM?](#part-1-rq1-is-the-frozen-encoder-model-more-accurate-than-salfom)
- [Part 2. RQ2: Does the predictor's error show where people look?](#part-2-rq2-does-the-predictors-error-show-where-people-look)
- [Part 3. RQ3: Which design factors matter when the encoder is frozen?](#part-3-rq3-which-design-factors-matter-when-the-encoder-is-frozen)
- [Part 4. RQ4: Does the approach transfer, and does sound help?](#part-4-rq4-does-the-approach-transfer-and-does-sound-help)
- [Part 5. RQ5: What does it cost, and can the cost be cut further?](#part-5-rq5-what-does-it-cost-and-can-the-cost-be-cut-further)
- [Part 6. How the five questions fit together](#part-6-how-the-five-questions-fit-together)
- [Part 7. Check your understanding](#part-7-check-your-understanding)

---

## Part 0. The toolkit you need before the questions

### 0.1 The task (§4.2)

When people watch a video in an eye-tracking experiment, the tracker records where each observer looks in every frame.
Video saliency prediction asks a model to predict, for each frame, where those gaze points will fall.

For each frame *n*, a dataset provides two things:
- a binary **fixation map** *F_n*, which marks the pixels that some observer fixated;
- a continuous **fixation density** *Y_n*, which is the same fixations blurred with a Gaussian.

The model must output a **probability distribution** over pixels: non-negative values that sum to one (Eq. 4.1). The
thesis treats saliency as a distribution, not as an arbitrary "importance" score, because this makes the training loss
and several metrics well defined. It follows Kümmerer et al. in this.

> *Hypothetical example.* Seventeen observers (the number in DHF1K) watch a dog running along a beach. Most fixations
> land on the dog, and a few on a person throwing a ball. A good prediction puts most of its probability on the dog,
> some on the person and almost none on the sand.

### 0.2 The model, SURGE-D (§4.4, Fig. 4.1)

Think of SURGE-D as two parts. The first is a large, already trained "eye" that is never changed. The second is a
small "interpreter" that is trained to turn what the eye sees into a saliency map.

**The frozen encoder (§4.4.1).** The "eye" is a V-JEPA 2 video transformer.
- **How V-JEPA learns.** V-JEPA is trained without labels. Parts of a clip are hidden, and a separate **predictor**
  network learns to predict the internal representations of the hidden parts, not their pixels, from the visible
  parts (Eq. 2.1, §2.3.4). The predictor is kept with the released model, and RQ2 uses it.
- **Frozen.** The encoder's weights are never updated, and no gradient flows into it.
- **Two sizes.** ViT-L has 303.9 million parameters and ViT-g has 1,012.2 million.
- **Tokens.** A clip of 16 frames at 256×256 pixels becomes 8×16×16 = 2,048 tokens. Each token covers 2 frames of
  16×16 pixels, a block called a *tubelet*.

**The trained decoder.**
- **Four taps.** The decoder reads four intermediate layers of the encoder: blocks 6, 12, 18 and 24 of ViT-L, or 10,
  20, 30 and 40 of ViT-g. Several levels are read because dense prediction tasks benefit from features at several
  levels of abstraction.
- **Feature pyramid (Eq. 4.3).** The four taps become feature maps at 64×64, 32×32 and twice 16×16 resolution.
- **Top-down (2+1)D decoder (Eqs. 4.4–4.5).** The decoder refines the maps from coarse to fine. Each block applies a
  convolution within each frame, then a convolution across time. Splitting a 3D convolution this way mixes space and
  time more cheaply.
- **Deep supervision.** Two auxiliary heads, at coarser levels, are trained on the same targets. They are dropped at
  inference.
- **Centre prior (Eq. 4.6).** People tend to look at the centre of the screen. A learned mixture of eight Gaussians
  (41 parameters), the same for every input, adds this bias.
- **Output (Eq. 4.7).** A softmax over the 64×64 positions turns each frame's scores into a probability distribution.

Only the decoder and the prior are trained: 6.99 million parameters with ViT-L and 7.39 million with ViT-g.

**Windows and time (§4.2).** The model reads 16 frames, sampling every second frame. That is about one second of video
at 30 frames per second. It predicts maps for all 16 sampled positions but keeps only the central 8 (Eq. 4.2). Every
kept prediction therefore has at least about 0.27 s of context on each side. This makes the model **non-causal**: the
map of a frame depends on frames that come after it. A **causal variant**, trained as a control, predicts each frame
from the 16 sampled frames that end at it.

**Inference (§4.7, Algorithm 4.1).** Windows advance by 16 original frames, so their central positions tile the video
without gaps. One forward pass therefore yields usable maps for 16 original frames. Each map is then:
1. interpolated in time between sampled frames;
2. upsampled to the native resolution;
3. blurred with a Gaussian of width σ, which is chosen on the holdout (§0.3);
4. normalised to sum to one.

**Training (§4.6).** The loss (Eq. 4.19) combines three terms, plus the same terms for the auxiliary heads:
- the Kullback–Leibler divergence from the ground-truth density;
- one minus the correlation coefficient;
- a small normalised-scanpath-saliency term.

Each model trains for three epochs, and the best epoch is chosen on the holdout.

**The name.** SURGE stands for *SURprise-Gated Encoder*–decoder. In the original design, the predictor's error was to
modulate the decoder. The suffix D marks the reduced design that remained after the surprise branch failed its gate
(§4.4). That history is RQ2.

### 0.3 Data: where models are chosen and where they are tested (§4.1, §4.3, Table 4.3)

The thesis keeps three kinds of data apart:
1. **Training data.** For DHF1K, 540 of the 600 official training videos.
2. **Holdout.** The other 60 training videos. All model selection happens here: the best epoch, the output blur and
   every gate.
3. **Test data.** The official test annotations of DHF1K are withheld, so its 100-video official validation set serves
   as the test set. Each model is scored on it once, after all of that model's settings are fixed. Table A.4 lists
   every use of this set.

Predictions for the 300 official test videos were produced, but their scores were not available when the thesis was
written, so no test-set result is claimed (§5.5.1).

The other datasets:
- **UCF Sports.** Gaze was recorded while observers performed action recognition.
- **Hollywood-2 with the openly licensed gaze of Vig et al.** This is a different ground truth from the standard one,
  so the results are not compared with published Hollywood-2 numbers.
- **Six audio-visual datasets** (DIEM, Coutrot1, Coutrot2, ETMD, AVAD and SumMe), with the folds released with STAViS.
- **SALICON** static images, used for one ablation only.

### 0.4 The metrics (§4.8.1)

Every metric is computed per frame at the native resolution, then averaged within each video and then over videos. The
video is the unit of analysis.

| Metric | Question it asks | Better | What to keep in mind |
|---|---|---|---|
| **CC** (correlation coefficient) | Does the predicted map rise and fall where the true density does? | higher | The metric of every decision rule. It is the Pearson correlation between the two maps. |
| **NSS** (normalised scanpath saliency) | How high is the predicted map at the fixated pixels, in standard deviations above its mean? | higher | NSS 3.4 means fixated pixels lie 3.4 standard deviations above the map's average. A **negative** NSS means people looked where the map is *below* average. |
| **SIM** (similarity) | How much do the two distributions overlap? | higher | A histogram intersection; 1 means identical. |
| **KLD** (Kullback–Leibler divergence) | How much information is lost when the prediction stands in for the truth? | lower | Punishes near-zero probability where people look. |
| **AUC-J** (AUC-Judd) | If pixels are ranked by predicted saliency, are fixated pixels ranked above the rest? | higher | 0.5 is chance and 1 is perfect. |
| **sAUC** (shuffled AUC) | The same, but the "negatives" are fixations from *other* videos | higher | Does not reward simply predicting the centre. Its sampling is specific to the thesis, so it is compared only within the thesis. |

RQ2 also uses **information gain** in bits per fixation (Eq. 4.10).

**How far the evaluator was checked (§5.2).**
- CC and NSS were checked against reference code.
- CC, NSS, SIM and KLD agreed per video with a second, independently written evaluator, and AUC-J agreed in the mean.
- sAUC differed slightly between the two, because its negatives are drawn at random.

> *Hypothetical example: the scale of CC differences.* The square of a correlation is roughly the share of variation
> that one map explains in the other. A CC of 0.564 corresponds to about 32 % and a CC of 0.603 to about 36 %, so a
> gain of 0.039 CC captures roughly a seventh more of the pattern of where gaze concentrates. This reading is
> approximate, because CC is averaged over frames, but it shows that differences of a few hundredths are substantial
> and differences of a few thousandths are small.

> *Hypothetical example: bits per fixation.* An information gain of 0.33 bits per fixation means that, in geometric
> mean, the model gives the fixated location 2^0.33 ≈ 1.26 times the probability that the baseline gives it, that is
> 26 % more. A gain of 0.07 bits means about 5 % more.

The thesis does not measure an inter-observer ceiling on DHF1K. It therefore does not say how far these scores are from
the best that can be achieved.

### 0.5 How the thesis decides (§4.1, §4.8.2, Tables 4.1 and 4.2)

**Rules written in advance.** For each uncertain component, a decision rule was written into a dated research log
before the experiment and applied unchanged afterwards. A rule states what is measured, on which data, and which
threshold decides the outcome. This guards against two familiar sources of optimism: choosing among variants after
seeing the test results, and moving a threshold after seeing which side of it a result falls on.

Three points about the rules:
- **Follow-ups.** Rules marked † and ‡ were added in two follow-ups after the first draft, but still before the runs
  they govern.
- **One exception.** The tie margin of 0.01 for published numbers was introduced after the first audio-visual
  comparison had been seen (§4.1).
- **Choices without a rule.** Some choices were fixed without any test: the decoder architecture, the tapped layers,
  the loss weights, the squashing of the input and the learning rates (§4.1).

**The paired bootstrap over videos (§4.8.2).** To compare two models on the same test videos, the thesis computes the
difference on each video. It then resamples the videos with replacement 10,000 times and recomputes the mean
difference each time. The middle 95 % of these resampled means is the **95 % interval**. A model is called better only
if this interval excludes zero. Videos are resampled rather than frames because frames of the same video are strongly
correlated.

> *Hypothetical example.* Model A beats model B by 0.04 CC on average over 100 videos, but loses on 30 of them.
> Resampling asks: had a different set of 100 videos been drawn from the same population, how different might the
> average be? If nearly all resampled averages are above zero, the advantage is not an accident of which videos
> happened to be in the test set.

**Seeds.** Training the same model again with another random seed changes holdout CC by about 0.003 (§4.1, "Unit of
analysis"); the seed changes the initialisation and the data order. A single-run difference smaller than this could be
luck. Most component tests therefore use three seeds and a **two-level bootstrap**, which resamples first the seeds and
then the videos (§4.8.2).

**The three-seed rule (Table 4.1, "Ablations, three seeds").** It has three possible outcomes, with a margin of
0.003 CC:
- **Effect:** the 95 % interval excludes zero *and* the mean difference is at least 0.003.
- **Equivalence:** the 90 % interval lies entirely within ±0.003. The data then rule out any difference large enough
  to matter.
- **Inconclusive:** neither of the above.

> *Hypothetical example.* Three outcomes for removing a component:
> - a 95 % interval of [−0.004, −0.001] with a mean of −0.0025 excludes zero, but the mean is below the margin, so the
>   outcome is *inconclusive*;
> - a 90 % interval of [−0.002, +0.001] is *equivalent*;
> - a 95 % interval of [−0.006, +0.004] is *inconclusive*: the experiment cannot tell.

"Inconclusive" does not mean "no effect". Only "equivalence" shows that something does not matter.

**Gates and screens.**
- A **gate** is a yes-or-no decision taken before a larger investment. Gate G1 chose the encoder, and gate G2 decided
  whether the surprise branch would be built.
- A **screen** runs an extension with one seed and continues only if it clears a threshold (Table 4.2).

**Published numbers.** Results copied from other papers were computed with other evaluation code. They cannot be
paired with the thesis's results, so margins below 0.01 CC count as ties (Table 4.1).

---

## Part 1. RQ1: Is the frozen-encoder model more accurate than SalFoM?

### 1.1 The question

> **RQ1.** Can a frozen self-supervised video encoder of the V-JEPA family, combined with a lightweight
> spatio-temporal decoder, predict human fixations in video more accurately than the closest foundation-model approach,
> SalFoM, when both are evaluated under an identical protocol? (§1.4)

**In plain words.** Take a large video model that learned about video on its own, leave it untouched, and train only a
small add-on to read saliency out of it. Can that predict where people look better than SalFoM, which retrains its
whole large model on saliency data? And is the comparison fair?

**The uncertainty it addresses.** The strongest recent saliency models draw their accuracy from large pretrained
encoders, which they adapt to small eye-tracking datasets (§1.3). SalFoM fine-tunes a 300-million-parameter encoder on
a few hundred training videos and runs it once for every output frame (§1.2). Two things were unknown. Is this
adaptation necessary? And does an encoder trained by predicting features, such as V-JEPA, already contain enough
information about where people look? The research gap (§3.7) notes that such encoders had not been evaluated for this
task in the peer-reviewed studies the thesis could consult.

A second uncertainty concerns measurement. Published comparisons rely on numbers copied from other papers, computed with
different code and rarely with uncertainty estimates (§1.2, §3.7). "Better than SalFoM" means something only if both
models are scored the same way.

**Why it matters.** The motivation (§1.2) gives three reasons:
- **Practical.** Fine-tuning a very large encoder on few videos risks overfitting, and running it once per frame makes
  inference expensive. A frozen encoder with a decoder that predicts many frames at once avoids both problems, *if* its
  features are good enough.
- **Scientific.** It tests whether a representation learned without labels, by predicting features, captures what
  drives human attention.
- **Methodological.** Small differences between models are hard to interpret in this field. A re-scored, paired
  comparison gives a more reliable answer.

**What the thesis tries to show.** That SURGE-D is more accurate than SalFoM, and that the advantage survives four
challenges:
1. the choice of training seed;
2. giving SalFoM its best possible post-processing;
3. removing SURGE-D's access to future frames;
4. attributing the gain to the right cause, the encoder rather than freezing as such.

**Assumptions behind the question.**
1. **SalFoM is the right comparator.** It is the closest prior work: the first video saliency model built on a video
   foundation encoder, and the best on the DHF1K test set among the methods it compared with. Its released weights also
   allow it to be re-scored (§5.5.1).
2. **The DHF1K validation set can serve as a test set,** provided nothing is selected on it (§4.1).
3. **The six metrics, with CC as the decision metric, capture "predicting human fixations".**
4. **Running SalFoM's released code reproduces SalFoM as its authors intended.** The wrapper program is not part of
   the thesis code (§5.2).
5. **"Identical protocol" means identical evaluation:** the same videos, frames, evaluator and post-processing options.
   It does not mean identical training.

**What would count as a convincing answer.** A paired comparison on the same test videos, scored by the same validated
evaluator, in which the 95 % interval of the advantage excludes zero. The advantage should hold for every seed, against
the baseline's best blur, and when SURGE-D is restricted to past frames. Ideally it would also hold on the official
test set and against more than one re-scored baseline.

### 1.2 How the thesis approaches it

**Approach.** Build the simplest model that could work: a frozen encoder with a small decoder (§4.4). Choose the encoder
by a gate (G1), train with three seeds, compare with SalFoM on the same videos with the same validated evaluator, and
then challenge the comparison with checks.

**Why this design.** The decoder's design follows from three considerations (§4.4):
1. The encoder carries most of the representational capacity, so the decoder can be small, about 2 % of the size of
   ViT-L.
2. The (2+1)D blocks add temporal mixing at every scale cheaply.
3. Predicting many frames per pass spreads the encoder's cost over many output frames (RQ5).

Freezing was not simply assumed: low-rank adaptation (LoRA) was tested as the alternative (RQ3). The multi-level taps
and the feature pyramid follow standard practice for dense prediction (§4.4.2).

**Hypothesis.** V-JEPA's features, learned by predicting representations rather than pixels, are reported to work well
with a frozen backbone (Bardes et al., cited in §2.3.4 and §4.4.1). A light decoder trained on 540 videos should
therefore be able to read saliency from them, and should overfit less than full fine-tuning.

### 1.3 The experiments

#### Experiment 1.0: Validating the evaluator (§5.2)

- **Objective.** Make sure the thesis's own metric code reproduces reference scores, so that every "same evaluator"
  comparison can be trusted.
- **Setup.** SalFoM's DHF1K validation maps, produced with its released code and weights, for which scores from the
  evaluation code of the TMFI-Net authors were available. A second, independently written evaluator that follows the
  MIT benchmark conventions served as a further reference.
- **Variables.** The evaluator varies; the maps are fixed.
- **Rule.** Re-scoring must agree within 0.002 for CC and SIM and within 0.01 for NSS and KLD (Table 4.1).
- **Results.** CC was 0.5639 against a reference of 0.5642, and NSS 3.3264 against 3.3268. Against the second
  evaluator, CC, NSS, SIM and KLD agreed per video to within 5×10⁻⁵, and AUC-J agreed in the mean. Two deviations from
  the reference procedures were found and corrected before any SURGE-D model was scored: the AUC-J tie-breaking and the
  sampling of sAUC negatives.
- **Interpretation.** The measuring instrument is reliable. SalFoM's re-scored values (CC 0.5639, NSS 3.326 and so on)
  become the baseline numbers for the whole thesis.
- **What it means for RQ1.** This is the foundation of every "same evaluator" claim.

#### Experiment 1.1: Gate G1, the choice of encoder (§5.3, Table 5.2, Fig. 5.1)

- **Objective.** Choose the encoder before the main program. The proposal had chosen V-JEPA 2.1, which was designed to
  have denser features.
- **Setup.** Three frozen ViT-L encoders with the same decoder, trained for ten epochs with seed 0. V-JEPA 2 and V-JEPA
  2.1 ran at 256×256 pixels. UMT-L, the encoder family of SalFoM, ran at 224×224, because its position table is fixed
  to that size.
- **Variables.** Only the encoder changes. The measure is holdout CC (with NSS, SIM and KLD) at the best epoch.
- **Rule.** The highest holdout CC wins, but V-JEPA 2.1 is kept if it is within 0.005 of the winner.
- **Results.** V-JEPA 2 scored 0.5782, V-JEPA 2.1 0.5692 (0.009 lower) and UMT-L 0.5464 (0.032 lower). V-JEPA 2 led on
  all four metrics.
- **The learning curves (Fig. 5.1).** For every encoder, holdout CC peaked at the second or third epoch and then fell,
  while training CC kept rising.
- **Interpretation.** V-JEPA 2 was adopted. The gap to UMT-L is several times the seed spread measured later (0.003).
  The gap between the two V-JEPA variants could partly be seed noise, since each was run with one seed. The overfitting
  curves changed the whole training schedule to three epochs.
- **What it means for RQ1.** It fixes the encoder used for RQ1, and it already hints at the answer: with the same
  decoder, a frozen V-JEPA 2 clearly beats a frozen UMT-L.

#### Experiment 1.2: The gate for the larger encoder (§5.6, "Encoder scale"; Table 5.14)

- **Objective.** Decide whether to use ViT-g.
- **Rule.** ViT-g is adopted if its holdout CC is at least 0.5833, the ViT-L three-seed mean plus 0.005.
- **Result.** 0.5866 with seed 0, so ViT-g was adopted. It was higher in every seed: 0.5866, 0.5867 and 0.5829
  against 0.5772, 0.5801 and 0.5776 for ViT-L.
- **What it means for RQ1.** RQ1 is answered with both encoders, and ViT-g provides the headline model.

#### Experiment 1.3: The main comparison with SalFoM (§5.5.1, Tables 5.5–5.6, Fig. 5.2)

- **Objective.** Find out whether SURGE-D is more accurate than SalFoM on the same unseen videos.
- **Setup.** The DHF1K validation set: 100 videos and 60,273 frames. SURGE-D with ViT-L and with ViT-g, each as a single
  model (seed 0) and as a three-seed ensemble. SalFoM as released, re-scored by the same evaluator.
- **Variables.** The model changes; the videos, frames and evaluator are fixed. The measures are the six metrics and
  their paired bootstrap differences.
- **Rule** ("Claim over a baseline"). The 95 % interval of the CC difference over test videos must lie above zero.
- **Scores (Table 5.5).** SalFoM 0.5639; ViT-L single 0.5836 and ensemble 0.5910; ViT-g single 0.6003 and ensemble
  0.6029.
- **Paired differences (Table 5.6).**
  - ViT-g single minus SalFoM: +0.036 [+0.026, +0.048]. ViT-g ensemble: +0.039 [+0.029, +0.050].
  - ViT-L single: +0.020 [+0.010, +0.031], although its NSS difference was not significant. ViT-L ensemble: +0.027
    [+0.017, +0.038].
  - The ViT-g single model and ensemble were better on all six metrics.
- **Per video (Fig. 5.2).** The ViT-g ensemble has the higher CC on 79 of the 100 videos.
- **Interpretation.** The advantage is about ten times the seed spread of 0.003, consistent across metrics, and spread
  over most videos rather than carried by a few.
- **What it means for RQ1.** It gives the first-level answer: yes, on the DHF1K validation set, against SalFoM as
  released. The highest published validation score of another method (lower block of Table 5.5) is 0.046 to 0.049
  below the ViT-g models, but it was computed with other code and cannot enter a paired test.

#### Experiment 1.4: Checks of the comparison (§5.5.1, Table 5.7)

These checks were specified after the first draft (rules marked †) and challenge the comparison in three ways.

1. **The seed check: could the reported model be a lucky seed?** Every seed of both encoders beat SalFoM as released,
   with intervals above zero. Seed 0 was an average ViT-L seed, and it was the best of the three ViT-g seeds (0.6003
   against a mean of 0.5969).
2. **The blur check: was SalFoM handicapped by post-processing?** SURGE-D's output blur was tuned on its holdout,
   whereas SalFoM's maps had been scored as released.
   - Blurring SalFoM's maps raises its CC to 0.574, at σ = 16 pixels. That value was chosen *on the test set itself*,
     so it favours SalFoM: it is an upper bound for the baseline.
   - **Rule.** A claim over SalFoM stands only if it also holds against this blurred baseline.
   - **Result.** Every ViT-g model and the ViT-L ensemble remain better. The ViT-g ensemble leads by +0.029 [+0.019,
     +0.039], and the ViT-g single model and ensemble are better on all six metrics (Table A.3). The single ViT-L models
     are not reliably better: two of the three intervals include zero.
3. **The encoder check: is the gain due to freezing, or to the encoder?** A frozen UMT-L with the same decoder reaches
   0.545. That is below SalFoM (−0.019 [−0.029, −0.010]) and 0.039 below the frozen V-JEPA 2 ViT-L. Freezing as such
   does not explain the gain; the V-JEPA 2 encoder does. This check used one seed, 224×224 input and ten epochs.

**What it means for RQ1.** The answer now depends on encoder size. With ViT-g, every model is better. At SalFoM's
encoder size (ViT-L), only the ensemble is shown to be more accurate.

#### Experiment 1.5: The causal control, or is it the future frames? (§5.5.2, Table 5.8)

- **Objective.** SURGE-D sees about 0.27 s of frames after the predicted one; SalFoM does not. This experiment removes
  that difference.
- **Setup.** The causal variant (§4.2), trained with the ViT-g recipe and three seeds, then repeated with ViT-L. It
  needs one forward pass per frame, so the validation set was scored on every tenth frame (6,076 frames). SalFoM and the
  offline models were re-scored on the same frames.
- **Rule** (†, written before the run). SURGE-D is better than SalFoM under an identical protocol only if the 95 %
  interval of the CC difference between the causal three-seed ensemble and SalFoM *at its best blur* lies above zero.
- **Results with ViT-g.** The causal ensemble reached 0.6021, +0.029 [+0.019, +0.040] over SalFoM at its best blur.
  Every causal seed was also better, by +0.018 to +0.025, and the ensemble was better on all six metrics.
- **Results with ViT-L.** The causal ensemble led by +0.017 [+0.007, +0.028]. The single causal ViT-L models were not
  reliably better.
- **Causal against offline.** The causal ensembles differed from the offline ensembles by +0.0016 [−0.0015, +0.0046]
  with ViT-g and +0.0012 [−0.0022, +0.0047] with ViT-L. The future frames therefore add no measurable accuracy.
- **Interpretation.** The advantage does not come from seeing the future. What the causal model gives up is efficiency,
  since it needs one pass per output frame (RQ5).
- **What it means for RQ1.** This is the comparison the thesis calls "under an identical protocol": the same videos,
  frames and evaluator, no future frames for either model, and SalFoM at its best blur.

#### Experiment 1.6: The official test set (§5.5.1)

Maps for the 300 official test videos were produced with the ViT-g ensemble. An export check reproduced the validation
scores to within 10⁻⁴ CC. The dataset authors compute the test scores on request, and the scores were not available
when the thesis was written. **No test-set result is claimed.**

### 1.4 The reasoning chain

1. **Question.** RQ1, as quoted above.
2. **Hypothesis.** Frozen V-JEPA features are enough, and a light multi-frame decoder overfits less than full
   fine-tuning.
3. **Method.** SURGE-D, with the encoder chosen by G1, trained with three seeds.
4. **Experiments.** Evaluator validation; the same-evaluator paired comparison; checks of seed, blur and encoder; the
   causal control.
5. **Results.** +0.036 to +0.039 CC with ViT-g, robust to the seed, the best blur and causality. A frozen UMT-L scores
   below SalFoM.
6. **Interpretation.** The advantage is real on this test set, does not depend on future frames, and comes from the
   V-JEPA 2 encoder rather than from freezing.
7. **Answer.** "With the larger encoder, yes" (§6.1). At SalFoM's encoder size, only the ensemble is better. All of this
   holds on the validation set.

### 1.5 The results in plain language

- **On 100 unseen DHF1K videos, SURGE-D's maps match where people look better than SalFoM's.** This holds on every
  metric, in about four of every five videos, and for every training seed.
- **The advantage is about +0.03 to +0.04 CC.** By the rough r² reading of §0.4, that captures about a seventh more of
  the pattern of where gaze concentrates. Published DHF1K results are often separated by less than 0.01: the top three
  in the lower block of Table 5.5 are 0.554, 0.553 and 0.549. By the field's standards this is a large margin, although
  those published numbers come from other code.
- **The advantage survives giving SalFoM its best possible blur.** That blur was chosen with knowledge of the test set's
  answers.
- **Hiding the future frames from SURGE-D does not measurably change its accuracy.**
- **A frozen version of SalFoM's own encoder, with SURGE-D's decoder, is *worse* than SalFoM.** So "freeze the encoder
  and add a light decoder" is not a recipe that works with any encoder. It works with V-JEPA 2.

### 1.6 Critical examination

**Do the experiments answer the question?** Mostly, yes. The paired, same-evaluator design is stronger than the field's
usual practice, and the three checks and the causal control target the most obvious alternative explanations.

**Is the comparison fair?** Some aspects favour SalFoM: its best blur was chosen on the test set. Others remain
unequal:
- **Encoder size.** ViT-g has more than three times as many encoder parameters as SalFoM. At equal size (ViT-L), only
  the ensemble is reliably better. The question says "of the V-JEPA family", so ViT-g qualifies, but the strongest
  version of "a frozen encoder beats a fine-tuned one" holds only for the larger encoder or for ensembles.
- **Ensembles against a single SalFoM.** The ViT-L result rests on three models against one.
- **Identical evaluation, not identical training.** The input resolution (256×256 against 224×224), the handling of the
  training data and the selection of checkpoints all differ. The thesis says that "identical protocol" refers to
  evaluation, but a reader may assume more.
- **How SalFoM's checkpoint was selected.** The thesis states that this is not documented. If the weights were selected
  on the DHF1K validation set, SalFoM's scores there are optimistic, which would make the reported differences
  conservative (§5.2, §5.9).

> *Note from outside the thesis.* The project's claims ledger (`report/claims-ledger.md`, row S6) records that SalFoM's
> paper does state early stopping on the validation set, and flags the thesis's "not documented" wording for
> correction. If so, the comparison is conservative on this point.

**Are the metrics adequate?** CC decides, and the blur is tuned for CC, which slightly favours CC over the other
metrics (§5.9). All six metrics are reported, and for ViT-g they agree.

**Alternative explanations.**
- **Overlap with pre-training data.** V-JEPA 2 was pre-trained on large collections of internet video, and DHF1K comes
  from YouTube. Overlap between the pre-training videos and the test videos cannot be ruled out; the same is true for
  SalFoM (§5.9).
- **Use of the validation set.** No reported model or setting was selected on it, but it did inform which maps were
  prepared for the official test set (Table A.4).
- **The objective as the cause.** The thesis attributes the gain to "the V-JEPA 2 encoder rather than freezing as
  such". It does not claim that the feature-prediction *objective* is the cause, and the evidence would not support
  that: V-JEPA 2.1, also trained by feature prediction, scored lower, and UMT-L differs in objective, training data and
  input size.

**Demonstrated, suggested, claimed, unproven.**
- **Demonstrated.** On DHF1K validation, every ViT-g model and both ensembles are more accurate than SalFoM, both as
  released and at its best blur. The causal ViT-g and ViT-L ensembles beat SalFoM at its best blur under identical
  evaluation. Future frames add no measurable accuracy (Tables 5.6–5.8).
- **Suggested.** That the advantage carries over to the official test set. Only an export check supports this.
- **Claimed.** "The gain comes from the V-JEPA 2 encoder rather than from freezing as such". It is supported by one
  seed of a frozen UMT-L at 224×224: reasonable, but not a controlled test of the cause.
- **Unproven.** The score on the official test set; superiority over other re-scored baselines; that a frozen encoder
  beats a fine-tuned one in general. Full fine-tuning was never tested (§5.6).

**What would strengthen the answer.** The official test-set scores (pending), re-scoring other released baselines with
the same evaluator, and more than one seed for the frozen UMT-L check.

---

## Part 2. RQ2: Does the predictor's error show where people look?

### 2.1 The question

> **RQ2.** Does the prediction error of the encoder's pretrained latent predictor ("predictive surprise") carry
> information about where people look, beyond that carried by simple appearance and motion cues? (§1.4)

**In plain words.** V-JEPA 2 comes with a predictor that guesses what a hidden part of a video looks like internally.
Where it guesses badly, something unexpected may be happening. Do people look at those places? And do the predictor's
errors tell us anything about gaze that simple cues (motion, unusual-looking regions, the centre of the screen) do not?

**The uncertainty it addresses.** The idea that the unexpected draws attention has strong support from models built on
hand-crafted features, notably the Bayesian surprise of Itti and Baldi (§1.2, §3.6). It had not been tested with the
learned predictor of a modern video model (§3.7). The original project was built on this hypothesis: the predictor's
error was meant to gate the decoder, hence "SURprise-Gated" (§4.4).

**Why it matters.**
- **If true,** prediction error would be a principled attention cue learned without any gaze data (§1.2).
- **If false,** understanding why clarifies what such predictors actually learn (§1.2).
- **For the thesis,** it decided the architecture: gate G2 determined whether the surprise branch would be built.

**What the thesis tries to find out.**
1. On their own, do the error maps point to fixations better than simple cues? (Gate G2)
2. Combined with the simple cues, does the error add information, and in which direction? (The analysis of
   incremental information)
3. Inside the model, can the decoder use the error? (The predictability-gated decoder)

**Assumptions behind the question.**
1. **The error stands for surprise.** The difference between predicted and actual representations (Eq. 4.8) is taken
   to be a reasonable measure of surprise. The thesis notes that it is a distance in feature space, not a change of
   belief as in Bayesian surprise (§5.9, construct validity).
2. **The released predictor behaves as it did in pre-training** when queried in these ways. This assumption needed two
   corrections, one for the mask token and one for the shape of the query (§2.2).
3. **These are the cues to beat.** Frame difference (motion), token atypicality (how unusual a region's features are)
   and the centre prior are the right simple cues.
4. **No training is needed.** Scoring the error maps zero-shot is a fair test of the cue itself.

**What would count as a convincing answer.**
- **For "yes".** The error maps beat the simple cues on their own, or they add information with a **positive** weight
  once the cues are controlled for. Either result should hold across readouts and frame rates.
- **For "no".** The failure should hold for different readouts (both forecasting and pre-training-like queries), for
  the correctly trained mask token, at more than one frame rate and after controlling for the centre bias. In addition,
  a decoder given the error should not benefit from it.

### 2.2 How the thesis approaches it (§4.5.1)

**The readout: "raw surprise".**
1. Take a 16-frame window. The first 8 frames are the context and the last 8 the target.
2. Encode the context on its own, so that no information leaks in from the target frames.
3. Let the predictor predict the target tokens from the context tokens.
4. Compare the predictions with the target tokens taken from an encoding of the whole window.

The raw surprise of a token is the mean absolute difference between the layer-normalised prediction and the actual
representation (Eq. 4.8).

**Contextual surprise.** The same error minus the error obtained when the context comes from another video. Only the
part of the error that depends on the true past remains.

**Reference cues.**
- **Token atypicality:** how far each target token lies from the average token of its frame, an "unusualness" measure.
- **Frame difference:** the absolute grey-level change between consecutive frames, a simple motion measure.
- **The empirical centre prior:** reported for scale.

**Scoring.** All maps were upsampled to 640×360, blurred with σ = 16 pixels and scored against the fixations of the 60
holdout videos, at about 4 frames per second (1,720 frames) and about 15 (7,904 frames). No training is involved.

**Two corrections made along the way.**
1. **The mask token.** The predictor marks each position it must predict with a learned mask token, Δ_y in Eq. 2.1.
   The released checkpoints store ten such tokens, but only the first was trained. The model-loading library reads the
   second by default, which is an untrained zero vector. G2's first run therefore used an untrained token. After this
   was found, every map was recomputed with the trained token and the gate was scored again by its own rule (§4.5.1).
2. **The shape of the query: the masked-tube error.** V-JEPA 2's pre-training masks spatial blocks across the whole clip
   in time. The predictor was therefore trained to fill in hidden regions, never to forecast the future. A second
   readout follows that geometry:
   - blocks of 4×4 tokens are hidden in all eight tubelets, a quarter of the tokens per pass;
   - four passes ensure that every token is predicted once.

   This readout matches pre-training in the *shape* of the mask but not in its *extent* (§4.5.1).

**Incremental information (Eqs. 4.9–4.11).** A map can be weak on its own and still add information to other cues, so
the thesis fits a **log-linear model** of where fixations fall (Eq. 4.9):
- The frame is divided into 45×80 cells. The probability that a fixation falls in a cell is proportional to
  exp(w₀·log(centre prior) + Σ wₖ·cueₖ).
- The model's quality is its information gain over the centre prior, in bits per fixation (Eq. 4.10).
- The **incremental information** U(e) of an error map is the gain with the error added to the cues {frame difference,
  token atypicality}, minus the gain without it (Eq. 4.11).
- The weights are fitted by maximum likelihood and **cross-fitted**: fitted on one half of the videos and scored on the
  other half, so the model cannot memorise the videos it is scored on.
- The **sign of the fitted weight** tells whether a higher error means more fixations or fewer.

### 2.3 The experiments

#### Experiment 2.1: Gate G2 (§5.4, Table 5.3)

- **Objective.** Decide whether to build the surprise branch.
- **Setup.** Zero-shot maps on the 60 holdout videos at about 4 and 15 frames per second: raw and contextual surprise,
  plus the masked-tube error (outside the gate), each with the trained and the untrained mask token.
- **Variables.** The type of map, the mask token and the frame rate. The measures are NSS, CC, and the information gain
  of an equal mixture of map and centre prior over the prior alone.
- **Rule (Table 4.1).** The branch is enabled only if a surprise map beats *both* token atypicality and frame difference
  on NSS, *and* has a positive information gain over the centre prior.
- **Results with the trained token.**
  - Raw surprise: NSS −0.665 [−0.763, −0.566] at 4 fps and −0.877 at 15 fps.
  - Contextual surprise: −0.239 and −0.660.
  - Masked-tube error: −0.958 and −0.942.
  - Reference maps: token atypicality +0.454 and +0.650; frame difference +0.810 and +0.803; centre prior +1.464 and
    +1.479.
- **Further results.** The information gain was negative for every map. Every paired difference between an error map
  and either simple cue lies below zero at both rates (Tables A.9–A.10). The untrained token gave the same picture, and
  correcting the token made the scores *more* negative.
- **Interpretation.** A negative NSS means that, on average, people looked at regions the predictor found *easier* than
  average to predict. The error maps are worse than the simple cues and point away from fixations.
- **What it means for RQ2.** The gate fails with both tokens. The surprise branch was not built, and the model became
  SURGE-D.

#### Experiment 2.2: Ruling out the "wrong query" explanation (§5.4)

- **Objective.** Test whether the negative result arises because forecasting the second half of a clip is a task the
  predictor never saw.
- **Setup.** The masked-tube error, a pre-training-like query, against the forecasting error, both with the trained
  token.
- **Result.** The masked-tube error is the *most* negative map in the table, with NSS −0.958 at 4 fps.
- **What it means for RQ2.** The form of the query does not explain the sign, with the reservation that this readout
  hides less of the clip than pre-training did. With the mask-token correction, two explanations of the first negative
  result are set aside.

#### Experiment 2.3: Incremental information (§5.4, Table 5.4)

- **Objective.** A stand-alone score can be negative merely because a map is low at the centre of the frame, where
  fixations concentrate. Does the error still add information once the centre prior, motion and atypicality are in the
  model?
- **Setup.** The cross-fitted log-linear model (Eq. 4.9) on the 60 holdout videos, with intervals from a bootstrap over
  videos.
- **Rule (‡).** A map adds information if the 95 % interval of U lies above zero. The gain is negligible if the 90 %
  interval lies within ±0.01 bits. Otherwise the outcome is inconclusive. The primary comparisons, fixed in advance, are
  the raw surprise and the masked-tube error at about 4 fps with the trained token.
- **Results.**
  - Raw surprise: U = +0.0693 [+0.0346, +0.1065] bits per fixation, with fitted weights of −0.34 and −0.40 in the two
    halves.
  - Masked-tube error: U = +0.0619 [+0.0320, +0.0926], with weights −0.34 and −0.41.
  - For scale, the two simple cues together add +0.3300 bits over the centre prior. Removing frame difference costs
    0.1190 bits, and removing token atypicality costs 0.1845.
- **Interpretation.** The error does carry information beyond the simple cues, but with a negative sign. A cell whose
  error lies one standard deviation above its frame's mean receives about 30 % *less* probability, other cues being
  equal. The amount is modest: about a fifth of what the two simple cues add. In the 2^IG reading of §0.4, that is
  about 5 % more probability on fixated cells, against about 26 % for the simple cues.
- **What it means for RQ2.** Taken literally, the question ("does it carry information beyond simple cues?") is
  answered **yes**. But the information is not the kind the surprise hypothesis predicts.

#### Experiment 2.4: A preliminary diagnostic (§5.4)

This diagnostic ran before the gate, on eleven clips without gaze data and with the untrained token. Two findings:
- **The predictor uses the context.** Replacing a clip's past with another clip's past raised the mean error, from
  0.644 to 0.731.
- **The error reflects how unusual a region is, whatever the past.** The error map correlated at about 0.60 with token
  atypicality and at about 0.62 with the error computed from the *wrong* past. It correlated at about −0.11 with a
  central Gaussian, so it was lower at the centre.

This motivated the analysis of incremental information. That analysis shows the error is more than a stand-in for
atypicality or the centre bias, at least as Eq. 4.9 represents them.

#### Experiment 2.5: Can the decoder use the error? The predictability gate (§4.5.7, §5.7, Table 5.17)

- **Objective.** G2 and the incremental analysis compare the error with fixations directly. Neither shows whether a
  decoder, which already sees the encoder's features, can exploit it.
- **Setup.**
  - The masked-tube error (trained token), standardised within each tubelet, enters each pyramid level through a
    zero-initialised modulation (Eq. 4.15). The model therefore starts as SURGE-D.
  - ViT-g, with a seed-0 screen.
  - As a control, the trained model is also scored with error maps rotated by 180°, which keep their statistics but no
    longer line up with the frame, and with zero maps.
- **Rule (Table 4.2).** Seeds 1 and 2 run only if seed 0 reaches a holdout CC of at least 0.5896. The gate is adopted
  only if the three-seed rule finds an effect both over SURGE-D and over the same model given the rotated maps.
- **Results.** The best epoch reached 0.5857, against 0.5866 for SURGE-D (−0.0009), so the screen was not passed. With
  its own, rotated or zero error maps, the model scored 0.5857, 0.5856 and 0.5856.
- **Interpretation.** The decoder learned to ignore the error. Two readings remain (§5.7): the encoder's features may
  already contain that information, or three epochs on 540 videos may be too few for the modulation to learn to use
  it. The experiment does not distinguish between them.
- **What it means for RQ2.** Whatever the error adds to the simple cues, the trained decoder did not use it.

### 2.4 The reasoning chain

1. **Question.** RQ2, as quoted above.
2. **Hypothesis.** A high prediction error marks surprising events, and surprising events attract gaze (Bayesian
   surprise).
3. **Method.** Read out the error of the released predictor in two ways; compare it with simple cues on its own and in
   combination; give it to the decoder.
4. **Experiments.** G2 with both tokens, the masked-tube readout, the analysis of incremental information, and the
   predictability gate.
5. **Results.** The error maps are anti-correlated with fixations and worse than simple cues. They add a little
   information, but with a negative weight. The decoder ignores them.
6. **Interpretation.** The error carries information, but observers look *less* often where it is high.
7. **Answer.** Yes, the error carries information beyond simple cues, but not in the way the surprise hypothesis
   predicts. For this predictor and these readouts, the error does not behave as a signal that attracts gaze (§6.1).

### 2.5 The results in plain language

- **People's eyes avoid the hot spots of the predictor's "surprise heatmap".**
- **The thesis checked three alternative explanations.** The result is not a software bug (the mask token), not a
  mismatched query (forecasting instead of filling in), and not the centre bias in disguise.
- **The heatmap helps a little, in reverse.** Once motion and "unusualness" are known, it still improves the prediction
  of where people look, by indicating where they will *not* look.
- **A trained model given the heatmap simply ignored it.**

> *Hypothetical example.* A person walks past a hedge. The predictor may find the hedge's fine leaves hard to predict
> (a high error) and the person's face easy (a low error). People look at the face, so the error map is "hot" where
> nobody looks and anti-correlates with gaze. This illustrates *one possible* mechanism; the thesis does not test it.
> In its words, "What the error measures that draws gaze away was not established" (§5.4). Its future work proposes
> measuring texture, motion and object boundaries in high-error regions (§6.3).

### 2.6 Critical examination

**Does the evidence answer the question?** For this predictor and these readouts, yes, and robustly. Two mask tokens,
two readouts, two frame rates, stand-alone and combined analyses, and a test inside the decoder all agree in sign.

**Strengths of the design.**
- The gate was written in advance and, after the mask-token error, re-scored by its own rule.
- Cross-fitting stops the log-linear weights from fitting the very videos they are scored on.
- The rotated-map control in the predictability gate separates use of the error's content from use of its
  statistics.

**Limits and alternative explanations.**
- **The construct.** The error is a distance between representations, not a change of belief; Itti and Baldi's
  Bayesian surprise is a different quantity (§5.9). The thesis therefore states that the result does not refute
  surprise-based accounts of attention (§6.1).
- **The predictor.** It was trained to fill in hidden regions, not to forecast. A predictor trained to forecast might
  behave differently (§5.4); this was not tested.
- **The readout's extent.** The masked-tube readout hides a quarter of the tokens per pass, less than pre-training
  did.
- **The data.** The holdout has only 60 videos, and the secondary comparisons received no correction for multiplicity,
  as §5.4 states.
- **The model's form.** The centre bias and the cues enter the log-linear model in a particular parametric form, so
  "beyond simple cues" holds *as Eq. 4.9 represents them*. Interactions and non-linear effects are not modelled.
- **The decoder test.** It used one seed and cannot tell redundancy apart from too little training.
- **The mechanism.** What the error measures is not established.

**Demonstrated, suggested, claimed, unproven.**
- **Demonstrated.** The error maps of the released V-JEPA 2 predictor are anti-correlated with fixations and worse
  than frame difference and token atypicality, with both tokens and at both frame rates. Combined with the centre prior
  and the simple cues, the raw and masked-tube errors add information with a negative weight (primary comparisons,
  ‡ rule).
- **Suggested.** The decoder cannot use the error (one seed; screen only).
- **Claimed.** "It does not behave as a surprise signal that attracts gaze". This is supported for this predictor and
  these readouts, and the thesis itself limits the claim to them.
- **Unproven.** Why the error repels gaze, and whether other measures of surprise (forecasting predictors, changes of
  belief) would attract it.

---

## Part 3. RQ3: Which design factors matter when the encoder is frozen?

### 3.1 The question

> **RQ3.** Which design factors determine accuracy when the encoder is frozen? (§1.4)

**In plain words.** Once a large pretrained encoder is fixed, which of the remaining choices actually change accuracy,
and which are just habit? The choices include which encoder and how large, whether to adapt it slightly, which decoder
components, what training data and how many frames.

**The uncertainty it addresses.** Saliency papers usually include many components and report only the final
configuration, with few controlled tests and rarely with uncertainty estimates (§1.3, §3.7). With a strong frozen
encoder, it is unclear which design choices beyond the encoder still matter (§1.3). The thesis also notes that progress
in the field has tracked progress in the encoder, while decoders have converged on variations of multi-scale fusion
(§3.7).

**Why it matters.**
- **It shows where effort pays off.** If commonly included components (the centre prior, deep supervision) do nothing
  measurable, models can be simpler. If the encoder dominates, progress depends on better encoders rather than on
  decoder engineering.
- **It explains RQ1's answer.**

**What the thesis tries to find out.** For each factor, whether it produces an effect, is equivalent to its absence, or
cannot be decided, together with an account of the overall pattern.

**Assumptions behind the question.**
1. Factors can be tested one at a time against a fixed reference, without strong interactions between them.
2. The 60-video holdout, drawn from the training distribution, is representative enough to rank variants.
3. A margin of 0.003 CC, about the seed spread, is the smallest difference worth calling an effect.
4. Results obtained with ViT-L, where most component ablations were run, carry over to ViT-g.
5. The factors tested are the ones that matter. The decoder architecture, the taps, the loss weights, the squashing of
   the input and the learning rates were fixed without a test (§4.1).

**What would count as a convincing answer.** For each factor, a paired comparison over several seeds whose outcome is
either an effect or equivalence. Inconclusive outcomes leave a factor's role open. A convincing *pattern* needs enough
decided factors on both sides.

### 3.2 How the thesis approaches it

Each ablation (removing something) or addition is run under a rule (Tables 4.1–4.2), on the DHF1K holdout or, where
stated, on the validation set.
- **Seeds.** Early tests used one seed. After the first draft, the four component ablations were repeated with three
  seeds and judged by the three-seed rule (§5.6).
- **The learning curve** tests the explanation the thesis offers for the overall pattern.
- **Five extensions,** tested after the main evaluation, examine further additions (§5.7).

### 3.3 The experiments

The factors fall into five groups.

#### A. The encoder

**Experiment 3.1: Pre-training (G1 and the UMT-L check; details in Experiments 1.1 and 1.4).**
- **Setup.** The same decoder, a frozen encoder of ViT-L size.
- **Holdout results** (one seed, Table 5.2): V-JEPA 2 0.5782, then V-JEPA 2.1 0.5692, then UMT-L 0.5464.
- **Validation results** (one seed, Table 5.7): frozen UMT-L scored 0.545, 0.039 below the frozen V-JEPA 2 ViT-L
  (0.5836).
- **Meaning.** The choice of pretrained encoder moves accuracy by a few hundredths, the largest effects in the thesis.
- **Caveats.**
  - Each encoder was run with one seed.
  - V-JEPA 2.1 was used below the resolution at which it was finally trained, 384×384.
  - The decoder's four taps may already supply the local detail that V-JEPA 2's last layer lacks.

  The thesis notes that these two explanations of the V-JEPA ranking were not separated (§5.3).

**Experiment 3.2: Scale (§5.6; Tables 5.6 and 5.14).**
- **Setup.** ViT-g against ViT-L, with everything else fixed.
- **Rule.** Adopted if the seed-0 holdout CC is at least 0.5833.
- **Holdout results.** ViT-g was higher in all three seeds, by +0.007 on average.
- **Validation results.** Single models +0.017 [+0.012, +0.022]; ensembles +0.012 [+0.008, +0.016], better on all
  six metrics.
- **Other datasets.**
  - Hollywood-2 (Vig gaze): better on all six metrics, +0.0046 [+0.0032, +0.0060].
  - Audio-visual datasets: significantly better on four of the six.
  - UCF Sports: *not* significantly different, +0.0019 [−0.0071, +0.0125].
- **Meaning.** A larger frozen encoder helps. The effect is robust on DHF1K and less clear on the small, task-driven
  UCF Sports.

**Experiment 3.3: Adaptation with LoRA (§4.5.2, §5.6, Table 5.14).**
- **What LoRA is.** It adds small trainable low-rank updates to frozen weight matrices (Eq. 4.12). Here the updates had
  rank r = α = 16 and were applied to the attention projections of the last twelve blocks of ViT-L: 1.57 million
  parameters.
- **Setup.** Each LoRA model started from the frozen model of the same seed and trained for three more epochs.
- **Rule.** Adopted if the three-seed mean gain is at least +0.003 and LoRA wins in at least two of three seeds.
- **Result.** LoRA lost in all three seeds (−0.0029, −0.0010 and −0.0037; mean −0.0025), and its best epoch was always
  the first.
- **Meaning.** This form of adaptation did not help. The thesis offers two compatible explanations, neither tested
  (§5.6):
  - with 540 videos, the adaptation adds more variance than signal;
  - adaptation degrades features that already suit the task (Kumar et al.).

#### B. Decoder components (three seeds, ViT-L)

**Experiment 3.4: The centre prior and deep supervision (Table 5.14).**
- **Setup.** Remove the learned centre prior, or remove the auxiliary heads, and apply the three-seed rule.
- **Results.** −0.0010 [−0.0026, +0.0005] and −0.0010 [−0.0023, +0.0003]. Both 90 % intervals lie within ±0.003, so
  both outcomes are **equivalent**.
- **Meaning.** These commonly included components make no difference that matters. The thesis suggests, without
  testing it, that the encoder and the convolutional decoder can already represent the centre bias, for instance
  through the image borders, which the squashed input keeps at fixed positions (§5.6).

> *Why "equivalent" is a strong result here.* An inconclusive result would say only "we could not detect a difference".
> Equivalence says "we can rule out any difference larger than 0.003 in either direction". It is the only way the
> thesis can *positively show* that a component does not matter.

**Experiment 3.5: Co-training on static images (§4.5.4, Table 5.14).**
- **Setup.** Add 10,000 SALICON images to each epoch, each image repeated over the 16 input frames.
- **Result.** +0.0009 [−0.0039, +0.0058], with a 90 % interval of −0.0031 to +0.0049, so the outcome is
  **inconclusive**. The gain of +0.0028 seen with seed 0 did not recur with seeds 1 and 2.
- **Meaning.** No measurable benefit, but a small effect in either direction cannot be ruled out. Static images had
  helped models of the CNN era (§5.6).

#### C. Time and data

**Experiment 3.6: A temporal stride of four (Table 5.14).**
- **Setup.** Sample every fourth frame (about 7.5 frames per second, a 2-second window) instead of every second frame.
- **Result.** +0.0031, +0.0037 and +0.0020 by seed; mean +0.0029 [−0.0009, +0.0068], so **inconclusive**.
- **Meaning.** Outside the outcomes of the rule, the 90 % interval (−0.0003 to +0.0062) excludes a loss as large as
  the margin: the cheaper stride was at least not measurably worse. This matters for RQ5, because the stride halves the
  cost.

**Experiment 3.7: 32-frame clips (Table 5.14).**
- **Setup.** ViT-g, seed 0. The gate was 0.5904, the 16-frame three-seed mean plus 0.005.
- **Result.** 0.5902, which missed the gate by 0.0002, so the longer clips were not pursued.
- **Meaning.** The model was still improving at its last epoch, and only one seed was run. The thesis calls the result
  **inconclusive rather than negative** (§5.6).

**Experiment 3.8: 600 instead of 540 training videos (validation set; Tables 5.14 and A.7).**
- **Rule.** The 600-video models replace the 540-video models if the ensemble's interval of the CC difference lies
  above zero and it is worse on at most two of the six metrics.
- **Result.** +0.0015 [−0.0006, +0.0037], so the larger set was not adopted. The other metrics moved in both
  directions.

**Experiment 3.9: The learning curve (§5.6, "Training-set size"; Table 5.15, Fig. 5.4).**
- **Objective.** Test the explanation for the overall pattern: if the amount of training data limits accuracy, halving
  the data should hurt.
- **Setup.**
  - The frozen ViT-L model, trained on nested random subsets of 270 and 135 of the 540 videos, with three seeds. The
    subsets were redrawn for each seed.
  - The number of clips per video was raised so that every model takes the same 1,620 optimisation steps.
  - The models were scored on the holdout in the same way as the ablations.
- **Rule (‡).** Accuracy at 540 videos is limited by data if halving to 270 is an effect with a negative sign, and not
  limited if halving is equivalent. Otherwise the outcome is inconclusive.
- **Results.**
  - The mean CC was 0.5783 with 540 videos, 0.5729 with 270 and 0.5645 with 135, and it fell in every seed.
  - The primary comparison, 270 against 540, gave −0.0054 [−0.0112, +0.0011]: **inconclusive**. Its 90 % interval
    (−0.0101 to −0.0001) extends beyond the margin, so it is not equivalent either.
  - The comparisons 135 against 270 (−0.0084) and 135 against 540 (−0.0138) were effects.
  - Descriptively, the gap between training and holdout accuracy widened as the data shrank.
- **Meaning.** The amount of data matters between 135 and 270 videos. Whether it still limits the model at 540 is
  neither established nor refuted. The curve cannot extend beyond 540, because DHF1K has no further annotated training
  videos apart from the 60 of the holdout.

#### D. Ensembles and post-processing

**Experiment 3.10: Ensembles and output blur (§5.6; Table A.8, Fig. 5.5).**
- **Ensembles on the validation set** (three seeds against seed 0): +0.0074 [+0.0050, +0.0099] with ViT-L and +0.0025
  [+0.0002, +0.0049] with ViT-g. NSS, KLD and AUC-J also improved, but sAUC fell slightly for both encoders (−0.0022 and
  −0.0029, intervals below zero). On Hollywood-2 (Vig gaze), ensembling *lowered* NSS and SIM.
- **Output blur.** The best blur is broad: holdout CC varied by at most 0.003 between σ = 0 and σ = 12.
- **Meaning.** Ensembles give a small, reliable gain in CC. It is smaller for the stronger encoder, which "leaves less
  variance for the ensemble to remove". It comes at a small cost on sAUC, the metric that does not reward the centre
  bias.

#### E. Extensions after the main evaluation (§5.7, Table 5.17)

**Experiment 3.11: A second frozen encoder (§4.5.5, Table 5.18).**
- **Setup.** A second, "expert" encoder, either DINOv2-L (an image encoder) or UMT-L, fused into each pyramid level
  through a zero-initialised gate (Eq. 4.14).
- **Rule.**
  - **Screen.** Seed 0 must reach a holdout CC of at least 0.5896.
  - **First look.** Adopted if the three-seed rule finds an effect.
  - **Second look.** Decided after the first look and fixed before its runs. Over six seeds, adopted if the mean gain
    is at least +0.003 and its 97.5 % interval lies above zero. The stricter level keeps the false-positive rate of
    the two looks together below 7.5 %.
- **Results.**
  - First look: DINOv2-L +0.0041 [−0.0007, +0.0094] and UMT-L +0.0047 [−0.0001, +0.0103], both inconclusive.
  - Second look, UMT-L over six seeds: positive in every seed (+0.0017 to +0.0070), mean +0.0038, 97.5 % interval
    [−0.0009, +0.0093]. Not adopted.
- **Meaning.** The gain is consistent but not established. It is comparable to the spread between SURGE-D's own seeds
  (0.5829 to 0.5881), and it would cost a second encoder of 0.3 billion parameters on every clip.

**Experiment 3.12: A decoder state across windows (§4.5.9, Table 5.20, Fig. 5.6).**
- **Objective.** Windows overlap by half because their outer positions lack context on one side. Can a memory of the
  previous window make all 16 positions usable, so that windows no longer need to overlap?
- **Mechanism.** A stream buffer, taken from MoViNets. Each of the decoder's six temporal convolutions reads the last
  time step of the previous window instead of zero padding. Group normalisation, computed per time step, confines the
  effect to the first time steps of a window. Training uses pairs of adjacent windows, with gradients flowing through
  the buffer.
- **Setup.** Three decoders trained in the same runs on one stream of encoder features, so their comparisons are paired
  by construction:
  - the standard decoder: loss on the central 8 positions, scored with overlapping windows;
  - the standard decoder with the loss on all 16 positions, scored without overlap and without a buffer;
  - the decoder with the state: loss on all 16 positions, no overlap, buffer.
- **Rule (Table 4.2).**
  - **Screen.** At least 0.5836, that is SURGE-D's seed-0 score minus 0.003.
  - **Success.** The three-seed rule finds equivalence with SURGE-D or an effect above zero.
  - **Secondary.** Comparisons within the runs.
- **Results.**
  - **Against SURGE-D.** 0.5898 against 0.5854, a difference of +0.0044 [−0.0009, +0.0098]: **inconclusive**, so not
    adopted.
  - **Within the runs.** +0.0051 [+0.0022, +0.0083] over the standard decoder, an effect (+0.0050, +0.0051 and
    +0.0052 by seed).
  - **By position.** Without memory, SURGE-D's first two positions in a non-overlapping window scored 0.073 below
    positions 8 and 9. With the trained state, they scored 0.018 below. A post hoc analysis that leaves out each video's
    first window reduces the gap to 0.003 with the state, against 0.041 and 0.066 for the two standard decoders.
- **Meaning.** How the decoder handles time across windows matters within a run. Whether the stateful decoder beats the
  separately trained SURGE-D could not be decided. Part of the reason is that the comparison with separately trained
  models was too noisy ever to find equivalence (§5.7, §5.9).

The **predictability gate** (Experiment 2.5) also belongs to this group: an addition that the decoder ignored.

### 3.4 The pattern, and the thesis's explanation of it

The thesis's answer (§5.8, §6.1) is that **the encoder dominates.**
- **Gains.** The encoder's pre-training and its scale produced the only gains that were large relative to seed noise
  on DHF1K. Seed ensembles gave a small one.
- **No difference.** The centre prior and deep supervision are equivalent to their absence.
- **Losses.** LoRA lost in every seed.
- **No established gain.** Static-image co-training, 32-frame clips and 11 % more videos.
- **Gains in every seed, not established.** The stride of four improved with every seed without reaching the margin. A
  second encoder improved holdout CC in every seed without the gain being established. The decoder state helped within
  its own runs only.

**The proposed explanation (§5.8), which is a hypothesis.** With a strong frozen encoder, the small number of training
videos limits further progress. The decoder overfits within three epochs (Fig. 5.1). Additions that would increase its
effective capacity do not help, whereas better features do, and the one addition that moved accuracy consistently was
another encoder's features. The learning curve tested this explanation and left it open.

### 3.5 The reasoning chain

1. **Question.** RQ3, as quoted above.
2. **Hypotheses.** Each component was expected to help. Examples are the proposal's expectations that LoRA, V-JEPA
   2.1's denser features and static images would improve accuracy.
3. **Method.** One factor at a time, under rules written in advance, with three seeds for the components and with
   equivalence testing.
4. **Experiments.** G1, scale, LoRA, the component ablations, the tests of time and data, the learning curve,
   ensembles and the extensions.
5. **Results.** Large effects only from the choice and the scale of the encoder. Equivalence for two decoder
   components. Several inconclusive outcomes. LoRA negative.
6. **Interpretation.** With a frozen encoder, accuracy is set mainly by the encoder; decoder additions add little; the
   amount of data may be the bottleneck.
7. **Answer.** "With a frozen encoder, the encoder itself dominated" (§6.1). The explanation that data limit accuracy
   remains open.

### 3.6 The results in plain language

- **Changing which frozen encoder is used moves accuracy by a few hundredths.** Most other choices move it by a few
  thousandths or less.
- **Two components that most saliency models include can be removed without a measurable difference:** a centre-bias
  prior and extra supervision of intermediate layers.
- **Lightly adapting the encoder (LoRA) made it slightly worse in every seed.**
- **The amount of training data mattered at the small end.** Doubling it from 135 to 270 videos clearly helped. From
  270 to 540, the gain pointed the same way but was not decisive.

> *Hypothetical example: why one-seed results mislead.* If results differ between seeds by about 0.003 CC, a single run
> that comes out 0.003 better could be luck. SALICON co-training is the thesis's own case: +0.0028 with seed 0, a gain
> that did not recur with seeds 1 and 2.

### 3.7 Critical examination

**Strengths.**
- Three seeds and a two-level bootstrap for the component ablations.
- The equivalence outcome, rarely used in the field, lets the thesis show that a component does *not* matter.
- Comparisons are paired within each seed.
- The learning curve was designed to test the thesis's own explanation.

**Weaknesses and limits.**
- **What was not tested.** The decoder's own architecture (the four taps, the 3D pyramid, the (2+1)D blocks), the loss
  weights, squashing against cropping and the learning rates were fixed without a test (§4.1). RQ3's answer covers only
  the factors that were tested. For the same reason, the thesis cannot say that its decoder design is responsible for
  the accuracy. The abstract's sentence "Design choices were decided by rules written in advance" is broader than §4.1,
  which lists the untested choices. "Every tested component was decided by a rule written in advance" would be
  accurate.
- **Single seeds where they matter.** The encoder comparison (G1), the clip-length test, the audio experiments and the
  screens of the extensions used one seed each (§1.7). The ranking of V-JEPA 2 over V-JEPA 2.1, a difference of 0.009,
  is within what a single seed could plausibly produce. The gap to UMT-L is not.
- **Confounds.**
  - V-JEPA 2.1 was used below its training resolution, and UMT-L ran at 224×224.
  - The LoRA models trained three more epochs on top of their frozen references, because §4.5.2 starts them from the
    frozen model. "LoRA lost" therefore mixes adaptation with extra training. The thesis does not discuss this point.
- **Ablations at ViT-L.** The component ablations used ViT-L, but the final model uses ViT-g. Equivalence at ViT-L does
  not guarantee equivalence at ViT-g.
- **The size of the holdout.** With 60 videos, the intervals are roughly ±0.004 to ±0.005 wide, wider than the margin
  of 0.003. Many outcomes are inconclusive because the experiments cannot resolve differences of that size. The decoder
  state shows the extreme case: its comparison with separately trained models could not have returned "equivalent"
  (§5.7).
- **Margins set by judgement.** The margins of 0.003 and 0.005 were chosen before seed variation had been measured
  (§4.1, §5.9).

**Demonstrated, suggested, claimed, unproven.**
- **Demonstrated.**
  - ViT-g beats ViT-L on DHF1K (three seeds and validation intervals) and on Hollywood-2 (Vig gaze).
  - The learned centre prior and deep supervision are equivalent to their absence within ±0.003 (ViT-L, three seeds).
  - LoRA, as tested, lost in every seed.
  - Halving the training data from 270 to 135 videos hurts.
  - Ensembles raise CC slightly.
- **Suggested.**
  - V-JEPA 2 is the best of the three encoders (one seed).
  - A second encoder helps a little (six seeds, not established).
  - The decoder state helps (within runs only).
  - The stride of four is at least not worse.
- **Claimed.**
  - "The encoder dominates" is well supported as a description of the factors that were tested.
  - "The small number of training videos limits further progress" is offered as one explanation and explicitly left
    open.
- **Unproven.**
  - Whether the decoder architecture matters.
  - Whether more than 540 training videos would help.
  - Whether adaptation would help ViT-g, or help with more data (§6.3).
  - Full fine-tuning, which was never tested.

---

## Part 4. RQ4: Does the approach transfer, and does sound help?

### 4.1 The question

> **RQ4.** How well does the approach transfer to eye-tracking datasets with other content and viewing tasks, and does
> adding the soundtrack improve it on audio-visual datasets? (§1.4)

**In plain words.** DHF1K records free viewing of YouTube videos. Does the model also work when people watch sports
while trying to recognise the action, or watch movie clips, or watch videos with sound? And when there is sound, does
giving the model the soundtrack help?

**The uncertainty it addresses.**
- **Overfitting to one benchmark.** A model trained on one dataset may exploit that dataset's particular content or
  viewing conditions.
- **The viewing task.** Gaze recorded during action recognition (UCF Sports, Hollywood-2) differs from free viewing
  (§5.5.3).
- **Sound.** On audio-visual datasets, models often include audio, but whether it helps is disputed (Agrawal et al.,
  cited in §3.5 and §5.8).

**Why it matters.** It concerns external validity. A saliency model is useful only if it works beyond the benchmark it
was trained on. And if sound helps, a purely visual model misses something; if not, audio modules add complexity for
nothing.

**What the thesis tries to find out.**
- How the model performs on eight further test sets, without training (zero-shot) and after brief fine-tuning.
- How it compares with the best published results.
- Whether two forms of the soundtrack improve it: a global audio description and maps of where the sound comes from.
  Maps of the active speaker were also planned but turned out impossible to build.

**Assumptions behind the question.**
1. Brief fine-tuning, three epochs starting from the DHF1K model (§4.6), is a fair way to transfer.
2. Published results computed with other code are comparable within a margin: differences below 0.01 CC count as ties.
3. The STAViS folds and preprocessed releases are the standard protocol for the audio-visual data.
4. If sound helps, the modules that were tested could exploit it.
5. The Vig gaze data for Hollywood-2 make a usable ground truth; their reliability was not measured.

**What would count as a convincing answer.**
- **For transfer.** Same-evaluator comparisons against re-scored competitors on each dataset, which are not available
  here. Failing that, margins clearly larger than the disagreement between published evaluations of the same model.
- **For audio.** A gain above a meaningful threshold, *and* better results with the matched soundtrack than with a
  wrong one, which shows that the sound itself is used.

### 4.2 How the thesis approaches it

- **Fine-tuning.** The DHF1K models (both encoders, three seeds, ensembles) are fine-tuned on each dataset with that
  dataset's own holdout, and each test set is scored once.
- **Baselines.** Zero-shot results are reported, and the fine-tuned results are compared with published peer-reviewed
  results under the tie rule.
- **Audio.** A fusion module (Eq. 4.13) is tested with a mismatched-soundtrack control. Then sound-source maps (§4.5.8)
  and speaker maps (§4.5.6) are tried.

### 4.3 The experiments

#### Experiment 4.1: UCF Sports (§5.5.3, Table 5.9)

- **Setup.** 103 training videos (93 for training and 10 for the holdout) and 47 test videos. Gaze was recorded during
  action recognition.
- **SURGE-D results.** Zero-shot, ViT-L reached CC 0.608. After fine-tuning, the ViT-g ensemble reached 0.735, the best
  NSS, SIM and AUC-J among the SURGE-D models.
- **Published results.** ViNet-E 0.744, ViNet-A 0.734, THTD-Net 0.711, TMFI-Net 0.707, SalFoM 0.631.
- **Comparison.**
  - The ViT-g ensemble ties ViNet-A (+0.001) and ViNet-E (−0.009, a tie under the rule).
  - Its NSS and SIM are below ViNet-E's, and its AUC-J is above.
  - ViT-g is not significantly better than ViT-L here: +0.0019 [−0.0071, +0.0125].
- **Interpretation.**
  - The jump from 0.608 zero-shot to 0.735 after fine-tuning confirms that task-driven gaze differs from free viewing.
  - SURGE-D is competitive: clearly above SalFoM (by about 0.10) and TMFI-Net, but not ahead of the best.
  - The ViNet models use encoders trained for action recognition or action localisation, which may suit this dataset
    (§5.5.3).

#### Experiment 4.2: Hollywood-2 with the Vig gaze (§5.5.3, Table 5.11)

- **Setup.** 884 test clips (257,733 scored frames). The ground truth was built from the recordings of Vig et al.
  (§4.3) and has at most five observers per frame. It is not comparable with published Hollywood-2 numbers.
- **Results.**
  - The ViT-g ensemble reached CC 0.688.
  - As single models, ViT-g beat ViT-L on all six metrics, +0.0046 [+0.0032, +0.0060].
  - Ensembling raised CC, KLD, AUC-J and sAUC, but *lowered* NSS and SIM for both encoders.
- **Interpretation.** With this many test clips, small differences are resolved precisely. The thesis attributes the
  fall in NSS and SIM to the sparse ground truth: averaging three models smooths the sharp peaks that those metrics
  reward (§5.5.3).

#### Experiment 4.3: Six audio-visual datasets, visual model only (§5.5.3; Tables 5.10 and A.12)

- **Setup.** The STAViS folds: three folds per dataset, and a single split for DIEM. Each fold was fine-tuned and
  selected on its own small holdout of 1 to 6 videos.
- **Effect of fine-tuning.** It improved every dataset, most of all Coutrot2 (+0.17 CC), whose videos show four people
  in a meeting.
- **Against the best verified peer-reviewed result on each dataset,** the ViT-g ensemble was:
  - higher by 0.018 to 0.028 on DIEM, Coutrot1, ETMD and SumMe;
  - tied on AVAD (+0.002);
  - lower on Coutrot2: −0.043 against the best visual-only model and −0.018 against the best audio-visual one.
- **Limits stated in §5.5.3.**
  - Published evaluations of the same model on these data disagree by a few hundredths. STAViS on Coutrot2, for
    instance, is reported at 0.734 in one paper and 0.652 in another, so margins of 0.02 to 0.03 are within that noise.
  - The comparison is restricted to peer-reviewed results. Some results that had not appeared in peer-reviewed venues
    are higher; against them, Coutrot1 and ETMD would also be ties.
  - It is also restricted to results whose tables could be accessed.

  In the thesis's words, "The table should consequently not be read as establishing a state of the art."

#### Experiment 4.4: Audio fusion (§4.5.3; §5.6, "Audio"; Table 5.16)

- **Objective.** Find out whether the soundtrack helps.
- **Setup.**
  - Audio features from the Audio Spectrogram Transformer (AST): one 768-dimensional vector about every 0.1 s,
    averaged per frame (§4.3).
  - The features enter three pyramid levels through feature-wise modulation, plus an audio-visual correspondence map
    (Eq. 4.13). The new weights start at zero, so training begins from the visual model.
  - ViT-L, seed 0, every fold of the six datasets.
  - **Control: "rotated audio".** Each test video is also scored with the soundtrack of another test video.
- **Rule.** Audio "helps" on a dataset only if the fold-mean CC gain is at least +0.010 *and* the matched soundtrack
  beats the rotated one by at least 0.005.
- **Results.** The largest change was +0.001 CC, so no dataset met the threshold. On Coutrot2, the wrong soundtrack
  cost 0.018: the model *had* learned to use the audio there. Yet with the right soundtrack, it only matched the visual
  model.
- **Interpretation.** In meeting scenes, the visual model already finds the faces from the image alone, and the global
  audio description adds nothing measurable. With five test videos per fold, the Coutrot2 observation is suggestive
  rather than conclusive (§5.6). The result agrees with Jain et al. and Agrawal et al. (§5.8).

#### Experiment 4.5: Sound-source localisation maps (§4.5.8; §5.7; Table 5.19)

- **Objective.** Tell the decoder *where* the sound comes from, instead of describing the soundtrack as a whole.
- **Setup.**
  - Maps from EZ-VSL, a frozen sound-source localisation model. Each map is a 7×7 grid of cosine similarities between
    visual features and 3 s of audio.
  - The maps enter the decoder through the modulation of Eq. 4.15. ViT-g, seed 0, on Coutrot2 first, where sound
    should matter most.
  - Control: maps computed from the soundtrack of the next test video.
- **Check before training.** Without gaze data, the maps mostly marked the people at the table. Maps computed from
  another conversation's soundtrack correlated 0.96 with the matched maps, and their peaks fell in the same cell in 53 %
  of frames.
- **Results.** Matched maps scored +0.0017 [+0.0001, +0.0036] over the visual model, and matched beat rotated by
  +0.0007 [−0.0002, +0.0021]. Both are far below the thresholds of 0.010 and 0.005, so the maps were not adopted and the
  other datasets were not run. Most of the small gain remained with the wrong soundtrack, so it does not come from the
  sound.
- **Interpretation.** The likely source of the small gain is the visual part of the localisation model, which marks
  people. This model hardly depended on the sound at all.

#### Experiment 4.6: Speaker maps (§4.5.6, §5.7)

- **Plan.** An active speaker detector (Light-ASD) would weight each face box by its probability of speaking. A face
  map without speaking scores was the planned control.
- **Check before training.** The released frames are 256×340 pixels, so faces are only about 12 to 15 pixels wide. On
  the first Coutrot2 video (35 s, four people), the detector marked speech in 1 % of the face frames and split the four
  people into 21 face tracks.
- **Outcome.** The maps would have been face maps, and the test could not have separated speaking from the mere
  presence of faces. The extension was dropped before training.

#### Experiment 4.7: The characteristic failure (§5.5.4, Fig. 5.3)

Five validation frames were chosen by a rule fixed before any map was inspected. In the video where SURGE-D loses most
to SalFoM (0663, a panel discussion), observers fixate one person, while SURGE-D spreads its prediction over several
faces. In conversations, gaze goes to the person speaking (Coutrot and Guyader, cited in §5.5.4), which an image-only
model cannot identify. The same limitation appears on Coutrot2.

### 4.4 The reasoning chain

1. **Question.** RQ4, as quoted above.
2. **Hypotheses.** Features that serve free viewing on DHF1K transfer with brief fine-tuning. On audio-visual data,
   sound adds information, especially about who is speaking.
3. **Method.** Fine-tuning per dataset; zero-shot baselines; comparisons with published results under the tie rule;
   audio modules with mismatched-sound controls.
4. **Experiments.** UCF Sports, Hollywood-2 (Vig gaze), the six audio-visual datasets, audio fusion, sound-source maps
   and speaker maps.
5. **Results.**
   - Transfer is strong: competitive with, or above, the published results on most datasets, but within the noise of
     comparisons across evaluation code.
   - The model fails on conversations.
   - No audio module helps, and the sound-source maps hardly depend on the sound.
   - Speaker maps cannot be built at this resolution.
6. **Interpretation.** The visual model transfers well. The sound information that would be needed, namely who is
   speaking, is not exploited by simple forms of audio and cannot be extracted from the released frames.
7. **Answer.** The approach transfers well after brief fine-tuning. Adding the soundtrack in the forms tested does not
   help (§6.1).

### 4.5 The results in plain language

- **The model adapts quickly.** Trained on free viewing of YouTube videos, it needs three short epochs on a new dataset
  to reach the level of the best published models.
- **"About as good as the best" is the safe reading.** On four of the six audio-visual datasets, its scores are a few
  hundredths above the best verified published numbers. But those numbers come from other evaluation code, and
  published evaluations of the *same* model differ by about that much.
- **It fails where gaze depends on who is speaking.** In a meeting, it highlights all the faces, while people look at
  the speaker.
- **The soundtrack did not help.** On one dataset the model learned to rely on the sound, but that did not make it more
  accurate than the model without sound.

### 4.6 Critical examination

**Fairness of the comparisons.**
- No competitor was re-scored on these datasets, so every comparison crosses evaluation code.
- The tie margin of 0.01 is smaller than the noise between evaluations that the thesis itself documents (a few
  hundredths), and it was fixed after the first such comparison had been seen (§4.1).
- The restriction to peer-reviewed results whose tables could be accessed leaves out some higher numbers.

Be careful with the word "exceeded" in the abstract: the thesis's own analysis says that these margins are within that
noise and that the table does not establish a state of the art (§5.5.3).

**Selection on tiny holdouts.** On the audio-visual datasets, the best epoch and the blur were chosen on one to six
videos per fold (§4.6, §5.9), so these choices are noisy.

**The audio tests are narrow.** One seed, ViT-L and one fusion design; the sound-source maps were tried on Coutrot2
only. The conclusion covers these forms of sound on these data. The thesis concludes that exploiting sound "probably
requires speaker-level information that the released data do not support, such as the videos at their original
resolution" (§5.8). That is a plausible account, not a tested one.

**Alternative explanations.** The Coutrot2 failure is consistent with gaze that follows the speaker, but this was not
tested directly. The gap to ViNet-E on UCF Sports may reflect encoders pre-trained for actions, a suggestion made in
§5.5.3.

**Demonstrated, suggested, claimed, unproven.**
- **Demonstrated.**
  - Fine-tuning improves every dataset over zero-shot.
  - ViT-g beats ViT-L on Hollywood-2 (Vig gaze) and on four audio-visual datasets (bootstrap over the pooled test
    videos).
  - Audio fusion did not meet its rule on any dataset.
  - Sound-source maps did not meet their rule on Coutrot2.
- **Suggested.**
  - SURGE-D is as good as, or better than, the best published models on most of these datasets (comparisons across
    evaluation code).
  - On Coutrot2, the fusion model learned to use the audio without benefiting from it.
- **Claimed.** The Coutrot2 failure arises because gaze follows the speaker. This is consistent with the literature and
  with Fig. 5.3, but not tested.
- **Unproven.**
  - A state of the art on any of these datasets, which the thesis itself does not claim.
  - Whether speaker-level audio would help.
  - How the Hollywood-2 (Vig gaze) results relate to the standard ground truth.

---

## Part 5. RQ5: What does it cost, and can the cost be cut further?

### 5.1 The question

> **RQ5.** How much computation does the approach need per output frame compared with SalFoM, and can that cost be
> reduced further without a loss of accuracy? (§1.4)

**Important context.** RQ5 was added when the results were written up, after they were known (§5.8, "Relation to the
original proposal"). Measuring the cost is one of the thesis's specific objectives (§1.5.2: "measure the
computational cost of the approach"), and the timing comparison with SalFoM had a criterion fixed before the
measurement. But no rule weighed cost against accuracy: the rules for the stride and the decoder state judged accuracy
alone.

**In plain words.** How much computing does SURGE-D need to produce one saliency map, compared with SalFoM? And can it
be made cheaper still without becoming worse?

**The uncertainty it addresses.** Recent models obtain their accuracy "at a high computational cost per output frame"
(§1.3). SalFoM runs its encoder once for every output frame (§1.2). Accuracy alone does not show whether a model is
usable at scale.

**Why it matters.** Cost decides whether a saliency model can serve real applications such as video compression or
summarisation (§4.2), and how much energy the research consumes (§4.9). A model that is both more accurate and much
cheaper has a far stronger practical case than one that is only more accurate.

**What the thesis tries to find out.**
- The cost per output frame, in operations and in measured time, against SalFoM, both offline and causally.
- Whether two changes that halve the cost, the stride of four and the decoder state, keep the accuracy.

**Assumptions behind the question.**
1. Operations (TFLOPs) and measured time are adequate measures of cost. Memory traffic and the efficiency of an
   implementation show up only in the timing (§5.5.5).
2. "Per output frame" is the right unit, because the models differ in how many frames one forward pass serves.
3. The offline setting is the relevant one for the efficiency claim; the causal comparison is descriptive.
4. A loss of accuracy can be judged on the 60-video holdout with the 0.003 margin.

**What would count as a convincing answer.**
- **For the cost.** Operation counts on the real modules plus timing under identical conditions, judged by a criterion
  fixed in advance. Both are present here.
- **For a further reduction.** A test written in advance that establishes equivalence or non-inferiority at the lower
  cost.

### 5.2 How the thesis approaches it

**Why SURGE-D could be cheap (§4.4).** One forward pass over a 16-frame clip yields usable maps for 16 original frames,
so the encoder's cost is shared among them. SalFoM predicts one frame per pass. Freezing the encoder does not reduce
the cost of inference, only the cost of training: a model trains in about 15 minutes with ViT-L and 31 minutes with
ViT-g (§5.1).

**The measurements.**
- **Operations.** Counted with the PyTorch FLOP counter on one 16×256×256 clip (Table 5.12). SalFoM's encoder was
  counted on a re-implementation loaded from its released checkpoint.
- **Time.** Measured on the same GPU: 10 warm-up and 50 timed forward passes, repeated in three runs (Table 5.13).

### 5.3 The experiments

#### Experiment 5.1: Operation counts (§5.5.5, Table 5.12)

- **Results.** Per output frame, SURGE-D needs 0.114 TFLOPs with ViT-L and 0.328 with ViT-g. SalFoM's encoder alone
  needs 2.87 TFLOPs for one 16×224×224 clip and runs once per output frame. SalFoM therefore needs about 25 times the
  computation per output frame of SURGE-D with ViT-L, and about 9 times that of SURGE-D with ViT-g.
- **Interpretation.** ViT-g's encoder is more than three times larger than SalFoM's, and one SURGE-D clip costs more
  than one SalFoM clip (5.26 against 2.87 TFLOPs). But each SURGE-D pass serves 16 frames.

#### Experiment 5.2: Measured time (§5.5.5, Table 5.13)

- **Rule** (‡, fixed before measuring). The statement that offline SURGE-D costs less per output frame than SalFoM is
  withdrawn if SalFoM's measured time per output frame is less than twice SURGE-D ViT-g's, at a batch size of one or
  eight.
- **Setup.** One RTX A5000 GPU, bfloat16 precision, random input of each model's size already on the GPU. This
  excludes frame decoding, resizing and the final blur for every model. The measurement was made three times.
- **Results.**
  - SalFoM needed 22 to 26 times the time per output frame of offline SURGE-D with ViT-g, and 41 to 64 times that of
    ViT-L. At batch 1, for example, SalFoM took 227.0 ms per frame against 10.3 ms for ViT-g.
  - Peak memory at batch 8 was 14.2 GB for SalFoM, against 3.9 GB and 8.2 GB for SURGE-D.
  - Offline, both SURGE-D models run faster than real time.
- **Interpretation.** The criterion is met by a wide margin. The time ratio is larger than the ratio of operations
  partly because of the implementations: SalFoM forms its attention matrix explicitly over 3,136 tokens, whereas
  SURGE-D uses a fused attention kernel over 2,048 tokens. How much of the difference this explains was not measured.

#### Experiment 5.3: The cost of causal prediction (§5.5.5, §5.5.2)

- **Results.**
  - A causal model needs one pass per output frame: 1.82 TFLOPs with ViT-L, about two thirds of SalFoM's encoder, and
    5.26 with ViT-g, nearly twice SalFoM's.
  - Measured, a causal ViT-L model took 0.25 to 0.39 times SalFoM's time per output frame, and a causal ViT-g model
    0.61 to 0.73 times.
  - The causal ViT-L ensemble, run as three models one after another, needs about 1.9 times SalFoM's computation and
    0.75 to 1.17 times its time.
- **Interpretation.** The saving exists only offline. In causal use, the accuracy advantage of RQ1 is bought with more
  computation, not less. These causal comparisons are descriptive; the criterion concerned the offline model only.

#### Experiment 5.4: Two ways to halve the offline cost (Tables 5.14 and 5.20, Fig. 5.6)

- **A stride of four** (ViT-L, three seeds). It halves the cost per output frame. Result: +0.0029 [−0.0009, +0.0068],
  inconclusive. The 90 % interval excludes a loss as large as 0.003.
- **The decoder state** (ViT-g, three seeds). Windows no longer overlap, so an output frame costs one thirty-second of
  a forward pass instead of one sixteenth, about 0.16 instead of 0.33 TFLOPs. Results:
  - +0.0044 against SURGE-D, inconclusive;
  - a 90 % interval of +0.0001 to +0.0089, which also excludes a loss as large as the margin;
  - +0.0051 over the standard decoder trained in the same runs, an effect.
- **Interpretation (§5.8).** "The evidence suggests, but does not establish, that the offline cost can be halved without
  a loss of accuracy." Neither change was measurably less accurate than the model it would replace. Both outcomes were
  nevertheless inconclusive under their rules, and the reading "no measurable loss" lies outside those rules. Whether
  the two savings combine, to a quarter of the cost, was not tested.

### 5.4 The reasoning chain

1. **Question.** RQ5, as quoted above.
2. **Hypothesis.** Predicting many frames per pass spreads the encoder's cost, so a frozen-encoder model can be far
   cheaper per output frame than a model that predicts one frame per pass, even with a larger encoder.
3. **Method.** Operation counts on the real modules; timing with a criterion fixed in advance; the causal cost; tests of
   two ways to halve the cost.
4. **Results.** Offline, about 9 times fewer operations and 22 to 26 times less time than SalFoM. No saving in causal
   use. Neither halving was measurably worse, but both were inconclusive.
5. **Interpretation.** The efficiency comes from the multi-frame output, and it disappears in streaming use.
6. **Answer.** Offline, about a ninth of SalFoM's encoder computation per output frame. Halving it again is suggested
   but not established (§6.1).

### 5.5 The results in plain language

- **Offline, SURGE-D is much cheaper than SalFoM.** To produce the map of one frame, it does about a ninth of the
  arithmetic of SalFoM's encoder, and on the same GPU it is more than twenty times faster.
- **The saving does not come from freezing the encoder.** It comes from producing 16 maps from one pass over the clip.
- **The saving disappears in live prediction.** Frame by frame, without seeing the future, SURGE-D loses that advantage
  and does more arithmetic than SalFoM, although in these implementations it still ran faster.
- **Halving the cost again looks possible but is not shown.** Two changes would halve the offline cost. Neither made the
  model measurably worse, but under their own rules the experiments could not confirm that nothing is lost.

### 5.6 Critical examination

- **A question framed after the results.** RQ5 was added after the results were known, and the thesis says so. The
  measurements themselves were planned and the timing criterion was fixed in advance, so the first half of RQ5 stands
  on firm ground. The second half rests on tests designed for accuracy, not for non-inferiority.
- **SalFoM's encoder only.** The factor of about nine counts SalFoM's encoder alone, not its decoder, so if anything it
  understates SalFoM's total cost (§5.5.5).
- **Time depends on implementation.** The time ratio mixes design and implementation, and the share of each was not
  measured.
- **Throughput, not latency.** Offline SURGE-D needs about 0.27 s or more of future frames for each prediction (§5.9).
  "Faster than real time" describes throughput, not latency.
- **Other dimensions of cost.** With about one billion encoder parameters, the model is large. The thesis does not
  claim that SURGE-D is cheaper than lightweight saliency models in general.

**Demonstrated, suggested, claimed, unproven.**
- **Demonstrated.** Offline SURGE-D needs about a ninth (ViT-g) to a twenty-fifth (ViT-L) of SalFoM's encoder
  computation per output frame, and much less measured time; the criterion fixed in advance was met.
- **Suggested.** The stride of four and the decoder state each halve the cost without a loss.
- **Claimed.** The saving comes from predicting sixteen frames per pass. This follows directly from the operation
  counts.
- **Unproven.**
  - That the cost can be halved without a loss.
  - That the two halvings combine.
  - Efficient streaming prediction, which lay outside the scope of the study (§1.7).

---

## Part 6. How the five questions fit together

### 6.1 The overall objective

The general objective (§1.5.1) is "to design, implement and rigorously evaluate a video saliency prediction model built
on a frozen self-supervised video encoder, and to test whether the encoder's own prediction error is a useful cue for
human attention."
- **RQ1, RQ3, RQ4 and RQ5 serve the first half:** building and evaluating the model, that is its accuracy, what drives
  that accuracy, how well it generalises and what it costs.
- **RQ2 serves the second half:** the prediction error.

### 6.2 Which questions are foundational

```
Evaluator validation (§5.2) ................ underlies every "same evaluator" claim
          |
G1: choice of encoder (an RQ3 experiment) --+
                                            +--> the SURGE-D model --> RQ1: accuracy against SalFoM
G2: surprise gate (an RQ2 experiment) ------+    (no surprise branch)       |
                                                                            +--> RQ3 explains it
                                                                            +--> RQ4 tests its generality
                                                                            +--> RQ5 prices it
```

- **The evaluator validation** is the base of every "same evaluator" claim.
- **Two gates come before RQ1.** G1, an RQ3 experiment, chose the encoder. G2, an RQ2 experiment, removed the surprise
  branch. The model evaluated for RQ1 is the product of these two decisions.
- **RQ1 is the central claim.** RQ3 explains it (the encoder dominates), RQ4 tests whether it generalises, and RQ5 adds
  its price.
- **RQ2 asks an independent scientific question, but it decided the architecture.** Its negative result is why the model
  is SURGE-D rather than the surprise-gated design.

### 6.3 Experiments that serve more than one question

| Experiment | Serves | How |
|---|---|---|
| G1 encoder comparison (Table 5.2) | RQ1, RQ3 | Chooses the encoder; shows that pre-training matters |
| Frozen UMT-L check (Table 5.7) | RQ1, RQ3 | Places the gain in the encoder rather than in freezing |
| ViT-g against ViT-L (Tables 5.6, 5.14) | RQ1, RQ3, RQ4 | The headline model; the effect of scale on DHF1K and on other datasets |
| Causal control (Table 5.8) | RQ1, RQ5 | Accuracy under identical evaluation; the price of causality |
| Stride of four (Table 5.14) | RQ3, RQ5 | Temporal sampling; halves the cost |
| Decoder state (Table 5.20) | RQ3, RQ5 | Use of time across windows; halves the cost |
| Predictability gate (§5.7) | RQ2, RQ3 | Whether the decoder can use the error; an addition that failed |
| Audio fusion, sound-source maps, speaker maps | RQ4, RQ3 | Sound on audio-visual data; additions of information |
| Learning curve (Table 5.15) | RQ3 | Tests the "data limit" explanation for the null results of decoder additions |
| Qualitative failure (Fig. 5.3) | RQ1, RQ4 | Links the largest DHF1K loss to the failure on Coutrot2 |

### 6.4 How the answers support the contributions (§6.2)

1. **SURGE-D** (RQ1, RQ4, RQ5). More accurate than SalFoM on DHF1K validation, offline and causally under identical
   evaluation, at a fraction of its offline computation; it transfers well.
2. **A negative result on predictive surprise** (RQ2).
3. **Controlled evidence on design choices** (RQ3). The encoder dominates; two common components are equivalent to their
   absence; adaptation and several additions did not help.
4. **An evaluation methodology** (all questions). The validated evaluator, the rules written in advance and the paired
   statistics are what make every answer above interpretable, including the negative ones.

### 6.5 What remains partly answered

| RQ | Answered | Still open |
|---|---|---|
| RQ1 | Yes on DHF1K validation (ViT-g; at ViT-L only the ensemble) | The official test set; other re-scored baselines; full fine-tuning |
| RQ2 | The sign and size of the information, for one predictor | What the error measures; other measures of surprise |
| RQ3 | The encoder's pre-training and scale dominate; two components are equivalent to their absence | The decoder architecture; whether data limit accuracy at 540 videos; several inconclusive factors |
| RQ4 | Strong transfer; simple forms of audio do not help | No re-scored competitors; speaker-level sound |
| RQ5 | About a ninth of SalFoM's computation offline; no saving in causal use | Halving the cost without a loss; combining the two halvings |

### 6.6 One-line answers for a defence

| RQ | Answer | Strongest evidence | Main caveat |
|---|---|---|---|
| 1 | With ViT-g, yes; at SalFoM's encoder size, only as an ensemble | Causal ViT-g ensemble +0.029 [+0.019, +0.040] against SalFoM at its best blur (Table 5.8) | Validation set; one re-scored baseline |
| 2 | The error carries information, but people look *away* from high error | U = +0.069 bits with weights −0.34 and −0.40 (Table 5.4); NSS −0.665 (Table 5.3) | One predictor; a feature-space error is not a change of belief |
| 3 | The encoder dominates; the centre prior and deep supervision make no difference | G1, scale and the equivalence results (Tables 5.2, 5.14) | Decoder architecture untested; many single-seed or inconclusive tests |
| 4 | Transfers well after brief fine-tuning; simple forms of audio do not help | Gains from fine-tuning (Tables 5.9–5.11); the audio rule was not met (Table 5.16) | Comparisons cross evaluation code; tiny holdouts |
| 5 | About a ninth of SalFoM's computation offline; halving it again is suggested | 22–26 times less time against a criterion of 2 times (Table 5.13) | Added after the results; the halvings were inconclusive |

---

## Part 7. Check your understanding

Try to answer each question before reading the answer.

1. **SalFoM's best blur was chosen on the test set. Why does that make the comparison stronger rather than weaker?**
   Because it favours the baseline. It gives SalFoM an advantage that it could not have in practice, an upper bound,
   so an advantage that survives it is conservative (§5.5.1).
2. **What exactly does the causal control remove, and what does it leave unequal?** It removes SURGE-D's access to
   frames after the predicted one, and it scores both models on the same frames with the same evaluator and SalFoM's
   best blur. It leaves the encoder, the input resolution and the training procedure unequal (§5.5.2).
3. **Why does the frozen UMT-L result matter for RQ1?** It shows that freezing an encoder and adding SURGE-D's decoder
   does not by itself beat SalFoM. The advantage depends on the V-JEPA 2 encoder (Table 5.7).
4. **Why is a negative NSS for the error maps not enough to reject the surprise hypothesis, and what does the
   incremental analysis add?** A map can score negatively just because it is low at the centre, where fixations
   concentrate. The incremental analysis controls for the centre prior and the simple cues. It finds that the error
   still carries information, with a negative weight (Table 5.4).
5. **What would RQ2's results have looked like if the surprise hypothesis were true?** A positive NSS that beats frame
   difference and token atypicality, and a positive weight in the log-linear model. Ideally, also a decoder that uses
   the error, scoring worse with rotated maps.
6. **What is the difference between "inconclusive" and "equivalent"? Give one RQ3 factor with each outcome.**
   Equivalent means that any difference larger than ±0.003 is ruled out: removing the centre prior or deep supervision.
   Inconclusive means the data cannot decide: static-image co-training, the stride of four and the learning curve's
   primary comparison (Table 5.14, Table 5.15).
7. **Why can the thesis not say that its decoder design is responsible for the accuracy?** The decoder architecture,
   the taps and the loss weights were fixed without a test (§4.1). Only the components that were ablated can be judged.
8. **Why are SURGE-D's wins over published audio-visual results only "suggested"?** They cross evaluation code, and
   published evaluations of the same model disagree by a few hundredths, the same size as the margins (§5.5.3).
9. **Where does SURGE-D's efficiency come from, and why does it vanish in causal use?** From predicting 16 frames per
   forward pass. A causal model needs one pass per output frame (§5.5.5).
10. **Which experiments answer two research questions at once?** For example, the causal control (RQ1 and RQ5), the
    decoder state (RQ3 and RQ5), the frozen UMT-L check (RQ1 and RQ3) and the predictability gate (RQ2 and RQ3); see
    §6.3 of this guide.
