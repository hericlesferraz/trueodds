# Decisions

Each decision is dated. When a decision changes, add a new entry that supersedes the old one
instead of editing it.

---

## D1 — Planning format: phases with measurable exit criteria, harness first
*2026-09-23*

**Decision:** the project uses the format of fluentloop and leakyverifier: `docs/PLAN.md` (phases,
tasks, exit criteria), `docs/ARCHITECTURE.md` and this file. Specs are written only for contracts
that are not obvious, at the start of the phase that needs them. The evaluation of the original plan
(its Phase 4) is built in Phase 0, before the model.

**Why:** in both earlier projects, the part that kept the work honest was the table of phases with
measurable exit criteria, and the harness being ready before the thing it measures. Here it also
means the first trained model is compared with its baselines on day one, instead of after a report
script is written.

---

## D2 — The model is a cross-encoder over (state, question, option)
*2026-09-23*

**Decision:** ModernBERT-base reads `[CLS] state [SEP] question [SEP] option [SEP]` once per option,
a pooled vector goes through one linear layer to a score, and the scores are softmaxed across the
options of a question with cross-entropy against `label_idx`. Pooling (the CLS vector or the mean of
the tokens) is chosen in Phase 2 by two short runs on dev.

**Why:**
- It is the simplest model that has all three properties of the target: typed options,
  probabilities from one pass, and any number of options.
- It is invariant to option order by construction, so position bias is not something to measure
  away later.
- It is the standard multiple-choice head (as in `AutoModelForMultipleChoice`), so a bug is easier to
  find by comparing with a known implementation.

How Jev itself is built is not public beyond its description (a state, questions evaluated against
it in parallel, one number per option, a calibration step). This model reproduces the interface,
not the internals. D13 covers the part of the interface it does not reproduce yet.

---

## D3 — Backbone: ModernBERT-base
*2026-09-23*

**Decision:** `answerdotai/ModernBERT-base` (Apache-2.0), with `-large` as a Phase 5 experiment.

**Why:** a modern encoder with a long context, trained on data that includes code, fast with
unpadded attention, and small enough to fine-tune fully on 16 GB. An encoder fits the task: the
model classifies and never generates.

---

## D4 — Training mix and caps
*2026-09-23*

**Decision:** train on BoolQ, MNLI, ARC-Easy and ARC-Challenge, HellaSwag, MMLU auxiliary train, AG
News and Yahoo Answers Topics, each capped at ~50k training examples *(initial)*. Smaller datasets
(BoolQ ~9k, ARC ~3k) are used whole, not upsampled at first.

**Why:** without caps, Yahoo (1.4M) and AG News (120k) would be most of the data, and the model would
become a topic classifier. Upsampling the small ones is a later option if their accuracy lags;
measured first, not assumed.

---

## D5 — No example may be in both training and evaluation
*2026-09-23*

**Decision:** after conversion, duplicates are removed by normalized text (lowercase, collapsed
whitespace, of question plus sorted options). Any training example that matches a dev or test
example of any source, or any example of a held-out dataset, is dropped from training, and the
count is reported.

**Why:** MMLU auxiliary train is built from other datasets, ARC among them, so ARC test questions
could reach training through it. Without the check, the ARC number would measure memory. The same
check also catches unexpected overlap with CommonsenseQA.

---

## D6 — Three splits: train, dev, test
*2026-09-23*

**Decision:** each training source has three splits:
- **train:** used for gradients;
- **dev:** carved out of the official training split (~2k examples or 10%, whichever is smaller);
  used for checkpoint selection, pooling choice, temperature fitting and every other choice;
- **test:** the official validation split (the official test labels of most of these datasets are
  hidden). Only read, never tuned against.

For MNLI, test is `validation_matched`; `validation_mismatched` is reported too. The held-out
datasets use their official validation or test split, whichever has labels, and have no dev.

**Why:** the original plan used one validation split for checkpoint selection, temperature fitting
and the final report. Temperature scaling fitted on the split it is scored on always looks better
than it is. The ECE improvement of Phase 3 is only meaningful on data the temperature did not see.

---

## D7 — Metrics: accuracy, ECE, NLL and Brier, with baselines
*2026-09-23*

**Decision:**
- **Accuracy:** the top option is the label.
- **ECE:** 15 equal-width bins over the top-1 probability, weighted by bin size. Also reported by
  number of options.
- **NLL** and **Brier score** over all options, because ECE depends on the binning and can be low
  for a useless model (one that always says 1/K on a balanced dataset).
- **Baselines:** random (1/K) and majority class, per source.

**Why:** calibration is the point of the project, and ECE alone can be gamed by the binning. NLL and
Brier are proper scoring rules: they are best only when the probabilities are right. ECE by K
matters because one temperature is shared by 2-option and 14-option questions, whose confidence
scales differ.

---

## D8 — Question templates, and one held-out template per task
*2026-09-23*

**Decision:** each task has 3–5 phrasings of its question, sampled at random in training, and one
more phrasing that is never used in training. The harness reports accuracy on the held-out phrasing
next to the trained ones.

**Why:** the reason for templates is that the model should read the question instead of recognizing
the task by its phrasing. The held-out phrasing measures whether that worked.

---

## D9 — Truncation cuts the state, never the question or the option
*2026-09-23*

**Decision:** 512 tokens per sequence. The question and the option are never cut; the state is cut
at its end to fit. Examples whose question and option alone do not fit are dropped and counted. The
share of truncated states is reported per source.

**Why:** a cut question or option changes what is being asked; a cut state only loses evidence.
512 keeps training affordable. Most states here are short; BoolQ and RACE-derived MMLU aux passages
are the long ones, and the truncation share says whether 512 is enough.

---

## D10 — Batches are sized in sequences and grouped by option count
*2026-09-23*

**Decision:** the micro-batch is set by the number of sequences (question × option) that fits, and
questions with similar K are batched together. The effective batch of ~32 questions is reached by
gradient accumulation.

**Why:** a 10-option Yahoo question is 10 sequences; a BoolQ question is 2. Batching mixed K pads
everything to the largest K and wastes most of the compute on masked options.

---

## D11 — Environment: uv, Python 3.12, PyTorch from the cu128 index
*2026-09-23*

**Decision:** the same toolchain as fluentloop: uv, Python 3.12, torch from
`https://download.pytorch.org/whl/cu128` (Blackwell needs CUDA 12.8+). flash-attn is tried in Phase
0; if it does not build for compute capability 12.0, PyTorch SDPA is used and the throughput cost is
recorded.

**Why:** that stack is already proved on this GPU. ModernBERT's speed comes partly from unpadded
flash attention, so whether it works here changes the epoch time, and it is measured rather than
assumed.

---

## D12 — Data and weights stay out of the repository
*2026-09-23*

**Decision:** downloaded and processed data live in `~/.trueodds/data/`, checkpoints and logs in
`~/.trueodds/runs/`, all regenerated by scripts. Results JSONs are committed.

**Why:** the datasets have different licenses (some non-commercial or share-alike), and none of them
needs to be redistributed. The results are small and are what a reader needs to check the claims.

---

## D13 — Shared state encoding is a Phase 5 experiment, not v1
*2026-09-23*

**Decision:** v1 encodes the state once per option (D2). Encoding the state once and scoring many
questions and options against it is a Phase 5 experiment, measured against v1 on the same harness.

**Why:** that design is closer to what Jev describes (one state, many questions answered in
parallel), and it is where the latency gain is. It is also harder: options no longer read the full
state with full attention, or they share one sequence and become order-sensitive. The cross-encoder
is first a working reference, so the experiment has something to be compared with.

---

## D14 — Which official split becomes test, per source
*2026-09-23, refines D6*

**Decision:** D6's "test = the official validation split" holds where the official test labels are
hidden (BoolQ, MNLI, HellaSwag, CommonsenseQA). Elsewhere:
- **ARC** publishes test labels, so test = the official test (2,376 Easy, 1,172 Challenge) and dev
  = the official validation.
- **AG News, Yahoo and DBpedia-14** have no validation split: test = the official test, dev carved
  from train.
- **MMLU auxiliary train** is one split. Dev and test (2,000 each) are carved from it **by
  passage**: all the questions on one RACE passage go to the same split.

**Why:** a labelled official test is larger and is what other results report, so it is the better
test. Carving MMLU aux by passage matters because RACE asks several questions per passage; a random
carve would put a passage in training and another question on it in test.

---

## D15 — Evaluation files are capped at 5,000 questions per source
*2026-09-23*

**Decision:** each test and held-out file is sampled down to 5,000 questions, stratified by label,
seed 0, once, in `prepare_data`. Every run is scored on the same files.

**Why:** DBpedia test (70k × 14 options) and Yahoo test (60k × 10) would cost about 1.6M sequences
per evaluation, much more than all the other sources together. At n = 5,000 the 95% interval of an
accuracy near 0.8 is about ±1.1 points, finer than any difference the exit criteria rest on.

---

## D16 — The dedup key includes the state and the template variables
*2026-09-23, refines D5*

**Decision:** two examples are duplicates when their normalized state, their normalized template
variables (or the question, for datasets that bring their own), and their sorted normalized options
are equal (spec 001).

**Why:** D5's key (question plus options) would make every AG News example a duplicate of every
other: they share the question "What is this text about?" and the same four labels. The rendered
question is left out of the key because it is only a template; two renderings of the same MNLI pair
are the same example.

---

## D17 — HellaSwag: ActivityNet rows only
*2026-09-23*

**Decision:** the HellaSwag converter drops every row whose `source_id` starts with `wikihow`. What
is left is the ActivityNet part: 14,740 training rows (before the dev carve) and 3,243 test rows.

**Why:** on 2026-09-14 GitHub blocked the HellaSwag repository after a DMCA notice from wikiHow,
which claims copyright on its articles. About two thirds of HellaSwag's contexts are wikiHow text.
The ActivityNet rows keep a "what happens next" task in the mix without that content.

---

## D18 — A third baseline: the label prior
*2026-09-23, adds to D7*

**Decision:** besides random and majority, every result reports a **prior** predictor. It gives each
option the frequency of that label in the source's training split (by label text when all
questions share the options, else by position), smoothed and renormalized over the question's
options.

**Why:** majority has no calibration to compare with (all its probability is on one option), and
random's NLL is only log K. The prior is the best a model can do without reading the text. Its ECE is
near 0 on a balanced set, which shows why ECE alone is not the target (D7), and its NLL and Brier
are the calibration numbers a trained model must beat.

---

## D19 — Attention backend: PyTorch SDPA with torch 2.11
*2026-09-23, closes the question left open by D11*

**Decision:** train and run with `attn_implementation="sdpa"` on the torch that `uv sync --extra
gpu` resolves (2.11.0+cu128). flash-attn is not a dependency.

**Why:** measured in `docs/SETUP.md`. The official flash-attn 2.8.3 wheel runs on compute capability
12.0, but only with torch 2.8. Next to SDPA it gives the same throughput at fixed length and at
most ~8% more with padded batches. In transformers 5.17 neither backend skips the compute on
padding, so grouping batches by length (D10) matters far more than the backend.

---

## D20 — Phase 2 targets, from the Phase 0 baselines
*2026-09-23, replaces the (initial) targets of Phase 2*

**Decision:** from `harness/results/2026-09-23-112729-baselines.json`. Each file must be passed by
the best checkpoint (chosen on dev):

1. **Accuracy at least 10 points above the higher of the random and majority baselines**, on every
   training source's `test` file and on both held-out datasets:

   | File | Random | Majority | Target ≥ |
   |---|---|---|---|
   | boolq/test | 0.500 | 0.622 | 0.722 |
   | mnli/test | 0.333 | 0.318 | 0.433 |
   | arc_easy/test | 0.250 | 0.246 | 0.350 |
   | arc_challenge/test | 0.250 | 0.265 | 0.365 |
   | hellaswag/test | 0.250 | 0.248 | 0.350 |
   | mmlu_aux/test | 0.250 | 0.272 | 0.372 |
   | ag_news/test | 0.250 | 0.250 | 0.350 |
   | yahoo/test | 0.100 | 0.100 | 0.200 |
   | commonsense_qa/test (held-out) | 0.201 | 0.210 | 0.310 |
   | dbpedia/test (held-out) | 0.071 | 0.072 | 0.172 |

2. **NLL below the prior baseline's (D18)** on every one of those files.
3. **Held-out templates** (D8): on each templated source, accuracy on `test-heldout-template` within
   3 points of `test`.

**Why:** 10 points is more than three times the 95% interval of the smallest test file
(ARC-Challenge, n = 1,172: about ±2.9 points), so passing it cannot be noise. It is a floor that
says the model learned to read each task, not a target for good accuracy: a fine-tuned base encoder
is expected to be far above it on AG News, Yahoo, MNLI and BoolQ, and the Phase 2 report states the
actual margins. The NLL condition is the calibration side of the same floor. A model that has only
learned how often each answer is right matches the prior; beating it means the probabilities use
the text.

---

## D21 — torch is a base dependency
*2026-09-23, changes the `gpu` extra of Phase 0*

**Decision:** `torch` moves from the `gpu` extra to the base dependencies, still from the cu128 index
(D11). The `gpu` extra keeps what only a training machine needs: `nvidia-ml-py` and `tensorboard`.

**Why:** from Phase 1 the package is a torch model, and the tests that prove its properties (a padded
option gets probability 0, shuffling options shuffles probabilities, the tiny model overfits a known
set) need torch but not a GPU: they run on a tiny random ModernBERT on the CPU in seconds. With torch
optional they would be skipped by a plain `uv run pytest`, and a skipped exit-criterion test is easy
to miss. The cost is a larger first `uv sync`.

---

## D22 — An optimizer step is 32 questions, split into micro-batches by a token budget
*2026-09-23, refines D10: the micro-batch is sized in padded tokens, not in sequences*

**Decision:** each optimizer step takes 32 questions (`questions_per_step`). An epoch shuffles the
questions, sorts them by (K, length) within chunks of 64 steps, cuts the chunks into steps and
shuffles the steps. Each step is split into micro-batches whose padded size (longest sequence ×
sequences) is at most 16,384 tokens (`max_tokens_per_microbatch`); the loss of each micro-batch is
its summed cross-entropy divided by the step's 32 questions, so the accumulated gradient is the mean
over the step whatever the split. Padded options are never encoded: only the N real sequences go
through the encoder, and their scores are scattered into a [B, K_max] matrix filled with −inf.

**Why:** the effective batch stays ~32 questions (PLAN.md) whether a step is 32 BoolQ questions (64
sequences) or 32 Yahoo questions (320). The budget is in tokens because throughput is constant in
tokens (SETUP.md), and 16,384 is a margin below the ~18.4k that fit in Phase 0. Measured in Phase 1
on the 50 costliest steps of the full mix: 13.07 GiB peak reserved, 12.34 GiB with
`expandable_segments`. Sorting within chunks makes most steps a single K (so little padding) while
keeping the order across steps random. The cost: most steps come from one source, which makes the
gradient noisier across tasks than a fully mixed batch.

---

## D23 — Training re-samples the question template every epoch
*2026-09-23, makes D8 concrete*

**Decision:** in training, each templated question is re-rendered from its `vars` with a random
trained template every time it is drawn (`resample_templates`). Dev and test keep the stored
question, whose template is fixed by the example id. The held-out template is never rendered in
training; `resample_template` raises if it meets one.

**Why:** the stored train files hold one template per example, fixed at conversion. Re-sampling
shows the model every phrasing of every question over the epochs, which is what D8 asks for, at the
cost of a few tokenizations per step. The Phase 1 overfit run turns it off, so that the 200
questions it memorizes are the ones it is scored on.

---

## D24 — MLflow tracks the runs and registers the best checkpoints; the harness stays the record
*2026-09-23*

**Decision:** MLflow 3.16 (the optional `tracking` extra), local only: runs and the model registry
in `~/.trueodds/mlflow.db` (sqlite), artifacts in `~/.trueodds/mlartifacts/`, one experiment and one
registered model, both named `trueodds`. A run with `tracking: mlflow` (or `--mlflow`) logs:
- its config as params, and every scalar the trainer logs (train loss, lr, grad norm, throughput,
  VRAM, the eval metrics) at its step;
- `config.yaml` and `summary.json` as artifacts;
- its `best` checkpoint (never `last`) as a new version of the `trueodds` model: a pyfunc model
  whose input is rows of `state`, `question`, `options` and whose output is one probability list per
  row, tagged with its run, pooling and dev accuracy, NLL and ECE.

`harness.evaluate --checkpoint` mirrors its result into the checkpoint's run (the JSON as an
artifact, per file accuracy, NLL and ECE under `harness/`). No code sets an alias: a version is
promoted to `champion` by hand, in the UI or with `python -m trueodds.tracking promote`, and on dev
numbers only (D6), which is why only dev numbers are version tags.

**Why:** to learn the tool, and because Phase 2 and Phase 5 compare many runs, which the UI makes
easier than a directory of `summary.json` files. The harness JSONs in `harness/results/` remain the
record the exit criteria are read from; MLflow shows them next to the training curves and does not
replace them. Phase 4's `predict()` can load `models:/trueodds@champion`. Each version stores its own
copy of the weights (573 MB), so only `best` is registered. Langfuse was considered and not taken: it
traces calls to generative models, and this model generates nothing; it may fit the Phase 5
experiment where a local LLM labels data.

---

## D25 — CLS pooling, chosen on dev by two short runs
*2026-09-23, closes the pooling choice left open in D2*

**Decision:** the model pools with the CLS vector. `configs/phase2-base.yaml` keeps `pooling: cls`.

**Why:** two runs identical except for pooling (`configs/phase2-pool-cls.yaml`,
`configs/phase2-pool-mean.yaml`: 1,500 steps of 32 questions, about 22% of an epoch, lr 3e-5 decayed
to 0 over those steps, seed 0), scored on the full dev split (11,286 questions):

| pooling | step 500 acc | step 1000 acc | step 1500 acc | NLL | ECE | Brier |
|---------|-------------:|--------------:|--------------:|----:|----:|------:|
| CLS     | 0.634 | 0.701 | **0.711** | **0.720** | 0.014 | **0.384** |
| mean    | 0.589 | 0.668 | 0.692 | 0.758 | 0.013 | 0.405 |

CLS leads at every evaluation, by 1.9 points at the end, about 3 standard errors of the difference
on 11,286 questions. It is one seed and a short run, so the gap may shrink with full training, but
nothing here favors mean pooling except an ECE that is equal within noise, and calibration is
Phase 3's job. Peak VRAM was the same for both (12.40 GiB reserved).

---

## D26 — Ten trained phrasings per task, 2 epochs, best checkpoint by dev NLL
*2026-09-23, amends D8 (3–5 phrasings) and the Phase 1 rule that `best` is chosen by dev accuracy*

**What prompted it, stated plainly:** the harness report of `phase2-base`
(`harness/results/2026-09-23-221918-eval.json`), which is a test and held-out result. It missed the
Phase 2 exit criteria in two ways: accuracy on the held-out template was 12.4 points below the
trained templates on MNLI and 4.6 on DBpedia-14 (limit 3), and ARC-Challenge's NLL was above the
prior's (1.410 vs 1.384, ECE 0.19). The rule is that test numbers are read, never tuned against
(D6). Nothing below is fitted to them: no number in this decision was chosen by looking at a test
score, and the held-out templates stay unseen. But the direction of the change was chosen because of
them, and the next run's test numbers are therefore less independent than the first run's were.

**Decision:**
1. **Ten trained phrasings per task** (`:0` to `:9` in `templates.py`), up from four. The six new
   ones are used only by the trainer's per-epoch redraw (D23). Dev and test are rendered with `:0`
   to `:3` only (`STABLE`), so their files are byte-identical to Phase 0's (checked: 34,929
   templated dev and test rows re-render the same) and the two runs are compared on the same text.
   A unit test forbids the new phrasings from using the words that make each held-out phrasing
   distinctive (e.g. *claim* and *supported* for MNLI, *file* and *heading* for topics).
2. **2 epochs, not 3.** From `phase2-base`'s dev curve alone: dev accuracy peaked in epoch 2 (0.776
   at step 12,000) and epoch 3 only overfit (dev NLL 0.65 → 1.23, ECE 0.06 → 0.16).
3. **`best` is chosen by dev NLL** (`select_by: nll`). Also from the dev curve: selecting by accuracy
   took step 12,000 (ECE 0.063) over step 6,000 (ECE 0.008) for 1.8 points of accuracy. NLL scores
   the probabilities, which are this project's product; accuracy scores only their argmax.

**Why:** the MNLI confusion matrix shows the model reading a phrasing rather than the question: under
the held-out phrasing, 748 of 1,772 true entailments become *maybe*, against 180 under the trained
ones. Four phrasings per task are few enough to be recognized as a set; ten is meant to make the
question's meaning the only thing that transfers. DBpedia-14's gap (predictions drifting to *Written
work*) is on labels never trained on, and may not close. Points 2 and 3 address ARC-Challenge's
overconfidence; whether they are enough, or temperature scaling in Phase 3 is needed as well, is what
the run will show. Config: `configs/phase2-templates.yaml`.

---

## D27 — One temperature, fitted on pooled dev by NLL, kept next to the weights
*2026-09-24, Phase 3*

**Decision:**
- **One T** for every source and every K, the plan's starting point. p = softmax(s / T).
- **Fitted on the pooled dev files** of the eight training sources (11,286 questions) by
  minimizing the mean NLL, with LBFGS over log T in float64. Padded options are masked after the
  division by T, because a −∞ score divided by T gives a NaN gradient. Nothing is fitted on test
  or on a held-out dataset (D6).
- **Stored as `<checkpoint>/temperature.json`** (T, the sources, n, dev metrics before and after),
  never folded into the weights or the head. `ModelPredictor.load` applies it, so
  `harness.evaluate --checkpoint` and Phase 4's `predict()` use it without a flag;
  `temperature=False` gives the raw model.
- **Scores once, temperatures after.** `harness.calibrate` runs the checkpoint once over dev and
  every evaluation file and caches the scores in `<checkpoint>/scores.npz` (not committed, D12).
  The fit, the before (T = 1) and after evaluations and the bootstrap all run from that cache
  through `harness.evaluate`, so before and after differ by T alone.
- **A paired bootstrap interval** (1,000 resamples, seed 0) is reported with every ECE change,
  and one file (`<stamp>-calibration.json`, spec 002) holds the fit, before, after and intervals.

**Why:** NLL is a proper scoring rule and smooth in T; ECE depends on binning and would be a noisy
thing to fit. Keeping T out of the weights keeps the Phase 2 checkpoint as it was, and lets the same
checkpoint be scored with and without it. The bootstrap is there because the Phase 2 model is
already calibrated in-domain (pooled test ECE 0.008 before this phase): a change that small has to
come with its noise, or "ECE went down" says nothing.

**Result** (`harness/results/2026-09-24-110652-calibration.json`): T = 1.0576. Pooled test ECE 0.0077
→ 0.0069, within noise (95% interval −0.0048 to +0.0045). DBpedia-14, underconfident on K = 14,
gets worse (0.144 → 0.166). CommonsenseQA does not change beyond noise. The Phase 3 criterion is met
as written. The finding is that a temperature fitted in-domain does not transfer to the held-out
data.

---

## D28 — predict() is a loaded object over the harness scoring path, left as measured
*2026-09-24, Phase 4*

**Decision:**
- **`trueodds.load(path)` returns a `TrueOdds` object** with `predict(state, question, options)
  -> {option: probability}` and `predict_batch(state, [(question, options), ...])` (spec 003). It
  wraps `ModelPredictor`, so `predict()` and `harness.evaluate` score through the same function
  and the harness numbers are the numbers a caller gets. The fitted temperature is applied
  (D27).
- **The default is a checkpoint path**, `~/.trueodds/runs/phase2-templates/best`
  (`paths.DEFAULT_CHECKPOINT`). Loading needs no MLflow; the registry (D24) stays a tool for
  comparing runs.
- **Requests are checked by spec 001's rules** (`Example.validate`): at least 2 options, none
  empty, no duplicates (a dict cannot hold both), a non-empty question. The MLflow pyfunc model
  now builds its examples through the same helper.
- **No inference optimization.** The model runs as the harness evaluated it: fp32 weights under
  bf16 autocast, PyTorch SDPA. The exit target is met without bf16 weights or `torch.compile`, so
  neither was tried; each would change the numbers slightly, and the harness would have to score
  that variant too.
- **No HTTP endpoint** in v1 (the plan's optional FastAPI task, skipped).

**Why:** a loaded object keeps the 0.6 GB model in memory between calls, which is what makes tens
of milliseconds possible; a module-level `predict()` would hide a global. Keeping one scoring path
means there is nothing to reconcile between what was measured and what is served.

**Result** (`harness/results/2026-09-24-122853-latency.json`): 4 options and a 256-token state,
p50 17.7 ms and p95 18.8 ms (target 30 and 60). Below about 1,000 tokens per call the time is
per-call overhead, 13–14 ms (2 options and 64 tokens take 13.7 ms); above it the forward pass is
compute-bound at 70–90k tokens/s, so 14 options and a 480-token state take 98 ms. **`predict_batch`
is no faster than a loop** (50 questions about one state: 1.31 s against 1.27 s): every option
repeats the state, so most single questions are already past the crossover, and a batch removes
overhead that was mostly hidden behind compute. Large micro-batches are also slower per token: at
a 4,096-token budget instead of the eval budget of 32,768, the same batch takes 1.06 s, about 20%
faster than the loop. The budget was left as it is (SETUP.md). Batch and loop differ by up
to 0.013 in probability on the GPU and by 9e-7 in fp32 on the CPU: bf16 rounding under different
batch shapes, not a bug. Encoding the state once (D13) is the experiment that would change the
batch numbers.

---

## D29 — ModernBERT-large: +5.4 points in-domain, calibrated the same, worse on DBpedia-14
*2026-09-25, Phase 5A*

**Decision:** v1 stays `phase2-templates/best` (ModernBERT-base) as the default of
`trueodds.load()`, and -large is kept as a measured alternative (`phase5-large/best`, T = 1.041).
The comparison is recorded; no setting was tuned for -large.

**Setup:** v1's config (`phase2-templates.yaml`) with `backbone: answerdotai/ModernBERT-large`; the
same 32-question step, split into 5,120-token micro-batches instead of 16,384 to fit (D22: same
gradient). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, peak 13.21 GiB reserved. 2 epochs,
13,916 steps, 10.9k padded tokens/s, about 11 hours with the dev evaluations. As with base, the
best checkpoint by dev NLL (D26) is the end of epoch 1 (step 6,000: dev accuracy 0.813, NLL 0.509,
ECE 0.007); epoch 2 raises dev accuracy to 0.828 but ECE to 0.07–0.09 and NLL to 0.55–0.64.

**Result** (`harness/results/2026-09-25-015315-compare.json`, each model at its own T, paired
bootstrap 95% intervals over questions):

- **In-domain, better everywhere but the topic tasks.** Pooled test accuracy 0.761 → **0.815**
  (+5.4 points, +5.0 to +5.9), NLL 0.614 → 0.508. The gain is where reading matters: ARC-Challenge
  +14.3, ARC-Easy +12.6, HellaSwag +11.7, MMLU +9.5, BoolQ +5.4, MNLI +3.5. AG News is unchanged
  (+0.1, interval crosses 0) and Yahoo gains +1.5.
- **Calibration in-domain is the same.** Pooled ECE 0.008 before and 0.011 at T = 1.041 (v1 0.007);
  the change is within noise (−0.003 to +0.006). The temperature again does nothing measurable
  (dev NLL 0.5085 → 0.5081), as in D27.
- **Held-out template gap ≤ 1.0 point** on every templated source (DBpedia-14 1.0, the rest ≤ 0.3).
- **CommonsenseQA (held out): +10.9 points** (0.530 → 0.639), NLL 1.199 → 0.913, ECE 0.036 → 0.025.
- **DBpedia-14 (held out): worse.** Accuracy 0.861 → 0.843 (−1.7, −2.8 to −0.7), NLL 0.569 →
  0.668, ECE 0.166 → 0.203 at T. It is more underconfident than base: mean confidence 0.66
  against accuracy 0.84, answers given 0.5 are right 76% of the time (base: 70%). The temperature
  fitted in-domain softens it further (ECE 0.186 → 0.203), as in D27. On the held-out DBpedia
  template, ECE is lower than base's (0.194 → 0.168).
- **Latency** (`2026-09-25-014805-latency.json`): K = 4 on a 256-token state, p50 **37.4 ms**
  (base 17.7), over the Phase 4 target of 30 ms that v1 was held to; 1.8 to 2.5× base on every cell
  past the overhead floor (K = 14, 480 tokens: 232 ms). Batching still saves nothing (50
  questions: 2.85 s batch, 2.88 s one by one).

**Why v1 stays the default:** -large is the better model on the kind of data it was trained on,
by a margin its interval makes certain, and on CommonsenseQA. But it costs twice the latency and
misses the Phase 4 target, and on the one unseen label set it generalizes worse and its
probabilities are further from true, which is the property this project measures first. A larger
encoder reads better; it does not make the probabilities transfer. What would, is the open
question D27 left (K itself or new labels), and a held-out set that is not topic classification.

---

## D30 — Shared state encoding: −1.3 points, same calibration, 16× faster on many questions
*2026-09-25, Phase 5B*

**Decision:** the shared-state model (`phase5-shared/best`, T = 1.068, spec 004) is kept as the
measured alternative for the use D13 was about, many questions about one state; v1 stays the
default of `trueodds.load()`. The comparison is recorded; nothing was tuned for it.

**Setup:** v1's config with `architecture: shared`: the state once, then each question and its
options in one packed sequence, masks keeping them apart (spec 004). Same 32-question step and
16,384-token budget, `expandable_segments`, peak 11.89 GiB. 2 epochs in **57 minutes** (v1: 246),
at the same ~25k tokens/s: each question is about 4.3× fewer tokens, since the state is not
repeated per option. Best by dev NLL again at the end of epoch 1 (step 6,000: dev accuracy 0.745,
NLL 0.646, ECE 0.015); epoch 2 again breaks calibration (ECE 0.06–0.10).

**Result** (`harness/results/2026-09-25-041921-compare.json`, each at its own T, paired 95%
intervals):

- **What giving the state no view of the question costs: 1.3 points pooled.** Pooled test
  accuracy 0.761 → 0.747 (−1.3, −1.8 to −0.9), NLL 0.614 → 0.643. The loss is where the answer
  depends on reading the state for a particular question: ARC-Easy −4.0, HellaSwag −3.4, MNLI −1.7
  to −2.2, MMLU −1.6. Where the state is a document to classify, nothing is lost: BoolQ −0.3, AG
  News 0.0, Yahoo −0.6, ARC-Challenge −0.4 (all four intervals cross 0).
- **Calibration in-domain is the same:** pooled ECE 0.010 at T against 0.007 (−0.002 to +0.007).
  The temperature again changes nothing measurable (dev NLL 0.6464 → 0.6453).
- **Held-out template gap ≤ 1.3 points** (MNLI 1.3, DBpedia-14 0.9, the rest ≤ 0.6).
- **Held-out datasets lose more:** CommonsenseQA −4.7 points (0.530 → 0.483), DBpedia-14 −3.9
  (0.861 → 0.822), ECE 0.166 → 0.186, underconfident like base (mean confidence 0.66 against
  accuracy 0.82).
- **Latency, the point of D13** (`2026-09-25-041810-latency.json`): **50 questions about a
  256-token state in 44 ms** with `predict_batch`, against 709 ms one by one and **1,270 ms for v1**
  one by one (the 5B target): 29× faster than v1, 16× than its own loop. 10 questions: 15 ms.
  Every single request, up to 14 options on a 480-token state, sits at the ~14 ms overhead floor
  (v1: 98 ms for that one), since the state is encoded once whatever K is. Packed and single
  answers differ by up to 0.011 in probability on the GPU (bf16; v1's batch 0.013); the fp32 CPU
  tests hold them to 1e-5.
- **On the CPU** (`2026-09-25-041935-latency.json`, 5C): a 4-option question on a 256-token
  state takes 120 ms (v1 517 ms), 14 options on 480 tokens 259 ms (v1 4.0 s, 15×), and 50
  questions about one state 1.54 s with `predict_batch` (v1 one by one: 53.7 s, 35×). On the CPU
  the saving shows on single requests too, since there is no overhead floor to hide it.

**Why it is not the default:** it is 1.3 points less accurate in-domain and 4–5 points less on the
held-out datasets, and the harness measures one question at a time, where v1 already meets its
latency target. For a caller asking many questions about the same state (the Jev-style use), it
answers 50 questions in less time than v1 takes for two (44 ms against 25 ms per question), with
the same calibration. Which one to load
depends on how many questions share a state, and both are measured.

## D31 — Farther held-out datasets: accuracy transfers, probabilities do not
*2026-09-25, Phase 5D*

**Decision:** RTE, WiC and Rotten Tomatoes are held-out sources from now on (spec 001), scored by
every model. Nothing changes in the models; the finding is recorded.

**Data:** every labeled official split of each is one `test` file (none is trained on), capped at
5,000: RTE 2,767 (plus its held-out-template copy, asked with the MNLI phrasings), WiC 5,000,
Rotten Tomatoes 5,000. WiC and Rotten Tomatoes are asked with a question the model never saw.
Regenerating the data showed that `prepare_data.py` no longer reproduces the stored training files
of five sources byte for byte: they were written before D26 and store phrasings 0–3, while a fresh
run stores 0–9. Rows, labels and dedup are identical, and training re-samples the phrasing every
epoch (D23), but batches are planned from the stored lengths, so the files the Phase 2–5 models
were trained on were restored and `stats.json` says so. Every other file is byte-identical, and
re-scoring all three models reproduced their temperatures exactly.

**Result** (`2026-09-25-12*/13*-calibration.json`, `-compare.json`; at each model's T):

| | v1 | -large | shared | prior NLL |
|---|---|---|---|---|
| RTE accuracy / NLL / conf − acc | 0.729 / 0.689 / **+0.155** | 0.771 / 0.607 / +0.127 | 0.759 / 0.585 / +0.110 | 0.693 |
| WiC accuracy / NLL / conf − acc | **0.475** / 0.760 / **+0.151** | 0.565 / 0.695 / +0.032 | 0.504 / 0.840 / +0.193 | 0.693 |
| Rotten Tomatoes accuracy / NLL / conf − acc | 0.826 / 0.461 / −0.137 | 0.882 / 0.405 / −0.172 | 0.808 / 0.433 / +0.015 | 0.693 |

- **Accuracy transfers where the skill exists.** RTE (inference, trained as MNLI) and sentiment
  (never trained) are well above chance in every model; -large gains 4 to 6 points on both.
- **WiC is a skill the model does not have**: v1 is below chance, -large 6.5 points above it.
- **Calibration does not transfer, and on new tasks it fails in the dangerous direction.** In
  training data ECE is under 0.01; on RTE it is 0.11–0.16 in every model and on WiC up to 0.19,
  **overconfident** in both (only -large is near calibrated on WiC, ECE 0.04): v1
  gives WiC answers 0.63 on average and is right 47.5% of the time, and its RTE NLL (0.689) is no
  better than knowing the label frequencies (0.693) despite 73% accuracy. On RTE, the likely cause
  is the label set: MNLI's "maybe" is gone, and its mass lands on "yes" or "no".
- **A new label set without a new task is underconfident** (Rotten Tomatoes for v1 and -large,
  like DBpedia-14), except in the shared model, which is calibrated there (+0.015). One run is not
  enough to say why.
- The answer to D27's question, extended: a temperature fitted on seen data cannot fix any of
  this, since the error changes sign with the kind of novelty. The probabilities are true on data
  like the training data; outside it, not even their direction can be assumed.

---

## D32 — Underconfidence on unseen labels comes from the labels, and grows with K
*2026-09-25, Phase 5E*

**Decision:** recorded as a finding; it answers D27's open question. `harness/options.py` stays
as the tool for it.

**Method:** each question of DBpedia-14 (unseen labels), Yahoo and AG News (trained labels) is
scored with its answer and K − 1 distractors drawn at random, the same draw for every model, from
the cached scores. Options never see each other (D2, spec 004), so this equals re-scoring (tested
on both architectures), and uniform distractors keep a calibrated model calibrated (tested).

**Result** (`2026-09-25-132523-options.json`, confidence − accuracy at each model's T, 95%
intervals ±0.004–0.010):

| K | DBpedia-14 v1 / large / shared | Yahoo v1 / large / shared | AG News v1 / large / shared |
|---|---|---|---|
| 2 | −0.024 / −0.033 / −0.032 | −0.004 / −0.009 / −0.005 | +0.000 / +0.001 / −0.002 |
| 4 | −0.063 / −0.078 / −0.075 | −0.003 / −0.015 / −0.012 | −0.007 / −0.008 / −0.008 |
| 10 | −0.132 / −0.166 / −0.156 | −0.014 / −0.029 / −0.037 | |
| 14 | −0.166 / −0.203 / −0.186 | | |

- **At the same K, unseen labels are underconfident and trained labels are not:** at K = 4,
  DBpedia-14's gap is −0.06 to −0.08 in every model, Yahoo's and AG News's −0.003 to −0.015.
- **K amplifies it:** DBpedia-14's gap grows steadily with K, from −0.02/−0.03 at K = 2 to
  −0.17/−0.20 at K = 14. Yahoo, with trained labels, also drifts underconfident at K = 10, but by
  a tenth to a quarter as much.
- So the cause is the labels, and K sets its size. A temperature per K fitted on dev cannot fix it
  (dev has no unseen labels), which is why one global T made DBpedia-14 worse (D27).

## D33 — Phase 5 closes; the score head and the distilled labels are deferred
*2026-09-25*

**Decision:** Phase 5 is done with five experiments measured against v1: ModernBERT-large (D29),
shared state encoding (D30), latency on the CPU (5C), farther held-out datasets (D31) and options
against labels (D32). The two experiments the plan listed first, a score head and distilled labels
for an own domain, are deferred, not dropped. The repository is published under Apache-2.0 (the
code only; the datasets keep their licenses, `docs/LICENSES.md`, and no weights are published).

**Why:** neither tests whether the model's probabilities are true, which is what this project
measures. The score head adds a third kind of output (a number), worth building only when numeric
answers are needed. Distilled labels would train the model on a domain, but an LLM's labels cannot
tell whether the model is calibrated there: after D31 that needs test labels with a known true
answer. What the experiments left open matters more: the probabilities are true on data like the
training data and not beyond it, where the error changes sign with the kind of novelty (D31, D32).

**What would bring them back:** a real need for numeric answers, or a specific domain to use the
model in, with a way to know the true answers. A next phase, if any, would start from calibration
outside the training data.

---

## Open questions

- **Are the held-out datasets far enough?** *Answered by 5D (D31): RTE, WiC and Rotten Tomatoes
  added.* Accuracy transfers where the skill exists; calibration does not, and on new tasks it is
  overconfident. Still open: why the shared model is calibrated on Rotten Tomatoes when the
  others are not.
- **A stronger reference than majority class.** A zero-shot NLI classifier (such as a BART-large MNLI
  model) on DBpedia-14 would say whether the model generalizes better than an off-the-shelf approach,
  not just better than chance.
- **Contamination of the backbone.** ModernBERT's pretraining data may include these datasets'
  text. That cannot be checked, only stated.
- **Does one temperature fit all?** *Answered by Phase 3 (D27): no.* In-domain, ECE by K moves in
  both directions under the one T (K = 2 improves, K = 3 and 10 get worse), and DBpedia-14 (K = 14,
  never trained) is underconfident while dev asks for a T slightly above 1. A temperature per K
  cannot be fitted for K = 14 from dev, which has no 14-option questions. What is left open is
  whether underconfidence on unseen label sets comes from K itself (more options spreading the
  probability) or from the new labels. *Answered by 5E (D32): the labels, amplified by K.*
