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

## Open questions

- **Are the held-out datasets far enough?** CommonsenseQA is multiple choice like ARC and HellaSwag,
  and DBpedia-14 is topic classification like AG News and Yahoo. They measure new data and new label
  sets, not a new kind of task. A held-out yes/no or inference dataset (for example, one of RTE, WiC
  or a question type absent from training) would test more.
- **A stronger reference than majority class.** A zero-shot NLI classifier (such as a BART-large MNLI
  model) on DBpedia-14 would say whether the model generalizes better than an off-the-shelf approach,
  not just better than chance.
- **Contamination of the backbone.** ModernBERT's pretraining data may include these datasets'
  text. That cannot be checked, only stated.
- **Does one temperature fit all?** If ECE by K differs widely after Phase 3, a temperature per K or
  per task type is the next step.
