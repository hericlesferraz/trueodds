# trueodds

A small decision model that answers multiple-choice questions about a piece of text with a
probability for each option, from one forward pass, and measures whether those probabilities are
true.

Given a **state** (any text), a **question** and a list of **options**, it returns
`{option: probability}`. It never generates text. The design follows the idea behind "System One"
decision models such as TypeSafe AI's Jev, rebuilt small, from public datasets, on one consumer
GPU, to understand how such a model works and where it fails.

Three things are measured:

- **Accuracy** on the validation data of every training dataset.
- **Generalization:** accuracy on datasets the model never saw (CommonsenseQA, DBpedia-14),
  against random and majority-class baselines.
- **Calibration:** expected calibration error (ECE), negative log-likelihood and Brier score,
  before and after temperature scaling, on the training datasets and on the unseen ones.

This is a learning project, not a new method.

## Status

**v1 (2026-09-24).** ModernBERT-base fine-tuned for one epoch on eight public datasets, with one
temperature fitted on the dev split. Phases 0 to 4 of the plan are done. Phase 5 (experiments)
is in progress: ModernBERT-large is measured below; shared state encoding is next. See [`docs/PLAN.md`](docs/PLAN.md) for the phases and exit criteria,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for why things are the way they are.

## Usage

```python
import trueodds

model = trueodds.load()   # ~/.trueodds/runs/phase2-templates/best and its temperature
state = "The striker scored twice in the second half to seal the title."

model.predict(state, "What is this text about?",
              ["World", "Sports", "Business", "Science and technology"])
# {'World': 0.049, 'Sports': 0.948, 'Business': 0.002, 'Science and technology': 0.0}

model.predict_batch(state, [
    ("Is this a news report?", ["yes", "no"]),
    ("Does it follow that a team won a championship?", ["yes", "maybe", "no"]),
])
# [{'yes': 0.842, 'no': 0.158}, {'yes': 0.696, 'maybe': 0.3, 'no': 0.005}]
```

The contract (errors, truncation, the batch form) is in [`specs/003-predict.md`](specs/003-predict.md).
Weights are not in this repository; `scripts/prepare_data.py` and `configs/phase2-templates.yaml`
rebuild them (see `docs/SETUP.md`).

## Results

Test split of every training dataset, and the two datasets never trained on. The best baseline is
the higher of random and majority class; the prior baseline knows the label frequencies and reads
no text (D18). NLL is at the fitted temperature, which is what `predict()` returns. From
[`2026-09-24-110652-calibration.json`](harness/results/2026-09-24-110652-calibration.json).

| dataset | n | accuracy | best baseline | NLL (prior) | ECE, T = 1 | ECE, T = 1.058 |
|---|---|---|---|---|---|---|
| BoolQ | 3,270 | 0.806 | 0.622 | 0.427 (0.663) | 0.037 | 0.028 |
| MNLI | 5,000 | 0.859 | 0.333 | 0.379 (1.099) | 0.008 | 0.013 |
| ARC-Easy | 2,376 | 0.595 | 0.250 | 0.999 (1.388) | 0.044 | 0.057 |
| ARC-Challenge | 1,172 | 0.456 | 0.265 | 1.230 (1.384) | 0.058 | 0.049 |
| HellaSwag | 3,243 | 0.592 | 0.250 | 0.975 (1.387) | 0.032 | 0.028 |
| MMLU auxiliary | 2,000 | 0.702 | 0.272 | 0.749 (1.384) | 0.021 | 0.016 |
| AG News | 5,000 | 0.934 | 0.250 | 0.192 (1.386) | 0.008 | 0.012 |
| Yahoo Answers | 5,000 | 0.744 | 0.100 | 0.779 (2.303) | 0.016 | 0.019 |
| **Pooled test** | 27,061 | **0.761** | 0.282 | 0.614 (1.415) | 0.008 | 0.007 |
| CommonsenseQA *(held out)* | 1,221 | 0.530 | 0.210 | 1.199 (1.605) | 0.036 | 0.038 |
| DBpedia-14 *(held out)* | 5,000 | 0.861 | 0.072 | 0.569 (2.639) | 0.144 | 0.166 |

What the table says:

- Accuracy is far above the baselines everywhere, including the two datasets never seen in
  training, and NLL is below the prior's everywhere.
- In-domain the model was already calibrated before temperature scaling (pooled ECE 0.008); the
  fitted T = 1.058 changes it only within noise (95% interval −0.005 to +0.005).
- **Calibration does not transfer to unseen label sets.** On DBpedia-14 the model is
  underconfident (answers given 0.5 are right about 70% of the time), and the temperature fitted
  on seen data makes it worse. The probabilities are trustworthy on data like the training data,
  and not yet beyond it (D27).
- Accuracy on a question phrasing never trained on is within 1.4 points of the trained phrasings
  on every templated dataset (D8, D26).

Reliability diagrams (confidence against observed accuracy, before and after the temperature):

| Pooled test | CommonsenseQA (held out) | DBpedia-14 (held out) |
|---|---|---|
| ![pooled](harness/results/figures/2026-09-24-110652-reliability-pooled.png) | ![commonsense_qa](harness/results/figures/2026-09-24-110652-reliability-commonsense_qa.png) | ![dbpedia](harness/results/figures/2026-09-24-110652-reliability-dbpedia.png) |

### Latency

`predict()` end to end (tokenization, forward pass, softmax) on an RTX 5060 Ti, bf16, after warmup,
200 requests per cell, from
[`2026-09-24-122853-latency.json`](harness/results/2026-09-24-122853-latency.json):

| options | state 64 tokens | 256 tokens | 480 tokens |
|---|---|---|---|
| 2 | 13.7 ms (p95 14.6) | 13.8 ms (14.6) | 16.5 ms (17.4) |
| 4 | 14.1 ms (15.0) | **17.7 ms (18.8)** | 26.9 ms (28.1) |
| 14 | 16.9 ms (17.8) | 45.8 ms (47.3) | 98.2 ms (99.9) |

Many questions about one 256-token state (options K = 2 to 14): 50 questions take 1.31 s with
`predict_batch` and 1.27 s one by one. Batching saves nothing. Each option is its own sequence
that repeats the state, so one question already carries enough tokens to keep the GPU busy, and a
batch only removes per-call overhead that was already hidden behind the compute. Encoding the
state once is the Phase 5 shared-state experiment (D13).

**On the CPU** (Ryzen 7 5700X, 8 threads, fp32, nothing optimized;
[`2026-09-25-015515-latency.json`](harness/results/2026-09-25-015515-latency.json)):

| options | state 64 tokens | 256 tokens | 480 tokens |
|---|---|---|---|
| 2 | 71 ms | 208 ms | 492 ms |
| 4 | 118 ms | **517 ms** | 1.01 s |
| 14 | 420 ms | 2.09 s | 4.00 s |

5 to 46 times the GPU. The CPU has no overhead floor to hide behind, so the time follows the
number of options times the state length. `predict_batch` is slower than one by one here (50
questions: 66 s against 54 s).

### Experiment: ModernBERT-large (Phase 5A)

The same training with ModernBERT-large (~395M parameters) instead of base, compared question by
question with v1, each at its own fitted temperature, with paired bootstrap 95% intervals
([`2026-09-25-015315-compare.json`](harness/results/2026-09-25-015315-compare.json), D29):

| dataset | accuracy base → large | Δ (95% interval) | ECE base → large |
|---|---|---|---|
| BoolQ | 0.806 → 0.860 | +5.4 (+4.3, +6.7) | 0.028 → 0.021 |
| MNLI | 0.859 → 0.894 | +3.5 (+2.7, +4.3) | 0.013 → 0.020 |
| ARC-Easy | 0.595 → 0.721 | +12.6 (+10.4, +14.8) | 0.057 → 0.041 |
| ARC-Challenge | 0.456 → 0.599 | +14.3 (+11.2, +17.5) | 0.049 → 0.037 |
| HellaSwag | 0.592 → 0.710 | +11.7 (+10.1, +13.5) | 0.028 → 0.044 |
| MMLU auxiliary | 0.702 → 0.797 | +9.5 (+7.7, +11.3) | 0.016 → 0.025 |
| AG News | 0.934 → 0.935 | +0.1 (−0.3, +0.5) | 0.012 → 0.011 |
| Yahoo Answers | 0.744 → 0.760 | +1.5 (+0.8, +2.3) | 0.019 → 0.030 |
| **Pooled test** | **0.761 → 0.815** | **+5.4 (+5.0, +5.9)** | 0.007 → 0.011 |
| CommonsenseQA *(held out)* | 0.530 → 0.639 | +10.9 (+7.9, +13.8) | 0.038 → 0.025 |
| DBpedia-14 *(held out)* | 0.861 → 0.843 | −1.7 (−2.8, −0.7) | 0.166 → 0.203 |

Large reads better: the gain is largest where the answer takes reasoning over the text (ARC,
HellaSwag, MMLU) and on CommonsenseQA, never trained on. It is not better calibrated, and on
DBpedia-14, the unseen label set, it is less accurate and more underconfident than base. It also
takes about twice as long (K = 4, 256-token state: 37.4 ms p50 against 17.7), so v1 stays the
default model.

## Datasets

Downloaded from the Hugging Face Hub by a script and never stored in this repository. Each one keeps
its own license. The licenses below are the Hub's tags as of 2026-09-23; each is checked against its
original source in Phase 0.

**Training**

| Dataset | Hub ID | Task | Rows (all splits) | License (Hub tag) |
|---|---|---|---|---|
| BoolQ | `google/boolq` | yes/no about a passage | 12.7k | CC-BY-SA 3.0 |
| MNLI | `nyu-mll/multi_nli` | yes / maybe / no | 412k | mixed: CC-BY 3.0, CC-BY-SA 3.0, MIT, other |
| ARC-Easy, ARC-Challenge | `allenai/ai2_arc` | science multiple choice | 5.2k + 2.6k | CC-BY-SA 4.0 |
| HellaSwag | `Rowan/hellaswag` | choose the ending | 60k | none on the Hub |
| MMLU auxiliary train | `cais/mmlu`, config `auxiliary_train` | multiple choice | 100k | MIT |
| AG News | `fancyzhx/ag_news` | 4 topics | 128k | unknown |
| Yahoo Answers Topics | `community-datasets/yahoo_answers_topics` | 10 topics | 1.46M | unknown |

**Held out, never trained on**

| Dataset | Hub ID | Task | Rows (all splits) | License (Hub tag) |
|---|---|---|---|---|
| CommonsenseQA | `tau/commonsense_qa` | 5-option commonsense | 12.1k | MIT |
| DBpedia-14 | `fancyzhx/dbpedia_14` | 14 topics | 630k | CC-BY-SA 3.0 |

Together they are about 1 GB of parquet to download.

## Backbone

[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (Answer.AI and LightOn,
Apache-2.0).
