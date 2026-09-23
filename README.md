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

Planning. See [`docs/PLAN.md`](docs/PLAN.md) for the phases and exit criteria,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design, and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for why things are the way they are.

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
