# Architecture

Nothing is built yet. This is the design the plan starts from; each part is confirmed or changed by
a measurement in the phase that builds it.

## Data flow

```
 Hugging Face Hub
        │  scripts/prepare_data.py
        ▼
 converters (one per dataset) ──▶ {id, source, split, state, question, options[], label_idx, template_id}
        │  cap per dataset, dedup, question templates        ~/.trueodds/data/  (spec 001)
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ one question with K options  →  K sequences                                  │
 │   [CLS] state [SEP] question [SEP] option_k [SEP]   (state cut first, D9)    │
 │                          │                                                   │
 │                          ▼                                                   │
 │                 ModernBERT-base (bf16)                                       │
 │                          │  pooled vector (CLS or mean, D2)                  │
 │                          ▼                                                   │
 │                 linear → one score s_k                                       │
 │                          │                                                   │
 │   padded options: s_k = −inf     softmax over k     cross-entropy on label   │
 └──────────────────────────────────────────────────────────────────────────────┘
        │  training: best checkpoint by dev accuracy        ~/.trueodds/runs/
        ▼
 temperature T, fitted on dev (Phase 3)  →  p_k = softmax(s_k / T)
        │
        ▼
 harness: accuracy, ECE, NLL, Brier, baselines   →   harness/results/*.json   (spec 002)
 predict(state, question, options) → {option: p}   (Phase 4, spec 003)
```

## The model

A cross-encoder (D2). Every option is read together with the state and the question, so the model
can attend from the option's words to the evidence in the state. Each option is scored on its own,
and only the softmax puts the options of one question together.

Consequences, all deliberate:

- **No position bias across options.** Shuffling the options shuffles the probabilities the same
  way; a unit test checks it. Multiple-choice LLM prompts do not have this property.
- **Any number of options.** K is a batch dimension, padded and masked, so a 2-option yes/no and a
  14-option topic question go through the same head.
- **Options cannot see each other.** "All of the above" or "none of these" cannot be scored
  correctly. None of the datasets depend on it.
- **Cost grows with K.** A question with K options is K sequences, and the state is encoded K times.
  That is what the shared-state experiment of Phase 5 removes (D13).

Answer formats:

| Task | Options |
|---|---|
| Yes/no (BoolQ) | `yes`, `no` |
| Inference (MNLI) | `yes`, `maybe`, `no`, with the hypothesis in the question ("Does it follow that …?") |
| Multiple choice (ARC, HellaSwag, MMLU aux, CommonsenseQA) | the answer texts |
| Topic (AG News, Yahoo, DBpedia-14) | the label names, written as words ("Sports", "Science & Mathematics") |

## Input and truncation

Maximum 512 tokens per sequence (D9). The question and the option are tokenized first and never
cut; the state gets the remaining budget and is cut at its end. If the question and the option alone
exceed the budget, the example is dropped at conversion and counted.

## Memory and throughput

Estimates, to be replaced by Phase 0 measurements:

| Item | Estimate |
|---|---|
| ModernBERT-base parameters | ~150M, ~0.3 GB in bf16 |
| AdamW state and fp32 master weights | ~2.4 GB |
| Activations | the rest; set by the micro-batch in sequences, not questions |

One question with 10 options (Yahoo) costs as much as five yes/no questions. The micro-batch is
therefore sized in sequences, and batches group questions of similar K to reduce padding (D10).

## Where things live

| What | Where | In git |
|---|---|---|
| Converters, model, training, inference | `src/trueodds/` | yes |
| Evaluations and their results | `harness/`, `harness/results/*.json` | yes |
| Run settings | `configs/*.yaml` | yes |
| Downloaded and processed data | `~/.trueodds/data/` | no (D12) |
| Checkpoints and TensorBoard logs | `~/.trueodds/runs/` | no (D12) |
