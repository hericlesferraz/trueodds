# Architecture

Phase 0 built the data and the harness, Phase 1 the model and the training loop. This is the design
the plan starts from; each part is confirmed or changed by a measurement in the phase that builds it.

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

### Batches and masking (D22)

A batch of B questions with up to K_max options is flattened to its N real sequences only; padded
options are never encoded. The N scores are scattered into a [B, K_max] matrix filled with −inf, so
the softmax gives padded options exactly 0 and the cross-entropy never sees them. This is what the
unit tests check on a tiny random ModernBERT (`tests/test_model.py`): padded options get 0, each row
sums to 1, and permuting the options permutes the probabilities.

An optimizer step is 32 questions of similar K and length, split into micro-batches of at most
16,384 padded tokens, with the loss divided by the step's question count so that the accumulated
gradient does not depend on the split.

## Input and truncation

Maximum 512 tokens per sequence (D9). The question and the option are tokenized first and never
cut; the state gets the remaining budget and is cut at its end. If the question and the option alone
exceed the budget, the example is dropped at conversion and counted.

`src/trueodds/encode.py` does it (`build_ids`); the empty state of ARC and CommonsenseQA keeps its
`[SEP]`, so every sequence has the same four special tokens.

## Memory and throughput

Measured in Phase 0 (`docs/SETUP.md`), with fp32 weights, bf16 autocast, fused AdamW and PyTorch
SDPA attention (D19):

| Item | Measured |
|---|---|
| ModernBERT-base parameters | ~150M; a bf16 forward pass of 16 × 512 tokens peaks at 0.65 GiB |
| Largest training micro-batch under 15 GiB | 36 sequences at 512 tokens, 76 at 256, 152 at 128 (no gradient checkpointing) |
| Training throughput | ~25k tokens/s at any length; gradient checkpointing costs ~22% |
| One epoch of the Phase 2 mix | 154.6M tokens, ~100 min if batches are grouped by length |
| Phase 2 training loop, 50 costliest steps (Phase 1) | 13.07 GiB peak reserved; 12.34 GiB with `expandable_segments`; ~23.5k padded tokens/s |

Throughput is constant in tokens, and neither attention backend skips the compute on padding here,
so batches are grouped by length as well as by K. One question with 10 options (Yahoo) costs as much
as five yes/no questions, so the micro-batch is sized in padded tokens, not questions (D10, D22).

## Code

| Module | Role |
|---|---|
| `trueodds/encode.py` | tokenization and truncation (D9), no torch |
| `trueodds/model.py` | `DecisionModel`: encoder, pooling (CLS or mean), linear head, masked scores; save and load |
| `trueodds/batching.py` | steps, token-budget micro-batches, collation (D22) |
| `trueodds/train.py` | the training loop, one YAML config per run |
| `trueodds/predict.py` | `ModelPredictor`, the harness `Predictor` for a checkpoint |

A checkpoint directory (`~/.trueodds/runs/<run>/best/` and `last/`) holds `model.safetensors`, the
encoder's `config.json`, `model.json` (backbone, pooling, max length) and the tokenizer, so it loads
without the Hub. The run directory also holds `config.yaml`, `summary.json` and the TensorBoard log
in `tb/`.

## Where things live

| What | Where | In git |
|---|---|---|
| Converters, model, training, inference | `src/trueodds/` | yes |
| Evaluations and their results | `harness/`, `harness/results/*.json` | yes |
| Run settings | `configs/*.yaml` | yes |
| Downloaded and processed data | `~/.trueodds/data/` | no (D12) |
| Checkpoints and TensorBoard logs | `~/.trueodds/runs/` | no (D12) |
