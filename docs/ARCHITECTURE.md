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
        │  training: best checkpoint by dev NLL (D26)       ~/.trueodds/runs/
        ▼
 temperature T, fitted on dev (Phase 3, D27)  →  p_k = softmax(s_k / T)   <checkpoint>/temperature.json
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

### The shared-state variant (Phase 5B, D13, spec 004)

`architecture: shared` in a config, and in a checkpoint's `model.json`. One packed sequence holds
the state once, then each question and its options:

```
[CLS] state [SEP] | q1 [SEP] | [CLS] o11 [SEP] | [CLS] o12 [SEP] | q2 [SEP] | [CLS] o21 [SEP] ...
```

Boolean attention masks, built on the device from per-token question and option ids, let the
state see only the state, a question the state and itself, and an option the state, its question
and itself. Every question starts at the same position after the state and every option right
after its question, so a question scores the same alone or packed with others, and permuting
options still permutes their probabilities. ModernBERT's local layers get the same mask AND its
64-token window, measured by position id. Each option is scored from its own `[CLS]` by the same
head. The masks and `position_ids` go straight into `ModernBertModel` (transformers 5.17 accepts
a mask per layer type), so the encoder itself is unchanged.

What changes: the state is encoded without seeing the question or the options, and
`predict_batch` puts every question about one state into one sequence. Training and the harness
still read one question per sequence, since the datasets have one question per state.

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

## Inference (D28)

`trueodds.load()` reads a checkpoint directory once and keeps the model on the GPU;
`predict(state, question, options)` builds the same `Example` the harness scores, checks it by
spec 001's rules, and goes through `ModelPredictor` (encoding, micro-batches, bf16 autocast,
softmax at the fitted T). There is no second code path to drift from what the harness measured.
`predict_batch` sends many questions about one state through the same call, sharing
micro-batches.

Measured on the RTX 5060 Ti (`docs/SETUP.md`): a 4-option question on a 256-token state takes
17.7 ms (p50). Below about 1,000 tokens a call costs its overhead (13–14 ms); above it, compute.
Since the state is encoded once per option (D2), cost grows with K × length, and a batch of
questions about one state is no faster than asking them one by one; the shared-state experiment
(D13) is what would change that.

## Code

| Module | Role |
|---|---|
| `trueodds/encode.py` | tokenization and truncation (D9), no torch |
| `trueodds/shared.py` | the shared-state variant (spec 004): packed sequences, their masks, collation, `SharedEncoder` |
| `trueodds/model.py` | `DecisionModel`: encoder, pooling (CLS or mean), linear head, masked scores; save and load; the cross-encoder or the shared-state variant |
| `trueodds/batching.py` | steps, token-budget micro-batches, collation (D22) |
| `trueodds/train.py` | the training loop, one YAML config per run |
| `trueodds/predict.py` | `ModelPredictor`, the harness `Predictor` for a checkpoint; scores, and probabilities at the checkpoint's temperature |
| `trueodds/infer.py` | `trueodds.load()` and `TrueOdds.predict` / `predict_batch` (spec 003, D28): requests checked, scored through `ModelPredictor` |
| `trueodds/calibrate.py` | fitting the temperature by NLL and writing `temperature.json` (D27) |
| `trueodds/tracking.py` | MLflow runs, registering and promoting versions, mirroring harness results (D24) |
| `trueodds/mlflow_model.py` | the checkpoint as an MLflow pyfunc model (rows of state, question, options → probabilities) |

A checkpoint directory (`~/.trueodds/runs/<run>/best/` and `last/`) holds `model.safetensors`, the
encoder's `config.json`, `model.json` (backbone, pooling, max length) and the tokenizer, so it loads
without the Hub. After Phase 3, `best/` also holds `temperature.json` (T and how it was fitted),
which `ModelPredictor.load` applies, and `scores.npz`, the cached scores `harness.calibrate` fits
and evaluates from (D27). The weights themselves are never changed by calibration. The run directory also holds `config.yaml`, `summary.json` and the TensorBoard log
in `tb/`.

## Where things live

| What | Where | In git |
|---|---|---|
| Converters, model, training, inference | `src/trueodds/` | yes |
| Evaluations and their results | `harness/`, `harness/results/*.json`, `harness/results/figures/*.png` | yes |
| Run settings | `configs/*.yaml` | yes |
| Downloaded and processed data | `~/.trueodds/data/` | no (D12) |
| Checkpoints and TensorBoard logs | `~/.trueodds/runs/` | no (D12) |
| MLflow runs and model registry, and its artifacts | `~/.trueodds/mlflow.db`, `~/.trueodds/mlartifacts/` | no (D12, D24) |
