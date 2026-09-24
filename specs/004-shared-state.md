# 004 — Shared state encoding

**Phase 5 (5B, D13).** The v1 cross-encoder reads the state once per option (D2), so K options
and many questions about one state repeat the state's work each time. The shared-state variant
reads the state once and scores every question and option against it in one packed sequence.
Written by `trueodds/shared.py`, read by `trueodds/model.py`, the trainer, `predict.py` and
`predict_batch`.

## Packed sequence

```
[CLS] state [SEP] | q1 [SEP] | [CLS] o11 [SEP] | [CLS] o12 [SEP] | q2 [SEP] | [CLS] o21 [SEP] ...
```

Each token carries a position id, a question id (`-1` for the state) and an option id (`-1` for
the state and for question tokens).

| Rule | Contract |
|---|---|
| Attention | Token *i* may attend to *j* when *j* is in the state, or in *i*'s question outside its options, or in *i*'s own option. So the state sees only the state; a question sees the state and itself; an option sees the state, its question and itself. Questions never see each other, options never see each other. |
| Sliding-window layers | ModernBERT alternates global and local (128-token window) layers. Local layers use the same rule AND \|pos_i − pos_j\| ≤ `config.sliding_window` (64), measured by **position id**, not by row, which is ModernBERT's own rule on an unpacked sequence. |
| Positions | The state takes 0 … S−1 (with its `[CLS]` and `[SEP]`). Every question starts at S; every option of a question starts right after that question's `[SEP]`. |
| Padding | A padding token attends to itself only (no empty attention row); nothing attends to it. |
| Pooling | Each option is scored from its leading `[CLS]` by the same linear head as v1 (the CLS pooling of D25). Scores go into [B, K_max] with −inf on padded options, as in v1 (D22). |
| Truncation (D9) | The question and options are never cut. A question keeps `max_len − (question + longest option + 4) − 1` state tokens, so its longest option sits within `max_len` (512) positions, as in v1. When several questions are packed, the state is cut to the smallest of their budgets. A question and option too long for any state raise `ValueError`. The packed sequence itself may exceed 512 tokens (ModernBERT's context is 8,192). |

Consequences, all by construction:

- **A question scores the same alone or packed** with other questions (same positions, same visible
  tokens), up to floating-point rounding, as long as the state is not cut differently.
- **Permuting options permutes their probabilities**, and changing one option changes no other
  option's score.
- **The state is encoded without seeing the question or the options.** This is the cost D13 names,
  and what 5B measures against v1.

The masks are passed to `ModernBertModel` as `attention_mask={"full_attention": …,
"sliding_attention": …}` (boolean, [R, 1, L, L], built on the device from the segment ids) with
`position_ids`; transformers 5.17 uses them as given. With every token in the state, they equal
ModernBERT's own masks (tested).

## Training and scoring

- Training and the harness put **one question per packed sequence**: the dataset has one question
  per state. A packed question counts as one sequence of its packed length in the micro-batch
  budget (D22).
- `predict_batch(state, questions)` puts **all questions about the state in one sequence**
  (`pack=True`); `predict` is the same with one question. Both go through
  `predict.predict_scores`, the harness's path (D28).

## Checkpoint

`model.json` gains `"architecture": "cross" | "shared"`. A checkpoint without it (all of Phases 1–4)
is a cross-encoder. `ModelPredictor.load` and `trueodds.load` pick the encoder from it; a config
selects it with `architecture: shared`. The shared model uses CLS pooling only.

## Acceptance

`tests/test_shared.py`, on the CPU with a tiny random ModernBERT (one global and one local layer,
states longer than the window):

- with every token in the state, the masks give ModernBERT's own hidden states;
- the mask structure: the state sees only the state, no token sees another question, no option
  sees another option, the sliding mask is the full mask AND the window;
- positions restart for every question and option; truncation cuts only the state;
- a question scores the same alone, packed with others, in any order, and packed next to other
  states' questions;
- changing one option leaves the others' scores; padded options get 0 and each question sums
  to 1; permuting options permutes probabilities;
- the architecture survives save and load; an old `model.json` loads as a cross-encoder;
  `predict_batch` equals `predict` one by one;
- the tiny shared model overfits a small known set.
