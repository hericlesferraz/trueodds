# 003 — predict()

**Phase 4.** How a caller asks the trained model a question, and the latency results JSON that
measures it. Written by `trueodds/infer.py` and `harness/latency.py`, read by anyone using the model
and by `harness/report.py`.

## Interface

```python
import trueodds

model = trueodds.load()                     # ~/.trueodds/runs/phase2-templates/best, its temperature
model.predict(state, question, options)     # -> {option: probability}
model.predict_batch(state, [(question, options), ...])   # -> [{option: probability}, ...]
```

| Call | Contract |
|---|---|
| `trueodds.load(path=DEFAULT_CHECKPOINT, device=None, temperature=True) -> TrueOdds` | Loads a checkpoint directory (`model.safetensors`, `model.json`, the tokenizer, and `temperature.json` if one was fitted, D27). `device` defaults to `cuda` when available. `temperature=False` gives the raw model (T = 1). No MLflow is needed. |
| `TrueOdds.predict(state: str, question: str, options: Sequence[str]) -> dict[str, float]` | One probability per option, keys in the order given, summing to 1: softmax(scores / T). |
| `TrueOdds.predict_batch(state: str, questions: Sequence[tuple[str, Sequence[str]]]) -> list[dict[str, float]]` | Many questions about one state, in one call. One dict per question, in order. The results equal `predict` on each question, up to bf16 noise (the sequences share micro-batches with different padding). |
| `TrueOdds.temperature: float` | The T in use. |

- The input to the model is exactly what training and the harness use:
  `[CLS] state [SEP] question [SEP] option [SEP]`, 512 tokens at most, the state cut at its end to
  fit, the question and the option never cut (D9, `encode.build_ids`). A caller whose state is cut
  gets no warning; the state should be kept under ~480 tokens.
- `predict` and the harness score through the same function (`predict.predict_scores`), so the
  harness numbers are the numbers `predict` gives.
- **Errors** (`ValueError`), raised before the forward pass (`Example.validate` and `build_ids`):
  - fewer than 2 options, an empty or blank option, duplicate options (a dict cannot hold both);
  - an empty question;
  - a question and an option that together exceed the 512-token budget.
- An empty state is allowed (ARC and CommonsenseQA are trained with one).

## Latency results

`harness/results/YYYY-MM-DD-HHMMSS-latency.json`, written by `harness/latency.py`, committed.

```json
{
  "kind": "latency",
  "created": "...", "run": "phase2-templates", "checkpoint": "best", "temperature": 1.0576,
  "env": {"gpu": "NVIDIA GeForce RTX 5060 Ti", "device": "cuda:0", "torch": "...", "transformers": "...",
          "attention": "sdpa", "weights": "float32", "autocast": "bfloat16"},
  "warmup": 20, "repeats": 200,
  "requests": [                      // one per (K, state length): K = 2, 4, 14 x 64, 256, 480 tokens
    {"k": 4, "state_tokens": 256, "longest_sequence": 269,
     "p50_ms": 17.7, "p95_ms": 18.8, "mean_ms": 17.9, "max_ms": 20.5}
  ],
  "batches": [                       // 10 and 50 questions about the 256-token state
    {"questions": 50, "sequences": 330, "ks": [2, 3, 4, 10, 14], "repeats": 20,
     "batch_p50_ms": 1314, "loop_p50_ms": 1270, "batch_ms_per_question": 26.3,
     "loop_ms_per_question": 25.4, "speedup": 0.97, "max_abs_diff": 0.013}
  ],
  "peak_vram_gib": 2.5
}
```

- A request is timed end to end, as a caller sees it: tokenization, the forward pass, the copy of
  the scores to the CPU (which waits for the GPU) and the softmax. Timed with `time.perf_counter`
  after `warmup` untimed calls; p50 and p95 are over `repeats` calls.
- `batches` compares `predict_batch` with a loop of `predict` on the same questions; `max_abs_diff`
  is the largest difference between their probabilities.

## Acceptance

- `tests/test_infer.py`, on the CPU with a tiny random model: the keys are the options in order and
  sum to 1; the numbers equal `ModelPredictor.predict` on the same example; `temperature.json` is
  applied and `temperature=False` ignores it; shuffling the options keeps each option's
  probability; `predict_batch` equals `predict` one by one with mixed K; each error above raises
  `ValueError`.
- Phase 4 exit (PLAN.md): the request with K = 4 and a 256-token state has p50 ≤ 30 ms and
  p95 ≤ 60 ms on the RTX 5060 Ti.
