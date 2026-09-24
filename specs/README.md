# Specs

Specs define **contracts that are not obvious from the code**: data formats and anything two parts
of the system (or the harness and the trainer) must agree on.

## Rules

- **Write a spec when the phase that needs it starts**, not before. A spec for a component nobody
  is building yet goes stale (see D1 in `docs/DECISIONS.md`).
- **Keep it short:** purpose, the contract itself (fields, types, examples), and acceptance criteria
  that the harness or tests can check.
- **Update the spec in the same commit** as any change that makes it inaccurate.
- Name files `NNN-short-name.md`.

## Written

| Spec | Phase | Contract |
|---|---|---|
| `001-example-format.md` | 0 | The unified example (`id, source, split, state, question, options[], label_idx, template_id`), the split rule, the question templates with the held-out one per task, and the dedup rule |
| `002-run-report.md` | 0 | The results JSON of one evaluation, and the exact definitions of accuracy, ECE, NLL, Brier and the baselines |
| `003-predict.md` | 4 | `trueodds.load()`, `predict(state, question, options) -> {option: probability}` and `predict_batch`, many questions about one state; the latency results JSON |
| `004-shared-state.md` | 5 | The packed sequence of the shared-state model (D13): layout, attention masks, positions, pooling, truncation, and the `architecture` field of a checkpoint |
