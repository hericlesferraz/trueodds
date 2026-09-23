# Harness

The evaluations that decide whether a phase is done. Built in Phase 0, before the model.

Everything is scored through one interface: anything that returns a probability per option for a
list of examples. The baselines, the trained model, the calibrated model and any Phase 5 variant are
all measured the same way.

## Evaluations

### Accuracy and calibration (`harness/evaluate.py`)

Per source and split (test for the training sources, the labelled split for the held-out ones):

| Metric | Meaning |
|---|---|
| Accuracy | The top option is the label |
| ECE | Expected calibration error, 15 equal-width bins over the top-1 probability (D7) |
| NLL | Mean negative log-probability of the label |
| Brier | Mean squared error between the probabilities and the one-hot label, over all options |
| Baselines | Random (1/K) and majority class, next to every number |
| By template | Accuracy on the trained question phrasings and on the held-out one (D8) |
| By K | ECE and accuracy by number of options |

A reliability diagram (confidence against observed accuracy, per bin) is saved for the pooled test
split and each held-out dataset.

A trained checkpoint is scored with `--checkpoint` (Phase 1 added it); the model goes through
`trueodds.predict.ModelPredictor`, which is a predictor like the baselines:

```bash
uv run --extra gpu python -m harness.evaluate --checkpoint ~/.trueodds/runs/<run>/best
```

### Latency (`harness/latency`, Phase 4)

p50 and p95 per request by number of options and state length, and the time for many questions
about one state as a batch.

## Results

One JSON per evaluation in `harness/results/`, named `YYYY-MM-DD-HHMMSS-<kind>.json`, committed
(spec 002). `harness/report.py` compares them in one table.
