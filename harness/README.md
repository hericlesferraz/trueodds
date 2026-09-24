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

A trained checkpoint is scored at its fitted temperature when it has one (`temperature.json`,
Phase 3); the eval JSON records the `temperature` it used.

### Calibration (`harness/calibrate.py`, Phase 3)

Fits one temperature on the pooled dev split and reports what it does on every test and held-out
file (D27, spec 002). The checkpoint is scored once, and the scores are cached in the checkpoint
directory, so the fit and both evaluations run from the same numbers:

```bash
uv run --extra gpu python -m harness.calibrate --checkpoint ~/.trueodds/runs/<run>/best
uv run python -m harness.report harness/results/<stamp>-calibration.json     # before/after tables
uv run --extra plots python -m harness.plots harness/results/<stamp>-calibration.json
```

The report gives ECE, NLL and Brier before and after per file, the held-out datasets in their own
block, a paired bootstrap 95% interval on each ECE change, and ECE by number of options. The
reliability diagrams (confidence against observed accuracy per bin, before and after on the same
axes) are written for the pooled test split and each held-out dataset to `harness/results/figures/`.

A trained checkpoint is scored with `--checkpoint` (Phase 1 added it); the model goes through
`trueodds.predict.ModelPredictor`, which is a predictor like the baselines:

```bash
uv run --extra gpu python -m harness.evaluate --checkpoint ~/.trueodds/runs/<run>/best
```

If the checkpoint's run was tracked in MLflow (D24), the result is also mirrored into that run: the
JSON as an artifact, and per file accuracy, NLL and ECE as `harness/...` metrics. The JSON in
`harness/results/` stays the record.

### Latency (`harness/latency.py`, Phase 4)

p50 and p95 per request of `predict()` (spec 003) at K = 2, 4 and 14 options and states of 64, 256
and 480 tokens, after warmup, and the time for 10 and 50 questions about one state with
`predict_batch` against the same questions one by one:

```bash
uv run --extra gpu python -m harness.latency [--checkpoint ~/.trueodds/runs/<run>/best]
uv run python -m harness.report harness/results/<stamp>-latency.json
```

## Results

One JSON per evaluation in `harness/results/`, named `YYYY-MM-DD-HHMMSS-<kind>.json`, committed
(spec 002). `harness/report.py` compares them in one table.
