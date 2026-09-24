# 002 — Run report

**Phase 0.** The results JSON of one evaluation, and the exact definitions of its metrics and
baselines (D7). Written by `harness/evaluate.py`, read by `harness/report.py` and by anyone checking
a claim in the README.

## File

`harness/results/YYYY-MM-DD-HHMMSS-<kind>.json`, committed. `kind` is `baselines` (no model) or
`eval` (a model). Other kinds in the same folder: `gpu-bench-<attn>`, `data-stats`,
`calibration` (below) and `latency` (spec 003).

```json
{
  "kind": "eval",
  "created": "2026-09-23T14:02:11",
  "run": "phase2-base-cls",            // the run's name, null for baselines
  "checkpoint": "step-12000",           // null for baselines
  "predictor": "modernbert-cross-encoder",
  "temperature": 1.0,                   // the model's T (D27); null for baselines, 1.0 when none fitted
  "data_stats_created": "2026-09-23T12:40:05",   // ties the result to one build of the data
  "results": {
    "<source>/<file>": {
      "source": "boolq", "split": "test", "role": "train", "n": 3270,
      "metrics": {                      // null in a baselines file
        "n": 3270, "accuracy": 0.81, "ece": 0.04, "nll": 0.44, "brier": 0.27,
        "reliability": [{"lo": 0.0, "hi": 0.0667, "n": 0, "confidence": null, "accuracy": null}, ...],
        "by_k":        {"2": {"n": 3270, "accuracy": 0.81, "ece": 0.04}},
        "by_template": {"boolq:0": {"n": 820, "accuracy": 0.80, "ece": 0.05}, ...}
      },
      "baselines": {
        "random":   {"n", "accuracy", "ece": null, "nll", "brier", "by_k", "by_template", "fit_on": null},
        "majority": {..., "fit_on": "train", "label": "yes"},
        "prior":    {..., "fit_on": "train"}
      }
    },
    "pooled/test": { ... }              // every training source's test file together
  }
}
```

`<file>` is `test`, `test-heldout-template` or `test_mismatched` (spec 001). `role` is `train` or
`heldout`. `pooled/test` concatenates the `test` files of the training sources. It is the set of
the Phase 3 reliability diagram and ECE target.

## Metric definitions

For question i with K_i options, probabilities p_i (summing to 1) and label y_i, over N questions:

| Metric | Definition |
|---|---|
| accuracy | mean of [argmax_k p_ik = y_i]; ties go to the lowest index |
| confidence | c_i = max_k p_ik (the top-1 probability) |
| ECE | 15 equal-width bins on c_i: bin b holds [b/15, (b+1)/15), and the last bin also holds 1.0. ECE = Σ_b (n_b/N) · \|acc_b − conf_b\|, over non-empty bins |
| NLL | −mean log max(p_iy_i, 1e-12) |
| Brier | mean over questions of Σ_k (p_ik − [k = y_i])²; 0 is perfect, 2 is sure and wrong |
| reliability | per bin: range, n_b, conf_b (mean confidence), acc_b (accuracy) |

`by_k` and `by_template` repeat accuracy and ECE per option count and per template id.

## Baselines

All three are fitted on the source's `train` file. A held-out source has no train file, so they are
fitted on its own `test` file, and `fit_on` says so. This gives a held-out majority baseline more
than it would get in practice, which makes it a stricter bar.

| Baseline | Prediction | Notes |
|---|---|---|
| random | uniform 1/K | accuracy is **mean 1/K**, the expected accuracy of a random pick, not the argmax of a tie; ECE is `null` |
| majority | all the probability on the most frequent label: its **text** when every question has the same options (yes/no, topic labels), else its **index** (ties go to the smallest) | uniform when that label is not among a question's options; NLL is near −log 1e-12 and only shows that a sure guess is heavily penalized |
| prior | the label frequencies (by text or by index as above), +1 smoothing, renormalized over the question's options | knows how often each answer is right but reads no text. Its NLL and Brier are the bar a model must pass on calibration; its ECE is near 0 on its fit set, which is why ECE alone is not enough (D7) |

## Calibration results (Phase 3)

`harness/results/YYYY-MM-DD-HHMMSS-calibration.json`, written by `harness/calibrate.py` (D27). One
file holds the whole before/after comparison, so the claim and its evidence cannot drift apart.

```json
{
  "kind": "calibration",
  "created": "...", "run": "phase2-templates", "checkpoint": "best",
  "predictor": "model:phase2-templates/best", "data_stats_created": "...",
  "temperature": {
    "value": 1.03, "fit_on": "dev", "sources": ["boolq", ...], "n": 11286,
    "dev_before": {"accuracy", "ece", "nll", "brier"},    // at T = 1
    "dev_after":  {"accuracy", "ece", "nll", "brier"}     // at the fitted T
  },
  "before": { "<source>/<file>": {...}, "pooled/test": {...} },   // the `results` of an eval at T = 1
  "after":  { ... },                                              // the same at the fitted T
  "ece_change": { "<source>/<file>": {"delta": -0.002, "ci95": [-0.004, 0.000], "n_boot": 1000}, ... }
}
```

- **Temperature:** p = softmax(s / T) over a question's scores s. T minimizes the mean NLL over
  the pooled `dev` files of the training sources, and nothing else. It is written to
  `<checkpoint>/temperature.json`, which `ModelPredictor.load` applies.
- **`before` and `after`** are computed from the same cached scores through `harness.evaluate`,
  so they differ only by T. Accuracy is identical in both: dividing by T > 0 does not change the
  order of the options.
- **`ece_change`:** delta = ECE_after − ECE_before on that file. ci95 is the 2.5th and 97.5th
  percentile of the same difference over `n_boot` paired resamples of the file's questions (drawn
  with replacement, seed 0; the same resample scores both). An interval that contains 0 means the
  change is within sampling noise.

## Acceptance

- `tests/test_metrics.py` checks known answers: a perfect model (accuracy 1, ECE 0, NLL 0,
  Brier 0), always sure and right half the time (ECE 0.5), labels sampled from the model's own
  probabilities (ECE < 0.01), the uniform predictor (NLL = mean log K), and always-"A" (accuracy = the
  majority share).
- `tests/test_evaluate.py` checks the baselines, the file, the pooled set and the table on a small
  fixture data directory, and that an oracle predictor scores accuracy 1 everywhere.
- `tests/test_calibrate.py` checks the temperature fit on known answers (labels drawn from
  softmax(z / T0) give back T0, with mixed K and padding), that the fit never raises its own NLL,
  that a checkpoint's `temperature.json` is applied, that the vectorized ECE of the bootstrap
  equals `metrics.ece`, and the calibration file end to end on the fixture data directory.
