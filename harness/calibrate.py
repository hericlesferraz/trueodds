"""Phase 3: fit one temperature on dev and report its effect on test and held-out (D27, spec 002).

The checkpoint is run once over dev and every evaluation file, and the scores are cached next to
it. Everything after that runs on the CPU from the cache: the fit, the before (T = 1) and after
evaluations through `harness.evaluate`, and a paired bootstrap of the ECE change.

    uv run --extra gpu python -m harness.calibrate --checkpoint ~/.trueodds/runs/<run>/best
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from harness import metrics
from harness.evaluate import eval_sets, evaluate, mirror_to_mlflow, write
from trueodds import paths
from trueodds.calibrate import fit_temperature, write_temperature
from trueodds.data.converters import SOURCES
from trueodds.data.schema import Example, load_examples
from trueodds.predict import softmax

SCORES_FILE = "scores.npz"  # in the checkpoint dir; under ~/.trueodds, never committed (D12)
N_BOOT = 1000


class Scorer(Protocol):
    name: str

    def scores(self, examples: Sequence[Example]) -> list[np.ndarray]: ...


def _key(ex: Example) -> tuple[str, str, str]:
    # An id alone repeats across a source's test and held-out-template files; with the template
    # and the question text it is unique over every file.
    return ex.id, ex.template_id, ex.question


class CachedScores:
    """A harness predictor that looks up cached scores and applies a temperature."""

    def __init__(self, scores: dict[tuple, np.ndarray], temperature: float, name: str) -> None:
        self.table = scores
        self.temperature = temperature
        self.name = name if temperature == 1.0 else f"{name}@T={temperature:.4f}"

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return [softmax(self.table[_key(ex)], self.temperature) for ex in examples]


def dev_sets(data_dir: Path) -> dict[str, list[Example]]:
    """`<source>/dev` for every training source: the only data T is fitted on (D6)."""
    return {
        f"{name}/dev": load_examples(data_dir / name / "dev.jsonl")
        for name, src in SOURCES.items()
        if src.role == "train" and (data_dir / name / "dev.jsonl").exists()
    }


def load_or_score(
    scorer: Scorer, sets: dict[str, list[Example]], cache: Path | None, stamp: str | None
) -> dict[str, list[np.ndarray]]:
    """Scores per set, from `cache` when it was written for the same sets and data build."""
    if cache is not None and cache.exists():
        with np.load(cache) as z:
            meta = json.loads(str(z["meta"]))
            if meta["data_stats_created"] == stamp and meta["sets"] == {
                k: len(v) for k, v in sets.items()
            }:
                return {k: np.split(z[f"{k}/flat"], z[f"{k}/offsets"][1:-1]) for k in meta["sets"]}
    out = {}
    for k, examples in sets.items():
        print(f"scoring {k} ({len(examples)})", flush=True)
        # float32, as the model returns them and as the cache stores them: the first run and
        # a cached one fit the same T.
        out[k] = [np.asarray(x, dtype=np.float32) for x in scorer.scores(examples)]
    if cache is not None:
        arrays = {
            "meta": json.dumps(
                {"data_stats_created": stamp, "sets": {k: len(v) for k, v in sets.items()}}
            )
        }
        for k, s in out.items():
            arrays[f"{k}/flat"] = np.concatenate(s)
            arrays[f"{k}/offsets"] = np.cumsum([0] + [len(x) for x in s])
        np.savez(cache, **arrays)
    return out


def ece_arrays(conf: np.ndarray, correct: np.ndarray, n_bins: int = metrics.N_BINS) -> float:
    """`metrics.ece` from top-1 confidences and hits, vectorized for the bootstrap."""
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    gap = np.bincount(idx, weights=correct - conf, minlength=n_bins)
    return float(np.abs(gap).sum() / len(conf))


def ece_change(
    before: Sequence[np.ndarray],
    after: Sequence[np.ndarray],
    labels: Sequence[int],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict:
    """ECE after minus before, and its 95% interval over paired resamples of the questions."""
    pred, conf_b = metrics.top1(before)
    _, conf_a = metrics.top1(after)  # same argmax: a temperature does not reorder the options
    correct = (pred == np.asarray(labels)).astype(float)
    delta = ece_arrays(conf_a, correct) - ece_arrays(conf_b, correct)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, len(correct), len(correct))
        boot[b] = ece_arrays(conf_a[ix], correct[ix]) - ece_arrays(conf_b[ix], correct[ix])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": delta, "ci95": [float(lo), float(hi)], "n_boot": n_boot}


def calibrate(
    scorer: Scorer,
    data_dir: Path = paths.DATA,
    checkpoint: Path | None = None,
    run: str | None = None,
    rescore: bool = False,
) -> dict:
    """Fit T on dev, write it to the checkpoint (if given), and return the calibration result."""
    stats_path = data_dir / "stats.json"
    stamp = json.loads(stats_path.read_text()).get("created") if stats_path.exists() else None
    dev, evals = dev_sets(data_dir), eval_sets(data_dir)
    if not dev:
        raise SystemExit(f"no dev files under {data_dir}; run scripts/prepare_data.py")
    sets = {**dev, **evals}
    cache = checkpoint / SCORES_FILE if checkpoint is not None else None
    if rescore and cache is not None and cache.exists():
        cache.unlink()
    scores = load_or_score(scorer, sets, cache, stamp)

    dev_ex = [ex for k in dev for ex in dev[k]]
    dev_scores = [s for k in dev for s in scores[k]]
    dev_labels = [ex.label_idx for ex in dev_ex]
    t = fit_temperature(dev_scores, dev_labels)

    def dev_stats(temp: float) -> dict:
        m = metrics.summarize([softmax(s, temp) for s in dev_scores], dev_labels, bins=False)
        return {k: m[k] for k in ("accuracy", "ece", "nll", "brier")}

    fit = {
        "sources": [k.split("/")[0] for k in dev],
        "n": len(dev_ex),
        "dev_before": dev_stats(1.0),
        "dev_after": dev_stats(t),
    }
    if checkpoint is not None:
        write_temperature(checkpoint, t, fit)

    table = {_key(ex): s for k, exs in sets.items() for ex, s in zip(exs, scores[k], strict=True)}
    ckpt_name = checkpoint.name if checkpoint is not None else None
    before = evaluate(CachedScores(table, 1.0, scorer.name), data_dir, run, ckpt_name)
    after = evaluate(CachedScores(table, t, scorer.name), data_dir, run, ckpt_name)

    change = {}
    pooled_ex: list[Example] = []
    for k, exs in evals.items():
        s = scores[k]
        change[k] = ece_change([softmax(x) for x in s], [softmax(x, t) for x in s],
                               [ex.label_idx for ex in exs])  # fmt: skip
        if k.endswith("/test") and SOURCES[k.split("/")[0]].role == "train":
            pooled_ex.extend(exs)
    if pooled_ex:
        s = [table[_key(ex)] for ex in pooled_ex]
        change["pooled/test"] = ece_change([softmax(x) for x in s], [softmax(x, t) for x in s],
                                           [ex.label_idx for ex in pooled_ex])  # fmt: skip

    return {
        "kind": "calibration",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run": run,
        "checkpoint": ckpt_name,
        "predictor": scorer.name,
        "data_stats_created": stamp,
        "temperature": {"value": t, "fit_on": "dev", **fit},
        "before": before["results"],
        "after": after["results"],
        "ece_change": change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA)
    parser.add_argument("--rescore", action="store_true", help="ignore the cached scores")
    args = parser.parse_args()
    from trueodds.predict import ModelPredictor

    ckpt = args.checkpoint.expanduser().resolve()
    scorer = ModelPredictor.load(ckpt, temperature=False)
    result = calibrate(scorer, args.data_dir, ckpt, ckpt.parent.name, args.rescore)
    path = write(result)
    mirror_to_mlflow(result, path, ckpt)
    t = result["temperature"]
    pooled = result["ece_change"].get("pooled/test")
    print(
        f"T = {t['value']:.4f} (dev NLL {t['dev_before']['nll']:.4f} -> {t['dev_after']['nll']:.4f})"
    )
    if pooled:
        lo, hi = pooled["ci95"]
        print(f"pooled test ECE change {pooled['delta']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
