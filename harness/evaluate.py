"""Score a predictor on every evaluation file and write the results JSON (spec 002).

A predictor is anything with a `name` and `predict(examples) -> list of probability vectors`.
The baselines are computed and written next to every result, so each number has its reference.

    uv run python -m harness.evaluate --baselines      # the baselines alone, before any model
    uv run --extra gpu python -m harness.evaluate --checkpoint ~/.trueodds/runs/<run>/best
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from harness import metrics
from harness.baselines import Majority, Prior, Uniform
from trueodds import paths
from trueodds.data.converters import SOURCES
from trueodds.data.schema import Example, load_examples

EVAL_FILES = ("test", "test-heldout-template", "test_mismatched")


class Predictor(Protocol):
    name: str

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]: ...


def eval_sets(data_dir: Path = paths.DATA) -> dict[str, list[Example]]:
    """`<source>/<file>` -> examples, for every evaluation file that exists."""
    out = {}
    for name in SOURCES:
        for file in EVAL_FILES:
            path = data_dir / name / f"{file}.jsonl"
            if path.exists():
                out[f"{name}/{file}"] = load_examples(path)
    return out


def score(probs: list[np.ndarray], examples: Sequence[Example], bins: bool = True) -> dict:
    labels = [ex.label_idx for ex in examples]
    out = metrics.summarize(probs, labels, bins=bins)
    for field, key in (("by_k", lambda ex: ex.k), ("by_template", lambda ex: ex.template_id)):
        groups: dict = defaultdict(list)
        for i, ex in enumerate(examples):
            groups[key(ex)].append(i)
        out[field] = {
            str(g): {
                "n": len(ix),
                "accuracy": metrics.accuracy([probs[i] for i in ix], [labels[i] for i in ix]),
                "ece": metrics.ece([probs[i] for i in ix], [labels[i] for i in ix]),
            }
            for g, ix in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }
    return out


def baseline_score(name: str, probs: list[np.ndarray], examples: Sequence[Example]) -> dict:
    """Like `score`, except that the random baseline's accuracy is that of a random pick.

    The uniform vector's argmax is a tie that always goes to option 0, so its measured accuracy
    would be the share of label 0. A random pick is right with probability 1/K, and has no
    confidence to calibrate, so its ECE is not reported.
    """
    out = score(probs, examples, bins=False)
    if name == "random":
        out["accuracy"] = float(np.mean([1.0 / ex.k for ex in examples]))
        out["ece"] = None
        for field, key in (
            ("by_k", lambda ex: str(ex.k)),
            ("by_template", lambda ex: ex.template_id),
        ):
            for group, g in out[field].items():
                g["accuracy"] = float(np.mean([1.0 / ex.k for ex in examples if key(ex) == group]))
                g["ece"] = None
    return out


def baseline_predictors(source: str, data_dir: Path) -> tuple[list[Predictor], str]:
    """Fitted on the source's train split, or for a held-out source on its own test split."""
    fit_split = "train" if SOURCES[source].role == "train" else "test"
    fit_on = load_examples(data_dir / source / f"{fit_split}.jsonl")
    return [Uniform(), Majority(fit_on), Prior(fit_on)], fit_split


def evaluate(
    predictor: Predictor | None,
    data_dir: Path = paths.DATA,
    run: str | None = None,
    checkpoint: str | None = None,
) -> dict:
    sets = eval_sets(data_dir)
    if not sets:
        raise SystemExit(f"no evaluation files under {data_dir}; run scripts/prepare_data.py")
    stats_path = data_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    results: dict[str, dict] = {}
    pooled: dict[str, list] = defaultdict(list)  # predictor name -> probs over pooled test
    pooled_examples: list[Example] = []
    fitted: dict[str, tuple[list[Predictor], str]] = {}
    for key, examples in sets.items():
        source, file = key.split("/")
        if source not in fitted:
            fitted[source] = baseline_predictors(source, data_dir)
        baselines, fit_split = fitted[source]
        pooled_here = file == "test" and SOURCES[source].role == "train"
        entry: dict = {
            "source": source,
            "split": file,
            "role": SOURCES[source].role,
            "n": len(examples),
        }
        entry["metrics"] = None
        if predictor:
            probs = predictor.predict(examples)
            entry["metrics"] = score(probs, examples)
            if pooled_here:
                pooled["model"].extend(probs)
        entry["baselines"] = {}
        for b in baselines:
            probs = b.predict(examples)
            entry["baselines"][b.name] = baseline_score(b.name, probs, examples)
            entry["baselines"][b.name]["fit_on"] = None if b.name == "random" else fit_split
            if b.name == "majority":
                entry["baselines"][b.name]["label"] = b.label
            if pooled_here:
                pooled[b.name].extend(probs)
        if pooled_here:
            pooled_examples.extend(examples)
        results[key] = entry

    if pooled_examples:
        results["pooled/test"] = {
            "source": "pooled",
            "split": "test",
            "role": "train",
            "n": len(pooled_examples),
            "metrics": score(pooled["model"], pooled_examples) if predictor else None,
            "baselines": {
                name: baseline_score(name, probs, pooled_examples)
                for name, probs in pooled.items()
                if name != "model"
            },
        }

    return {
        "kind": "eval" if predictor else "baselines",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run": run,
        "checkpoint": checkpoint,
        "predictor": predictor.name if predictor else None,
        "data_stats_created": stats.get("created"),
        "results": results,
    }


def write(result: dict, results_dir: Path = paths.RESULTS) -> Path:
    stamp = dt.datetime.fromisoformat(result["created"]).strftime("%Y-%m-%d-%H%M%S")
    path = results_dir / f"{stamp}-{result['kind']}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--baselines", action="store_true", help="score the baselines alone")
    which.add_argument("--checkpoint", type=Path, help="a checkpoint dir written by the trainer")
    parser.add_argument("--data-dir", type=Path, default=paths.DATA)
    args = parser.parse_args()
    if args.baselines:
        path = write(evaluate(None, args.data_dir))
    else:
        from trueodds.predict import ModelPredictor

        ckpt = args.checkpoint.expanduser().resolve()
        predictor = ModelPredictor.load(ckpt)
        path = write(evaluate(predictor, args.data_dir, run=ckpt.parent.name, checkpoint=ckpt.name))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
