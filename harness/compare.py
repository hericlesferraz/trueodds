"""Phase 5: compare two checkpoints question by question, with paired bootstrap intervals.

Both checkpoints must have been through `harness.calibrate`, which caches their scores on every
evaluation file (`scores.npz`) and fits their temperature; each is compared at its own T. For every
test and held-out file, and for the pooled test split, the difference B - A in accuracy, NLL and
ECE comes with a 95% interval over paired resamples of the questions. Nothing here is used for a
choice (D6): test and held-out numbers are read, as everywhere else.

    uv run python -m harness.compare --b ~/.trueodds/runs/<run>/best     # A defaults to v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from harness.calibrate import N_BOOT, SCORES_FILE, _key, dev_sets, ece_arrays, load_or_score
from harness.evaluate import eval_sets, write
from trueodds import paths
from trueodds.data.converters import SOURCES
from trueodds.data.schema import Example
from trueodds.predict import read_temperature, softmax


class _CacheOnly:
    """A scorer for `load_or_score` that refuses to score: the comparison reads caches only."""

    def __init__(self, checkpoint: Path) -> None:
        self.name = str(checkpoint)

    def scores(self, examples: Sequence[Example]) -> list[np.ndarray]:
        raise SystemExit(
            f"no up-to-date {SCORES_FILE} in {self.name}; run harness.calibrate on it first"
        )


def per_question(probs: Sequence[np.ndarray], labels: Sequence[int]) -> dict[str, np.ndarray]:
    """Hit, NLL term and top-1 confidence of each question."""
    y = np.asarray(labels)
    p_true = np.array([p[t] for p, t in zip(probs, y, strict=True)])
    return {
        "correct": np.array([float(np.argmax(p) == t) for p, t in zip(probs, y, strict=True)]),
        "nll": -np.log(np.clip(p_true, 1e-12, None)),
        "conf": np.array([float(np.max(p)) for p in probs]),
    }


def paired(a: dict, b: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """B - A in accuracy, NLL and ECE, each with a 95% interval over the same resamples."""

    def stats(q: dict, ix: np.ndarray) -> np.ndarray:
        return np.array(
            [
                q["correct"][ix].mean(),
                q["nll"][ix].mean(),
                ece_arrays(q["conf"][ix], q["correct"][ix]),
            ]
        )

    n = len(a["correct"])
    everything = np.arange(n)
    point = stats(b, everything) - stats(a, everything)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 3))
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        boot[i] = stats(b, ix) - stats(a, ix)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = {"n": n, "n_boot": n_boot}
    for j, name in enumerate(("accuracy", "nll", "ece")):
        out[name] = {
            "a": float(stats(a, everything)[j]),
            "b": float(stats(b, everything)[j]),
            "delta": float(point[j]),
            "ci95": [float(lo[j]), float(hi[j])],
        }
    return out


def data_stamp(data_dir: Path) -> str | None:
    stats_path = data_dir / "stats.json"
    return json.loads(stats_path.read_text()).get("created") if stats_path.exists() else None


def cached_scores(
    ckpt: Path, evals: dict[str, list[Example]], data_dir: Path = paths.DATA
) -> tuple[dict[tuple, np.ndarray], float]:
    """A checkpoint's cached scores on every evaluation file, by example key, and its T."""
    sets = {**dev_sets(data_dir), **evals}  # the cache holds dev too; its key must match
    scores = load_or_score(_CacheOnly(ckpt), sets, ckpt / SCORES_FILE, data_stamp(data_dir))
    table = {_key(ex): s for k in evals for ex, s in zip(evals[k], scores[k], strict=True)}
    return table, read_temperature(ckpt)


def compare(a: Path, b: Path, data_dir: Path = paths.DATA) -> dict:
    stamp = data_stamp(data_dir)
    evals = eval_sets(data_dir)
    side = {}
    for name, ckpt in (("a", a), ("b", b)):
        table, t = cached_scores(ckpt, evals, data_dir)
        side[name] = {"t": t, "table": table}

    def questions(name: str, exs: list[Example]) -> dict:
        t, table = side[name]["t"], side[name]["table"]
        return per_question(
            [softmax(table[_key(ex)], t) for ex in exs], [ex.label_idx for ex in exs]
        )

    results = {}
    pooled: list[Example] = []
    for k, exs in evals.items():
        results[k] = {
            "role": SOURCES[k.split("/")[0]].role,
            **paired(questions("a", exs), questions("b", exs)),
        }
        if k.endswith("/test") and SOURCES[k.split("/")[0]].role == "train":
            pooled.extend(exs)
    if pooled:
        results["pooled/test"] = {
            "role": "train",
            **paired(questions("a", pooled), questions("b", pooled)),
        }
    return {
        "kind": "compare",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "a": {"run": a.parent.name, "checkpoint": a.name, "temperature": side["a"]["t"]},
        "b": {"run": b.parent.name, "checkpoint": b.name, "temperature": side["b"]["t"]},
        "data_stats_created": stamp,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, default=paths.DEFAULT_CHECKPOINT, help="default: v1")
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA)
    args = parser.parse_args()
    result = compare(args.a.expanduser().resolve(), args.b.expanduser().resolve(), args.data_dir)
    path = write(result)
    pooled = result["results"].get("pooled/test")
    if pooled:
        acc = pooled["accuracy"]
        lo, hi = acc["ci95"]
        print(f"pooled test accuracy {acc['a']:.4f} -> {acc['b']:.4f}: "
              f"{acc['delta']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")  # fmt: skip
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
