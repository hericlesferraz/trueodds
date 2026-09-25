"""Phase 5E: is underconfidence on unseen labels about the number of options, or the labels?

DBpedia-14 (14 labels never trained on) is underconfident in every model. Here each question is
scored again with fewer options: its answer plus K - 1 distractors drawn at random. Options never
see each other, in the cross-encoder (D2) and in the shared-state model (spec 004), so the scores
of a subset are exactly the subset of the scores `harness.calibrate` cached: no forward pass.

The same is done to Yahoo and AG News, whose labels were trained on. If DBpedia at K = 4 is still
underconfident while Yahoo and AG News at K = 4 are not, the cause is the new labels; if its gap
closes as K falls, it is K.

Distractors are drawn uniformly, so every subset holding the answer is equally likely whichever
option the answer is: a calibrated model stays calibrated under this subsetting, and a gap that
appears is the model's, not the method's.

    uv run python -m harness.options      # v1, -large and shared, from their cached scores
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from harness import metrics
from harness.calibrate import N_BOOT, _key, ece_arrays
from harness.compare import cached_scores, data_stamp
from harness.evaluate import eval_sets, write
from trueodds import paths
from trueodds.data.schema import Example
from trueodds.predict import softmax

SEED = 0
CELLS = {  # evaluation file -> option counts; the last is the file's own K
    "dbpedia/test": (2, 4, 10, 14),
    "yahoo/test": (2, 4, 10),
    "ag_news/test": (2, 4),
}
CHECKPOINTS = ("phase2-templates", "phase5-large", "phase5-shared")


def keep(ex: Example, k: int, seed: int = SEED) -> list[int]:
    """Indices of the options kept: the answer and k - 1 others, the same draw for every model."""
    if k >= ex.k:
        return list(range(ex.k))
    digest = hashlib.sha256(f"{seed}/{ex.id}/{k}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    others = [j for j in range(ex.k) if j != ex.label_idx]
    return sorted([ex.label_idx, *rng.choice(others, k - 1, replace=False).tolist()])


def restrict(
    scores: Sequence[np.ndarray], examples: Sequence[Example], k: int, temperature: float
) -> tuple[list[np.ndarray], list[int]]:
    """Probabilities over each question's kept options, and the answer's index among them."""
    probs, labels = [], []
    for s, ex in zip(scores, examples, strict=True):
        kept = keep(ex, k)
        probs.append(softmax(np.asarray(s)[kept], temperature))
        labels.append(kept.index(ex.label_idx))
    return probs, labels


def cell(probs: Sequence[np.ndarray], labels: Sequence[int], n_boot: int = N_BOOT) -> dict:
    """Accuracy, mean confidence, their gap (with a 95% interval) and ECE."""
    pred, conf = metrics.top1(probs)
    correct = (pred == np.asarray(labels)).astype(float)
    rng = np.random.default_rng(SEED)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, len(conf), len(conf))
        boot[b] = conf[ix].mean() - correct[ix].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "confidence": float(conf.mean()),
        "gap": float(conf.mean() - correct.mean()),  # > 0 overconfident, < 0 underconfident
        "gap_ci95": [float(lo), float(hi)],
        "ece": ece_arrays(conf, correct),
    }


def options(checkpoints: Sequence[Path], data_dir: Path = paths.DATA) -> dict:
    evals = eval_sets(data_dir)
    models = {}
    for ckpt in checkpoints:
        table, t = cached_scores(ckpt, evals, data_dir)
        results = {}
        for file, ks in CELLS.items():
            exs = evals[file]
            scores = [table[_key(ex)] for ex in exs]
            results[file] = {str(k): cell(*restrict(scores, exs, k, t)) for k in ks}
        models[ckpt.parent.name] = {"checkpoint": ckpt.name, "temperature": t, "results": results}
    return {
        "kind": "options",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "data_stats_created": data_stamp(data_dir),
        "seed": SEED,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[paths.RUNS / run / "best" for run in CHECKPOINTS],
    )
    parser.add_argument("--data-dir", type=Path, default=paths.DATA)
    args = parser.parse_args()
    ckpts = [c.expanduser().resolve() for c in args.checkpoints]
    result = options(ckpts, args.data_dir)
    for run, m in result["models"].items():
        for file, cells in m["results"].items():
            gaps = ", ".join(f"K={k} {c['gap']:+.3f}" for k, c in cells.items())
            print(f"{run} {file}: confidence - accuracy {gaps}")
    print(f"wrote {write(result)}")


if __name__ == "__main__":
    main()
