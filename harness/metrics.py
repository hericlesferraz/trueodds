"""Accuracy, ECE, NLL and Brier over questions with any number of options (D7, spec 002).

`probs` is a list with one probability vector per question (K may differ between questions), and
`labels` the index of the correct option of each.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_BINS = 15
EPS = 1e-12


def _check(probs: Sequence[np.ndarray], labels: Sequence[int]) -> None:
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} probability vectors for {len(labels)} labels")
    for i, (p, y) in enumerate(zip(probs, labels, strict=True)):
        if not 0 <= y < len(p):
            raise ValueError(f"question {i}: label {y} out of range for K = {len(p)}")
        if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-4):
            raise ValueError(f"question {i}: not a probability vector (sum {p.sum():.6f})")


def top1(probs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(predicted index, its probability) per question; ties go to the lowest index."""
    pred = np.array([int(np.argmax(p)) for p in probs], dtype=int)
    conf = np.array([float(p[i]) for p, i in zip(probs, pred, strict=True)])
    return pred, conf


def accuracy(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    pred, _ = top1(probs)
    return float(np.mean(pred == np.asarray(labels)))


def nll(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    return float(
        -np.mean([np.log(max(float(p[y]), EPS)) for p, y in zip(probs, labels, strict=True)])
    )


def brier(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    """Squared error against the one-hot label, summed over options, mean over questions (0 to 2)."""
    total = 0.0
    for p, y in zip(probs, labels, strict=True):
        onehot = np.zeros_like(p, dtype=float)
        onehot[y] = 1.0
        total += float(np.sum((p - onehot) ** 2))
    return total / len(labels)


def reliability(
    probs: Sequence[np.ndarray], labels: Sequence[int], n_bins: int = N_BINS
) -> list[dict]:
    """Per equal-width bin of top-1 probability: its range, count, mean confidence and accuracy.

    Bin b holds confidences in [b/n, (b+1)/n); the last bin also holds 1.0.
    """
    pred, conf = top1(probs)
    correct = pred == np.asarray(labels)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        bins.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": n,
                "confidence": float(conf[mask].mean()) if n else None,
                "accuracy": float(correct[mask].mean()) if n else None,
            }
        )
    return bins


def ece(probs: Sequence[np.ndarray], labels: Sequence[int], n_bins: int = N_BINS) -> float:
    """Sum over bins of (bin share) x |accuracy - mean confidence|."""
    total = len(labels)
    return float(
        sum(
            b["n"] / total * abs(b["accuracy"] - b["confidence"])
            for b in reliability(probs, labels, n_bins)
            if b["n"]
        )
    )


def summarize(probs: Sequence[np.ndarray], labels: Sequence[int], bins: bool = True) -> dict:
    _check(probs, labels)
    out = {
        "n": len(labels),
        "accuracy": accuracy(probs, labels),
        "ece": ece(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
    }
    if bins:
        out["reliability"] = reliability(probs, labels)
    return out
