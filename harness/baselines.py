"""Predictors that read no text: uniform (random), majority class and label prior (D7, spec 002).

Each one fits on the examples it is given (the source's train split, or for a held-out source its
own test split) and predicts a probability vector per question, like a trained model does, so the
harness scores all of them the same way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from trueodds.data.schema import Example


def _key(ex: Example, by_text: bool) -> str | int:
    return ex.label if by_text else ex.label_idx


def fixed_options(examples: Sequence[Example]) -> bool:
    """True when every question offers the same options (yes/no, topic labels): labels are texts."""
    return len({tuple(ex.options) for ex in examples}) == 1


class Uniform:
    name = "random"

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return [np.full(ex.k, 1.0 / ex.k) for ex in examples]


class Majority:
    """All probability on the most frequent label: its text for fixed option sets, else its index."""

    name = "majority"

    def __init__(self, fit_on: Sequence[Example]) -> None:
        self.by_text = fixed_options(fit_on)
        counts = Counter(_key(ex, self.by_text) for ex in fit_on)
        # most_common breaks ties by first occurrence; sort first so ties go to the smallest key
        self.label = max(sorted(counts, key=str), key=lambda k: counts[k])

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        out = []
        for ex in examples:
            p = np.zeros(ex.k)
            if self.by_text and self.label in ex.options:
                p[ex.options.index(self.label)] = 1.0
            elif not self.by_text and self.label < ex.k:
                p[self.label] = 1.0
            else:
                p[:] = 1.0 / ex.k
            out.append(p)
        return out


class Prior:
    """The label frequencies of the fit examples, renormalized over each question's K options.

    A model that knows only how often each answer is right: perfectly calibrated on average and no
    better than majority at ranking. Its NLL and Brier are the bar a model must pass to show it
    reads the text.
    """

    name = "prior"

    def __init__(self, fit_on: Sequence[Example], smoothing: float = 1.0) -> None:
        self.by_text = fixed_options(fit_on)
        self.counts = Counter(_key(ex, self.by_text) for ex in fit_on)
        self.smoothing = smoothing

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        out = []
        for ex in examples:
            keys = ex.options if self.by_text else range(ex.k)
            p = np.array([self.counts.get(k, 0) + self.smoothing for k in keys], dtype=float)
            out.append(p / p.sum())
        return out
