"""The metrics on inputs whose answer is known (CLAUDE.md: known answers first)."""

import math

import numpy as np
import pytest

from harness import metrics as m


def test_perfect_model():
    probs = [np.array([0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
    labels = [1, 0]
    s = m.summarize(probs, labels)
    assert s["accuracy"] == 1.0
    assert s["ece"] == pytest.approx(0.0)
    assert s["nll"] == pytest.approx(0.0, abs=1e-9)
    assert s["brier"] == pytest.approx(0.0)


def test_always_sure_right_half_the_time_has_ece_half():
    probs = [np.array([1.0, 0.0, 0.0, 0.0])] * 1000
    labels = [0, 1] * 500
    assert m.accuracy(probs, labels) == 0.5
    assert m.ece(probs, labels) == pytest.approx(0.5)
    assert m.brier(probs, labels) == pytest.approx(1.0)  # half 0, half 2
    assert m.nll(probs, labels) == pytest.approx(0.5 * -math.log(m.EPS))


def test_labels_sampled_from_the_model_give_ece_near_zero():
    rng = np.random.default_rng(0)
    probs, labels = [], []
    for _ in range(50_000):
        k = int(rng.integers(2, 15))
        p = rng.dirichlet(np.full(k, 0.5))
        probs.append(p)
        labels.append(int(rng.choice(k, p=p)))
    assert m.ece(probs, labels) < 0.01
    # the same probabilities with the labels shuffled are miscalibrated
    shuffled = [int(rng.integers(len(p))) for p in probs]
    assert m.ece(probs, shuffled) > 0.2


def test_uniform_predictor():
    probs = [np.full(4, 0.25)] * 10 + [np.full(2, 0.5)] * 10
    labels = [3] * 10 + [1] * 10
    assert m.nll(probs, labels) == pytest.approx((math.log(4) + math.log(2)) / 2)
    assert m.brier(probs, labels) == pytest.approx((0.75 + 0.5) / 2)
    assert m.accuracy(probs, labels) == 0.0  # ties go to index 0


def test_always_a_scores_the_majority_share():
    labels = [0] * 70 + [1] * 20 + [2] * 10
    probs = [np.array([0.9, 0.05, 0.05])] * 100
    assert m.accuracy(probs, labels) == pytest.approx(0.70)
    # 90% sure, 70% right: one occupied bin, ECE = 0.2
    assert m.ece(probs, labels) == pytest.approx(0.2)


def test_reliability_bins():
    probs = [np.array([1.0, 0.0]), np.array([0.5, 0.5]), np.array([0.3, 0.7])]
    bins = m.reliability(probs, [0, 1, 1])
    assert len(bins) == 15 and sum(b["n"] for b in bins) == 3
    assert bins[-1]["n"] == 1 and bins[-1]["accuracy"] == 1.0  # 1.0 goes in the last bin
    assert bins[7]["n"] == 1 and bins[7]["confidence"] == 0.5 and bins[7]["accuracy"] == 0.0


def test_bad_input_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        m.summarize([np.array([0.5, 0.5])], [2])
    with pytest.raises(ValueError, match="probability"):
        m.summarize([np.array([0.5, 0.6])], [0])
