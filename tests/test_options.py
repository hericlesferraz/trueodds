"""Phase 5E (`harness.options`): fewer options per question, on answers known in advance."""

import json
import random

import numpy as np
import pytest

from harness import metrics
from harness.options import cell, keep, restrict
from harness.report import options_tables
from tests.conftest import WORDS, make_model
from trueodds.data.schema import Example
from trueodds.encode import Encoder
from trueodds.predict import predict_scores, softmax
from trueodds.shared import SharedEncoder


def example(i, k, label, state="", options=None):
    options = options or [f"w{j}" for j in range(k)]
    return Example(id=f"t/test/{i}", source="t", split="test", state=state, question="w1 w2",
                   options=options, label_idx=label, template_id="native")  # fmt: skip


def test_the_answer_is_always_kept_and_the_draw_is_fixed():
    for i in range(50):
        ex = example(i, 14, label=i % 14)
        kept = keep(ex, 4)
        assert len(kept) == 4 and ex.label_idx in kept and kept == sorted(set(kept))
        assert keep(ex, 4) == kept  # the same draw every time, for every model
    assert keep(example(0, 4, 1), 4) == [0, 1, 2, 3]  # the full K keeps everything


def test_at_the_full_k_the_numbers_are_the_plain_evaluation():
    rng = np.random.default_rng(0)
    exs = [example(i, 10, int(rng.integers(10))) for i in range(300)]
    scores = [rng.normal(size=10) for _ in exs]
    probs, labels = restrict(scores, exs, 10, temperature=1.3)
    plain = metrics.summarize(
        [softmax(s, 1.3) for s in scores], [ex.label_idx for ex in exs], bins=False
    )
    c = cell(probs, labels)
    assert c["accuracy"] == pytest.approx(plain["accuracy"])
    assert c["ece"] == pytest.approx(plain["ece"])


def test_a_calibrated_model_stays_calibrated_with_fewer_options():
    """Answers drawn from the model's own probabilities: no gap at any K, up to noise."""
    rng = np.random.default_rng(1)
    exs, scores = [], []
    for i in range(20_000):
        s = rng.normal(scale=2.0, size=14)
        exs.append(example(i, 14, int(rng.choice(14, p=softmax(s)))))
        scores.append(s)
    for k in (2, 4, 10, 14):
        c = cell(*restrict(scores, exs, k, temperature=1.0), n_boot=200)
        assert abs(c["gap"]) < 0.01, (k, c["gap"])
        assert c["gap_ci95"][0] <= 0 <= c["gap_ci95"][1]


@pytest.mark.parametrize("architecture", ["cross", "shared"])
def test_cached_subset_equals_scoring_the_subset(tokenizer, architecture):
    """Options never see each other, so dropping options is a lookup, not a new forward pass."""
    model = make_model(architecture=architecture)
    enc = (SharedEncoder if architecture == "shared" else Encoder)(tokenizer, max_len=64)
    rng = random.Random(0)
    exs = []
    for i in range(6):
        opts = [" ".join(rng.choices(WORDS[:60], k=2)) + f" w{j}" for j in range(8)]
        exs.append(example(i, 8, rng.randrange(8), " ".join(rng.choices(WORDS, k=12)), opts))
    full = predict_scores(model, enc, exs)
    for ex, s in zip(exs, full, strict=True):
        kept = keep(ex, 3)
        sub = Example(**{**ex.__dict__, "options": [ex.options[j] for j in kept],
                         "label_idx": kept.index(ex.label_idx)})  # fmt: skip
        [again] = predict_scores(model, enc, [sub])
        np.testing.assert_allclose(again, s[kept], rtol=1e-5, atol=1e-5)


def test_the_report_renders():
    probs, labels = restrict([np.array([2.0, 0.0, -1.0])], [example(0, 3, 0)], 2, 1.0)
    result = {
        "models": {
            "run": {"temperature": 1.0, "results": {"x/test": {"2": cell(probs, labels, 10)}}}
        }
    }
    assert "| x/test | 2 | 1 |" in options_tables(json.loads(json.dumps(result)))
