"""`predict()` (spec 003), on a tiny random model saved as a checkpoint."""

import math

import numpy as np
import pytest

import trueodds
from tests.conftest import make_model
from trueodds.calibrate import write_temperature
from trueodds.infer import TrueOdds, request_example
from trueodds.predict import ModelPredictor

STATE = "w1 w2 w3 w4 red cat w5 w6"


@pytest.fixture
def ckpt(tmp_path, tokenizer):
    path = tmp_path / "run/best"
    make_model().save(path, tokenizer, max_len=32)
    return path


@pytest.fixture
def model(ckpt):
    return trueodds.load(ckpt, device="cpu")


def test_load_is_exposed_on_the_package(ckpt):
    assert isinstance(trueodds.load(ckpt, device="cpu"), TrueOdds)


def test_keys_are_the_options_in_order_and_sum_to_one(model):
    options = ["red", "blue", "green", "w9 w10"]
    out = model.predict(STATE, "w7 w8", options)
    assert list(out) == options
    assert math.isclose(sum(out.values()), 1.0, abs_tol=1e-9)
    assert all(0.0 < p < 1.0 for p in out.values())


def test_same_numbers_as_the_harness_predictor(ckpt, model):
    options = ["yes", "no", "maybe"]
    ex = request_example(0, STATE, "w7 w8", options)
    expected = ModelPredictor.load(ckpt, device="cpu").predict([ex])[0]
    np.testing.assert_allclose(list(model.predict(STATE, "w7 w8", options).values()), expected)


def test_the_checkpoint_temperature_is_applied(ckpt):
    raw = trueodds.load(ckpt, device="cpu").predict(STATE, "w7", ["cat", "dog"])
    write_temperature(ckpt, 2.0, {"n": 1})
    hot = trueodds.load(ckpt, device="cpu")
    assert hot.temperature == 2.0
    soft = hot.predict(STATE, "w7", ["cat", "dog"])
    assert abs(soft["cat"] - 0.5) < abs(raw["cat"] - 0.5)
    ignored = trueodds.load(ckpt, device="cpu", temperature=False)
    assert ignored.temperature == 1.0
    assert ignored.predict(STATE, "w7", ["cat", "dog"]) == pytest.approx(raw)


def test_shuffling_the_options_keeps_each_probability(model):
    options = ["red", "blue", "green", "cat", "dog"]
    a = model.predict(STATE, "w7 w8", options)
    b = model.predict(STATE, "w7 w8", options[::-1])
    assert b == pytest.approx(a, abs=1e-6)


def test_a_batch_equals_one_question_at_a_time(model):
    questions = [
        ("w7", ["yes", "no"]),
        ("w8 w9", ["red", "blue", "green", "w1"]),
        ("w10", ["yes", "maybe", "no"]),
        ("w11 w12 w13", [f"w{i}" for i in range(20, 30)]),
    ]
    batch = model.predict_batch(STATE, questions)
    assert len(batch) == len(questions)
    for (q, opts), got in zip(questions, batch, strict=True):
        assert got == pytest.approx(model.predict(STATE, q, opts), abs=1e-5)


def test_an_empty_state_is_allowed(model):
    assert sum(model.predict("", "w7", ["yes", "no"]).values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("question", "options", "match"),
    [
        ("w7", ["yes"], "K = 1"),
        ("w7", ["yes", "yes"], "duplicate options"),
        ("w7", ["yes", " "], "empty option"),
        (" ", ["yes", "no"], "empty question"),
        (" ".join(["w1"] * 40), ["yes", "no"], "more than 32"),
    ],
)
def test_bad_requests_raise(model, question, options, match):
    with pytest.raises(ValueError, match=match):
        model.predict(STATE, question, options)


def test_latency_cuts_the_state_to_the_token_count(tokenizer):
    from harness.latency import cut_state

    text = " ".join(f"w{i % 60}" for i in range(100))
    state = cut_state(text, tokenizer, 37)
    assert len(tokenizer(state, add_special_tokens=False)["input_ids"]) == 37
    with pytest.raises(ValueError, match="fewer than"):
        cut_state("w1 w2", tokenizer, 37)


@pytest.mark.parametrize("n", [10, 50])
def test_latency_batch_questions_are_all_different(n):
    from harness.latency import batch_questions

    qs = batch_questions(n)
    assert len({(q, tuple(o)) for q, o in qs}) == n
    assert {len(o) for _, o in qs} == {2, 3, 4, 10, 14}
