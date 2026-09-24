"""Phase 5 comparisons (`harness.compare`): paired intervals, on answers known in advance."""

import json
from dataclasses import replace

import numpy as np
import pytest

from harness.calibrate import calibrate
from harness.compare import compare, paired, per_question
from harness.report import compare_tables
from tests.test_calibrate import TooSure
from tests.test_evaluate import data_dir  # noqa: F401  (the fixture data directory)
from trueodds.data.schema import load_examples, write_examples


def test_a_model_against_itself_differs_by_nothing():
    rng = np.random.default_rng(0)
    probs = [p / p.sum() for p in rng.random((200, 4))]
    q = per_question(probs, rng.integers(0, 4, 200))
    r = paired(q, q)
    for m in ("accuracy", "nll", "ece"):
        assert r[m]["delta"] == 0 and r[m]["ci95"] == [0, 0]


def test_always_right_against_always_wrong():
    wrong = per_question([np.array([0.9, 0.1])] * 50, [1] * 50)
    right = per_question([np.array([0.9, 0.1])] * 50, [0] * 50)
    r = paired(wrong, right)
    assert r["accuracy"]["delta"] == 1 and r["accuracy"]["ci95"] == [1, 1]
    assert r["nll"]["delta"] == pytest.approx(np.log(0.1) - np.log(0.9))  # B's NLL is lower
    assert r["ece"]["a"] == pytest.approx(0.9) and r["ece"]["b"] == pytest.approx(0.1)


def test_compare_end_to_end(data_dir, tmp_path):  # noqa: F811
    for name in ("boolq", "arc_easy"):  # dev files, as in test_calibrate_end_to_end
        train = load_examples(data_dir / name / "train.jsonl")
        dev = [replace(e, id=e.id.replace("train", "dev"), split="dev") for e in train]
        write_examples(data_dir / name / "dev.jsonl", dev)
    a, b = tmp_path / "a/best", tmp_path / "b/best"
    for ckpt, seed in ((a, 0), (b, 1)):
        ckpt.mkdir(parents=True)
        calibrate(TooSure(seed=seed), data_dir, ckpt, run=ckpt.parent.name)
    same = compare(a, a, data_dir)
    assert same["results"]["pooled/test"]["accuracy"]["delta"] == 0
    r = compare(a, b, data_dir)
    assert r["a"]["run"] == "a" and r["b"]["run"] == "b"
    assert "pooled/test" in r["results"] and "commonsense_qa/test" in r["results"]
    acc = r["results"]["pooled/test"]["accuracy"]
    assert acc["ci95"][0] <= acc["delta"] <= acc["ci95"][1]
    text = compare_tables(json.loads(json.dumps(r)))
    assert "Held-out datasets" in text and "boolq/test" in text


def test_compare_needs_the_calibration_cache(data_dir, tmp_path):  # noqa: F811
    (tmp_path / "x/best").mkdir(parents=True)
    with pytest.raises(SystemExit, match=r"harness\.calibrate"):
        compare(tmp_path / "x/best", tmp_path / "x/best", data_dir)
