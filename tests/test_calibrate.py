"""Temperature scaling (Phase 3, D27), proved on known answers before it touches a real model."""

import importlib.util
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from harness import metrics
from harness.calibrate import (
    SCORES_FILE,
    CachedScores,
    _key,
    calibrate,
    dev_sets,
    ece_arrays,
    ece_change,
    load_or_score,
)
from harness.evaluate import eval_sets, evaluate
from harness.report import calibration_tables
from tests.conftest import make_model
from tests.test_evaluate import data_dir  # noqa: F401  (the fixture data directory)
from tests.test_model import mixed_k  # noqa: F401
from trueodds.calibrate import fit_temperature, nll_at, write_temperature
from trueodds.data.schema import load_examples, write_examples
from trueodds.predict import ModelPredictor, predict_probs, predict_scores, read_temperature


def sampled(t0, n=20_000, ks=(4,), seed=0):
    """Scores z and labels drawn from softmax(z / t0): the NLL-optimal temperature is t0."""
    rng = np.random.default_rng(seed)
    scores, labels = [], []
    for i in range(n):
        z = rng.normal(0, 2.0, ks[i % len(ks)])
        p = np.exp(z / t0 - (z / t0).max())
        scores.append(z)
        labels.append(int(rng.choice(len(z), p=p / p.sum())))
    return scores, labels


@pytest.mark.parametrize("t0", [0.5, 1.0, 2.0])
def test_fit_recovers_the_true_temperature(t0):
    scores, labels = sampled(t0)
    assert fit_temperature(scores, labels) == pytest.approx(t0, rel=0.05)


def test_mixed_k_is_padded_and_fits_like_each_part():
    scores, labels = sampled(1.5, n=12_000, ks=(2, 4, 14))
    t = fit_temperature(scores, labels)
    assert np.isfinite(nll_at(scores, labels, t))
    assert t == pytest.approx(1.5, rel=0.05)
    # the same fit as a grid search over the unpadded questions
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 2001))
    unpadded = [
        -np.mean([np.log(np.exp(s / g - (s / g).max())[y] / np.exp(s / g - (s / g).max()).sum())
                  for s, y in zip(scores[:3000], labels[:3000], strict=True)])
        for g in grid[::20]
    ]  # fmt: skip
    best = grid[::20][int(np.argmin(unpadded))]
    assert fit_temperature(scores[:3000], labels[:3000]) == pytest.approx(best, rel=0.05)


def test_fit_never_raises_the_nll_on_its_fit_set():
    scores, labels = sampled(0.7, n=3000)
    t = fit_temperature(scores, labels)
    assert nll_at(scores, labels, t) <= nll_at(scores, labels, 1.0)


def test_temperature_scales_the_scores_and_keeps_the_argmax(encoder, mixed_k):  # noqa: F811
    model = make_model()
    scores = predict_scores(model, encoder, mixed_k)
    for t in (1.0, 0.5, 3.0):
        probs = predict_probs(model, encoder, mixed_k, temperature=t)
        for s, p in zip(scores, probs, strict=True):
            expect = torch.softmax(torch.tensor(s, dtype=torch.float64) / t, -1).numpy()
            assert np.allclose(p, expect)
            assert np.argmax(p) == np.argmax(s)


def test_predictor_reads_the_checkpoint_temperature(tmp_path, tokenizer):
    ckpt = tmp_path / "run/best"
    make_model().save(ckpt, tokenizer, max_len=32)
    assert read_temperature(ckpt) == 1.0
    assert ModelPredictor.load(ckpt, device="cpu").temperature == 1.0
    write_temperature(ckpt, 0.8, {"n": 1})
    assert ModelPredictor.load(ckpt, device="cpu").temperature == 0.8
    assert ModelPredictor.load(ckpt, device="cpu").name.endswith("@T=0.8000")
    assert ModelPredictor.load(ckpt, device="cpu", temperature=False).temperature == 1.0


def test_vectorized_ece_matches_the_metric():
    scores, labels = sampled(1.0, n=2000)
    probs = [np.exp(s - s.max()) / np.exp(s - s.max()).sum() for s in scores]
    pred, conf = metrics.top1(probs)
    correct = (pred == np.array(labels)).astype(float)
    assert ece_arrays(conf, correct) == pytest.approx(metrics.ece(probs, labels), abs=1e-12)


def test_identical_before_and_after_change_nothing():
    scores, labels = sampled(1.0, n=500)
    probs = [np.exp(s - s.max()) / np.exp(s - s.max()).sum() for s in scores]
    c = ece_change(probs, probs, labels, n_boot=50)
    assert c["delta"] == 0.0 and c["ci95"] == [0.0, 0.0]


class TooSure:
    """Puts the label on top often, with scores inflated twice over: overconfident, so T > 1."""

    name = "too-sure"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def scores(self, examples):
        out = []
        for ex in examples:
            z = self.rng.normal(0, 1.0, ex.k)
            z[ex.label_idx] += 1.5
            out.append(2.0 * z)
        return out


def test_calibrate_end_to_end(data_dir, tmp_path):  # noqa: F811
    for name in ("boolq", "arc_easy"):  # dev files: the train questions, repeated for a stable fit
        train = load_examples(data_dir / name / "train.jsonl")
        dev = [
            replace(e, id=f"{e.id.replace('train', 'dev')}-{r}", split="dev")
            for r in range(40)
            for e in train
        ]
        write_examples(data_dir / name / "dev.jsonl", dev)
    ckpt = tmp_path / "run/best"
    ckpt.mkdir(parents=True)
    r = calibrate(TooSure(), data_dir, ckpt, run="run")
    t = r["temperature"]["value"]
    assert t > 1.0
    assert r["temperature"]["dev_after"]["nll"] <= r["temperature"]["dev_before"]["nll"]
    assert read_temperature(ckpt) == pytest.approx(t)
    assert set(r["before"]) == set(r["after"]) == set(r["ece_change"])
    for key in r["before"]:  # a temperature does not move the argmax
        assert r["before"][key]["metrics"]["accuracy"] == r["after"][key]["metrics"]["accuracy"]
    # `before` is a plain evaluation of the cached scores at T = 1
    sets = {**dev_sets(data_dir), **eval_sets(data_dir)}
    scores = load_or_score(None, sets, ckpt / SCORES_FILE, None)
    table = {_key(ex): s for k, exs in sets.items() for ex, s in zip(exs, scores[k], strict=True)}
    dev = [ex for k in dev_sets(data_dir) for ex in sets[k]]
    assert t == fit_temperature([table[_key(ex)] for ex in dev], [ex.label_idx for ex in dev])
    plain = evaluate(CachedScores(table, 1.0, "too-sure"), data_dir)
    assert plain["results"]["pooled/test"]["metrics"] == r["before"]["pooled/test"]["metrics"]
    # a second run reads the cache (a scorer with another seed would give other scores)
    again = calibrate(TooSure(seed=99), data_dir, ckpt, run="run")
    assert again["temperature"]["value"] == pytest.approx(t)
    text = calibration_tables(json.loads(json.dumps(r)))  # the result survives JSON
    assert "boolq/test" in text and "commonsense_qa/test" in text and "| pooled/test | 4 |" in text
    if importlib.util.find_spec("matplotlib"):  # the plots extra
        from harness.plots import diagram

        assert diagram(r, "pooled/test", tmp_path / "rel.png").stat().st_size > 0
