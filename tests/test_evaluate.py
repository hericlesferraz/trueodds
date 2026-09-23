import numpy as np
import pytest

from harness.baselines import Majority, Prior, Uniform
from harness.evaluate import evaluate, write
from harness.report import table
from trueodds.data.schema import Example, write_examples


def ex(i, split, label, options=("yes", "no"), source="boolq", template_id="boolq:0"):
    return Example(id=f"{source}/{split}/{i}", source=source, split=split, state=f"s{i}",
                   question="Q?", options=list(options), label_idx=label,
                   template_id=template_id, vars={"question": "Q?"} if template_id != "native" else {})  # fmt: skip


def mc(i, split, label, k=4):
    return ex(
        i,
        split,
        label,
        options=[f"q{i}o{j}" for j in range(k)],
        source="arc_easy",
        template_id="native",
    )


def test_majority_by_text_and_by_index():
    train = [ex(i, "train", 0) for i in range(7)] + [ex(i, "train", 1) for i in range(7, 10)]
    maj = Majority(train)
    assert maj.by_text and maj.label == "yes"
    flipped = ex(0, "test", 1, options=("no", "yes"))  # the label text decides, not the position
    assert np.array_equal(maj.predict([flipped])[0], [0.0, 1.0])

    mc_train = [mc(i, "train", i % 3 and 2) for i in range(9)]  # label 2 six times, 0 three times
    maj = Majority(mc_train)
    assert not maj.by_text and maj.label == 2
    assert np.array_equal(maj.predict([mc(0, "test", 0, k=3)])[0], [0.0, 0.0, 1.0])
    assert np.allclose(maj.predict([mc(0, "test", 0, k=2)])[0], [0.5, 0.5])  # index 2 absent


def test_prior_is_label_frequency():
    train = [ex(i, "train", 0) for i in range(98)] + [ex(i, "train", 1) for i in range(98, 100)]
    p = Prior(train, smoothing=0.0).predict([ex(0, "test", 0)])[0]
    assert np.allclose(p, [0.98, 0.02])
    assert np.allclose(Uniform().predict([mc(0, "test", 0, k=5)])[0], 0.2)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    write_examples(
        tmp_path / "boolq/train.jsonl",
        [ex(i, "train", int(rng.random() < 0.4)) for i in range(100)],
    )
    write_examples(
        tmp_path / "boolq/test.jsonl", [ex(i, "test", int(rng.random() < 0.4)) for i in range(50)]
    )
    write_examples(tmp_path / "arc_easy/train.jsonl", [mc(i, "train", i % 4) for i in range(40)])
    write_examples(tmp_path / "arc_easy/test.jsonl", [mc(i, "test", i % 4) for i in range(20)])
    held = [ex(i, "test", i % 5, options=[f"c{j}" for j in range(5)], source="commonsense_qa",
               template_id="native") for i in range(10)]  # fmt: skip
    write_examples(tmp_path / "commonsense_qa/test.jsonl", held)
    return tmp_path


class Oracle:
    name = "oracle"

    def predict(self, examples):
        return [np.eye(e.k)[e.label_idx] for e in examples]


def test_baselines_run(data_dir, tmp_path):
    r = evaluate(None, data_dir)
    assert r["kind"] == "baselines" and r["predictor"] is None
    assert set(r["results"]) == {
        "boolq/test",
        "arc_easy/test",
        "commonsense_qa/test",
        "pooled/test",
    }
    boolq = r["results"]["boolq/test"]
    assert boolq["metrics"] is None
    assert boolq["baselines"]["random"]["accuracy"] == 0.5
    assert boolq["baselines"]["random"]["ece"] is None
    assert boolq["baselines"]["majority"]["fit_on"] == "train"
    assert r["results"]["commonsense_qa/test"]["baselines"]["majority"]["fit_on"] == "test"
    pooled = r["results"]["pooled/test"]
    assert pooled["n"] == 70  # training sources only
    assert pooled["baselines"]["random"]["accuracy"] == pytest.approx((50 * 0.5 + 20 * 0.25) / 70)
    path = write(r, tmp_path)
    assert path.name.endswith("-baselines.json")
    assert "boolq/test" in table([path])


def test_model_run_scores_the_predictor(data_dir):
    r = evaluate(Oracle(), data_dir, run="oracle")
    for entry in r["results"].values():
        assert entry["metrics"]["accuracy"] == 1.0
        assert entry["metrics"]["nll"] == pytest.approx(0.0, abs=1e-9)
    assert r["results"]["arc_easy/test"]["metrics"]["by_k"] == {
        "4": {"n": 20, "accuracy": 1.0, "ece": 0.0}
    }
