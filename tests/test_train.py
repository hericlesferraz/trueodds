import random

import pytest
import yaml

from tests.conftest import make_model
from trueodds.data.schema import Example
from trueodds.data.templates import TRAINED
from trueodds.train import TrainConfig, fit, linear_schedule, resample_template


def test_schedule_warms_up_then_decays_to_zero():
    f = linear_schedule(warmup=10, total=110)
    assert f(0) == 0
    assert f(5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.5)
    assert f(110) == 0 and f(200) == 0


def test_config_loads_from_yaml_and_refuses_test_sets(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"run": "x", "lr": 1e-4, "eval_sets": ["train", "dev"]}))
    cfg = TrainConfig.from_yaml(path)
    assert cfg.lr == 1e-4 and cfg.questions_per_step == 32
    for bad in (["test"], ["test-heldout-template"], ["dev", "test_mismatched"]):
        with pytest.raises(ValueError, match="eval_sets"):
            TrainConfig(run="x", eval_sets=bad)
    with pytest.raises(ValueError, match="not training sources"):
        TrainConfig(run="x", sources=["commonsense_qa"])  # held-out, never trained on


def test_resampled_templates_are_trained_ones_and_keep_the_answer():
    ex = Example(id="mnli/train/0", source="mnli", split="train", state="A man sleeps.",
                 question="Does it follow that a man sleeps?", options=["yes", "maybe", "no"],
                 label_idx=0, template_id="mnli:0", vars={"hypothesis": "A man sleeps."})  # fmt: skip
    rng = random.Random(0)
    seen = {resample_template(ex, rng).template_id for _ in range(50)}
    assert seen == {f"mnli:{n}" for n in TRAINED["mnli"]}
    again = resample_template(ex, rng)
    assert (again.options, again.label_idx, again.state) == (ex.options, ex.label_idx, ex.state)
    native = Example(id="arc_easy/train/0", source="arc_easy", split="train", state="",
                     question="Q?", options=["a", "b"], label_idx=1, template_id="native")  # fmt: skip
    assert resample_template(native, rng) is native


def test_the_tiny_model_overfits_a_known_set(encoder):
    """Known answer: the right option is the colour named in the state."""
    colours = ["red", "blue", "green"]
    rng = random.Random(0)
    train = []
    for i in range(12):
        label = i % 3
        state = " ".join([*rng.choices([f"w{j}" for j in range(20)], k=5), colours[label]])
        options = colours[:]
        rng.shuffle(options)
        train.append(Example(id=f"t/train/{i}", source="t", split="train", state=state,
                             question="w50 w51", options=options,
                             label_idx=options.index(colours[label]), template_id="native"))  # fmt: skip
    cfg = TrainConfig(run="tiny", lr=3e-3, warmup_ratio=0.1, max_steps=150, questions_per_step=6,
                      eval_every=50, log_every=50, resample_templates=False, save=False)  # fmt: skip
    model = make_model().train()
    summary = fit(model, encoder, train, {"train": train}, cfg)
    final = summary["final"]["train"]
    assert final["accuracy"] == 1.0
    assert final["nll"] < 0.05
    assert summary["history"][0]["train"]["nll"] > final["nll"]
