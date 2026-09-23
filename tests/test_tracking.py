"""MLflow tracking and the registry (D24), against a throwaway sqlite store.

Needs the tracking extra: `uv run --extra tracking pytest`. Without it, only the error test runs.
"""

import random
import sys

import pytest
import torch

from tests.conftest import make_model
from trueodds.data.schema import Example
from trueodds.model import DecisionModel
from trueodds.predict import predict_probs
from trueodds.train import TrainConfig, fit


def questions(n=12):
    colours = ["red", "blue", "green"]
    rng = random.Random(0)
    out = []
    for i in range(n):
        label = i % 3
        state = " ".join([*rng.choices([f"w{j}" for j in range(20)], k=5), colours[label]])
        out.append(Example(id=f"t/dev/{i}", source="t", split="dev", state=state,
                           question="w50 w51", options=colours[:], label_idx=label,
                           template_id="native"))  # fmt: skip
    return out


def test_without_the_extra_mlflow_tracking_fails_clearly(monkeypatch):
    from trueodds import tracking

    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` raise ImportError
    with pytest.raises(RuntimeError, match="--extra tracking"):
        tracking.MlflowTracker(uri="sqlite:///unused.db")


def test_a_tracked_run_is_logged_registered_promoted_and_loadable(encoder, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    import pandas as pd

    from trueodds import tracking

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # compare on the CPU, in fp32
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_dir = tmp_path / "tiny"
    run_dir.mkdir()
    data = questions()
    cfg = TrainConfig(run="tiny", lr=3e-3, max_steps=20, questions_per_step=6, eval_every=10,
                      log_every=5, resample_templates=False, tracking="mlflow")  # fmt: skip

    tracker = tracking.MlflowTracker(uri, tmp_path / "artifacts")
    run_id = tracker.start(cfg.run, cfg.__dict__, run_dir)
    assert tracking.run_id_of(run_dir) == run_id
    model = make_model().train()
    fit(model, encoder, data, {"dev": data}, cfg, run_dir, tracker)
    best = run_dir / "best"
    version = tracking.register(best, run_id, tracking.checkpoint_tags(best), uri)
    tracker.end()

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert {"train/loss", "train/lr", "dev/accuracy", "dev/nll"} <= set(run.data.metrics)
    assert run.data.params["lr"] == "0.003"
    assert version == "1"
    tags = client.get_model_version(tracking.MODEL_NAME, version).tags
    assert tags["run"] == "tiny" and tags["pooling"] == "cls" and "dev_accuracy" in tags

    [pickle] = (tmp_path / "artifacts").rglob("python_model.pkl")
    # The wrapper alone is ~300 bytes; with the tiny model's weights in it, ~110 kB.
    assert pickle.stat().st_size < 10_000

    tracking.promote(version, uri=uri)
    loaded = mlflow.pyfunc.load_model(f"models:/{tracking.MODEL_NAME}@champion")
    frame = pd.DataFrame([{"state": ex.state, "question": ex.question, "options": ex.options}
                          for ex in data])  # fmt: skip
    served = loaded.predict(frame)
    # The registered version is `best` (by dev accuracy), not the model at the last step.
    expected = predict_probs(DecisionModel.load(best), encoder, data)
    assert len(served) == len(data)
    for p, q in zip(served, expected, strict=True):
        assert torch.allclose(torch.tensor(p, dtype=torch.float64), torch.tensor(q), atol=1e-5)

    result = {"results": {"t/test": {"metrics": {"accuracy": 0.5, "nll": 0.7, "ece": 0.1}}}}
    results_path = tmp_path / "eval.json"
    results_path.write_text("{}")
    assert tracking.log_eval(result, results_path, best, uri)
    assert client.get_run(run_id).data.metrics["harness/t/test/accuracy"] == 0.5
