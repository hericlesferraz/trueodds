"""MLflow run tracking and the model registry (D24). Local only: a sqlite store and file artifacts.

MLflow mirrors what the trainer and the harness produce; the results JSONs in `harness/results/`
stay the source of truth, and a version is promoted on dev numbers only (D6).

    uv run --extra tracking python -m trueodds.tracking register ~/.trueodds/runs/<run>/best
    uv run --extra tracking python -m trueodds.tracking promote 3            # alias "champion"
    uv run --extra tracking mlflow ui --backend-store-uri sqlite:///$HOME/.trueodds/mlflow.db
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path

import yaml

from trueodds import paths

EXPERIMENT = "trueodds"
MODEL_NAME = "trueodds"
RUN_FILE = "mlflow.json"  # in a run dir: the MLflow run id of that training run


def _mlflow():
    try:
        import mlflow
    except ImportError as e:
        raise RuntimeError(
            "MLflow tracking needs the tracking extra: uv run --extra tracking ..."
        ) from e
    return mlflow


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{paths.MLFLOW_DB}"


def setup(uri: str | None = None, artifact_root: Path | None = None):
    """Point MLflow at the store and make `trueodds` the experiment; return the mlflow module."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow = _mlflow()
    mlflow.set_tracking_uri(uri or tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        root = artifact_root or paths.MLARTIFACTS
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=root.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _params(config: dict) -> dict[str, str]:
    return {k: json.dumps(v) if isinstance(v, list | dict) else str(v) for k, v in config.items()}


class MlflowTracker:
    """One MLflow run per training run. Used by the trainer's `Logger` through `metrics`."""

    def __init__(self, uri: str | None = None, artifact_root: Path | None = None) -> None:
        self.mlflow = setup(uri, artifact_root)
        self.run_id: str | None = None

    def start(self, run_name: str, config: dict, run_dir: Path | None = None) -> str:
        tags = {"run_dir": str(run_dir)} if run_dir else {}
        run = self.mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = run.info.run_id
        self.mlflow.log_params(_params(config))
        if run_dir:
            (run_dir / RUN_FILE).write_text(json.dumps({"run_id": self.run_id}) + "\n")
        return self.run_id

    def metrics(self, step: int, values: dict[str, float]) -> None:
        self.mlflow.log_metrics({k: float(v) for k, v in values.items()}, step=step)

    def artifact(self, path: Path, subdir: str | None = None) -> None:
        self.mlflow.log_artifact(str(path), subdir)

    def end(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status)


@contextlib.contextmanager
def _in_run(mlflow, run_id: str):
    """Log into `run_id`, whether or not it is the active run."""
    active = mlflow.active_run()
    if active and active.info.run_id == run_id:
        yield
    else:
        with mlflow.start_run(run_id=run_id):
            yield


def run_id_of(run_dir: Path) -> str | None:
    path = run_dir / RUN_FILE
    return json.loads(path.read_text())["run_id"] if path.exists() else None


def register(
    checkpoint: Path,
    run_id: str,
    tags: dict[str, str] | None = None,
    uri: str | None = None,
) -> str:
    """Log `checkpoint` as a pyfunc model in `run_id` and register it; return the version."""
    mlflow = setup(uri)
    import pandas as pd
    import torch
    import transformers

    from trueodds.mlflow_model import TrueOddsModel

    example = pd.DataFrame(
        [
            {
                "state": "The sky was clear all day.",
                "question": "Did it rain?",
                "options": ["yes", "no"],
            }
        ]
    )
    with _in_run(mlflow, run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=TrueOddsModel(),
            artifacts={"checkpoint": str(checkpoint)},
            code_paths=[str(Path(__file__).parent)],
            input_example=example,
            pip_requirements=[
                f"torch=={torch.__version__.split('+')[0]}",
                f"transformers=={transformers.__version__}",
                "safetensors",
                "pandas",
            ],
            registered_model_name=MODEL_NAME,
        )
    version = str(info.registered_model_version)
    client = mlflow.MlflowClient()
    for k, v in (tags or {}).items():
        client.set_model_version_tag(MODEL_NAME, version, k, str(v))
    return version


def checkpoint_tags(checkpoint: Path) -> dict[str, str]:
    """What a registered version is tagged with: its run, pooling and dev numbers (D6)."""
    tags = {"run": checkpoint.parent.name, "checkpoint": checkpoint.name}
    meta = checkpoint / "model.json"
    if meta.exists():
        tags["pooling"] = json.loads(meta.read_text())["pooling"]
    summary = checkpoint.parent / "summary.json"
    if summary.exists():
        best = json.loads(summary.read_text()).get("best_dev") or {}
        for k in ("accuracy", "nll", "ece"):
            if k in best.get("dev", {}):
                tags[f"dev_{k}"] = f"{best['dev'][k]:.4f}"
    return tags


def register_checkpoint(checkpoint: Path, uri: str | None = None) -> str:
    """Register a checkpoint written by the trainer, creating an MLflow run for it if needed."""
    checkpoint = checkpoint.expanduser().resolve()
    run_dir = checkpoint.parent
    run_id = run_id_of(run_dir)
    if run_id is None:
        tracker = MlflowTracker(uri)
        config = run_dir / "config.yaml"
        tracker.start(
            run_dir.name, yaml.safe_load(config.read_text()) if config.exists() else {}, run_dir
        )
        for name in ("config.yaml", "summary.json"):
            if (run_dir / name).exists():
                tracker.artifact(run_dir / name)
        run_id = tracker.run_id
        try:
            return register(checkpoint, run_id, checkpoint_tags(checkpoint), uri)
        finally:
            tracker.end()
    return register(checkpoint, run_id, checkpoint_tags(checkpoint), uri)


def promote(version: str, alias: str = "champion", uri: str | None = None) -> None:
    """The scripted form of setting an alias in the UI."""
    mlflow = setup(uri)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, alias, str(version))


def log_eval(result: dict, results_path: Path, checkpoint: Path, uri: str | None = None) -> bool:
    """Mirror a harness result into the checkpoint's training run; False if it has none."""
    run_id = run_id_of(checkpoint.parent)
    if run_id is None:
        return False
    mlflow = setup(uri)
    values = {}
    # A calibration result (Phase 3) logs its temperature and the calibrated numbers apart.
    calibration = result.get("kind") == "calibration"
    prefix = "harness_calibrated" if calibration else "harness"
    if calibration:
        values["temperature"] = result["temperature"]["value"]
    for key, entry in result["after" if calibration else "results"].items():
        m = entry.get("metrics")
        if m:
            values[f"{prefix}/{key}/accuracy"] = m["accuracy"]
            values[f"{prefix}/{key}/nll"] = m["nll"]
            values[f"{prefix}/{key}/ece"] = m["ece"]
    with _in_run(mlflow, run_id):
        mlflow.log_artifact(str(results_path), "harness")
        mlflow.log_metrics({k: float(v) for k, v in values.items() if v is not None})
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="register a checkpoint as a new model version")
    reg.add_argument("checkpoint", type=Path)
    pro = sub.add_parser("promote", help="point an alias at a model version")
    pro.add_argument("version")
    pro.add_argument("--alias", default="champion")
    args = parser.parse_args()
    if args.command == "register":
        version = register_checkpoint(args.checkpoint)
        print(f"registered {MODEL_NAME} version {version} ({tracking_uri()})")
    else:
        promote(args.version, args.alias)
        print(f"{MODEL_NAME}@{args.alias} -> version {args.version}")


if __name__ == "__main__":
    main()
