"""Train the decision model from one YAML config in `configs/`.

    uv run --extra gpu python -m trueodds.train configs/phase1-overfit.yaml
    uv run --extra gpu python -m trueodds.train configs/phase2-base.yaml --max-steps 50 \
        --longest-first --no-eval --no-tracking             # the VRAM probe
    uv run --extra gpu --extra tracking python -m trueodds.train <config> --mlflow   # tracked (D24)

fp32 weights under bf16 autocast, fused AdamW, linear warmup then linear decay. Every `eval_every`
steps the eval sets are scored with the harness metrics; the best checkpoint by dev `select_by` is
kept in `<runs>/<run>/best/`, the final one in `last/`. Only train and dev are ever read here (D6).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from harness import metrics
from trueodds import paths
from trueodds.batching import plan_epoch, split_microbatches
from trueodds.data.converters import SOURCES
from trueodds.data.schema import Example, load_examples
from trueodds.data.templates import HELDOUT, pick_random, render
from trueodds.encode import BACKBONE, Encoder
from trueodds.model import ARCHITECTURES, DecisionModel
from trueodds.predict import ENCODERS, predict_probs
from trueodds.shared import SharedEncoder

TRAIN_SOURCES = [name for name, s in SOURCES.items() if s.role == "train"]
EVAL_SETS = ("train", "dev")  # "train" is the training subset itself, scored as stored
GIB = 1024**3


@dataclass
class TrainConfig:
    run: str
    backbone: str = BACKBONE
    architecture: str = "cross"  # "shared": the state encoded once (D13, spec 004)
    pooling: str = "cls"
    attn: str = "sdpa"
    max_len: int = 512
    sources: list[str] = field(default_factory=lambda: list(TRAIN_SOURCES))
    train_limit: int | None = None  # questions per source, sampled with `seed`
    eval_sets: list[str] = field(default_factory=lambda: ["dev"])
    eval_limit: int | None = None  # questions per source and eval set
    resample_templates: bool = True  # a random trained template per question per epoch (D23)
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: float = 3
    max_steps: int | None = None
    questions_per_step: int = 32
    max_tokens_per_microbatch: int = 16_384
    grad_checkpointing: bool = False
    clip: float = 1.0
    eval_every: int = 500
    log_every: int = 10
    save: bool = True
    tracking: str = "none"  # "mlflow": log the run and register the best checkpoint (D24)
    select_by: str = "accuracy"  # dev metric that picks `best`: "accuracy" (higher) or "nll" (D26)
    longest_first: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        bad = [s for s in self.eval_sets if s not in EVAL_SETS]
        if bad:
            # Test and held-out files are never read for any choice (D6).
            raise ValueError(f"eval_sets may only be {EVAL_SETS}; got {bad}")
        if self.select_by not in ("accuracy", "nll"):
            raise ValueError(f"select_by must be 'accuracy' or 'nll', not {self.select_by!r}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if self.tracking not in ("none", "mlflow"):
            raise ValueError(f"tracking must be 'none' or 'mlflow', not {self.tracking!r}")
        unknown = [s for s in self.sources if s not in TRAIN_SOURCES]
        if unknown:
            raise ValueError(f"not training sources: {unknown}")

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        return cls(**yaml.safe_load(path.read_text()))


def sample(examples: list[Example], limit: int | None, seed: int) -> list[Example]:
    if limit is None or len(examples) <= limit:
        return examples
    return random.Random(seed).sample(examples, limit)


def load_split(cfg: TrainConfig, split: str, limit: int | None, data_dir: Path) -> list[Example]:
    out: list[Example] = []
    for source in cfg.sources:
        out.extend(sample(load_examples(data_dir / source / f"{split}.jsonl"), limit, cfg.seed))
    return out


def resample_template(ex: Example, rng: random.Random) -> Example:
    """The same question with a random trained template (the `vars` of spec 001 exist for this)."""
    if ex.template_id == "native":
        return ex
    task, n = ex.template_id.split(":")
    if n == HELDOUT:
        raise ValueError(f"{ex.id}: the held-out template is never trained on (D8)")
    question, template_id = render(task, pick_random(task, rng), ex.vars)
    return replace(ex, question=question, template_id=template_id)


def better(new: dict, old: dict, cfg: TrainConfig) -> bool:
    """Whether dev metrics `new` beat `old` on `cfg.select_by`."""
    if cfg.select_by == "nll":
        return new["nll"] < old["nll"]
    return new["accuracy"] > old["accuracy"]


def linear_schedule(warmup: int, total: int) -> Callable[[int], float]:
    """LR multiplier: 0 → 1 over `warmup` steps, then 1 → 0 at `total`."""

    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return f


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def evaluate_sets(
    model: DecisionModel, encoder: Encoder | SharedEncoder, eval_sets: dict[str, list[Example]]
) -> dict[str, dict]:
    out = {}
    for name, examples in eval_sets.items():
        probs = predict_probs(model, encoder, examples)
        out[name] = metrics.summarize(probs, [ex.label_idx for ex in examples], bins=False)
    return out


class Logger:
    """Writes to TensorBoard when it is installed (the gpu extra), and to a tracker if given."""

    def __init__(self, log_dir: Path | None, tracker=None) -> None:
        self.tracker = tracker  # anything with `metrics(step, values)`, e.g. MlflowTracker
        self.writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(log_dir))
            except ImportError:
                print("tensorboard is not installed; logging to stdout only")

    def scalars(self, step: int, values: dict[str, float]) -> None:
        if self.tracker:
            self.tracker.metrics(step, values)
        if self.writer:
            for k, v in values.items():
                self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()


def fit(
    model: DecisionModel,
    encoder: Encoder | SharedEncoder,
    train: Sequence[Example],
    eval_sets: dict[str, list[Example]],
    cfg: TrainConfig,
    run_dir: Path | None = None,
    tracker=None,
) -> dict:
    """Train `model` in place on `train`; return the summary (also written to `run_dir`)."""
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    logger = Logger(run_dir / "tb" if run_dir else None, tracker)

    lengths = encoder.lengths(train)
    ks = [encoder.n_sequences(ex) for ex in train]  # a packed question is one sequence
    steps_per_epoch = math.ceil(len(train) / cfg.questions_per_step)
    total = cfg.max_steps or math.ceil(cfg.epochs * steps_per_epoch)
    warmup = round(cfg.warmup_ratio * total)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, fused=cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_schedule(warmup, total))
    print(f"{len(train)} questions, {steps_per_epoch} steps per epoch, {total} steps")

    history: list[dict] = []
    best: dict | None = None
    step, epoch = 0, 0
    window_tokens, window_loss, window_start = 0, 0.0, time.perf_counter()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    while step < total:
        plan = plan_epoch(
            lengths,
            ks,
            cfg.questions_per_step,
            rng,
            cfg.max_tokens_per_microbatch,
            longest_first=cfg.longest_first,
        )
        for indices in plan:
            if step >= total:
                break
            examples = [train[i] for i in indices]
            if cfg.resample_templates:
                examples = [resample_template(ex, rng) for ex in examples]
            items = encoder.encode(examples)
            mbs = split_microbatches(
                [e.length for e in items],
                [e.n_sequences for e in items],
                cfg.max_tokens_per_microbatch,
            )
            step_loss = 0.0
            for mb in mbs:
                batch = encoder.collate([items[i] for i in mb]).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                    scores = model(**batch.model_inputs())
                loss = F.cross_entropy(scores, batch.labels, reduction="sum") / len(items)
                loss.backward()
                step_loss += loss.item()
                window_tokens += batch.input_ids.numel()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            window_loss += step_loss

            if step % cfg.log_every == 0 or step == total:
                elapsed = time.perf_counter() - window_start
                n = cfg.log_every if step % cfg.log_every == 0 else step % cfg.log_every
                values = {
                    "train/loss": window_loss / n,
                    "train/lr": sched.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm),
                    "train/padded_tokens_per_s": window_tokens / elapsed,
                }
                if cuda:
                    values["gpu/max_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
                    values["gpu/max_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
                logger.scalars(step, values)
                print(f"step {step}/{total} " + " ".join(f"{k}={v:.4g}" for k, v in values.items()))
                window_tokens, window_loss, window_start = 0, 0.0, time.perf_counter()

            if eval_sets and (step % cfg.eval_every == 0 or step == total):
                result = {"step": step, "epoch": epoch, **evaluate_sets(model, encoder, eval_sets)}
                history.append(result)
                for name, m in result.items():
                    if isinstance(m, dict):
                        logger.scalars(step, {f"{name}/{k}": v for k, v in m.items() if k != "n"})
                print(json.dumps(result))
                if "dev" in result and (best is None or better(result["dev"], best["dev"], cfg)):
                    best = result
                    if run_dir and cfg.save:
                        model.save(run_dir / "best", encoder.tokenizer, cfg.max_len)
                window_start = time.perf_counter()
        epoch += 1

    summary = {
        "run": cfg.run,
        "config": dataclasses.asdict(cfg),
        "steps": step,
        "questions": len(train),
        "final": history[-1] if history else None,
        "best_dev": best,
        "history": history,
    }
    if cuda:
        summary["peak_vram_gib"] = {
            "max_reserved": torch.cuda.max_memory_reserved() / GIB,
            "max_allocated": torch.cuda.max_memory_allocated() / GIB,
        }
    if run_dir:
        if cfg.save:
            model.save(run_dir / "last", encoder.tokenizer, cfg.max_len)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-steps", type=int, help="override the config")
    parser.add_argument("--longest-first", action="store_true", help="costliest steps first")
    parser.add_argument("--no-eval", action="store_true", help="no evaluation, no checkpoints")
    parser.add_argument("--run", help="override the run name")
    parser.add_argument("--mlflow", action="store_true", help="track the run in MLflow (D24)")
    parser.add_argument("--no-tracking", action="store_true", help="no MLflow, e.g. for a probe")
    parser.add_argument("--data-dir", type=Path, default=paths.DATA)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    overrides: dict = {}
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.longest_first:
        overrides["longest_first"] = True
    if args.no_eval:
        overrides.update(eval_sets=[], save=False)
    if args.run:
        overrides["run"] = args.run
    if args.mlflow:
        overrides["tracking"] = "mlflow"
    if args.no_tracking:
        overrides["tracking"] = "none"
    cfg = replace(cfg, **overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = paths.RUNS / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))

    encoder = ENCODERS[cfg.architecture].from_pretrained(cfg.backbone, cfg.max_len)
    train = load_split(cfg, "train", cfg.train_limit, args.data_dir)
    eval_sets: dict[str, list[Example]] = {}
    for name in cfg.eval_sets:
        # "train" re-reads the same sampled subset, with the stored (not re-sampled) questions.
        limit = cfg.train_limit if name == "train" else cfg.eval_limit
        eval_sets[name] = load_split(cfg, name, limit, args.data_dir)
    model = DecisionModel.from_pretrained(cfg.backbone, cfg.pooling, cfg.attn, cfg.architecture)
    model = model.to(device)

    tracker = None
    (run_dir / "mlflow.json").unlink(
        missing_ok=True
    )  # a re-used run dir must not point at an old run
    if cfg.tracking == "mlflow":
        from trueodds import tracking

        tracker = tracking.MlflowTracker()
        tracker.start(cfg.run, dataclasses.asdict(cfg), run_dir)
    status = "FAILED"
    try:
        summary = fit(model, encoder, train, eval_sets, cfg, run_dir, tracker)
        if tracker:
            tracker.artifact(run_dir / "config.yaml")
            tracker.artifact(run_dir / "summary.json")
            best = run_dir / "best"
            if best.exists():  # only best is registered, never last (D24)
                version = tracking.register(best, tracker.run_id, tracking.checkpoint_tags(best))
                print(f"registered {tracking.MODEL_NAME} version {version}")
        status = "FINISHED"
    finally:
        if tracker:
            tracker.end(status)
    print(json.dumps({k: summary.get(k) for k in ("final", "peak_vram_gib")}, indent=2))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
