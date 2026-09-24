"""Scores and probabilities from a model, for the trainer's evaluation and for the harness.

`ModelPredictor` follows the harness `Predictor` protocol (`name`, `predict(examples)`), so a
checkpoint is scored by `harness.evaluate --checkpoint <dir>` like any baseline. It applies the
checkpoint's temperature when one has been fitted (Phase 3, D27). Phase 4's `predict()` (spec 003)
is built on it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trueodds.batching import split_microbatches
from trueodds.data.schema import Example
from trueodds.encode import Encoder
from trueodds.model import DecisionModel, read_meta
from trueodds.shared import SharedEncoder

ENCODERS = {"cross": Encoder, "shared": SharedEncoder}

EVAL_BUDGET = 32_768  # padded tokens per forward pass; no activations are kept for backward
TEMPERATURE_FILE = "temperature.json"  # in a checkpoint dir, written by trueodds.calibrate


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


@torch.no_grad()
def predict_scores(
    model: DecisionModel,
    encoder: Encoder | SharedEncoder,
    examples: Sequence[Example],
    budget: int = EVAL_BUDGET,
    chunk: int = 4096,
    pack: bool = False,
) -> list[np.ndarray]:
    """One score vector per example (its K real options), in the order given.

    With `pack`, a shared-state encoder puts the questions about one state into one sequence
    (spec 004); the cross-encoder scores the same either way.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray | None] = [None] * len(examples)
    for start in range(0, len(examples), chunk):
        items = encoder.encode(examples[start : start + chunk], pack=pack)
        lengths, seqs = [e.length for e in items], [e.n_sequences for e in items]
        for mb in split_microbatches(lengths, seqs, budget):
            batch = encoder.collate([items[i] for i in mb]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model(**batch.model_inputs()).cpu().numpy()
            # one score row per question, in item order (a packed item holds several)
            members = [
                (q, k) for i in mb for q, k in zip(items[i].questions, items[i].ks, strict=True)
            ]
            for row, (q, k) in enumerate(members):
                out[start + q] = scores[row, :k].copy()
    model.train(was_training)
    return out  # type: ignore[return-value]


def predict_probs(
    model: DecisionModel,
    encoder: Encoder | SharedEncoder,
    examples: Sequence[Example],
    temperature: float = 1.0,
    budget: int = EVAL_BUDGET,
    pack: bool = False,
) -> list[np.ndarray]:
    """One probability vector per example: softmax(scores / temperature)."""
    scores = predict_scores(model, encoder, examples, budget, pack=pack)
    return [softmax(s, temperature) for s in scores]


def read_temperature(path: Path) -> float:
    """The checkpoint's fitted temperature, or 1.0 when none has been fitted."""
    file = path / TEMPERATURE_FILE
    return float(json.loads(file.read_text())["temperature"]) if file.exists() else 1.0


class ModelPredictor:
    def __init__(
        self,
        model: DecisionModel,
        encoder: Encoder | SharedEncoder,
        name: str,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.temperature = temperature
        self.name = name if temperature == 1.0 else f"{name}@T={temperature:.4f}"

    @classmethod
    def load(
        cls, path: Path, device: str | None = None, temperature: bool = True
    ) -> ModelPredictor:
        """`temperature=False` ignores a fitted temperature (the raw model, T = 1)."""
        from transformers import AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = read_meta(path)
        model = DecisionModel.load(path, device=device)
        encoder_cls = ENCODERS[model.architecture]
        encoder = encoder_cls(AutoTokenizer.from_pretrained(path), meta["max_len"] or 512)
        t = read_temperature(path) if temperature else 1.0
        return cls(model, encoder, f"model:{path.parent.name}/{path.name}", t)

    def scores(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return predict_scores(self.model, self.encoder, examples)

    def predict(self, examples: Sequence[Example], pack: bool = False) -> list[np.ndarray]:
        """`pack`: questions about the same state share one sequence (shared-state models)."""
        return predict_probs(self.model, self.encoder, examples, self.temperature, pack=pack)
