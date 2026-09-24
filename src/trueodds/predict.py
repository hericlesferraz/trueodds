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

from trueodds.batching import collate, split_microbatches
from trueodds.data.schema import Example
from trueodds.encode import Encoder
from trueodds.model import DecisionModel, read_meta

EVAL_BUDGET = 32_768  # padded tokens per forward pass; no activations are kept for backward
TEMPERATURE_FILE = "temperature.json"  # in a checkpoint dir, written by trueodds.calibrate


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


@torch.no_grad()
def predict_scores(
    model: DecisionModel,
    encoder: Encoder,
    examples: Sequence[Example],
    budget: int = EVAL_BUDGET,
    chunk: int = 4096,
) -> list[np.ndarray]:
    """One score vector per example (its K real options), in the order given."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray | None] = [None] * len(examples)
    for start in range(0, len(examples), chunk):
        items = encoder.encode(examples[start : start + chunk])
        for mb in split_microbatches([e.length for e in items], [e.k for e in items], budget):
            batch = collate([items[i] for i in mb], encoder.pad).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model(**batch.model_inputs()).cpu().numpy()
            for row, i in enumerate(mb):
                out[start + i] = scores[row, : items[i].k].copy()
    model.train(was_training)
    return out  # type: ignore[return-value]


def predict_probs(
    model: DecisionModel,
    encoder: Encoder,
    examples: Sequence[Example],
    temperature: float = 1.0,
    budget: int = EVAL_BUDGET,
) -> list[np.ndarray]:
    """One probability vector per example: softmax(scores / temperature)."""
    return [softmax(s, temperature) for s in predict_scores(model, encoder, examples, budget)]


def read_temperature(path: Path) -> float:
    """The checkpoint's fitted temperature, or 1.0 when none has been fitted."""
    file = path / TEMPERATURE_FILE
    return float(json.loads(file.read_text())["temperature"]) if file.exists() else 1.0


class ModelPredictor:
    def __init__(
        self, model: DecisionModel, encoder: Encoder, name: str, temperature: float = 1.0
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
        encoder = Encoder(AutoTokenizer.from_pretrained(path), meta["max_len"] or 512)
        t = read_temperature(path) if temperature else 1.0
        return cls(model, encoder, f"model:{path.parent.name}/{path.name}", t)

    def scores(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return predict_scores(self.model, self.encoder, examples)

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return predict_probs(self.model, self.encoder, examples, self.temperature)
