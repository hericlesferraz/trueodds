"""Probabilities from a model, for the trainer's evaluation and for the harness.

`ModelPredictor` follows the harness `Predictor` protocol (`name`, `predict(examples)`), so a
checkpoint is scored by `harness.evaluate --checkpoint <dir>` like any baseline. Phase 4's
`predict()` (spec 003) is built on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trueodds.batching import collate, split_microbatches
from trueodds.data.schema import Example
from trueodds.encode import Encoder
from trueodds.model import DecisionModel, read_meta

EVAL_BUDGET = 32_768  # padded tokens per forward pass; no activations are kept for backward


@torch.no_grad()
def predict_probs(
    model: DecisionModel,
    encoder: Encoder,
    examples: Sequence[Example],
    budget: int = EVAL_BUDGET,
    chunk: int = 4096,
) -> list[np.ndarray]:
    """One probability vector per example, in the order given."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray | None] = [None] * len(examples)
    for start in range(0, len(examples), chunk):
        items = encoder.encode(examples[start : start + chunk])
        for mb in split_microbatches([e.length for e in items], [e.k for e in items], budget):
            batch = collate([items[i] for i in mb], encoder.pad).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = model(**batch.model_inputs())
            probs = torch.softmax(scores, dim=-1).cpu().numpy().astype(np.float64)
            for row, i in enumerate(mb):
                out[start + i] = probs[row, : items[i].k]
    model.train(was_training)
    return out  # type: ignore[return-value]


class ModelPredictor:
    def __init__(self, model: DecisionModel, encoder: Encoder, name: str) -> None:
        self.model = model
        self.encoder = encoder
        self.name = name

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> ModelPredictor:
        from transformers import AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta = read_meta(path)
        model = DecisionModel.load(path, device=device)
        encoder = Encoder(AutoTokenizer.from_pretrained(path), meta["max_len"] or 512)
        return cls(model, encoder, f"model:{path.parent.name}/{path.name}")

    def predict(self, examples: Sequence[Example]) -> list[np.ndarray]:
        return predict_probs(self.model, self.encoder, examples)
