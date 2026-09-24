"""Temperature scaling (Phase 3, D27): one T, fitted on the pooled dev split by NLL.

The model's scores s are divided by T before the softmax, p = softmax(s / T). T > 1 softens the
probabilities, T < 1 sharpens them, and no T changes which option is on top, so accuracy is the
same before and after. The weights are never changed: T lives in `<checkpoint>/temperature.json`.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from trueodds.predict import TEMPERATURE_FILE


def _padded(scores: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K_max] float64 scores (0 at padding) and a mask that is -inf at padding, 0 elsewhere.

    The mask is added after dividing by T: a -inf score divided by T would give the padded options
    a gradient of 0 · ∞ = NaN with respect to T.
    """
    k_max = max(len(s) for s in scores)
    out = torch.zeros((len(scores), k_max), dtype=torch.float64)
    mask = torch.full((len(scores), k_max), float("-inf"), dtype=torch.float64)
    for i, s in enumerate(scores):
        out[i, : len(s)] = torch.as_tensor(np.asarray(s, dtype=np.float64))
        mask[i, : len(s)] = 0.0
    return out, mask


def _nll(s: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, t: torch.Tensor | float):
    return torch.nn.functional.cross_entropy(s / t + mask, y)


def nll_at(scores: Sequence[np.ndarray], labels: Sequence[int], temperature: float) -> float:
    s, mask = _padded(scores)
    return float(_nll(s, mask, torch.as_tensor(labels, dtype=torch.long), temperature))


def fit_temperature(scores: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    """The T that minimizes the mean NLL of softmax(s / T) over these questions.

    Optimized over log T so T stays positive; the problem is one-dimensional and smooth, and
    LBFGS converges in a few steps.
    """
    s, mask = _padded(scores)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [log_t], lr=0.5, max_iter=200, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = _nll(s, mask, y, log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def write_temperature(checkpoint: Path, temperature: float, fit: dict) -> Path:
    """`<checkpoint>/temperature.json`: T and how it was fitted (`fit` holds sources, n, dev stats)."""
    path = checkpoint / TEMPERATURE_FILE
    record = {
        "temperature": temperature,
        "fit_on": "dev",
        **fit,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
