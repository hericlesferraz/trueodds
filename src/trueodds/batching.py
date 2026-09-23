"""Steps, micro-batches and collation (D10, D22).

An optimizer step is a fixed number of questions. Its questions are drawn so that they have similar
option counts and lengths, and it is split into micro-batches whose padded size (longest sequence x
sequences) stays under a token budget. Throughput on this GPU is constant in tokens (SETUP.md), so
the budget is set in tokens, and padding is what grouping by K and length saves.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from trueodds.encode import Encoded


def padded_tokens(lengths: Sequence[int], ks: Sequence[int]) -> int:
    """Tokens a micro-batch of these questions costs: every sequence padded to the longest."""
    return max(lengths) * sum(ks) if lengths else 0


def split_microbatches(lengths: Sequence[int], ks: Sequence[int], budget: int) -> list[list[int]]:
    """Positions of one step's questions, cut into micro-batches of at most `budget` padded tokens.

    Questions are taken longest first, so each micro-batch holds similar lengths. A question that is
    over the budget on its own gets a micro-batch of its own.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out: list[list[int]] = []
    current: list[int] = []
    longest = seqs = 0
    for i in order:
        if current and max(longest, lengths[i]) * (seqs + ks[i]) > budget:
            out.append(current)
            current, longest, seqs = [], 0, 0
        current.append(i)
        longest = max(longest, lengths[i])
        seqs += ks[i]
    if current:
        out.append(current)
    return out


def plan_epoch(
    lengths: Sequence[int],
    ks: Sequence[int],
    questions_per_step: int,
    rng: random.Random,
    budget: int,
    chunk_steps: int = 64,
    longest_first: bool = False,
) -> list[list[int]]:
    """One epoch as a list of steps, each a list of question indices; every question once.

    The questions are shuffled, then sorted by (K, length) within chunks of `chunk_steps` steps
    ("sortish"), cut into steps, and the steps are shuffled again. With `longest_first`, the steps
    are ordered by their costliest micro-batch instead, so that running out of memory shows at the
    start (the VRAM probe).
    """
    order = list(range(len(lengths)))
    rng.shuffle(order)
    chunk = questions_per_step * chunk_steps
    steps: list[list[int]] = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start : start + chunk], key=lambda i: (ks[i], lengths[i]))
        steps.extend(
            part[j : j + questions_per_step] for j in range(0, len(part), questions_per_step)
        )
    rng.shuffle(steps)
    if longest_first:

        def cost(step: list[int]) -> int:
            ls, kk = [lengths[i] for i in step], [ks[i] for i in step]
            return max(
                padded_tokens([ls[i] for i in mb], [kk[i] for i in mb])
                for mb in split_microbatches(ls, kk, budget)
            )

        steps.sort(key=lambda s: (-cost(s), -max(lengths[i] for i in s)))
    return steps


@dataclass
class Batch:
    input_ids: torch.Tensor  # [N, L]
    attention_mask: torch.Tensor  # [N, L]
    question_index: torch.Tensor  # [N]
    option_index: torch.Tensor  # [N]
    labels: torch.Tensor  # [B]
    n_questions: int
    k_max: int

    def to(self, device: str | torch.device) -> Batch:
        return Batch(
            self.input_ids.to(device, non_blocking=True),
            self.attention_mask.to(device, non_blocking=True),
            self.question_index.to(device, non_blocking=True),
            self.option_index.to(device, non_blocking=True),
            self.labels.to(device, non_blocking=True),
            self.n_questions,
            self.k_max,
        )

    def model_inputs(self) -> dict:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "question_index": self.question_index,
            "option_index": self.option_index,
            "n_questions": self.n_questions,
            "k_max": self.k_max,
        }


def collate(items: Sequence[Encoded], pad_id: int) -> Batch:
    seqs = [s for e in items for s in e.sequences]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, s in enumerate(seqs):
        ids[row, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[row, : len(s)] = 1
    q_index = torch.tensor([q for q, e in enumerate(items) for _ in range(e.k)], dtype=torch.long)
    o_index = torch.tensor([k for e in items for k in range(e.k)], dtype=torch.long)
    labels = torch.tensor([e.label_idx for e in items], dtype=torch.long)
    return Batch(ids, mask, q_index, o_index, labels, len(items), max(e.k for e in items))
