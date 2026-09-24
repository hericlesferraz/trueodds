"""Shared state encoding (D13, spec 004): the state once, then each question and its options.

    [CLS] state [SEP] | question [SEP] | [CLS] option [SEP] | [CLS] option [SEP] | question [SEP] ...

One packed sequence per state. The attention masks keep questions and options apart: the state
sees only the state, a question the state and itself, an option the state, its question and
itself. Every question starts at the same position right after the state, and every option of a
question right after its question, so a question scores the same alone or packed with others, and
permuting the options permutes their scores. Each option is scored from its leading [CLS].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from trueodds.data.pipeline import MAX_TOKENS, SPECIAL_TOKENS
from trueodds.data.schema import Example
from trueodds.encode import BACKBONE

STATE = -1  # question and option id of state tokens; option id of question tokens


@dataclass(frozen=True)
class Packed:
    """One packed sequence: a state and the questions about it, with per-token segment ids."""

    ids: list[int]
    positions: list[int]
    question: list[int]  # per token: STATE, or the question's number in this pack
    option: list[int]  # per token: STATE for state and question tokens, else the option's number
    pools: list[list[int]]  # per question, the index of each option's [CLS]
    labels: list[int]
    questions: list[int]  # per question, its position in the examples given to `encode`

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def n_sequences(self) -> int:
        return 1

    @property
    def k(self) -> int:
        return max(len(p) for p in self.pools)

    @property
    def ks(self) -> list[int]:
        return [len(p) for p in self.pools]


def state_budget(
    question: Sequence[int], options: Sequence[Sequence[int]], max_len: int = MAX_TOKENS
) -> int:
    """State tokens kept for one question: its longest option's view fits in `max_len` positions.

    One more special token than the cross-encoder (the option's [CLS]); a question and option that
    would not fit without any state are rejected as in D9.
    """
    fixed = len(question) + max(len(o) for o in options) + SPECIAL_TOKENS
    if fixed > max_len:
        raise ValueError(f"question and option take {fixed} tokens, more than {max_len}")
    return max(0, max_len - fixed - 1)


def build_packed(
    state: Sequence[int],
    questions: Sequence[tuple[Sequence[int], Sequence[Sequence[int]]]],
    cls: int,
    sep: int,
    max_len: int = MAX_TOKENS,
) -> tuple[list[int], list[int], list[int], list[int], list[list[int]]]:
    """ids, positions, question ids, option ids and pooling indices of one packed sequence.

    The state is cut at its end to the smallest budget among the questions, so each question sees
    at most what it would see alone.
    """
    keep = min(state_budget(q, opts, max_len) for q, opts in questions)
    ids = [cls, *state[:keep], sep]
    positions = list(range(len(ids)))
    qid = [STATE] * len(ids)
    oid = [STATE] * len(ids)
    pools: list[list[int]] = []
    start = len(ids)
    for n, (q, opts) in enumerate(questions):
        ids += [*q, sep]
        positions += range(start, start + len(q) + 1)
        qid += [n] * (len(q) + 1)
        oid += [STATE] * (len(q) + 1)
        at = start + len(q) + 1
        pool = []
        for k, o in enumerate(opts):
            pool.append(len(ids))
            ids += [cls, *o, sep]
            positions += range(at, at + len(o) + 2)
            qid += [n] * (len(o) + 2)
            oid += [k] * (len(o) + 2)
        pools.append(pool)
    return ids, positions, qid, oid, pools


def attention_masks(
    positions: torch.Tensor,
    question: torch.Tensor,
    option: torch.Tensor,
    valid: torch.Tensor,
    window: int,
) -> dict[str, torch.Tensor]:
    """[R, L] segment ids -> the boolean [R, 1, L, L] masks of both ModernBERT layer types.

    Token i may attend to j when j is in the state, or in i's question outside the options, or in
    i's option. Sliding-window layers add |pos_i - pos_j| <= window, by position and not by row,
    as ModernBERT's own mask does. A padding token attends to itself only, so no row is empty.
    """
    qi, qj = question[:, :, None], question[:, None, :]
    oi, oj = option[:, :, None], option[:, None, :]
    same_question = (qi == qj) & (qi != STATE)
    allowed = (qj == STATE) | (same_question & ((oj == STATE) | (oi == oj)))
    allowed &= valid[:, :, None].bool() & valid[:, None, :].bool()
    allowed |= torch.eye(positions.shape[1], dtype=torch.bool, device=positions.device)
    near = (positions[:, :, None] - positions[:, None, :]).abs() <= window
    return {"full_attention": allowed[:, None], "sliding_attention": (allowed & near)[:, None]}


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # [R, L]
    attention_mask: torch.Tensor  # [R, L], 1 on real tokens
    position_ids: torch.Tensor  # [R, L]
    question_ids: torch.Tensor  # [R, L]
    option_ids: torch.Tensor  # [R, L]
    pool_row: torch.Tensor  # [N]
    pool_col: torch.Tensor  # [N]
    question_index: torch.Tensor  # [N]
    option_index: torch.Tensor  # [N]
    labels: torch.Tensor  # [B]
    n_questions: int
    k_max: int

    def to(self, device: str | torch.device) -> PackedBatch:
        tensors = {
            k: v.to(device, non_blocking=True)
            for k, v in self.__dict__.items()
            if isinstance(v, torch.Tensor)
        }
        return PackedBatch(**tensors, n_questions=self.n_questions, k_max=self.k_max)

    def model_inputs(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "labels"}


def collate_packed(items: Sequence[Packed], pad_id: int) -> PackedBatch:
    """Packed sequences, padded to the longest; one score row per question, in item order."""
    width = max(p.length for p in items)
    shape = (len(items), width)
    ids = torch.full(shape, pad_id, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.long)
    pos = torch.zeros(shape, dtype=torch.long)
    qid = torch.full(shape, STATE, dtype=torch.long)
    oid = torch.full(shape, STATE, dtype=torch.long)
    rows, cols, q_index, o_index, labels = [], [], [], [], []
    for r, p in enumerate(items):
        n = p.length
        ids[r, :n] = torch.tensor(p.ids)
        mask[r, :n] = 1
        pos[r, :n] = torch.tensor(p.positions)
        qid[r, :n] = torch.tensor(p.question)
        oid[r, :n] = torch.tensor(p.option)
        for pool, label in zip(p.pools, p.labels, strict=True):
            row = len(labels)
            rows += [r] * len(pool)
            cols += pool
            q_index += [row] * len(pool)
            o_index += range(len(pool))
            labels.append(label)
    return PackedBatch(
        ids,
        mask,
        pos,
        qid,
        oid,
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
        torch.tensor(q_index, dtype=torch.long),
        torch.tensor(o_index, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        len(labels),
        max(p.k for p in items),
    )


class SharedEncoder:
    """The `Encoder` interface for the shared-state model: `encode`, `collate`, `lengths`."""

    architecture = "shared"

    def __init__(self, tokenizer, max_len: int = MAX_TOKENS) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, name: str = BACKBONE, max_len: int = MAX_TOKENS) -> SharedEncoder:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(name), max_len)

    def _tokenize(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def encode(self, examples: Sequence[Example], pack: bool = False) -> list[Packed]:
        """One packed sequence per question; with `pack`, one per distinct state, in first-seen
        order, holding every question about that state in the order given."""
        groups: dict[str, list[int]] = {}
        for i, ex in enumerate(examples):
            groups.setdefault(ex.state if pack else str(i), []).append(i)
        members = list(groups.values())
        states = self._tokenize([examples[m[0]].state for m in members])
        questions = self._tokenize([ex.question for ex in examples])
        unique = sorted({o for ex in examples for o in ex.options})
        options = dict(zip(unique, self._tokenize(unique), strict=True))
        out = []
        for m, state in zip(members, states, strict=True):
            qs = [(questions[i], [options[o] for o in examples[i].options]) for i in m]
            ids, pos, qid, oid, pools = build_packed(state, qs, self.cls, self.sep, self.max_len)
            labels = [examples[i].label_idx for i in m]
            out.append(Packed(ids, pos, qid, oid, pools, labels, m))
        return out

    def collate(self, items: Sequence[Packed]) -> PackedBatch:
        return collate_packed(items, self.pad)

    def n_sequences(self, example: Example) -> int:
        return 1

    def lengths(self, examples: Sequence[Example], chunk: int = 4096) -> list[int]:
        """The packed length of each question on its own; used to group batches."""
        out: list[int] = []
        for start in range(0, len(examples), chunk):
            out.extend(p.length for p in self.encode(examples[start : start + chunk]))
        return out
