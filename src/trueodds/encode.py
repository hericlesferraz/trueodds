"""From examples to token ids: `[CLS] state [SEP] question [SEP] option [SEP]`, cut to fit (D9).

The question and the option are never cut; the state is cut at its end to what is left. Only the
tokenizer is needed here, not torch, so the truncation is tested on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trueodds.data.pipeline import MAX_TOKENS, SPECIAL_TOKENS
from trueodds.data.schema import Example

if TYPE_CHECKING:
    from trueodds.batching import Batch

BACKBONE = "answerdotai/ModernBERT-base"


def build_ids(
    state: Sequence[int],
    question: Sequence[int],
    option: Sequence[int],
    cls: int,
    sep: int,
    max_len: int = MAX_TOKENS,
) -> list[int]:
    fixed = len(question) + len(option) + SPECIAL_TOKENS
    if fixed > max_len:
        # prepare_data drops these examples (D9), so reaching this is a bug upstream.
        raise ValueError(f"question and option take {fixed} tokens, more than {max_len}")
    return [cls, *state[: max_len - fixed], sep, *question, sep, *option, sep]


@dataclass(frozen=True)
class Encoded:
    """One question: a token id list per option, and the label."""

    sequences: list[list[int]]
    label_idx: int
    questions: list[int] = field(default_factory=lambda: [0])  # its position in `encode`'s input

    @property
    def k(self) -> int:
        return len(self.sequences)

    @property
    def ks(self) -> list[int]:
        return [self.k]

    @property
    def n_sequences(self) -> int:
        return self.k

    @property
    def length(self) -> int:
        return max(len(s) for s in self.sequences)


class Encoder:
    architecture = "cross"

    def __init__(self, tokenizer, max_len: int = MAX_TOKENS) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls = tokenizer.cls_token_id
        self.sep = tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, name: str = BACKBONE, max_len: int = MAX_TOKENS) -> Encoder:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(name), max_len)

    def _tokenize(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def encode(self, examples: Sequence[Example], pack: bool = False) -> list[Encoded]:
        """One `Encoded` per question; `pack` is for the shared-state encoder and changes nothing
        here, since every option sequence repeats the state anyway."""
        states = self._tokenize([ex.state for ex in examples])
        questions = self._tokenize([ex.question for ex in examples])
        unique = sorted({o for ex in examples for o in ex.options})
        options = dict(zip(unique, self._tokenize(unique), strict=True))
        return [
            Encoded(
                [build_ids(s, q, options[o], self.cls, self.sep, self.max_len) for o in ex.options],
                ex.label_idx,
                [i],
            )
            for i, (ex, s, q) in enumerate(zip(examples, states, questions, strict=True))
        ]

    def collate(self, items: Sequence[Encoded]) -> Batch:
        from trueodds.batching import collate  # torch, which this module otherwise does not need

        return collate(items, self.pad)

    def n_sequences(self, example: Example) -> int:
        return example.k

    def lengths(self, examples: Sequence[Example], chunk: int = 4096) -> list[int]:
        """The longest sequence of each question, after truncation; used to group batches."""
        out: list[int] = []
        for start in range(0, len(examples), chunk):
            out.extend(e.length for e in self.encode(examples[start : start + chunk]))
        return out
