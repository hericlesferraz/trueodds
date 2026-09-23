"""The unified example (spec 001) and its validation, reading and writing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPLITS = ("train", "dev", "test", "test_mismatched")


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    split: str
    state: str
    question: str
    options: list[str]
    label_idx: int
    template_id: str
    vars: dict[str, str] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.options)

    @property
    def label(self) -> str:
        return self.options[self.label_idx]

    def validate(self) -> None:
        """Raise ValueError if the example breaks spec 001."""
        problems = []
        if not self.id:
            problems.append("empty id")
        if self.split not in SPLITS:
            problems.append(f"unknown split {self.split!r}")
        if not self.question.strip():
            problems.append("empty question")
        if len(self.options) < 2:
            problems.append(f"K = {len(self.options)} < 2")
        if any(not o.strip() for o in self.options):
            problems.append("empty option")
        if len(set(self.options)) != len(self.options):
            problems.append("duplicate options")
        if not 0 <= self.label_idx < len(self.options):
            problems.append(f"label_idx {self.label_idx} out of range for K = {len(self.options)}")
        if self.template_id == "native" and self.vars:
            problems.append("a native question has no template vars")
        if problems:
            raise ValueError(f"{self.id}: " + "; ".join(problems))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, row: dict) -> Example:
        return cls(
            id=row["id"],
            source=row["source"],
            split=row["split"],
            state=row["state"],
            question=row["question"],
            options=list(row["options"]),
            label_idx=int(row["label_idx"]),
            template_id=row["template_id"],
            vars=dict(row.get("vars") or {}),
        )


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            ex.validate()
            f.write(ex.to_json() + "\n")
            n += 1
    return n


def iter_examples(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = Example.from_dict(json.loads(line))
                ex.validate()
                yield ex


def load_examples(path: Path) -> list[Example]:
    return list(iter_examples(path))
