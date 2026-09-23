"""From converted examples to the files of spec 001: split, cap, templates, dedup and length.

Every function here is pure and deterministic given its seed, so it is unit-tested without
downloads. `scripts/prepare_data.py` wires it to the Hugging Face datasets.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, replace

from trueodds.data.schema import Example
from trueodds.data.templates import HELDOUT, pick_random, pick_stable, render

MAX_TOKENS = 512
SPECIAL_TOKENS = 4  # [CLS] state [SEP] question [SEP] option [SEP]


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def dedup_key(ex: Example) -> tuple:
    """Spec 001: the state, the template variables (or the native question), the sorted options.

    The rendered question of a templated task is not in the key: it is a template, the same for
    every example of that task.
    """
    if ex.template_id == "native":
        asked = normalize(ex.question)
    else:
        asked = "|".join(f"{k}={normalize(v)}" for k, v in sorted(ex.vars.items()))
    return (normalize(ex.state), asked, tuple(sorted(normalize(o) for o in ex.options)))


# --- splits and sampling -------------------------------------------------------------------------


def dev_size(n_train: int, fraction: float = 0.1, max_dev: int = 2000) -> int:
    return min(max_dev, int(n_train * fraction))


def carve(
    examples: Sequence[Example],
    n: int,
    seed: int,
    split: str,
    group: Callable[[Example], Hashable] | None = None,
) -> tuple[list[Example], list[Example]]:
    """Move about `n` examples out of `examples` into `split`; return (rest, carved).

    With `group`, whole groups move together (MMLU aux passages, D14), so the carved count is the
    first total that reaches `n`.
    """
    group = group or (lambda ex: ex.id)
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[group(ex)].append(i)
    order = list(groups)
    random.Random(seed).shuffle(order)
    taken: set[int] = set()
    for g in order:
        if len(taken) >= n:
            break
        taken.update(groups[g])
    rest = [ex for i, ex in enumerate(examples) if i not in taken]
    carved = [_resplit(ex, split) for i, ex in enumerate(examples) if i in taken]
    return rest, carved


def _resplit(ex: Example, split: str) -> Example:
    source, _, key = ex.id.split("/", 2)
    return replace(ex, split=split, id=f"{source}/{split}/{key}")


def sample(examples: Sequence[Example], n: int, seed: int) -> list[Example]:
    """A uniform sample of `n`, in the original order; everything if there are `n` or fewer."""
    if len(examples) <= n:
        return list(examples)
    keep = set(random.Random(seed).sample(range(len(examples)), n))
    return [ex for i, ex in enumerate(examples) if i in keep]


def stratified_sample(examples: Sequence[Example], n: int, seed: int) -> list[Example]:
    """A sample of `n` that keeps the share of each `label_idx` (largest remainder), in order."""
    if len(examples) <= n:
        return list(examples)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_label[ex.label_idx].append(i)
    total = len(examples)
    quota = {lab: n * len(ix) // total for lab, ix in by_label.items()}
    remainders = sorted(by_label, key=lambda lab: -(n * len(by_label[lab]) % total))
    for lab in remainders[: n - sum(quota.values())]:
        quota[lab] += 1
    rng = random.Random(seed)
    keep: set[int] = set()
    for lab in sorted(by_label):
        keep.update(rng.sample(by_label[lab], quota[lab]))
    return [ex for i, ex in enumerate(examples) if i in keep]


# --- templates -----------------------------------------------------------------------------------


def task_of(ex: Example) -> str | None:
    return None if ex.template_id == "native" else ex.template_id.split(":", 1)[0]


def with_template(ex: Example, n: str) -> Example:
    task = task_of(ex)
    if task is None:
        return ex
    question, template_id = render(task, n, ex.vars)
    return replace(ex, question=question, template_id=template_id)


def assign_templates(examples: Iterable[Example], seed: int) -> list[Example]:
    """Train: a trained template drawn at random. Dev and test: one chosen by the id (spec 001)."""
    rng = random.Random(seed)
    out = []
    for ex in examples:
        task = task_of(ex)
        if task is None:
            out.append(ex)
        elif ex.split == "train":
            out.append(with_template(ex, pick_random(task, rng)))
        else:
            out.append(with_template(ex, pick_stable(task, ex.id)))
    return out


def heldout_template_copy(examples: Iterable[Example]) -> list[Example]:
    """The templated examples rendered with the held-out template; native ones are left out."""
    return [with_template(ex, HELDOUT) for ex in examples if task_of(ex) is not None]


# --- dedup ---------------------------------------------------------------------------------------


@dataclass
class DedupReport:
    overlap_with_eval: Counter
    within_train: Counter


def dedup_train(
    train: dict[str, list[Example]], evals: Iterable[Example]
) -> tuple[dict[str, list[Example]], DedupReport]:
    """Drop training examples that match any eval example, then duplicates within train (D5).

    Sources are processed in the order of `train`, so "first" is well defined.
    """
    eval_keys = {dedup_key(ex) for ex in evals}
    seen: set[tuple] = set()
    report = DedupReport(Counter(), Counter())
    out: dict[str, list[Example]] = {}
    for source, examples in train.items():
        kept = []
        for ex in examples:
            key = dedup_key(ex)
            if key in eval_keys:
                report.overlap_with_eval[source] += 1
            elif key in seen:
                report.within_train[source] += 1
            else:
                seen.add(key)
                kept.append(ex)
        out[source] = kept
    return out, report


def overlap(train: Iterable[Example], evals: Iterable[Example]) -> int:
    eval_keys = {dedup_key(ex) for ex in evals}
    return sum(dedup_key(ex) in eval_keys for ex in train)


# --- length (D9) ---------------------------------------------------------------------------------


@dataclass
class LengthReport:
    kept: int = 0
    dropped: int = 0
    truncated: int = 0
    tokens: int = 0  # sum over kept examples and their options, after truncation

    @property
    def truncated_share(self) -> float:
        return self.truncated / self.kept if self.kept else 0.0


def length_filter(
    examples: Sequence[Example],
    count_tokens: Callable[[list[str]], list[int]],
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[Example], LengthReport]:
    """Drop examples whose question and longest option leave no room for the state.

    `count_tokens` maps texts to token counts without special tokens (the ModernBERT tokenizer in
    `prepare_data`, a word counter in tests).
    """
    states = count_tokens([ex.state for ex in examples])
    questions = count_tokens([ex.question for ex in examples])
    unique_options = sorted({o for ex in examples for o in ex.options})
    option_len = dict(zip(unique_options, count_tokens(unique_options), strict=True))
    report = LengthReport()
    kept = []
    for ex, s, q in zip(examples, states, questions, strict=True):
        opts = [option_len[o] for o in ex.options]
        fixed = q + max(opts) + SPECIAL_TOKENS
        if fixed + 1 > max_tokens:
            report.dropped += 1
            continue
        kept.append(ex)
        report.kept += 1
        report.truncated += s + fixed > max_tokens
        report.tokens += sum(min(max_tokens, s + q + o + SPECIAL_TOKENS) for o in opts)
    return kept, report


# --- stats ---------------------------------------------------------------------------------------


def split_stats(examples: Sequence[Example]) -> dict:
    return {
        "n": len(examples),
        "k": dict(sorted(Counter(ex.k for ex in examples).items())),
        "label_idx": dict(sorted(Counter(ex.label_idx for ex in examples).items())),
        "templates": dict(sorted(Counter(ex.template_id for ex in examples).items())),
        "sequences": sum(ex.k for ex in examples),
    }
