"""Question templates per task: four trained phrasings and one held-out phrasing (D8)."""

from __future__ import annotations

import hashlib
import random

HELDOUT = "heldout"

# The `{question}` of BoolQ is the dataset's question, capitalized and ending in "?".
TEMPLATES: dict[str, dict[str, str]] = {
    "boolq": {
        "0": "{question}",
        "1": "According to the passage, {question_lc}",
        "2": "Based on the text, {question_lc}",
        "3": "Answer from the passage: {question_lc}",
        HELDOUT: "Going only by what the text says, {question_lc}",
    },
    "mnli": {
        "0": "Does it follow that {hypothesis_lc}?",
        "1": "If the text is true, is this also true: {hypothesis}",
        "2": "Can we conclude that {hypothesis_lc}?",
        "3": "Given the text, is the following statement correct? {hypothesis}",
        HELDOUT: 'Is the claim "{hypothesis}" supported by the text?',
    },
    "hellaswag": {
        "0": "What happens next?",
        "1": "How does this continue?",
        "2": "Which ending is the most likely?",
        "3": "What is the most plausible continuation?",
        HELDOUT: "Pick the sentence that best finishes the description.",
    },
    "topic": {
        "0": "What is this text about?",
        "1": "Which topic does this text belong to?",
        "2": "What category best describes this text?",
        "3": "What is the subject of this text?",
        HELDOUT: "If you had to file this text under one heading, which would it be?",
    },
}

TRAINED = {task: [n for n in t if n != HELDOUT] for task, t in TEMPLATES.items()}


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s[:1].isupper() and not s[1:2].isupper() else s


def _derived(vars: dict[str, str]) -> dict[str, str]:
    out = dict(vars)
    for key, value in vars.items():
        out[f"{key}_lc"] = _lower_first(value)
    if "hypothesis" in vars:
        # "{hypothesis_lc}?" must not end in ".?"
        out["hypothesis_lc"] = _lower_first(vars["hypothesis"].strip()).rstrip(".!? ")
    return out


def render(task: str, n: str, vars: dict[str, str]) -> tuple[str, str]:
    """Return (question, template_id) for template `n` of `task`."""
    return TEMPLATES[task][n].format(**_derived(vars)), f"{task}:{n}"


def pick_random(task: str, rng: random.Random) -> str:
    return rng.choice(TRAINED[task])


def pick_stable(task: str, example_id: str) -> str:
    """A trained template chosen by the id, the same in every run (dev and test)."""
    h = int.from_bytes(hashlib.sha256(example_id.encode()).digest()[:8], "big")
    return TRAINED[task][h % len(TRAINED[task])]
