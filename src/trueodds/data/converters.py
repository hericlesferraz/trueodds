"""One converter per dataset, from a raw Hugging Face row to an `Example` (spec 001).

Each converter is a pure function `(row, split, index) -> Example | None`. It returns None for a
row that cannot be an example (no label, fewer than two distinct options). Options with the same
text are merged into one; the pipeline counts
those. Templated questions are rendered with trained template 0 here, and re-rendered by the
pipeline with the template the split calls for.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from trueodds.data.schema import Example
from trueodds.data.templates import render

AG_NEWS_LABELS = ["World", "Sports", "Business", "Science and technology"]
YAHOO_LABELS = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]
DBPEDIA_LABELS = [
    "Company",
    "Educational institution",
    "Artist",
    "Athlete",
    "Office holder",
    "Means of transportation",
    "Building",
    "Natural place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written work",
]
MNLI_OPTIONS = ["yes", "maybe", "no"]  # entailment, neutral, contradiction: GLUE's label order
# Held-out sources added in Phase 5D (D31). Their question is fixed: a phrasing never trained.
WIC_QUESTION = 'Is the word "{word}" used with the same meaning in both sentences?'
SENTIMENT_QUESTION = "Is this movie review positive or negative?"


def _example(
    source: str,
    split: str,
    key: object,
    state: str,
    options: list[str],
    label_idx: int,
    *,
    task: str | None = None,
    vars: dict[str, str] | None = None,
    question: str | None = None,
) -> Example | None:
    options, label_idx = merge_repeated([" ".join(o.split()) for o in options], label_idx)
    if task is not None:
        question, template_id = render(task, "0", vars or {})
    else:
        template_id, vars = "native", {}
    ex = Example(
        id=f"{source}/{split}/{key}",
        source=source,
        split=split,
        state=state.strip(),
        question=(question or "").strip(),
        options=options,
        label_idx=label_idx,
        template_id=template_id,
        vars=vars or {},
    )
    try:
        ex.validate()
    except ValueError:
        return None
    return ex


def merge_repeated(options: list[str], label_idx: int) -> tuple[list[str], int]:
    """Keep the first of options with the same text; the label follows its text."""
    label = options[label_idx]
    merged = list(dict.fromkeys(options))
    return merged, merged.index(label)


def _letter_choice(source: str, row: dict, split: str, key: object) -> Example | None:
    """ARC and CommonsenseQA: `choices = {text, label}` and an `answerKey` among the labels."""
    labels = list(row["choices"]["label"])
    if row["answerKey"] not in labels:
        return None
    return _example(
        source,
        split,
        key,
        "",
        list(row["choices"]["text"]),
        labels.index(row["answerKey"]),
        question=row["question"],
    )


def boolq(row: dict, split: str, index: int) -> Example | None:
    q = " ".join(row["question"].split()).rstrip("?")
    q = q[:1].upper() + q[1:] + "?"
    return _example(
        "boolq",
        split,
        index,
        row["passage"],
        ["yes", "no"],
        0 if row["answer"] else 1,
        task="boolq",
        vars={"question": q},
    )


def mnli(row: dict, split: str, index: int) -> Example | None:
    if row["label"] not in (0, 1, 2):
        return None
    return _example(
        "mnli",
        split,
        row.get("idx", index),
        row["premise"],
        MNLI_OPTIONS,
        row["label"],
        task="mnli",
        vars={"hypothesis": " ".join(row["hypothesis"].split())},
    )


def arc_easy(row: dict, split: str, index: int) -> Example | None:
    return _letter_choice("arc_easy", row, split, row.get("id", index))


def arc_challenge(row: dict, split: str, index: int) -> Example | None:
    return _letter_choice("arc_challenge", row, split, row.get("id", index))


def commonsense_qa(row: dict, split: str, index: int) -> Example | None:
    return _letter_choice("commonsense_qa", row, split, row.get("id", index))


def hellaswag(row: dict, split: str, index: int) -> Example | None:
    if row.get("source_id", "").startswith("wikihow"):
        return None  # D17: wikiHow contexts are left out; ActivityNet ones are kept
    label = str(row["label"]).strip()
    if not label.isdigit():
        return None  # the official test split has no labels
    return _example(
        "hellaswag",
        split,
        row.get("ind", index),
        row["ctx"],
        list(row["endings"]),
        int(label),
        task="hellaswag",
    )


_SENTENCE_END = re.compile(r"(?<=[.?!])[\"')\]]?\s+(?=\S)")
# A period after these words, or after a single letter (an initial, "U.S."), does not end a sentence.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "mt", "no", "vs", "etc",
                  "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "co", "inc", "ltd"}  # fmt: skip


def _is_abbreviation(text: str, end: int) -> bool:
    """True when the period just before `end` closes an abbreviation rather than a sentence."""
    before = text[:end].rstrip()
    if not before.endswith("."):
        return False
    word = before[:-1].rsplit(" ", 1)[-1].lower().lstrip("(\"'")
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


def split_passage(text: str, min_passage: int = 300, min_question: int = 10) -> tuple[str, str]:
    """Split MMLU aux text into (passage, question) at the last sentence boundary.

    Texts up to `min_passage` characters are a question alone. A period after an abbreviation
    ("Mr.") or an initial is not a boundary, and the boundary is moved back while the question it
    leaves is shorter than `min_question` characters.
    """
    text = " ".join(text.split())
    if len(text) <= min_passage:
        return "", text
    for m in reversed(list(_SENTENCE_END.finditer(text))):
        if _is_abbreviation(text, m.start() + 1):
            continue
        if len(text) - m.end() >= min_question:
            return text[: m.start()].strip(), text[m.end() :].strip()
    return "", text


def mmlu_aux(row: dict, split: str, index: int) -> Example | None:
    row = row.get("train", row)  # the auxiliary_train config nests every row under "train"
    passage, question = split_passage(row["question"])
    return _example(
        "mmlu_aux",
        split,
        index,
        passage,
        list(row["choices"]),
        int(row["answer"]),
        question=question,
    )


def _topic(source: str, labels: list[str], state: str, label: int, split: str, key: object):
    return _example(source, split, key, state, labels, label, task="topic", vars={})


def ag_news(row: dict, split: str, index: int) -> Example | None:
    return _topic("ag_news", AG_NEWS_LABELS, row["text"], row["label"], split, index)


def yahoo(row: dict, split: str, index: int) -> Example | None:
    parts = [row["question_title"], row["question_content"], row["best_answer"]]
    state = "\n".join(p.strip() for p in parts if p and p.strip())
    return _topic("yahoo", YAHOO_LABELS, state, row["topic"], split, row.get("id", index))


def dbpedia(row: dict, split: str, index: int) -> Example | None:
    state = f"{row['title'].strip()}\n{row['content'].strip()}"
    return _topic("dbpedia", DBPEDIA_LABELS, state, row["label"], split, index)


# Phase 5D held-out sources. Every labeled split is read as one test file, so `idx` repeats across
# the concatenated splits; the row's position is the key instead.


def rte(row: dict, split: str, index: int) -> Example | None:
    """2-way inference, asked with the trained MNLI phrasings; label 0 is entailment."""
    if row["label"] not in (0, 1):
        return None
    return _example(
        "rte",
        split,
        index,
        row["sentence1"],
        ["yes", "no"],
        row["label"],
        task="mnli",
        vars={"hypothesis": " ".join(row["sentence2"].split())},
    )


def wic(row: dict, split: str, index: int) -> Example | None:
    """Word in context: label 1 means the word has the same meaning in both sentences."""
    if row["label"] not in (0, 1):
        return None
    state = f"1. {row['sentence1'].strip()}\n2. {row['sentence2'].strip()}"
    question = WIC_QUESTION.format(word=row["word"].strip())
    return _example("wic", split, index, state, ["yes", "no"], 1 - row["label"], question=question)


def rotten_tomatoes(row: dict, split: str, index: int) -> Example | None:
    """Sentiment of one review sentence: label 1 is positive."""
    if row["label"] not in (0, 1):
        return None
    options = ["positive", "negative"]
    return _example(
        "rotten_tomatoes", split, index, row["text"], options, 1 - row["label"],
        question=SENTIMENT_QUESTION,
    )  # fmt: skip


@dataclass(frozen=True)
class Source:
    """Where a source comes from on the Hub, and which official split fills each of ours."""

    name: str
    hf_id: str
    hf_config: str | None
    task: str  # boolq, mnli, hellaswag, topic or native
    role: str  # "train" or "heldout"
    splits: dict[str, str]  # our split -> official split; "dev" absent means carved from train
    convert: Callable[[dict, str, int], Example | None]


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source("boolq", "google/boolq", None, "boolq", "train",
               {"train": "train", "test": "validation"}, boolq),
        Source("mnli", "nyu-mll/glue", "mnli", "mnli", "train",
               {"train": "train", "test": "validation_matched",
                "test_mismatched": "validation_mismatched"}, mnli),
        Source("arc_easy", "allenai/ai2_arc", "ARC-Easy", "native", "train",
               {"train": "train", "dev": "validation", "test": "test"}, arc_easy),
        Source("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "native", "train",
               {"train": "train", "dev": "validation", "test": "test"}, arc_challenge),
        Source("hellaswag", "Rowan/hellaswag", None, "hellaswag", "train",
               {"train": "train", "test": "validation"}, hellaswag),
        Source("mmlu_aux", "cais/mmlu", "auxiliary_train", "native", "train",
               {"train": "train"}, mmlu_aux),  # dev and test are carved by passage (D14)
        Source("ag_news", "fancyzhx/ag_news", None, "topic", "train",
               {"train": "train", "test": "test"}, ag_news),
        Source("yahoo", "community-datasets/yahoo_answers_topics", "yahoo_answers_topics", "topic",
               "train", {"train": "train", "test": "test"}, yahoo),
        Source("commonsense_qa", "tau/commonsense_qa", None, "native", "heldout",
               {"test": "validation"}, commonsense_qa),
        Source("dbpedia", "fancyzhx/dbpedia_14", "dbpedia_14", "topic", "heldout",
               {"test": "test"}, dbpedia),
        # Phase 5D (D31): no official split is trained on, so every labeled one is test.
        Source("rte", "nyu-mll/glue", "rte", "mnli", "heldout",
               {"test": "train+validation"}, rte),
        Source("wic", "aps/super_glue", "wic", "native", "heldout",
               {"test": "train+validation"}, wic),
        Source("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", None, "native",
               "heldout", {"test": "train+validation+test"}, rotten_tomatoes),
    ]
}  # fmt: skip
