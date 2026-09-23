"""Download every dataset, convert it to spec 001 and write it to ~/.trueodds/data/.

    uv run python scripts/prepare_data.py

Prints and writes the counts per source and split (stats.json, also copied to harness/results/),
and exits with an error unless the overlap between training and every evaluation file is 0 (D5).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

from trueodds import paths
from trueodds.data import pipeline as p
from trueodds.data.converters import SOURCES, Source
from trueodds.data.schema import Example, write_examples

SEED = 0
TRAIN_CAP = 50_000  # D4
EVAL_CAP = 5_000  # D15
MMLU_CARVE = 2_000  # dev and test each (D14)
TOKENIZER = "answerdotai/ModernBERT-base"
SPLIT_ORDER = ["train", "dev", "test", "test-heldout-template", "test_mismatched"]


def convert(source: Source, official: str, split: str) -> tuple[list[Example], int]:
    rows = load_dataset(source.hf_id, source.hf_config, split=official, cache_dir=str(paths.RAW))
    out, skipped = [], 0
    for i, row in enumerate(rows):
        ex = source.convert(row, split, i)
        if ex is None:
            skipped += 1
        else:
            out.append(ex)
    return out, skipped


def build_source(source: Source) -> tuple[dict[str, list[Example]], dict]:
    """Official splits -> our splits, with carving and caps; templates are assigned last."""
    splits: dict[str, list[Example]] = {}
    skipped: dict[str, int] = {}
    for split, official in source.splits.items():
        splits[split], skipped[split] = convert(source, official, split)

    carved: dict[str, str] = {}
    if source.name == "mmlu_aux":
        # D14: carve by passage, so no passage is read in training and in evaluation.
        by_passage = lambda ex: ex.state or ex.id  # noqa: E731
        splits["train"], splits["test"] = p.carve(
            splits["train"], MMLU_CARVE, SEED, "test", by_passage
        )
        splits["train"], splits["dev"] = p.carve(
            splits["train"], MMLU_CARVE, SEED, "dev", by_passage
        )
        carved = {"test": "train, by passage", "dev": "train, by passage"}
    elif source.role == "train" and "dev" not in splits:
        n = p.dev_size(len(splits["train"]))
        splits["train"], splits["dev"] = p.carve(splits["train"], n, SEED, "dev")
        carved = {"dev": "train"}

    before_cap = {split: len(v) for split, v in splits.items()}
    if "train" in splits:
        splits["train"] = p.sample(splits["train"], TRAIN_CAP, SEED)
    for split in ("test", "test_mismatched"):
        if split in splits:
            splits[split] = p.stratified_sample(splits[split], EVAL_CAP, SEED)

    splits = {split: p.assign_templates(v, SEED) for split, v in splits.items()}
    info = {
        "hf_id": source.hf_id,
        "hf_config": source.hf_config,
        "role": source.role,
        "task": source.task,
        "official_splits": source.splits,
        "carved": carved,
        "skipped_at_conversion": skipped,
        "before_cap": before_cap,
    }
    return splits, info


def token_counter(name: str):
    tok = AutoTokenizer.from_pretrained(name)

    def count(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 4096):
            ids = tok(texts[i : i + 4096], add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in ids)
        return out

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument(
        "--no-results", action="store_true", help="do not copy the stats to harness/results/"
    )
    args = parser.parse_args()

    start = time.time()
    count_tokens = token_counter(TOKENIZER)
    data: dict[str, dict[str, list[Example]]] = {}
    stats: dict[str, dict] = {}
    for name in args.sources:
        print(f"[{time.time() - start:6.0f}s] {name}: download and convert", flush=True)
        data[name], stats[name] = build_source(SOURCES[name])
        stats[name]["length"] = {}
        for split, examples in data[name].items():
            data[name][split], report = p.length_filter(examples, count_tokens)
            stats[name]["length"][split] = {
                "dropped": report.dropped,
                "truncated_state_share": round(report.truncated_share, 4),
                "mean_tokens_per_sequence": round(report.tokens / max(1, sum(e.k for e in data[name][split])), 1),
            }  # fmt: skip

    evals = [e for d in data.values() for s, v in d.items() if s != "train" for e in v]
    train = {name: d["train"] for name, d in data.items() if "train" in d}
    train, dedup = p.dedup_train(train, evals)
    for name, examples in train.items():
        data[name]["train"] = examples
        stats[name]["dedup"] = {
            "dropped_overlap_with_eval": dedup.overlap_with_eval[name],
            "dropped_within_train": dedup.within_train[name],
        }

    all_train = [e for v in train.values() for e in v]
    overlap = p.overlap(all_train, evals)

    for name, splits in data.items():
        for split, examples in splits.items():
            write_examples(paths.DATA / name / f"{split}.jsonl", examples)
        held = p.heldout_template_copy(splits["test"])
        if held:
            write_examples(paths.DATA / name / "test-heldout-template.jsonl", held)
        stats[name]["splits"] = {split: p.split_stats(v) for split, v in splits.items()}
        if held:
            stats[name]["splits"]["test-heldout-template"] = p.split_stats(held)

    summary = {
        "kind": "data-stats",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "train_cap": TRAIN_CAP,
        "eval_cap": EVAL_CAP,
        "tokenizer": TOKENIZER,
        "train_eval_overlap": overlap,
        "totals": totals(data),
        "sources": stats,
    }
    text = json.dumps(summary, indent=2) + "\n"
    (paths.DATA / "stats.json").write_text(text)
    if not args.no_results:
        stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        (paths.RESULTS / f"{stamp}-data-stats.json").write_text(text)

    print_table(summary)
    print(f"done in {time.time() - start:.0f}s; wrote {paths.DATA}")
    if overlap:
        print(f"FAIL: {overlap} training examples overlap an evaluation file", file=sys.stderr)
        return 1
    print("overlap between train and every evaluation file: 0")
    return 0


def totals(data: dict[str, dict[str, list[Example]]]) -> dict:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"examples": 0, "sequences": 0})
    for splits in data.values():
        for split, examples in splits.items():
            out[split]["examples"] += len(examples)
            out[split]["sequences"] += sum(e.k for e in examples)
    return dict(out)


def print_table(summary: dict) -> None:
    head = f"{'source':16}{'split':23}{'n':>7}{'K':>10}{'trunc':>7}{'tok/seq':>8}{'dropped':>9}  labels"
    print(head)
    print("-" * len(head))
    for name, s in summary["sources"].items():
        for split, st in sorted(s["splits"].items(), key=lambda kv: SPLIT_ORDER.index(kv[0])):
            length = s["length"].get(split, {})
            dropped = length.get("dropped", 0)
            if split == "train":
                dropped += (
                    s["dedup"]["dropped_overlap_with_eval"] + s["dedup"]["dropped_within_train"]
                )
            ks = ",".join(str(k) for k in st["k"])
            labels = (
                " ".join(f"{v / st['n']:.2f}" for v in st["label_idx"].values()) if st["n"] else ""
            )
            print(f"{name:16}{split:23}{st['n']:>7}{ks:>10}"
                  f"{length.get('truncated_state_share', 0):>7.1%}"
                  f"{length.get('mean_tokens_per_sequence', 0):>8}{dropped:>9}  {labels}")  # fmt: skip
    for split, t in summary["totals"].items():
        print(f"total {split}: {t['examples']} examples, {t['sequences']} sequences")


if __name__ == "__main__":
    sys.exit(main())
