"""Latency of `predict()` on this GPU, per request and for many questions about one state (spec 003).

    uv run --extra gpu python -m harness.latency                      # the v1 checkpoint
    uv run --extra gpu python -m harness.latency --checkpoint ~/.trueodds/runs/<run>/best

A request is timed end to end, as a caller sees it: tokenization, the forward pass, the copy of the
scores to the CPU (which waits for the GPU) and the softmax. The state is real text (BoolQ test
passages, used only as text to time) cut to an exact number of tokens.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from harness.evaluate import write
from trueodds import paths
from trueodds.data.converters import AG_NEWS_LABELS, DBPEDIA_LABELS, YAHOO_LABELS
from trueodds.data.schema import load_examples
from trueodds.data.templates import TRAINED, render
from trueodds.infer import TrueOdds, load

STATE_TOKENS = (64, 256, 480)
REQUESTS = {  # K -> (question, options)
    2: ("Is this text about science?", ["yes", "no"]),
    4: ("What is this text about?", AG_NEWS_LABELS),
    14: ("What is this text about?", DBPEDIA_LABELS),
}
BATCH_SIZES = (10, 50)
BATCH_STATE = 256
YES_NO = ["Is the passage about a sport?", "Does the text mention a country?",
          "Is the author giving an opinion?", "Is this a news report?",
          "Does the text describe an event in the past?"]  # fmt: skip
HYPOTHESES = ["The text is about history.", "Someone is being interviewed.",
              "The passage describes a place.", "The events happened recently.",
              "The writer disagrees with a decision."]  # fmt: skip


def cut_state(text: str, tokenizer, n: int) -> str:
    """`text` cut to at most n tokens that re-tokenize to at most n (decode can merge or split)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < n:
        raise ValueError(f"the passage has {len(ids)} tokens, fewer than {n}")
    take = n
    while True:
        state = tokenizer.decode(ids[:take], clean_up_tokenization_spaces=False)
        if len(tokenizer(state, add_special_tokens=False)["input_ids"]) <= n:
            return state
        take -= 1


def passage(min_tokens: int, tokenizer, data_dir: Path = paths.DATA) -> str:
    """BoolQ test passages joined until the text is at least `min_tokens` long."""
    parts: list[str] = []
    for ex in load_examples(data_dir / "boolq" / "test.jsonl"):
        parts.append(ex.state)
        text = "\n\n".join(parts)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) >= min_tokens:
            return text
    raise ValueError("BoolQ test has too little text")


def batch_questions(n: int) -> list[tuple[str, list[str]]]:
    """n different questions about one state: topics (K = 4, 10, 14), yes/no (2), inference (3)."""
    kinds = [
        lambda p, v: (render("topic", p, {})[0], AG_NEWS_LABELS),
        lambda p, v: (render("topic", p, {})[0], YAHOO_LABELS),
        lambda p, v: (render("topic", p, {})[0], DBPEDIA_LABELS),
        lambda p, v: (render("boolq", p, {"question": YES_NO[v]})[0], ["yes", "no"]),
        lambda p, v: (
            render("mnli", p, {"hypothesis": HYPOTHESES[v]})[0],
            ["yes", "maybe", "no"],
        ),
    ]
    phrasings = TRAINED["topic"]  # "0" to "9", the same names in every task
    out = []
    for i in range(n):
        kind, round_ = i % len(kinds), i // len(kinds)
        out.append(kinds[kind](phrasings[round_ % len(phrasings)], round_ % len(YES_NO)))
    if len(set(map(repr, out))) != n:
        raise ValueError(f"{n} questions would repeat one")
    return out


def timings(call: Callable[[], object], warmup: int, repeats: int) -> np.ndarray:
    """Wall time of each call in ms, after `warmup` untimed calls."""
    for _ in range(warmup):
        call()
    out = np.empty(repeats)
    for r in range(repeats):
        start = time.perf_counter()
        call()
        out[r] = (time.perf_counter() - start) * 1000
    return out


def longest_sequence(model: TrueOdds, state: str, question: str, options: list[str]) -> int:
    from trueodds.infer import request_example

    encoded = model.predictor.encoder.encode([request_example(0, state, question, options)])
    return encoded[0].length


def measure(model: TrueOdds, warmup: int, repeats: int, batch_repeats: int) -> dict:
    tokenizer = model.predictor.encoder.tokenizer
    text = passage(max(STATE_TOKENS), tokenizer)
    states = {n: cut_state(text, tokenizer, n) for n in STATE_TOKENS}

    requests = []
    for k, (question, options) in REQUESTS.items():
        for n, state in states.items():
            t = timings(
                lambda s=state, q=question, o=options: model.predict(s, q, o), warmup, repeats
            )
            requests.append({
                "k": k,
                "state_tokens": len(tokenizer(state, add_special_tokens=False)["input_ids"]),
                "longest_sequence": longest_sequence(model, state, question, options),
                "p50_ms": float(np.percentile(t, 50)),
                "p95_ms": float(np.percentile(t, 95)),
                "mean_ms": float(t.mean()),
                "max_ms": float(t.max()),
            })  # fmt: skip
            print(f"K={k:>2} state={n:>3}: p50 {requests[-1]['p50_ms']:.1f} ms, "
                  f"p95 {requests[-1]['p95_ms']:.1f} ms")  # fmt: skip

    batches = []
    state = states[BATCH_STATE]
    for n in BATCH_SIZES:
        qs = batch_questions(n)
        batch = timings(lambda qs=qs: model.predict_batch(state, qs), 3, batch_repeats)
        loop = timings(lambda qs=qs: [model.predict(state, q, o) for q, o in qs], 3, batch_repeats)
        together = model.predict_batch(state, qs)
        alone = [model.predict(state, q, o) for q, o in qs]
        diff = max(abs(a[o] - b[o]) for a, b in zip(together, alone, strict=True) for o in a)
        b50, l50 = float(np.percentile(batch, 50)), float(np.percentile(loop, 50))
        batches.append({
            "questions": n,
            "sequences": sum(len(o) for _, o in qs),
            "ks": sorted({len(o) for _, o in qs}),
            "repeats": batch_repeats,
            "batch_p50_ms": b50,
            "loop_p50_ms": l50,
            "batch_ms_per_question": b50 / n,
            "loop_ms_per_question": l50 / n,
            "speedup": l50 / b50,
            "max_abs_diff": diff,
        })  # fmt: skip
        print(f"{n} questions: batch {b50:.0f} ms, one by one {l50:.0f} ms ({l50 / b50:.2f}x)")
    return {"requests": requests, "batches": batches}


def env(model: TrueOdds) -> dict:
    import transformers

    device = next(model.predictor.model.parameters()).device
    return {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attention": model.predictor.model.encoder.config._attn_implementation,
        "weights": str(next(model.predictor.model.parameters()).dtype).removeprefix("torch."),
        "autocast": "bfloat16" if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=paths.DEFAULT_CHECKPOINT)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--no-write", action="store_true", help="print only, write no JSON")
    args = parser.parse_args()

    model = load(args.checkpoint)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = {
        "kind": "latency",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "run": args.checkpoint.parent.name,
        "checkpoint": args.checkpoint.name,
        "temperature": model.temperature,
        "env": env(model),
        "warmup": args.warmup,
        "repeats": args.repeats,
        **measure(model, args.warmup, args.repeats, args.batch_repeats),
        "peak_vram_gib": (
            torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None
        ),
    }
    if not args.no_write:
        print(f"wrote {write(result)}")


if __name__ == "__main__":
    main()
