"""Measure a ModernBERT backbone (base by default) on this GPU: a bf16 forward pass, then training throughput.

Training step = forward, backward and AdamW step on a model with fp32 weights under bf16 autocast,
with a linear score head on the CLS vector (the Phase 1 model, D2). Inputs are synthetic token ids
at a fixed length, with no padding; with --varlen, padded batches of mixed lengths, where an
attention backend that skips padding has an advantage.

For each length and gradient-checkpointing setting, the largest micro-batch (in sequences) that
fits under the VRAM budget is searched for, with the allocator capped at the budget so that going
over it raises out-of-memory instead of silently using the last GB. Throughput is then timed at
that micro-batch.

    uv run --extra gpu python scripts/gpu_bench.py                    # sdpa
    uv run --extra gpu python scripts/gpu_bench.py --attn flash_attention_2
    uv run --extra gpu python scripts/gpu_bench.py --backbone answerdotai/ModernBERT-large
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModel

from trueodds.encode import BACKBONE

MODEL = BACKBONE  # set by --backbone
GIB = 1024**3
RESULTS = Path(__file__).resolve().parents[1] / "harness" / "results"


class Scorer(torch.nn.Module):
    def __init__(self, attn: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL, attn_implementation=attn)
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden[:, 0]).squeeze(-1)


def reset_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


VARLEN = False  # set by --varlen
ONLY_NO_CKPT = False  # set by --no-checkpointing-only


def batch(n: int, length: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and attention mask; with --varlen, real lengths are uniform in [length/8, length]."""
    ids = torch.randint(5, vocab, (n, length), device="cuda")
    mask = torch.ones_like(ids)
    if VARLEN:
        gen = torch.Generator(device="cuda").manual_seed(0)
        lens = torch.randint(length // 8, length + 1, (n,), device="cuda", generator=gen)
        mask = (torch.arange(length, device="cuda")[None, :] < lens[:, None]).long()
    return ids, mask


def forward_check(attn: str, length: int = 512, n: int = 16) -> dict:
    reset_memory()
    model = Scorer(attn).cuda().to(torch.bfloat16).eval()
    ids, mask = batch(n, length, model.encoder.config.vocab_size)
    with torch.no_grad():
        scores = model(ids, mask)
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(scores).all())
    out = {
        "batch": n,
        "length": length,
        "dtype": str(scores.dtype),
        "finite": ok,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
    }
    del model
    return out


def train_steps(model, opt, n: int, length: int, steps: int) -> float:
    """Run `steps` training steps; return seconds per step after a synchronize."""
    vocab = model.encoder.config.vocab_size
    ids, mask = batch(n, length, vocab)
    target = torch.zeros(n, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(ids, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores.float(), target)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def fits(model, opt, n: int, length: int) -> bool:
    try:
        train_steps(model, opt, n, length, steps=2)
        return True
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        reset_memory()
        return False


def max_batch(model, opt, length: int, start: int = 4, limit: int = 4096) -> int:
    lo, hi = 0, start
    while hi <= limit and fits(model, opt, hi, length):
        lo, hi = hi, hi * 2
    # lo fits, hi does not (or is past the limit); bisect to a multiple of 4.
    while hi - lo > 4:
        mid = (lo + hi) // 2 // 4 * 4
        if mid <= lo:
            break
        if fits(model, opt, mid, length):
            lo = mid
        else:
            hi = mid
    return lo


def measure(attn: str, lengths: list[int], budget_gib: float, steps: int) -> list[dict]:
    rows = []
    for checkpointing in (False,) if ONLY_NO_CKPT else (False, True):
        reset_memory()
        model = Scorer(attn).cuda().train()
        if checkpointing:
            model.encoder.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
        for length in lengths:
            reset_memory()
            n = max_batch(model, opt, length)
            if n == 0:
                rows.append({"length": length, "checkpointing": checkpointing, "max_batch": 0})
                continue
            probed = n
            while True:
                # A batch that passed the 2-step probe can still run out of memory over more steps
                # (fragmentation, seen with padded batches); step down until the timed run fits.
                try:
                    reset_memory()
                    train_steps(model, opt, n, length, steps=2)  # warmup
                    sec = train_steps(model, opt, n, length, steps=steps)
                    break
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    n -= 4
            row = {
                "length": length,
                "checkpointing": checkpointing,
                "probed_batch": probed,
                "max_batch": n,
                "sec_per_step": sec,
                "seqs_per_sec": n / sec,
                "tokens_per_sec": n * length / sec,  # padded tokens with --varlen
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
            }
            print(json.dumps(row))
            rows.append(row)
        del model, opt
    return rows


def nvml_used_gib() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / GIB
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--budget-gib", type=float, default=15.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--varlen", action="store_true", help="padded batches of mixed lengths")
    parser.add_argument("--no-checkpointing-only", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    global MODEL
    MODEL = args.backbone
    global VARLEN
    VARLEN = args.varlen
    global ONLY_NO_CKPT
    ONLY_NO_CKPT = args.no_checkpointing_only

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.budget_gib * GIB / total))

    result = {
        "kind": "gpu-bench",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "attn_implementation": args.attn,
        "varlen": args.varlen,
        "budget_gib": args.budget_gib,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_gib": total / GIB,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "forward_bf16": forward_check(args.attn),
    }
    print(json.dumps(result["forward_bf16"]))
    result["training"] = measure(args.attn, args.lengths, args.budget_gib, args.steps)
    result["nvml_used_gib_at_end"] = nvml_used_gib()

    if not args.no_save:
        stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        suffix = "-varlen" if args.varlen else ""
        if MODEL != BACKBONE:
            suffix += "-" + MODEL.rsplit("/", 1)[-1].lower()
        path = RESULTS / f"{stamp}-gpu-bench-{args.attn}{suffix}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
