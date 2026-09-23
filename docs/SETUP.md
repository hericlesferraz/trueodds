# Setup

How to get trueodds running on a machine like the author's, with every workaround that was needed.
Keep this file true: add a step only after it has worked.

## Reference machine

Checked on 2026-09-23: RTX 5060 Ti 16 GB (Blackwell, compute capability 12.0, 15.45 GiB usable),
driver 580.173, 32 GB RAM, Ubuntu, uv 0.9.18, Python 3.12. No CUDA toolkit (`nvcc`) is installed,
and none is needed.

## Python

```bash
uv sync                  # base package and dev tools; no GPU needed
uv sync --extra gpu      # adds torch (cu128 index) and nvidia-ml-py
uv run pytest            # unit tests; GPU tests are deselected
```

`uv sync --extra gpu` resolved **torch 2.11.0+cu128** and transformers 5.17.0 on 2026-09-23. It
runs on compute capability 12.0 with no extra steps.

Run GPU commands as `uv run --extra gpu ...`. A plain `uv sync` removes the GPU packages again; a
plain `uv run` leaves them installed (checked with uv 0.9.18), but naming the extra keeps GPU
commands working from a fresh environment.

## Data

```bash
uv run python scripts/prepare_data.py     # ~3 min with a warm download cache
```

Downloads the ten datasets to `~/.trueodds/data/raw/`, and writes the spec 001 files and
`stats.json` to `~/.trueodds/data/` (≈ 0.5 GB with the raw cache). It fails unless the overlap
between training and every evaluation file is 0. `TRUEODDS_HOME` moves the whole `~/.trueodds/`
tree. Use `--sources` to rebuild some sources, and `--no-results` to skip the copy of the stats to
`harness/results/`.

Result on 2026-09-23 (`harness/results/2026-09-23-112714-data-stats.json`): 222,639 training
questions (1,123,566 sequences), 11,286 dev, 33,282 test and held-out, 0 overlap. The state is cut to
fit 512 tokens in 6% of MMLU aux questions (RACE passages), 3% of Yahoo, 0.2% of BoolQ, and never
elsewhere.

## Baselines

```bash
uv run python -m harness.evaluate --baselines          # harness/results/<stamp>-baselines.json
uv run python -m harness.report harness/results/*-baselines.json
```

## GPU: ModernBERT-base

```bash
uv run --extra gpu python scripts/gpu_bench.py                       # fixed length, sdpa
uv run --extra gpu python scripts/gpu_bench.py --varlen --no-checkpointing-only
```

A training step is forward, backward and a fused AdamW step, with fp32 weights under bf16 autocast
and a linear head on the CLS vector. The micro-batch is the largest that fits under a 15 GiB
allocator cap (the VRAM rule in CLAUDE.md). A bf16 forward pass of 16 × 512 tokens runs, with finite
outputs, in 0.65 GiB.

**SDPA, torch 2.11, fixed length** (`2026-09-23-110050-gpu-bench-sdpa.json`):

| Tokens | Checkpointing | Max micro-batch | Sequences/s | Tokens/s | Peak reserved |
|---|---|---|---|---|---|
| 128 | no | 152 | 207 | 26.4k | 14.0 GiB |
| 256 | no | 76 | 100 | 25.6k | 14.0 GiB |
| 512 | no | 36 | 47 | 24.1k | 13.7 GiB |
| 128 | yes | 816 | 160 | 20.5k | 12.8 GiB |
| 256 | yes | 408 | 77 | 19.8k | 12.8 GiB |
| 512 | yes | 204 | 36 | 18.7k | 12.7 GiB |

- Throughput is about **25k tokens/s** whatever the length, so cost is set by the number of tokens,
  not of sequences.
- Gradient checkpointing costs about 22% of the throughput and multiplies the micro-batch by ~5.5.
  The effective batch of ~32 questions (D10) is a few hundred sequences at most, so checkpointing is
  not needed.

### flash-attn

No flash-attn wheel matches torch 2.11. The official 2.8.3.post1 wheels stop at torch 2.8 for CUDA
12, and a source build needs `nvcc`. To see whether flash-attn is worth pinning torch 2.8 for, it
was measured in a separate uv project (torch 2.8.0+cu128 and
`flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312`): **the wheel imports and runs on compute
capability 12.0.**

| Backend | Batches | 512-token micro-batch | Sequences/s at 128 / 256 / 512 |
|---|---|---|---|
| SDPA, torch 2.11 | fixed length | 36 | 207 / 100 / 47 |
| flash_attention_2, torch 2.8 | fixed length | 36 | 205 / 102 / 50 |
| SDPA, torch 2.11 | padded, lengths uniform in [L/8, L] | 36 | 209 / 100 / 46 |
| flash_attention_2, torch 2.8 | padded, lengths uniform in [L/8, L] | 44 | 201 / 100 / 50 |

The two flash_attention_2 files (`…-113031-gpu-bench-flash_attention_2.json` and
`…-114516-gpu-bench-flash_attention_2-varlen.json`) record torch 2.8 in their `env`.

- At fixed length the two backends are within 2–7% of each other.
- With padded batches, flash-attn fits about 20% more sequences per micro-batch (44 against 36 at
  512), but runs at the same sequences per second: in transformers 5.17, neither backend skips the
  compute on padding. Sequences per second are the same padded or not, for both.
- **Decision (D19): SDPA, torch 2.11.** flash-attn's gain is at most ~8% here, and would cost a
  torch pin to 2.8 and a wheel from outside the index.

## Epoch time estimate (Phase 2 mix)

From the data stats: 1,123,566 training sequences, 154.6M tokens after truncation (mean 138 per
sequence). MMLU aux (60.7M) and Yahoo (67.9M) are 83% of the tokens. At ~25k tokens/s, with
batches grouped by length so that padding stays small:

- **about 100 minutes per epoch**, so 4–5 hours for the 2–3 epochs of Phase 2, plus evaluation.
- With random batches padded to their longest sequence, most of the compute would go to padding,
  since neither backend skips it here (table above). Grouping by length matters as much as grouping
  by K (D10).

## Workarounds

- **A micro-batch that passes a short probe can still run out of memory later.** With padded
  mixed-length batches, a batch that fit for 2 steps ran out of memory in the timed run, with
  1.4 GiB reserved but unallocated (fragmentation). `gpu_bench.py` steps down until the timed run
  fits. Phase 1 should leave a margin below the measured maximum and consider
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, measuring both.
- transformers warns that "Flash Attention 2 only supports torch.float16 and torch.bfloat16" when
  the weights are fp32. Under bf16 autocast it runs anyway; the warning can be ignored.
- Loading `ModernBertModel` from the checkpoint reports `decoder.bias`, `head.dense.weight` and
  `head.norm.weight` as unexpected: those belong to the masked-LM head, which the encoder does not
  use.
