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

## Training (Phase 1)

```bash
uv run pytest                                                   # model tests run on the CPU (D21)
uv run --extra gpu python -m trueodds.train configs/phase1-overfit.yaml
uv run --extra gpu python -m trueodds.train configs/phase2-base.yaml \
    --max-steps 50 --longest-first --no-eval --run phase1-vram-probe  # VRAM probe
uv run --extra gpu tensorboard --logdir ~/.trueodds/runs
```

Each run writes `~/.trueodds/runs/<run>/`: `config.yaml`, `summary.json`, `tb/`, and the `best/` (by
dev accuracy) and `last/` checkpoints, about 0.6 GB each.

**Overfit, 2026-09-23** (`configs/phase1-overfit.yaml`): 200 training questions, 25 per training
source, templates not re-sampled, lr 3e-5, 40 epochs = 280 steps of 32 questions. Scored on the same
200 with `harness.metrics`:

| Step | Train accuracy | Train NLL | Dev accuracy (400) |
|---|---|---|---|
| 50 | 0.815 | 0.635 | 0.338 |
| 100 | 0.975 | 0.081 | 0.345 |
| 150 | 0.995 | 0.022 | 0.345 |
| 200 | 1.000 | 4.3e-5 | 0.343 |
| 280 | 1.000 | 1.6e-5 | 0.345 |

Dev stays near chance, as expected from 200 examples; the run only proves the loop can fit. Peak
13.75 GiB reserved, ~24.8k padded tokens/s.

**VRAM at the Phase 2 settings** (`configs/phase2-base.yaml`: the full mix, 32 questions per step,
16,384-token micro-batches, no gradient checkpointing). `--longest-first` orders the steps by their
costliest micro-batch, so the 50 steps run are the heaviest of the epoch:

| Allocator | Peak reserved | Peak allocated | Padded tokens/s |
|---|---|---|---|
| default | 13.07 GiB | 12.29 GiB | 23.5k |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 12.34 GiB | 12.25 GiB | 23.5k |

Both are under the 15 GiB budget. `expandable_segments` removes most of the gap between reserved and
allocated (fragmentation) at no cost in throughput, so it is worth setting for Phase 2.

**Scoring a checkpoint with the harness** (`harness.evaluate --checkpoint`, every test and held-out
file, 33,282 questions and 202,781 sequences plus the template and mismatched files): 565 s on this GPU
for the overfit checkpoint. Checked end to end; its numbers are near chance, as expected, and were
not kept.

## MLflow (D24)

```bash
uv sync --extra gpu --extra tracking      # mlflow 3.16.1 resolved on 2026-09-23
uv run --extra gpu --extra tracking python -m trueodds.train configs/<run>.yaml --mlflow
uv run --extra tracking python -m trueodds.tracking register ~/.trueodds/runs/<run>/best
uv run --extra tracking python -m trueodds.tracking promote <version>          # alias champion
uv run --extra tracking mlflow ui --backend-store-uri sqlite:///$HOME/.trueodds/mlflow.db
```

The UI is then at http://127.0.0.1:5000: runs under the `trueodds` experiment, versions under
Models → `trueodds`, where an alias such as `champion` is set on a version. A version loads with
`mlflow.pyfunc.load_model("models:/trueodds@champion")` (or `models:/trueodds/<n>`) after
`mlflow.set_tracking_uri("sqlite:///" + str(Path.home() / ".trueodds/mlflow.db"))`, and its
`predict` takes a DataFrame of `state`, `question`, `options`.

Checked on 2026-09-23: `phase1-overfit/best` registered as version 1, loaded back through the
registry and asked a BoolQ dev question; a 20-step tracked run (`phase2-mlflow-smoke`) logged its
curves and registered version 2 by itself. A version is 573 MB of artifacts, the weights once.

- `mlflow.pyfunc.log_model` calls `load_context` on the wrapper to infer the signature from the
  input example, then pickles the same object. Without `TrueOddsModel.__getstate__` dropping the
  loaded predictor, the pickle held a second copy of the weights (1.2 GB per version).
  `tests/test_tracking.py` checks the pickle stays small.
- MLflow warns that `predict` has no type hints; the signature is inferred from the input example
  instead (`state` and `question` strings, `options` an array of strings).
- The unit tests of the tracking extra run with `uv run --extra tracking pytest`; with a plain
  `uv sync`, `tests/test_tracking.py` skips its MLflow test.

## Inference latency (Phase 4)

```bash
uv run --extra gpu python -m harness.latency         # writes harness/results/<stamp>-latency.json
uv run python -m harness.report harness/results/<stamp>-latency.json
```

`predict()` end to end (tokenization, forward pass, copy to the CPU, softmax), `phase2-templates/best`
at T = 1.0576, fp32 weights under bf16 autocast, SDPA, torch 2.11. 20 warmup calls, then 200 timed
calls per cell; the state is BoolQ test text cut to the token count
(`2026-09-24-122853-latency.json`):

| K | state tokens | longest sequence | p50 ms | p95 ms |
|---|---|---|---|---|
| 2 | 64 | 75 | 13.7 | 14.6 |
| 2 | 256 | 267 | 13.8 | 14.6 |
| 2 | 480 | 491 | 16.5 | 17.4 |
| 4 | 64 | 77 | 14.1 | 15.0 |
| 4 | 256 | 269 | **17.7** | **18.8** |
| 4 | 480 | 493 | 26.9 | 28.1 |
| 14 | 64 | 78 | 16.9 | 17.8 |
| 14 | 256 | 270 | 45.8 | 47.3 |
| 14 | 480 | 494 | 98.2 | 99.9 |

Many questions about the 256-token state, mixed K (2, 3, 4, 10, 14), p50 of 20 repeats:

| questions | sequences | `predict_batch` | one by one | speedup | max prob. difference |
|---|---|---|---|---|---|
| 10 | 66 | 266 ms | 254 ms | 0.95× | 9.5e-3 |
| 50 | 330 | 1,314 ms | 1,270 ms | 0.97× | 1.3e-2 |

- **Two regimes, crossing at about 1,000 tokens per call.** Below it the call is bound by
  per-call overhead, about 13–14 ms whatever the size (K = 2 on 64 tokens is 150 tokens and takes
  13.7 ms). Above it the call is bound by compute. Timed on the forward pass alone, with fixed
  shapes: 4 × 270 tokens 14.0 ms (77k tokens/s), 16 × 270 48.5 ms (89k), 64 × 270 244 ms (71k),
  128 × 270 490 ms (71k). The 4-option, 256-token request (1,080 tokens) sits at the crossover;
  tokenization is 0.5 ms of it.
- **Why a batch does not help:** most of the 50 questions (K ≥ 3 on 270-token sequences) are
  already past the crossover, where the per-call overhead runs while the GPU computes. A batch
  removes overhead that was mostly hidden already.
- **Large micro-batches are slower per token.** The 50-question batch's forward time, by
  micro-batch budget: 4,096 padded tokens 1,058 ms (27 micro-batches), 8,192 1,193 ms, 16,384
  1,292 ms, 32,768 (the eval budget `predict_batch` uses) 1,306 ms, 65,536 1,318 ms. A 4,096-token
  budget would make `predict_batch` about 20% faster than the loop; not changed, as the target is
  met and the harness uses the same budget (D28).
- The batch/loop difference is bf16 rounding under different batch shapes: the same comparison in
  fp32 on the CPU gives 9e-7.
- Peak VRAM reserved during the whole run: 2.5 GiB (`peak_vram_gib`).

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
- **`uv sync --extra <one>` uninstalls every other extra.** `uv sync` makes the venv match exactly
  what it was asked for, so `uv sync --extra plots` removes mlflow and tensorboard, even under a
  running job that imports them later. Sync all the extras you use at once
  (`uv sync --extra gpu --extra tracking --extra plots`), or rely on `uv run --extra <x>`, which
  adds what is missing and removes nothing.
