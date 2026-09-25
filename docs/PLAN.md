# Plan

Work one phase at a time. A phase is done when **every exit criterion is met and measured by the
harness**. Numeric targets marked *(initial)* are starting points; revise them in `DECISIONS.md`
once Phase 0 produces the baselines.

| Phase | Goal | Exit when |
|---|---|---|
| 0 ✓ | Environment, data and harness, no training | ModernBERT runs in bf16 on this GPU; every dataset is converted with counts per split; the harness scores accuracy, ECE, NLL and Brier and reports the baselines on test and held-out; throughput is measured |
| 1 ✓ | Model and training loop, proved on a tiny set | The model overfits 200 examples; option masking, truncation and permutation are proved by tests |
| 2 ✓ | Full training | Accuracy on every training dataset and on both held-out datasets is clearly above its baselines |
| 3 ✓ | Calibration | Temperature scaling lowers ECE on test and on held-out, and the effect on held-out is reported apart |
| 4 ✓ | Inference (v1) | `predict()` answers in tens of milliseconds, and many questions about one state are measured as a batch |
| 5 | Experiments (after v1) | Each one is a run compared with the v1 report |

The steps of the original plan map onto these phases: setup and data are Phase 0, the model is
Phase 1, and evaluation is built in Phase 0 as the harness instead of after training (D1).

---

## Phase 0 — Environment, data and harness

Nothing is trained in this phase. It builds the measuring instruments and the data, and proves them
with known answers.

### Environment

Known state on 2026-09-23: RTX 5060 Ti 16 GB (Blackwell, compute capability 12.0), uv, Python 3.12
through uv, the PyTorch cu128 wheel index already used by fluentloop (D11).

- [x] Package skeleton with uv: `pyproject.toml`, `src/trueodds/`, `tests/`, ruff, pytest. GPU
      dependencies (torch, flash-attn if it builds) in an optional `gpu` extra so the unit tests
      run without a GPU.
- [x] Load `answerdotai/ModernBERT-base` on the GPU and run a bf16 forward pass on a batch of
      512-token sequences.
- [x] Find out whether flash-attn builds for Blackwell. ModernBERT uses it to skip padding; without
      it, PyTorch SDPA is the fallback. Record which one runs, and the throughput of each that
      works, in `docs/SETUP.md` (D11).
- [x] Measure training throughput (sequences per second, forward and backward, bf16) at 128, 256
      and 512 tokens, with and without gradient checkpointing, and the largest micro-batch that
      fits under 15 GB. From it, estimate the time of one epoch over the Phase 2 mix (D10).
      *Done: ~25k tokens/s with SDPA, ~100 min per epoch (SETUP.md); SDPA kept, D19.*
- [x] Record any workaround in `docs/SETUP.md`.

### Data

- [x] Write `specs/001-example-format.md`: the unified schema
      `{id, source, split, state, question, options[], label_idx, template_id}`, the split rule
      (D6), the question templates and the held-out template per task (D8), and the dedup rule
      (D5).
- [x] One converter per dataset, in `src/trueodds/data/`, each with a unit test on a few hand-made
      rows: BoolQ, MNLI, ARC-Easy and ARC-Challenge, HellaSwag (ActivityNet rows only, D17), MMLU
      auxiliary train, AG News, Yahoo Answers Topics, CommonsenseQA, DBpedia-14.
- [x] 3–5 question paraphrases per task, sampled at random in training, plus one paraphrase per
      task that is never trained on (D8).
- [x] Cap each training dataset (initially ~50k training examples) so the large ones do not
      dominate (D4).
- [x] Remove duplicates across datasets by normalized question and options text. In particular,
      MMLU auxiliary train contains ARC among its sources; nothing in the training data may also be
      in any test split or held-out dataset (D5).
- [x] `scripts/prepare_data.py` downloads, converts and writes everything to `~/.trueodds/data/`,
      and prints counts per source and split, option-count distribution and label distribution.
- [x] Check each dataset's license at its source and list them in `docs/LICENSES.md`.

### Harness

- [x] Write `specs/002-run-report.md`: the results JSON (run, checkpoint, per source and split:
      n, accuracy, ECE, NLL, Brier, and the baselines), and the exact metric definitions (D7).
- [x] `harness/metrics.py`: accuracy, ECE (15 equal-width bins on the top-1 probability), NLL,
      Brier, and the data for a reliability diagram. Proved on fakes: a model whose probabilities
      are sampled correctly has ECE near 0; a model that is always 100% sure and right half the
      time has ECE 0.5.
- [x] Baselines per source: random (mean of 1/K) and majority class (the most frequent label index
      or label text in that source's training split; for held-out, in its own test split, since it
      has no training split — stated as such in the report).
- [x] `harness/evaluate.py` takes anything that returns a probability per option and writes the
      results JSON; `harness/report.py` turns a set of JSONs into a comparison table.
- [x] Baseline run, before any model: the results JSON with only the baselines filled in, on every
      test split and both held-out datasets.

**Exit criteria** — met on 2026-09-23
- A bf16 forward pass of ModernBERT-base runs on the GPU; attention backend, throughput and peak
  VRAM are in `docs/SETUP.md`.
- `~/.trueodds/data/` holds every dataset in the spec 001 format; the counts are recorded; the
  overlap check reports 0 training examples shared with any test split or held-out dataset.
- `uv run pytest` passes, including the metric tests on known answers.
- The baseline results JSON exists in `harness/results/`.
- The Phase 2 targets are written in `DECISIONS.md`, from the baselines.

---

## Phase 1 — Model and training loop, proved on a tiny set

**Tasks**
- [x] The model (D2): ModernBERT-base encodes each `(state, question, option)` sequence, a pooled
      vector goes through a linear layer to one score, and the scores of one question are
      softmaxed across its options. *`src/trueodds/model.py`; CLS and mean pooling both built.*
- [x] Variable option counts: pad the options of a batch to the largest K, and give padded options
      a score of −inf before the softmax. *Padded options are not even encoded (D22).*
- [x] Truncation to 512 tokens: the question and the option are never cut; the state is cut to
      what is left (D9). *`src/trueodds/encode.py`.*
- [x] Training loop: bf16, AdamW, linear warmup then linear decay, gradient accumulation, optional
      gradient checkpointing, evaluation and checkpoint every N steps, the best checkpoint kept by
      dev accuracy. Logs to TensorBoard. One YAML per run in `configs/`. *`src/trueodds/train.py`;
      steps of 32 questions in token-budget micro-batches (D22); templates re-sampled (D23).*
- [x] Unit tests, on a tiny random model: a padded option never gets probability > 0; the
      probabilities of one question sum to 1; shuffling the options shuffles the probabilities the
      same way (the model has no position bias across options, by construction); truncation keeps
      the question and the option whole. *`tests/test_model.py`, `tests/test_encode.py`; torch is a
      base dependency so they run on the CPU (D21).*
- [x] Overfit run: 200 training examples, mixed across sources. *`configs/phase1-overfit.yaml`.*

**Exit criteria** — met on 2026-09-23
- On the 200 examples, training accuracy reaches 100% and loss falls below 0.05 *(initial)*.
  *Accuracy 1.000 and NLL 1.6e-5 after 280 steps (1.000 from step 200), scored with
  `harness.metrics`; `~/.trueodds/runs/phase1-overfit/summary.json`, numbers in SETUP.md.*
- The unit tests above pass. *`uv run pytest`: 66 passed.*
- Peak VRAM of the training loop at the Phase 2 settings is measured and under 15 GB.
  *13.07 GiB reserved (12.29 allocated) on the 50 costliest steps of the full mix; 12.34 GiB with
  `expandable_segments`.*

---

## Phase 2 — Full training

**Tasks**
- [x] Runs tracked in MLflow, the best checkpoint of each run registered as a version of the
      `trueodds` model (D24). *`src/trueodds/tracking.py`; `configs/phase2-base.yaml` tracks.*
- [x] Full run on the Phase 0 mix: learning rate in 2e-5 to 5e-5, 6% warmup, 2–3 epochs, effective
      batch of ~32 questions. Settings in `configs/`. *`configs/phase2-base.yaml`: 3 epochs, 20,874
      steps, 12.42 GiB peak. Best dev accuracy 0.776 at step 12,000 (epoch 2); epoch 3 overfits (dev
      NLL 0.65 → 1.23, ECE 0.06 → 0.16). Dev ECE was lowest at step 6,000 (0.008).*
- [x] Choose the pooling (CLS or mean) with two short runs compared on dev (D2).
      *CLS: dev accuracy 0.711 vs 0.692 after 1,500 steps, ahead at every evaluation (D25).*
- [x] Evaluate the best checkpoint with the harness on every test split and both held-out datasets.
      *`harness/results/2026-09-23-221918-eval.json`; pooled test accuracy 0.771, NLL 0.652.*
- [x] Accuracy on the held-out question templates, next to the trained templates (D8). *Gap ≤ 2.0
      points on BoolQ, HellaSwag, AG News, Yahoo; 12.4 on MNLI and 4.6 on DBpedia-14.*
- [x] Save the run's report next to the baseline in `harness/results/`.

**Exit criteria** *(set from the Phase 0 baselines in D20)*
- On every training dataset's test split and on CommonsenseQA and DBpedia-14 (never trained on):
  accuracy at least 10 points above the higher of the random and majority baselines (the numbers
  are in D20), and NLL below the prior baseline's (D18).
- On the held-out templates: accuracy within 3 points of the trained templates. A larger gap means
  the model reads a phrasing, not the question.

**Status (2026-09-23): not met** by `phase2-base/best` (first run). Accuracy passes on all 10 files. NLL is below
the prior on 9 of 10; ARC-Challenge fails (1.410 vs 1.384, ECE 0.19). The held-out template gap fails
on MNLI (12.4 points) and DBpedia-14 (4.6).
A second run follows D26: ten trained phrasings per task, 2 epochs, best checkpoint by dev NLL
(`configs/phase2-templates.yaml`).

**Met on 2026-09-24** by `phase2-templates/best` (step 6,000, end of epoch 1; dev accuracy 0.759, NLL
0.621, ECE 0.012), `harness/results/2026-09-24-040715-eval.json`:
- Accuracy above target and NLL below the prior on all 10 files. ARC-Challenge's NLL is now 1.237
  (prior 1.384, ECE 0.058). Pooled test: accuracy 0.761, NLL 0.615, ECE 0.008.
- Held-out template gap ≤ 1.4 points on every templated source: MNLI 12.4 → 1.1 (held-out accuracy
  0.736 → 0.848), DBpedia-14 4.6 → 1.4.
- The cost: test accuracy 1–3.5 points lower than `phase2-base` on BoolQ, ARC and MMLU, as the
  checkpoint is from epoch 1. DBpedia-14's ECE rose to 0.144 (from 0.073) while every training
  source's fell: a held-out dataset with its own miscalibration, for Phase 3.

---

## Phase 3 — Calibration

**Tasks**
- [x] Fit one temperature on the dev split (all training sources together) by minimizing NLL.
      *`trueodds/calibrate.py`, `harness/calibrate.py` (D27): T = 1.0576 on 11,286 dev questions;
      dev NLL 0.6206 → 0.6198, dev ECE 0.0121 → 0.0091. Written to `best/temperature.json`.*
- [x] Report ECE, NLL and Brier before and after, per source, on test and on held-out.
      *`harness/results/2026-09-24-110652-calibration.json`, `harness.report` renders it; each ECE
      change comes with a paired bootstrap 95% interval.*
- [x] Reliability diagrams before and after, for the pooled test split and for each held-out
      dataset. *`harness/results/figures/2026-09-24-110652-reliability-{pooled,commonsense_qa,dbpedia}.png`.*
- [x] Report ECE by number of options (2, 3, 4, 5, 10, 14), since one temperature is shared by
      questions with very different K (D7). *In the same report.*

**Exit criteria**
- ECE after temperature scaling is lower than before on the pooled test split.
- On the held-out datasets, the effect is reported whether it helps or not: the question is whether
  a temperature fitted on seen data transfers to unseen data.

**Met on 2026-09-24, with a caveat on the first criterion.** `phase2-templates/best`, T = 1.0576:
- Pooled test ECE 0.0077 → 0.0069 (NLL 0.615 → 0.614, accuracy unchanged at 0.761). The criterion
  is met as written, but the change is **within noise**: its 95% interval is −0.0048 to +0.0045.
  The model was already calibrated in-domain (D26's selection by dev NLL did most of this phase's
  work), so a dev-fitted T close to 1 has almost nothing left to fix. Per source the change is
  mixed: BoolQ, ARC-Challenge and HellaSwag improve, MNLI, ARC-Easy and AG News get slightly worse,
  as one T cannot move 2-option and 10-option questions the same way (by K: K = 2 0.037 → 0.028,
  K = 3 0.007 → 0.013, K = 4 0.012 → 0.010, K = 10 0.016 → 0.019).
- **Held-out: the temperature does not transfer, and on DBpedia-14 it makes things worse.**
  DBpedia-14's ECE 0.144 → 0.166 (interval +0.020 to +0.023; held-out template 0.169 → 0.194). The
  model is *under*confident there (confidence 0.50 → accuracy 0.70), and a T > 1 softens it further.
  CommonsenseQA: 0.036 → 0.038, within noise. The miscalibration on unseen data runs the opposite
  way from the small correction seen data asks for, so one global temperature fitted in-domain
  cannot fix it. This answers D27's open question: one temperature does not fit all.

## Phase 4 — Inference (v1)

**Tasks**
- [x] Write `specs/003-predict.md`: `predict(state, question, options) -> {option: probability}`
      and the batch form, many questions about one state. *`trueodds.load()` returns a `TrueOdds`
      with `predict` and `predict_batch` (D28); also the latency JSON.*
- [x] `predict()` loads the best checkpoint and its temperature. *`src/trueodds/infer.py`, through
      `ModelPredictor`, the harness's scoring path; default `~/.trueodds/runs/phase2-templates/best`;
      `tests/test_infer.py`.*
- [x] `harness/latency`: p50 and p95 per request at K = 2, 4 and 14 options and states of 64, 256
      and 480 tokens, after warmup; and the time for 10 and 50 questions about one state in one
      batch, against the same questions one by one. *`harness/latency.py`,
      `harness/results/2026-09-24-122853-latency.json`.*
- [ ] Optional: a small FastAPI endpoint around `predict()`. *Skipped for v1 (D28).*

**Exit criteria** *(initial)*
- One request with 4 options and a 256-token state: p50 ≤ 30 ms, p95 ≤ 60 ms on this GPU.
- The batch timings are reported.
- The README shows the results table and the reliability diagrams.

**Met on 2026-09-24** (`harness/results/2026-09-24-122853-latency.json`, 200 requests per cell after
20 warmup calls, end to end):
- K = 4, 256-token state: **p50 17.7 ms, p95 18.8 ms**, on the model as the harness scored it
  (fp32 weights, bf16 autocast, SDPA); nothing was optimized. The range: 13.7 ms (K = 2, 64 tokens)
  to 98.2 ms (K = 14, 480 tokens).
- Batch: 10 and 50 questions about a 256-token state (K = 2 to 14) take 266 ms and 1,314 ms with
  `predict_batch`, 254 ms and 1,270 ms one by one: **no speedup** (0.95–0.97×). Past about 1,000
  tokens per call the forward pass is compute-bound, and the state is encoded once per option,
  which a batch does not remove (D13, Phase 5).
- README: the results table, the three reliability diagrams and the latency table.

---

## Phase 5 — Experiments (after v1)

Each experiment is one run, compared with the v1 report by the same harness. The rules of the
earlier phases hold: checkpoint and temperature chosen on dev only (D6), peak VRAM measured under
15 GB, results JSONs committed, a dated decision for each outcome. An experiment is done when its
result is measured and reported, whether it beats v1 or not. The first two are planned in tasks
below; the score head and the distilled labels get their tasks when they start, since each needs
a dataset chosen first.

### 5A — ModernBERT-large (D3)

Large instead of base, at v1's settings (`configs/phase2-templates.yaml`): same optimizer step of
32 questions, only the micro-batch budget changed to fit memory, which D22 makes
gradient-equivalent.

**Tasks**
- [x] `scripts/gpu_bench.py --backbone`: the largest -large micro-batch under 15 GiB at 128, 256
      and 512 tokens, with and without gradient checkpointing; in `docs/SETUP.md`. *~6,144 tokens
      without checkpointing, ~11k tokens/s.*
- [x] `configs/phase5-large.yaml`; the longest-first VRAM probe (50 steps) under 15 GiB. *5,120-token
      micro-batches; 13.14 GiB with `expandable_segments` (14.71 without).*
- [x] Full run, then `harness.evaluate`, `harness.calibrate`, `harness.plots` and
      `harness.latency` on its best checkpoint. *13,916 steps, 13.21 GiB peak; best by dev NLL at
      step 6,000 (end of epoch 1). `harness/results/2026-09-25-*`.*
- [x] `harness.report` shows a run next to v1, with a paired bootstrap 95% interval on the pooled
      test accuracy difference. *`harness/compare.py` (B − A in accuracy, NLL and ECE per file and
      pooled, each at its checkpoint's T, from the scores `harness.calibrate` caches);
      `tests/test_compare.py`.*

**Exit criteria**
- Every test and held-out file, the held-out template gap, ECE before and after its temperature,
  and latency, reported next to v1 (`phase2-templates/best`).
- The pooled test accuracy difference with its interval, so "large is better" is a measured claim.

**Met on 2026-09-25** (D29, `harness/results/2026-09-25-015315-compare.json`):
- Pooled test accuracy 0.761 → **0.815** (+5.4 points, 95% interval +5.0 to +5.9), NLL 0.614 →
  0.508; ARC, HellaSwag and MMLU gain 9.5 to 14.3 points, AG News nothing.
- Pooled ECE the same within noise (0.007 → 0.011 at each model's T); held-out template gap ≤ 1.0.
- CommonsenseQA +10.9 points. **DBpedia-14 worse:** −1.7 points, ECE 0.166 → 0.203, more
  underconfident than base.
- Latency about twice base's: K = 4 on 256 tokens, p50 37.4 ms (base 17.7).
- v1 stays the default of `trueodds.load()`.

### 5B — Shared state encoding (D13)

Encode the state once and score every question and option against it, closer to how Jev answers
many questions about one state in one pass. One packed sequence holds the state, then each
question with its options; a mask keeps questions and options from seeing each other, so a
question scores the same alone or packed with others (`specs/004-shared-state.md`).

**Tasks**
- [x] `specs/004-shared-state.md`: sequence layout, attention masks, positions, pooling,
      truncation, checkpoint format.
- [x] The packed encoding, masks and model, with known answers proved by unit tests: a question
      scores the same alone and packed, permuting options permutes probabilities, padded options
      get 0, no option sees another. *`src/trueodds/shared.py`, `tests/test_shared.py` (12 tests);
      with every token in the state the masks give ModernBERT's own hidden states.*
- [x] `configs/phase5-shared.yaml` at v1 settings; the VRAM probe; full run; evaluate, calibrate,
      plots and latency as for 5A. *11.89 GiB peak; 2 epochs in 57 minutes (v1: 246); best by dev
      NLL at step 6,000. `harness/results/2026-09-25-04*`.*
- [x] `predict_batch` packs all questions about one state into one sequence. *`pack=True` through
      `predict.predict_scores`, the harness's path; tested equal to `predict` one by one.*

**Exit criteria**
- Accuracy and calibration reported next to v1, with the pooled accuracy difference and its
  interval: this is what giving the state no view of the question costs.
- The Phase 4 batch timings measured again. Target: 50 questions about a 256-token state faster
  than v1 answering them one by one (1.27 s).

**Met on 2026-09-25** (D30, `harness/results/2026-09-25-041921-compare.json`):
- Pooled test accuracy 0.761 → 0.747 (**−1.3 points**, 95% interval −1.8 to −0.9), NLL 0.614 →
  0.643: the cost of a state that never sees the question. ARC-Easy −4.0 and HellaSwag −3.4; BoolQ,
  AG News, Yahoo and ARC-Challenge unchanged within their intervals.
- Pooled ECE the same within noise (0.007 → 0.010 at each model's T); held-out template gap ≤ 1.3.
- Held-out: CommonsenseQA −4.7, DBpedia-14 −3.9.
- **50 questions about a 256-token state: 44 ms** with `predict_batch` (target: under v1's
  1,270 ms), 29× faster than v1 and 16× faster than its own loop. Single requests stay at the
  ~14 ms floor up to 14 options on 480 tokens.
- v1 stays the default; the shared model is the one to load for many questions about one state.

### 5C — Latency on the CPU

Every latency so far is on the GPU (Phase 4, D28). This measures `predict()` on the CPU (Ryzen 7
5700X, 8 cores, AVX2), with the same inputs, after the 5A run (a CPU measurement during training
would slow the run and be noisy).

**Tasks**
- [x] `harness.latency --device cpu`: the same JSON with `"device": "cpu"` and the thread count.
      *Also `--threads`; the CPU's name in `env`.*
- [x] v1 on the CPU across the Phase 4 grid: K = 2, 4 and 14 options × states of 64, 256 and 480
      tokens, p50 and p95 per request; and 10 and 50 questions about one state, batch against one
      by one. *`harness/results/2026-09-25-015515-latency.json` (v1) and `-021711-latency.json`
      (-large); 8 threads, fp32, 30 requests per cell after 3 warmup calls.*
- [x] The same grid for the 5A and 5B checkpoints once trained (the shared state should gain most
      on the CPU). *`2026-09-25-021711-latency.json` (-large), `-041935-latency.json` (shared).*

**Exit criteria**
- The CPU grid reported next to the GPU one for v1, with the CPU/GPU ratio per cell. Speed-ups
  (thread count, lower-precision weights, ONNX Runtime) are separate measurements, each scored by
  the harness if it changes the numbers.

**Measured on 2026-09-25 for v1 and -large** (p50, CPU against GPU):

| K | state tokens | v1 CPU | v1 GPU | ratio | -large CPU | -large GPU | ratio |
|---|---|---|---|---|---|---|---|
| 2 | 64 | 71 ms | 13.7 ms | 5× | 254 ms | 17.1 ms | 15× |
| 2 | 256 | 208 ms | 13.8 ms | 15× | 843 ms | 24.8 ms | 34× |
| 2 | 480 | 492 ms | 16.5 ms | 30× | 1.65 s | 34.1 ms | 48× |
| 4 | 64 | 118 ms | 14.1 ms | 8× | 438 ms | 18.6 ms | 24× |
| 4 | 256 | **517 ms** | **17.7 ms** | 29× | 1.77 s | 37.4 ms | 47× |
| 4 | 480 | 1.01 s | 26.9 ms | 38× | 3.48 s | 62.0 ms | 56× |
| 14 | 64 | 420 ms | 16.9 ms | 25× | 1.74 s | 36.4 ms | 48× |
| 14 | 256 | 2.09 s | 45.8 ms | 46× | 6.35 s | 113 ms | 56× |
| 14 | 480 | 4.00 s | 98.2 ms | 41× | 12.0 s | 232 ms | 52× |

- On the CPU the time is compute from the smallest request on: v1 reads 1,700 to 2,100 tokens/s
  (-large 580 to 610), so the time follows K × state length, and a 4-option question on a 256-token
  state takes half a second. The GPU's ~14 ms overhead floor hides this for small requests, so the
  ratio grows from 5× to 30–46× as the request grows. -large costs 3.0 to 4.1× v1 on the CPU,
  more than the ~2× it costs on the GPU.
- **Batching is slower than a loop on the CPU:** 50 questions about a 256-token state take 66.3 s
  with `predict_batch` and 53.7 s one by one (0.81×; -large 0.92×). The likely cause, not yet
  measured: the batch's micro-batches (the eval budget, 32,768 tokens) pad sequences of different
  lengths together, and the CPU pays for padding in full; one by one, no sequence is padded.

**Shared state (5B) on the CPU** (p50; ratio to v1 on the CPU):

| K | state 64 tokens | 256 tokens | 480 tokens |
|---|---|---|---|
| 2 | 52 ms (1.4×) | 117 ms (1.8×) | 221 ms (2.2×) |
| 4 | 54 ms (2.2×) | **120 ms (4.3×)** | 217 ms (4.6×) |
| 14 | 66 ms (6.4×) | 135 ms (15×) | 259 ms (15×) |

The state is read once, so K barely matters: 14 options cost 15 to 27% more than 2. Many questions
about one state: 50 in 1.54 s with `predict_batch`, 6.3 s one by one, and 53.7 s for v1 one by one
(35×). Here packing helps even on the CPU (4.1×), because one packed sequence per state leaves no
padding between questions.

**Met on 2026-09-25:** all three models across the grid, with the CPU/GPU ratios above. The
speed-ups (thread count, lower-precision weights, ONNX Runtime) remain untried.

### Later

- **Score head:** a numeric answer (regression) as a third kind of output, with its own metric and
  a dataset to train it.
- **Distilled labels for an own domain:** a local LLM labels questions about states of a domain
  (for example, robot sensor readings), and the model is trained and measured on them.
