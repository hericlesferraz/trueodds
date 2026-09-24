# 001 — Example format

**Phase 0.** The contract between the converters (`src/trueodds/data/`), the training loop and the
harness. Everything downstream reads examples in this form and nothing else.

## The example

One JSON object per line.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `<source>/<split>/<original id or row index>`; unique across all files |
| `source` | str | one of the source names below |
| `split` | str | `train`, `dev`, `test` (plus `test_mismatched` for MNLI) |
| `state` | str | the text the question is about; may be `""` (ARC, CommonsenseQA) |
| `question` | str | the rendered question, exactly what the model reads |
| `options` | list[str] | K ≥ 2 distinct, non-empty strings, in the dataset's order |
| `label_idx` | int | index of the correct option, `0 ≤ label_idx < K` |
| `template_id` | str | `<task>:<n>` for a trained template, `<task>:heldout`, or `native` |
| `vars` | dict[str, str] | the template's variables (always `{}` for `native`, and for templates without variables); lets the trainer re-render the question with another template each epoch |

Example (MNLI):

```json
{"id": "mnli/test/12", "source": "mnli", "split": "test",
 "state": "The new rights are nice enough.",
 "question": "Does it follow that everyone really likes the newest benefits?",
 "options": ["yes", "maybe", "no"], "label_idx": 1,
 "template_id": "mnli:0", "vars": {"hypothesis": "Everyone really likes the newest benefits."}}
```

## Files

```
~/.trueodds/data/
  <source>/train.jsonl  dev.jsonl  test.jsonl      # training sources
  <source>/test.jsonl                             # held-out sources (test only)
  <source>/test-heldout-template.jsonl            # the test rows again, with the held-out template
  mnli/test_mismatched.jsonl
  stats.json                                      # counts written by scripts/prepare_data.py
```

## Sources and splits

`dev` is for every choice; `test` is only read (D6). `dev` is carved from the official training split
with seed 0: 10% or 2,000 examples, whichever is smaller. Where a dataset's official test split has
public labels and a validation split exists too, test is the official test and dev is the official
validation (D14).

| Source | Hugging Face | Role | Task | train | dev | test |
|---|---|---|---|---|---|---|
| `boolq` | `google/boolq` | train | boolq | train | carved | validation |
| `mnli` | `nyu-mll/glue` `mnli` | train | mnli | train | carved | validation_matched (+ `test_mismatched` ← validation_mismatched) |
| `arc_easy` | `allenai/ai2_arc` `ARC-Easy` | train | native | train | validation | test |
| `arc_challenge` | `allenai/ai2_arc` `ARC-Challenge` | train | native | train | validation | test |
| `hellaswag` | `Rowan/hellaswag` | train | hellaswag | train | carved | validation |
| `mmlu_aux` | `cais/mmlu` `auxiliary_train` | train | native | train | carved | carved (D14) |
| `ag_news` | `fancyzhx/ag_news` | train | topic | train | carved | test |
| `yahoo` | `community-datasets/yahoo_answers_topics` | train | topic | train | carved | test |
| `commonsense_qa` | `tau/commonsense_qa` | held-out | native | — | — | validation |
| `dbpedia` | `fancyzhx/dbpedia_14` | held-out | topic | — | — | test |

- `mmlu_aux` dev and test (2,000 each) are carved **by passage**: all questions sharing a `state`
  go to the same split, so no passage is read in both training and evaluation.
- **Caps (D4, D15):** train is sampled down to 50,000 per source; test and held-out to 5,000 per
  source, stratified by `label_idx`. All sampling uses seed 0, so every run sees the same files.

## Conversion per task

| Task | state | question | options |
|---|---|---|---|
| boolq | passage | template over the dataset question (capitalized, `?` added) | `yes`, `no` |
| mnli | premise | template over the hypothesis | `yes`, `maybe`, `no` (entailment, neutral, contradiction) |
| hellaswag | `ctx`; ActivityNet rows only, wikiHow rows skipped (D17) | template | the four endings |
| topic | AG News: text · Yahoo: title, content and best answer · DBpedia: title and content, joined by newlines | template | the label names, written as words |
| native (ARC, CommonsenseQA) | `""` | the dataset question | the answer texts |
| native (MMLU aux) | the passage, if the text has one | the last sentence of the text | the answer texts |

Options with the same text (after collapsing whitespace) are merged into one, keeping the first; the
label follows its text. CommonsenseQA repeats a distractor in about 2% of its questions, which then
have K = 4. A row left with fewer than 2 options, or without a label, is skipped and counted.

MMLU aux joins a RACE passage and its question with a single space. The converter splits the text
at the last sentence boundary (`.`, `?` or `!` followed by a space) when the text is longer than 300
characters; shorter texts are a question alone and keep `state = ""`.

Topic label names: AG News `World`, `Sports`, `Business`, `Science and technology`; Yahoo as in the
dataset (`Society & Culture`, …); DBpedia `Company`, `Educational institution`, `Artist`, `Athlete`,
`Office holder`, `Means of transportation`, `Building`, `Natural place`, `Village`, `Animal`,
`Plant`, `Album`, `Film`, `Written work`.

## Templates (D8)

Defined in `src/trueodds/data/templates.py`. Each templated task has 10 trained templates (`:0` to
`:9`) and one held-out template (`:heldout`), which is never used in `train` or `dev`. `:4` to `:9`
were added in Phase 2 (D26) and are used only by the trainer's redraw: `dev` and `test` are rendered
with `:0` to `:3` only, so their files are the same as before.

- `train`: the stored question uses a template drawn at random (seed 0); the trainer may redraw
  among the trained templates each epoch from `vars`.
- `dev`, `test`: one of `:0` to `:3` chosen by a hash of `id`, so it is the same in every run.
- `test-heldout-template.jsonl`: the same rows as `test.jsonl`, rendered with `:heldout`. The
  harness reports its accuracy next to `test`'s.

## Deduplication (D5)

The key of an example is its normalized state, its normalized template variables (the question for
`native`), and its normalized options, sorted:

```
normalize(s) = " ".join(s.lower().split())
key = normalize(state) | normalize(vars or question) | sorted(normalize(o) for o in options)
```

The state and the variables are part of the key because for templated tasks the question and the
options alone (e.g. "What is this text about?" with the four AG News labels) are shared by every
example (D16).

1. Any `train` example whose key equals the key of any `dev`, `test` or held-out example, of any
   source, is dropped.
2. Duplicate keys within `train` are reduced to one (the first in source order).
3. The counts dropped by each rule are reported per source. After the rules run, `prepare_data`
   recomputes the overlap between train and every eval file, and fails unless it is 0.

## Length (D9)

A sequence is `[CLS] state [SEP] question [SEP] option [SEP]`: 4 special tokens. Counted with the
ModernBERT tokenizer, an example whose `len(question) + len(longest option) + 4` leaves no token for
the state within 512 is dropped and counted. An example whose state does not fit whole is kept (the
trainer cuts the state at its end) and counted as truncated; the share is reported per source and
split.

## Acceptance

- Every converter has a unit test on hand-made rows that checks each field.
- `load_examples()` validates each row: fields present, K ≥ 2, `label_idx` in range, options distinct
  and non-empty, `split` and `source` known.
- `stats.json` reports, per source and split: n, the K distribution, the label distribution, the
  dedup and length drops, and the truncated-state share; the train–eval overlap is 0.
