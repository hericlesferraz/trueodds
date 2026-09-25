# Licenses

trueodds downloads the datasets and the model below to `~/.trueodds/` and `~/.cache/huggingface/`,
and never redistributes them (D12). What it commits is its own code and the results JSONs, which
hold counts and metrics, not dataset text. Every line was checked on 2026-09-23 against the Hugging
Face card or the dataset's own source, not from memory.

## Datasets

| Source | Hugging Face | License | Notes |
|---|---|---|---|
| BoolQ | `google/boolq` | CC BY-SA 3.0 | |
| MNLI | `nyu-mll/glue` (`mnli`) | card says "other"; GLUE refers to each dataset's own license | MultiNLI is built on the Open American National Corpus; most of it is under the OANC's permissive terms, and some fiction under CC BY-SA 3.0 (MultiNLI paper and site) |
| ARC (Easy, Challenge) | `allenai/ai2_arc` | CC BY-SA 4.0 | |
| HellaSwag | `Rowan/hellaswag` | none stated on the card | The source repository (`rowanz/hellaswag`) was blocked on GitHub on 2026-09-14 by a wikiHow DMCA notice ([notice](https://github.com/github/dmca/blob/master/2026/09/2026-09-14-wikihow.md)). trueodds keeps only the ActivityNet rows and drops every wikiHow row at conversion (D17) |
| MMLU auxiliary train | `cais/mmlu` (`auxiliary_train`) | MIT (card) | Built from ARC, OBQA, MCTest and RACE, each with its own terms |
| AG News | `fancyzhx/ag_news` | unknown on the card | The card says it is provided by the academic community for research purposes |
| Yahoo Answers Topics | `community-datasets/yahoo_answers_topics` | unknown on the card ("More Information Needed") | Derived from Yahoo! Answers; treat as research use only |
| CommonsenseQA | `tau/commonsense_qa` | MIT | held-out |
| DBpedia-14 | `fancyzhx/dbpedia_14` | CC BY-SA 3.0 | held-out |
| RTE | `nyu-mll/glue` (`rte`) | card says "other"; GLUE refers to each dataset's own license | From the PASCAL RTE challenges; no license found stated for them. Held-out (Phase 5D), checked 2026-09-25 |
| WiC | `aps/super_glue` (`wic`) | CC BY-NC 4.0 ([WiC site](https://pilehvar.github.io/wic/)) | The SuperGLUE card defers to each dataset's license. Held-out (Phase 5D), checked 2026-09-25 |
| Rotten Tomatoes (sentence polarity) | `cornell-movie-review-data/rotten_tomatoes` | none stated on the card or on [the source page](https://www.cs.cornell.edu/people/pabo/movie-review-data/) | Held-out (Phase 5D), checked 2026-09-25; research use only |

Several of these are share-alike or have no clear license. That is why the data stays out of the
repository, and why trained weights are not published with this project.

## Model

| Model | License |
|---|---|
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | Apache-2.0 |
| [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | Apache-2.0 (Phase 5A) |

## This repository

The code is under the Apache License 2.0 (`LICENSE`, D33). It covers the code and the committed
results, not the datasets or any trained weights.
