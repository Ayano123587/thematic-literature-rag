# Narrative QA training report

The next study fine-tunes a real narrative answer generator with human-authored
questions and linked story sections. It is being prepared; there are no v5
training results yet. Historical low scores and failed calibration studies remain
in the [README](README.md) and versioned experiment documents.

## Dataset and filtering

Use the official [FairytaleQA source](https://github.com/uci-soe/FairytaleQAData)
at commit `a24ddc17364666b7c13a425b9970c87368b03417`, licensed Apache-2.0.
Its human-coded evidence sections support an evaluation of both answer quality
and evidence relevance. The [dataset paper](https://aclanthology.org/2022.acl-long.34/)
motivates narrative domain adaptation; our custom split and retrieval setting
are not directly comparable with its published benchmark.

Source-family separation is stronger than trusting origin labels literally.
The dataset's Blue group includes The Bunyip, listed in Gutenberg's
[Brown collection](https://www.gutenberg.org/ebooks/3282), and The Brownie of the
Lake, listed in the [Lilac collection](https://www.gutenberg.org/ebooks/3454).
Therefore all Lang groups are validation, together with Andersen. Potter and
Grimm are sealed test. Training uses only official training stories from other
eligible families. Ambiguous or unresolved rights groups are excluded. Exact
text/title and high-overlap source duplicates are checked without test answers.
Recurring plots can still cross families; pretraining exposure is also unknown.

## Training and evaluation

The [prespecified protocol](docs/v5-training-protocol.json) records model pins,
source filters, checkpoint selection, metrics and limitations before scoring.
Root `train.py` delegates to `src.narrative_training`; notebooks remain EDA only.
Private data, logs and checkpoints stay ignored. An isolated repository virtual
environment is required because these research repositories share a `src` name.

Primary evaluation retrieves sections from the full question's story, without
human evidence IDs. Generator training can use annotated evidence, but this is
not secretly supplied at validation/test inference. Report answer EM/token F1
separately from evidence recall/precision and exact source citation validity.
Generated answers are free-form; exact citations establish their source location,
not entailment. Answerable-only data cannot establish unknown-question calibration.

| Policy | Validation answer F1 | Test answer F1 | Status |
| --- | --- | --- | --- |
| Untuned T5-small, BM25 contexts | Pending | Not accessed | Preparation |
| Fine-tuned T5-small, BM25 contexts | Pending | Not accessed | Preparation |
| Learned retrieval ablation | Pending | Not accessed | Not trained |

The first staging attempt failed after writing private files because its rights
path variable was overwritten by parsed metadata. That partial directory is
preserved; the corrected staging is separate. No inference or test QA access
occurred in that failure.

Before model scoring, the account's free GPU quota was exhausted. The study was
prospectively adapted to a free Kaggle CPU run with pinned T5-small, 512 input
tokens and learning rate 1e-4. The initial BART/1024-token plan remains in Git
history. Architecture, context and optimization change together for feasibility;
this is not evidence that T5 outperforms BART. A small real CPU training smoke
must establish practical runtime before full training.
