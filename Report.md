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
The question's story identity is supplied by the dataset. This evaluates narrative
reading and within-story retrieval, not global book identification or cross-book
thematic reasoning. Existing thematic diagnostics retain their separate limits.
Generated answers are free-form; exact citations establish their source location,
not entailment. Answerable-only data cannot establish unknown-question calibration.
Evidence recall after truncation counts the presence of an annotated section,
including a partial prefix; it does not establish retention of every relevant
sentence. ROUGE-L uses the documented word-token LCS implementation without
stemming, so it is not presented as a reproduction of the paper's scorer.

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

The initial embedded script submission returned HTTP 400 before execution. The
private payload was moved to a separate private Kaggle dataset. A Windows CLI
relative-path upload failure was corrected by invoking upload from its folder.
Kaggle automatically extracted the ZIP; the small kernel verifies the extracted
manifest and every source/data file hash. CPU smoke version 1 was submitted and
confirmed running, but subsequent plain CLI reads lost authentication. Refreshing
the existing OAuth session recovered its failure: duplicate official question
IDs were rejected before model loading. No duplicate live job was submitted.

A strict local loader subsequently caught nonunique official question IDs within
seven stories: two extra training IDs and five validation IDs, each attached to
different questions. This guard precedes model loading in the submitted smoke.
Canonical IDs now include the stable CSV row ordinal and retain the original ID;
the validation cohort was prospectively rebuilt before any model outcome. This
correction does not inspect test QA. The failed smoke's exact older payload is
preserved rather than silently changed. Corrected CPU smoke version 2 completed
in 79.77 seconds on four CPU cores. It trained on four real QA, validated on two,
saved weights and reproduced both answers after reload. Loading diagnostics had
no missing, unexpected or mismatched keys. These tiny smoke examples establish
mechanics, not a performance estimate. Full training is being prepared.

The corrected 400-question validation cohort's BM25 evidence recall@4 is 83.96%
and evidence precision@4 is 22.13%, before token truncation. These measure human
section relevance, **not answer accuracy**. Answer F1 remains pending. The source
corpus has 3,314 training QA and 2,372 validation QA; the 1,972 validation questions
outside the hash-selected cohort never enter training. Test QA remains unparsed.

## Reproduction entrypoints

Create a fresh virtual environment for this repository, install the project and
its reader extra, and use pinned runtime versions captured by the cloud runner.
The real training command is:

```powershell
python train.py --data data/interim/v5-fairytale-cpu --output models/v5-run --micro-batch 2 --threads 4
```

The test evaluation is a separate explicit command. It requires frozen selection
and exact trained weight hashes and writes an exclusive marker before opening
test QA; a failure cannot be silently overwritten:

```powershell
python -m evaluation_scripts.evaluate_narrative --data data/interim/v5-fairytale-cpu --run models/v5-run
```

`evaluation_scripts.predict_narrative` provides an experimental saved-checkpoint
CLI whose schema labels free-form generated answers separately from retrieved
source spans. Its checkpoint-loading inference has not yet been validated on a
completed v5 training artifact; it is not advertised as a verified production
replacement for the existing extractive reader.
