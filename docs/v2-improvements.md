# Controlled reader improvement round

The v2 hypothesis is that short answer-focused retrieval windows and a reader
that covers the whole retrieved context can recover evidence lost by coarse
retrieval or token truncation. Citation assembly happens outside the model:
accepted answers must match exact source character offsets. This removes the
Qwen experiment's fragile generated-quote JSON requirement without claiming
that an exact quote is necessarily a correct answer.

The prespecified protocol identifies *A Sentimental Journey Through France and
Italy* as a fresh official NarrativeQA test book before its questions are read.
The [Gutenberg source](https://www.gutenberg.org/ebooks/804) identifies Laurence
Sterne (1713–1768) and marks this English text public domain in the USA. Users
must check their local rights. No novel text, QA answers, predictions or model
weights are redistributed. NarrativeQA retains its previously pinned Apache-2.0
metadata revision. Actual source hashes and exact model revisions are recorded
in the machine-readable selection evidence.

Three configurations compare 180-word passages/top five contexts, 96-word
passages/top twelve contexts, and the fine passages with 700-character source
expansion on each side. The last two use overlapping reader windows rather than
silently truncating long contexts. Four null thresholds and three retrieval-rank
penalties were initially selected using the 29 Dorian validation questions and four authored
unknown controls. The objective is conventional answer token F1 minus one quarter
of the control false-answer rate. These four controls do not establish broad
no-answer calibration. Reusing Dorian validation limits independence.

The fine-window configuration initially scored worse than the coarse baseline.
A validation-only follow-up tested stronger retrieval-rank penalties of 1.5,
3 and 6 to reduce high-confidence distractors from weakly ranked contexts.
Fine-window F1 rose from 7.16% to 8.11% at 1.5 and then plateaued; the same change
hurt the coarse configuration. This follow-up was recorded before fresh-test
selection in [validation follow-up](v2-validation-followup.json). The final grid
therefore contains six rank penalties, with 72 configuration/threshold/rank
combinations. Small validation gains can reflect overfitting.

Final validation selection was serialized at 2026-10-15 21:08:02 UTC, before
Journey test inference. Best policies per configuration were:

| Configuration | Answer F1 | EM | Abstention | Threshold / rank penalty |
| --- | ---: | ---: | ---: | --- |
| Coarse 180-word / top five | 7.62% | 6.90% | 41.38% | 2 / 0.8 |
| Fine 96-word / top twelve | 8.11% | 6.90% | 17.24% | 0 / 1.5 |
| Fine plus expanded sliding context | 8.78% | 6.90% | 6.90% | 0 / 0.8 |

All three best policies answered zero of the four authored unknown controls.
The expanded configuration wins the stated objective, but its lower abstention
also exposes more incorrect answers; the small F1 increase does not establish
safe no-answer behavior. Exact match does not improve. All 72 candidates remain
in [selection evidence](v2-validation-selection.json), including failures.

## Locked fresh test

The selected policy was run once on all 27 official NarrativeQA Journey test
questions: **8.33% answer token F1, 3.70% exact match (1/27), 18.52% abstention
(5/27)**. All 22 emitted answers exactly matched their cited source offsets.
That is a source-integrity result, not proof of entailment or correct literary
interpretation. There was no independent unknown-question cohort on this test.
See [locked aggregate evidence](v2-reader-locked-test.json).

The bounded CPU round finished at 21:12:36 UTC, about 35.8 minutes after its
original declaration, including the failed checkpoint attempt and continuation.
The validation changes produced a modest gain and unchanged exact match; fresh
accuracy remains weak. The pretrained SQuAD2 reader and exact-span answer format
do not resolve paraphrased, multi-hop or interpretive gold answers. Further work
requires a controlled reader hypothesis, not tuning from this now-observed test.

The selection is serialized before fresh test questions are read. The evaluator
refuses to overwrite its locked test report. Prior Persuasion and Call of the Wild
results, including Qwen's zero score, remain unchanged. Different test books do
not support a direct numerical improvement claim over those historical tests.

The first attempt failed after its first validation configuration when writing
to an absent ignored `data/interim` directory. It produced no saved validation
score and never accessed fresh test questions. The continuation fixes directory
creation, retains private per-query inference checkpoints, and reuses verified
persisted indexes without changing the retrieval or reader policies. The parent
authorized a 45-minute total wall limit because shared memory paging slowed CPU
inference; [continuation metadata](v2-continuation.json) records this explicitly.

## Usage

Use this repository's separate virtual environment, with `.[models,reader]`
installed. After the locally retained selection and fitted index exist:

```bash
python -m pip install -e '.[models,reader]'
python -m src.cli fetch-qa --acknowledge-local-rights
python -m src.cli fetch-reader-v2 --acknowledge-local-rights
python -m src.cli index --backend sentence-transformer --words 96 --overlap 48 --books dorian journey --output models/v2-96.lrag
python -m src.cli answer-v2 "Your question" journey
```

`--selection` and `--index` accept explicit artifact paths. The selected policy
requires an index built with its stated chunk configuration. Production inference
checks all source fingerprints before learned inference, and then checks exact
answer coordinates before returning a citation. The API abstains on empty
questions and absent books. It does not verify semantic entailment.

Post-experiment guards reject nonfinite reader logits, margins and policy values,
negative context expansion, oversized question tokenization, inconsistent span
book identifiers, and mismatched index chunk contracts. Chunk metadata is stored
in new archives and verified by reconstructing exact passages from fingerprinted
sources. Historical archives remain readable for their original uses, but an
archive without a verified chunk contract cannot serve `answer-v2`. Only local v2
archives were migrated after evaluation: vectors, passages and source fingerprints
were bitwise/structurally unchanged; original files remain privately preserved.
See [migration audit](v2-chunk-contract-migration.json). No test was rescored.

The new index CLI accepts `--words`, `--overlap`, `--books` and `--output`, so
the selected passage policy can be rebuilt rather than requiring a private
fitted artifact. `fetch-reader-v2 --acknowledge-local-rights` downloads only the
Journey source and rejects drift against the evaluated raw and normalized hashes.
It does not modify the older NarrativeQA subset or historical evaluations.

An actual pinned reader smoke inference placed a named answer after 140 repeated
sentences, beyond the initial reader window. Sliding inference recovered the
answer in its third window at source character 3110 and the source slice matched
exactly. This is a truncation regression check, not literature QA accuracy.
