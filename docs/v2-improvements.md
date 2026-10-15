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
penalties are selected using the 29 Dorian validation questions and four authored
unknown controls. The objective is conventional answer token F1 minus one quarter
of the control false-answer rate. These four controls do not establish broad
no-answer calibration. Reusing Dorian validation limits independence.

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
python -m src.cli answer-v2 "Your question" journey
```

`--selection` and `--index` accept explicit artifact paths. The selected policy
requires an index built with its stated chunk configuration. Production inference
checks all source fingerprints before learned inference, and then checks exact
answer coordinates before returning a citation. The API abstains on empty
questions and absent books. It does not verify semantic entailment.

An actual pinned reader smoke inference placed a named answer after 140 repeated
sentences, beyond the initial reader window. Sliding inference recovered the
answer in its third window at source character 3110 and the source slice matched
exactly. This is a truncation regression check, not literature QA accuracy.
