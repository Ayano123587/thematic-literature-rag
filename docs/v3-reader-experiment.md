# Stronger extractive reader experiment

V2's validation changes improved answer F1 only modestly and left exact match
unchanged. The next falsifiable hypothesis is that a larger reader trained for
extractive QA can select better spans from the same evidence. Retrieval contexts,
window size, overlap, maximum answer length, precision and scoring stay fixed.
MiniLM and RoBERTa run sequentially on the same free GPU, with one reader resident
at a time. No new model training or generated-quote JSON is involved.

Both models are attributed to **deepset** and their pinned model cards specify
**CC BY 4.0**: [MiniLM SQuAD2](https://huggingface.co/deepset/minilm-uncased-squad2/raw/934656cdda79824eabf503ed56e15c01ddbdbe3f/README.md)
and [RoBERTa SQuAD2](https://huggingface.co/deepset/roberta-base-squad2/raw/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4/README.md).
These are pretrained QA models; their original SQuAD scores do not establish
literature performance. Model revision, model-card hashes and actual weight hashes
are retained in the protocol/results; weights remain private local artifacts.

The reused Dorian validation set contains 29 official NarrativeQA questions and
four authored unknown controls. All 396 supplied context slices exactly matched
the fingerprinted source before upload. Gold files are separate and are not read
until both reader forward passes finish. The same 36 threshold/rank policies are
compared for each reader. Selection maximizes answer F1 minus one quarter of the
control false-answer rate, followed by prespecified tie breaks. Reused validation
and four narrow unknown controls limit model-selection and calibration claims.

Before inspecting fresh QA, source metadata identified the official test document
for *The Cask of Amontillado*. [Project Gutenberg](https://www.gutenberg.org/ebooks/1063)
identifies Edgar Allan Poe (1809–1849) and marks the source public domain in the
USA. Local rights still apply. The processed story has 2,317 words and 48 fine
passages; this short-story test does not represent long novels or cross-book
thematic reasoning. Its 28 official questions are packaged separately from gold.
No summaries or QA answers enter the unsupervised source index.

Validation selection must be serialized before the fresh question file is parsed.
Fresh retrieval uses the pinned v2 hybrid/reranker configuration against only the
new story, then the selected reader runs once. Fresh gold is read after prediction.
All emitted answers must match exact source offsets. That establishes source
integrity, not entailment. Persuasion, Call of the Wild and Journey results remain
preserved, and Journey is never presented as fresh again.

The private version-1 kernel is `aikyatansinha/portfolio-literary-reader-v3`.
It has a 900-second server timeout and process watchdog including installation
and model downloads. Audio GPU ownership was released before launch. Raw books,
questions, answers, predictions and the generated private bootstrap stay ignored.
The public code contains preparation and inference logic, not literary payloads.
If the budget or dependencies prevent completion, partial checkpoints and failure
logs must be retained without manufacturing a test score.

Preflight: 31 tests passed; the generated private bootstrap compiled; actual pinned
RoBERTa tokenization preserved offsets beyond the first context window; actual
MiniLM inference through the portable reader recovered the late-window fixture
answer exactly. See [preflight](v3-preflight.json), [smoke evidence](v3-smoke.json)
and [prespecified protocol](v3-prespecified-protocol.json).

## Completed version 1

The one private kernel completed in **146.3 seconds** on a Tesla T4. Actual pinned
forward inference and weight hashes are recorded in [locked results](v3-reader-locked-test.json).

| Reader | Validation F1 | EM | Abstention | Control false answers |
| --- | ---: | ---: | ---: | ---: |
| MiniLM | 8.78% | 6.90% | 6.90% | 0/4 |
| RoBERTa | 3.94% | 3.45% | 79.31% | 0/4 |

**The stronger-reader hypothesis failed.** MiniLM retained the selected threshold
0 and rank penalty 0.8, matching the earlier CPU validation result. RoBERTa's best
prespecified policy used threshold 2 and rank penalty 0. All policies are retained.

After selection was serialized, the selected MiniLM ran once on the 28 fresh Poe
questions: **19.84% F1, 14.29% EM (4/28), 17.86% abstention (5/28)**. Its 23 emitted
answers all matched exact source offsets. This is a short-story result, not a gain
over a matched long-novel test, and there is no fresh unknown-question cohort.

The local artifact audit verified runner/input/gold hashes, disjoint validation
and fresh IDs, and every candidate source span: 789 MiniLM validation spans,
801 RoBERTa validation spans, and 663 selected-reader fresh spans, with zero
invalid citations or nonfinite margins. This does not establish semantic
correctness. Actual Python/Torch/Transformers/tokenizers versions and exact
weight hashes are recorded. The fixed sentence-transformers installation is
declared, but version 1 did not explicitly capture base-image NumPy/scikit-learn
versions. See [artifact audit](v3-artifact-audit.json).

A separate cached-validation calibration follow-up tests whether RoBERTa's high
abstention partly reflects a negative null-margin scale. It introduces no new
inference or GPU use and has no independent held-out score. Its existence cannot
change the version-1 fresh test or retroactively make those questions fresh.
