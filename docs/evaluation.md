# Recorded baseline evaluation

Executed on 2026-10-16 with Python 3.14.2 on CPU. Two real Gutenberg downloads produced 1,446 overlapping passages. Exact raw-download hashes are in [source-manifest.json](source-manifest.json). The fitted LSA index used up to 128 dense dimensions, BM25 lexical scores, and reciprocal-rank fusion with constant 60.

Four known-passage diagnostics yielded **hit@5 = 1.0 and MRR = 0.875**. Full case outputs appear in [evaluation.json](evaluation.json). Austen's opening passage ranked second; the other three anchors ranked first. Relevance labels are literal normalized anchor matches; the broad vanity anchor matches 19 passages. This tiny, hand-selected diagnostic is easy and is not evidence of thematic reasoning or novel-query generalization. It must not be used as a headline benchmark.

`python -m src.cli query "isolation rejection responsibility"` was executed against both books. `python -m src.cli arc Elizabeth pride` returned explicit name evidence from narrative thirds. Citation-only output is recorded in [demo-summary.json](demo-summary.json); no third-party book text is committed. The complete evidence can be regenerated from the recorded corpus.

Nine offline tests cover source offsets, overlap, malformed-license rejection, data opt-in, lexical/dense ranking, book filters, unknown-query abstention, model persistence, cross-book citation grounding, evaluation-anchor validation and character name boundaries. Tests intentionally use original short fictional text rather than downloading books in CI.

Baseline environment: NumPy 2.4.2, scikit-learn 1.8.0, pytest 9.1.1. Optional model environment: PyTorch 2.14.0+cpu, sentence-transformers 5.7.0. Dependency ranges permit compatible updates; these versions record the executed run.

The real SentenceTransformer and CrossEncoder were also executed on two original fixture passages. `all-MiniLM-L6-v2` produced a `(2, 384)` embedding matrix. Query `alienation` reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` returned citation `a:0-61` (the lonely creature seeking friendship) ahead of `a:42-108`. This validates model wiring and semantic retrieval on one smoke example, not literary-domain relevance quality.

Full-corpus pretrained dense retrieval was then run on all 1,446 passages. BM25 plus MiniLM dense fusion achieved hit@5 = 1.0, MRR = 0.8333 on the same diagnostic. See [neural-evaluation.json](neural-evaluation.json). It did not improve aggregate MRR over the LSA baseline on these four examples; no superiority claim is supported. The separate local neural index is `models/neural-index.pkl`, ignored by Git.

Docling 2.139.0 converted `tests/fixtures/original.html` through the actual CLI and `DocumentConverter`. The resulting Markdown preserved the title, both chapter headings, and both paragraphs. The converted fixture was moved into `data/interim/original-fixture-docling.md` after inspection, leaving the two-book evaluation corpus intact. HTML conversion is verified; PDF OCR layout fidelity remains unevaluated.

## Real published QA evaluation

The expanded NarrativeQA experiment indexes four complete novels (2,605 passages), never Wikipedia summaries or benchmark answers. All 29 Dorian Gray official validation questions select retrieval policy; all 29 Persuasion official test questions are evaluated after selection. The books and questions do not overlap across those splits. Source commit, license and exact hashes are recorded in [the benchmark data card](narrativeqa-data-card.md).

The actual pretrained extractive reader achieved **15.27% conventional normalized token F1 and 6.90% exact match** on the 29 test questions. It abstained on 37.93% of them. Four separately authored impossible-modern-technology questions produced no false answers; this tiny control set does not establish general no-answer calibration. The null-logit threshold of 2 was selected on validation, not test. The reader is pinned MiniLM pretrained on SQuAD2, not a newly finetuned literary model. Its exact answer spans are linked to passage citations, while correctness remains limited by retrieval and contiguous-span assumptions.

[reader-locked-test.json](reader-locked-test.json) preserves the first predictions and original punctuation-separated token-F1 calculation (14.90%). [reader-score-audit.json](reader-score-audit.json) transparently corrects scoring to conventional ASCII-punctuation removal without changing predictions, threshold or model; exact match is unchanged. This metric-only correction is retrospective, not another newly held-out performance claim. [reader-validation-selection.json](reader-validation-selection.json) preserves threshold choices and the original selection objective.

Retrieval evidence proxies are reported **separately from QA accuracy**:

| Validation retrieval policy | Mean gold answer content-token recall in top 5 | Literal answer presence in top 5 |
|---|---:|---:|
| BM25 | 32.07% | 20.69% |
| BM25 + LSA | 36.09% | 24.14% |
| Pretrained MiniLM dense | 33.10% | 17.24% |
| BM25 + MiniLM | 37.82% | 24.14% |
| BM25 + MiniLM + cross-encoder | 39.54% | 34.48% |

The validation-selected reranked policy produced **44.09% weak content-token recall** and **17.24% literal answer presence** on Persuasion test. These numbers are not QA scores or passage-relevance judgments. Many published answers are paraphrases of summaries; a literal occurrence can also be irrelevant. Full policy and case evidence is preserved in [narrativeqa-validation-selection.json](narrativeqa-validation-selection.json) and [narrativeqa-locked-test.json](narrativeqa-locked-test.json). Source-filter grounding is mechanically enforced, not an entailment metric.

## Cross-book theme and character evidence

[thematic-diagnostic.json](thematic-diagnostic.json) contains three original cross-book evidence queries and two turning-point anchors. An implementing agent read source excerpts and wrote those anchors; there is no independent human adjudication or published thematic gold standard. The two held-out theme queries each recovered only one book's annotated anchor at top 10; **neither retrieved both gold sides of the comparison**. Extractive synthesis returned zero sufficiently supported sentences for the self-recognition query. These failures are preserved. Topic-aware chronological retrieval did recover both character-turning-point anchors across the two arc diagnostics, but speaker attribution and coreference remain unverified. Two successful anchors do not prove character-arc interpretation.

Sentence synthesis emits only exact source substrings and links cross-book sentences via explicit shared query terms. This proves traceability, not literary entailment or interpretation. Unsupported sources and zero informative evidence abstain. No invented thematic accuracy score is reported.

## Persistence and bad-input audit

Nineteen offline tests pass, including safe numeric archive roundtrips, legacy pickle refusal, tampered offset rejection, changed-source detection, unknown source/character handling, exact synthesized sentence offsets, disjoint benchmark document declarations, fail-closed generated-answer citation/quote validation, versioned source serialization and exact neural-revision requirements. Current indexes use JSON plus numeric arrays with pickle disabled and bounded array sizes. CLI inference verifies normalized-source fingerprints and exact passage slices. Local default source fingerprints are newline-normalized UTF-8 text hashes, distinct from immutable raw-download hashes. Legacy pickle files and revisionless neural indexes must be rebuilt; forensic loading cannot enable unpinned query encoding.

Environment: Python 3.14.2, NumPy 2.4.2, scikit-learn 1.8.0, PyTorch 2.14.0+cpu, transformers 4.57.6, sentence-transformers 5.7.0. Use a fresh per-repository virtual environment; sharing the generic `src` namespace across portfolio installations is unsupported.

[embedding-pin-audit.json](embedding-pin-audit.json) records a separate CPU re-encoding of all 2,605 historical passages with the explicitly pinned MiniLM revision. The new vectors were **bitwise identical** to the preserved historical vectors (maximum absolute difference 0). The production artifact was replaced with the newly attested revision-bearing archive; the original revisionless archive remains local for audit. This establishes reproducible current vectors without retroactively claiming the old archive recorded a revision. No model/policy selection, frozen cloud payload or benchmark test was rerun.

Public reader reports retain scores, source IDs and exact answer offsets while predicted answer strings remain in ignored private files, bound by SHA-256. The literal source-span audit confirmed 29/29 candidates exactly matched persisted source text. [provenance-audit.json](provenance-audit.json) records the metadata-only newline hash correction and 236/236 matching cloud payload passages. The corpus and live experiment were preserved.

## Actual bounded generative result

The pinned Qwen2.5-1.5B-Instruct private GPU experiment compared two prompts only on Dorian validation, then used the selected strict prompt once on 30 fresh official NarrativeQA *Call of the Wild* test questions. It scored **0% conventional token F1 and 0% EM**, with 93.33% final abstention after enforcing citation and exact-quote requirements. The two accepted answers were wrong. This failed experiment is preserved in [generative-locked-test.json](generative-locked-test.json), with complete protocol, model weights hash, environment and runtime/input digests. [Generative experiment details](generative-experiment.md) explain output-format and supporting-quote failures; no scores were improved by relaxing the gate or rerunning test.

Valid citation labels demonstrate only source linkage. The experiment does not prove entailment, strong generative literary QA, autonomous character interpretation or calibrated no-answer behavior. It uses a different test book from the extractive reader, so no model-superiority comparison is supported.
