# Recorded baseline evaluation

Executed on 2026-10-16 with Python 3.14.2 on CPU. Two real Gutenberg downloads produced 1,446 overlapping passages. Exact raw-download hashes are in [source-manifest.json](source-manifest.json). The fitted LSA index used up to 128 dense dimensions, BM25 lexical scores, and reciprocal-rank fusion with constant 60.

Four known-passage diagnostics yielded **hit@5 = 1.0 and MRR = 0.875**. Full case outputs appear in [evaluation.json](evaluation.json). Austen's opening passage ranked second; the other three anchors ranked first. Relevance labels are literal normalized anchor matches; the broad vanity anchor matches 19 passages. This tiny, hand-selected diagnostic is easy and is not evidence of thematic reasoning or novel-query generalization. It must not be used as a headline benchmark.

`python -m src.cli query "isolation rejection responsibility"` was executed against both books. `python -m src.cli arc Elizabeth pride` returned explicit name evidence from narrative thirds. Citation-only output is recorded in [demo-summary.json](demo-summary.json); no third-party book text is committed. The complete evidence can be regenerated from the recorded corpus.

Nine offline tests cover source offsets, overlap, malformed-license rejection, data opt-in, lexical/dense ranking, book filters, unknown-query abstention, model persistence, cross-book citation grounding, evaluation-anchor validation and character name boundaries. Tests intentionally use original short fictional text rather than downloading books in CI.

Baseline environment: NumPy 2.4.2, scikit-learn 1.8.0, pytest 9.1.1. Optional model environment: PyTorch 2.14.0+cpu, sentence-transformers 5.7.0. Dependency ranges permit compatible updates; these versions record the executed run.

The real SentenceTransformer and CrossEncoder were also executed on two original fixture passages. `all-MiniLM-L6-v2` produced a `(2, 384)` embedding matrix. Query `alienation` reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` returned citation `a:0-61` (the lonely creature seeking friendship) ahead of `a:42-108`. This validates model wiring and semantic retrieval on one smoke example, not literary-domain relevance quality.

Full-corpus pretrained dense retrieval was then run on all 1,446 passages. BM25 plus MiniLM dense fusion achieved hit@5 = 1.0, MRR = 0.8333 on the same diagnostic. See [neural-evaluation.json](neural-evaluation.json). It did not improve aggregate MRR over the LSA baseline on these four examples; no superiority claim is supported. The separate local neural index is `models/neural-index.pkl`, ignored by Git.

Docling 2.139.0 converted `tests/fixtures/original.html` through the actual CLI and `DocumentConverter`. The resulting Markdown preserved the title, both chapter headings, and both paragraphs. The converted fixture was moved into `data/interim/original-fixture-docling.md` after inspection, leaving the two-book evaluation corpus intact. HTML conversion is verified; PDF OCR layout fidelity remains unevaluated.
