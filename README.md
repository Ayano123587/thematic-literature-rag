# Thematic Literature RAG

Trace themes across books with BM25 and dense retrieval, reciprocal-rank fusion, optional neural reranking, and verifiable passage citations. Candidate cross-book connections expose pairwise dense cosine similarity between retrieved passages. Character evidence is arranged by early, middle and late narrative position. Answers are evidence dossiers: the default deliberately does not fabricate psychological conclusions from snippets.

## Run locally

Python 3.11+; a CPU is enough for the default model.

```bash
python -m pip install -e '.[dev]'
python -m src.cli fetch --acknowledge-local-rights
python -m src.cli index
python -m src.cli query "isolation rejection responsibility"
python -m src.cli arc Elizabeth pride
python -m src.cli evaluate
python -m pytest -q
```

Check local rights first: see [data card](docs/data-card.md). Downloads are needed once; default indexing and retrieval then run offline. The default dense model is TF-IDF followed by trained truncated SVD (LSA), not a pretrained sentence transformer. It supplies a transparent, reproducible baseline.

## Neural and document backends

```bash
python -m pip install -e '.[models]'
python -m src.cli index --backend sentence-transformer
python -m src.cli query "social rejection and alienation" --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
python -m src.cli ingest my-book.pdf my-book
```

SentenceTransformer produces real normalized dense embeddings. CrossEncoder scores query/passage pairs after hybrid candidate retrieval. Docling converts non-text documents into Markdown before indexing. These optional packages need a compatible PyTorch environment and may download weights; the default tests do not claim neural-backend or OCR quality. Only ingest documents you are authorized to process.

## Traceability and limits

Each citation is `book:start-end`, indexing Python character offsets in the processed UTF-8 text. Read `data/processed/book.txt` and slice `[start:end]` to reproduce it. For Docling inputs these are converted-text offsets, not original PDF pages. Word windows overlap; progress is normalized token position, not chapter numbering. Cross-book retrieval reserves results per book so one long book cannot erase another. Ranking scores are not probabilities. Out-of-vocabulary LSA queries abstain; in-vocabulary queries may still retrieve irrelevant passages.

Evaluation uses four manually specified known-passage queries on the downloaded corpus, with hit@5 and reciprocal rank. It is a diagnostic, not an independent thematic benchmark. Read [recorded evaluation](docs/evaluation.md). There is no claim of autonomous literary interpretation, coreference resolution, neural finetuning, or an LLM generation evaluation.

Source code lives in `src/{data,features,models,visualization}`; notebooks contain EDA only. Local data and model artifacts stay out of Git. CI runs offline behavioral tests. The software license does not relicense third-party books or model weights.
