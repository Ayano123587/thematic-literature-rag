# Thematic Literature RAG

Retrieve evidence across books, follow character mentions through narrative position, and answer questions with checkable source offsets. The index combines BM25, dense embeddings and optional neural reranking. Exact citations identify source text; they do not prove an interpretation follows from it.

The narrative adaptation dataset covers **all 10,580 FairytaleQA annotations**. A custom source-family partition preserves held-out collections: 7,883 training QA, all 2,372 validation QA and 325 sealed test QA. This differs from the official benchmark split. Full-data trained-model scores are pending; no subset score is presented as full-dataset performance.

## Setup and inference

Use Python 3.11+ and a fresh environment for this repository. Other portfolio projects share the `src` name and incompatible optional dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m src.cli fetch --acknowledge-local-rights
python -m src.cli index
python -m src.cli query "isolation rejection responsibility"
python -m src.cli synthesize "resentment revenge love" --books pride frankenstein
python -m src.cli arc Elizabeth pride --topic "vanity ignorance prejudice"
python -m pytest -q
```

On Linux/macOS activate with `source .venv/bin/activate`. Review [source rights](docs/data-card.md) before fetching. The offline baseline fits TF-IDF/SVD. Optional `.[models]` adds pinned sentence embeddings, cross-encoder reranking and Docling; `.[reader]` adds pretrained QA readers.

```powershell
python -m pip install -e ".[models,reader,notebook]"
python -m src.cli index --backend sentence-transformer
python -m src.cli ingest my-book.pdf my-book
python -m src.cli answer "Who persuaded Anne to reject Wentworth?" persuasion
```

Saved reader answers can be audited against source/model/chunk contracts without neural inference. The selected extractive reader is pinned MiniLM; dataset-specific generation is evaluated separately. Docling HTML conversion was exercised; PDF OCR accuracy has not been measured.

## Full-data training

[Report](Report.md) gives preparation and training commands. The [coverage audit](docs/full-coverage-audit.json), [training protocol](docs/full-coverage-protocol.json), [reranker protocol](docs/full-reranker-protocol.json) and [preflight](docs/full-coverage-preflight.json) bind the complete cohort and model pins. Source preparation retains every usable annotation and handles one repeated section ID without deleting text.

Training and validation resume from verified checkpoints and complete shards. The [selection command](evaluation_scripts/select_full_validation.py) compares every required checkpoint on all 2,372 validation questions. The test partition stays sealed until the final policy is frozen.

## Notebooks and results

[Main workflow](notebooks/00_main_literature_rag.ipynb), [source EDA](notebooks/01_source_family_eda.ipynb), [experiments](notebooks/02_validation_experiments.ipynb) and [citation audit](notebooks/03_citation_and_error_analysis.ipynb) show source coverage, evidence/context retention and checkpoint replay with executed outputs. Full model curves and selected-model inference await completed validation. Optional training/download/test cells are disabled by default.

## Limits

Story identity is supplied in FairytaleQA; this does not measure global book routing or adult literary reasoning. Previously observed validation questions remain in the expanded cohort, which is not an independent replication. Source-family separation cannot rule out recurring plots or pretrained exposure. Answerable-only data do not establish unknown-question calibration.

Citations use Python character offsets in newline-normalized processed text; Docling offsets address converted text rather than PDF pages. Margins are not calibrated probabilities. Safe JSON/numeric-array persistence checks source and model identities. Raw text, QA, predictions, checkpoints and credentials remain private and ignored. Dataset/model licenses are separate from the software license.
