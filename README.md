# Thematic Literature RAG

Retrieve evidence across books, follow character mentions through narrative position, and answer questions with checkable source offsets. The index combines BM25, dense embeddings and optional neural reranking. Exact citations identify source text; they do not prove an interpretation follows from it.

The narrative adaptation dataset covers **all 10,580 FairytaleQA annotations**: 7,883 training questions, 2,372 validation questions and 325 held-out test questions. This custom source-family split differs from the official benchmark split. The selected T5 reader with a trained cross-encoder and question-matched source windows scores **0.35289 token F1 / 0.19309 exact match** on complete validation and **0.45696 token F1 / 0.22154 exact match** on the separate full test cohort. Every test prediction passed source/reference metric replay; fresh package loads reproduced all 325 answers and citations.

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

[Main workflow](notebooks/00_main_literature_rag.ipynb), [source EDA](notebooks/01_source_family_eda.ipynb), [experiments](notebooks/02_validation_experiments.ipynb), [citation audit](notebooks/03_citation_and_error_analysis.ipynb), [context comparison](notebooks/04_context_policy_comparison.ipynb), [trained-reader validation](notebooks/05_trained_reader_full_validation.ipynb) and [full held-out test](notebooks/06_frozen_reader_full_test.ipynb) contain executed source analysis, training histories and full-cohort result plots. Neural training and inference ran on Kaggle; result notebooks replay their recorded measurements. Optional training/download/test cells are disabled by default.

The [frozen production package](https://www.kaggle.com/code/aikyatansinha/audited-narrative-production-package) contains the selected generator epoch 3 and reranker epoch 1. Its contract authenticates every checkpoint and inference source file. Fresh package loads reproduced the audited validation answers and citations. See [package use](docs/production-package.md) for inference and verification.

## Limits

Story identity is supplied in FairytaleQA; this does not measure global book routing or adult literary reasoning. Previously observed validation questions remain in the expanded cohort, which is not an independent replication. Source-family separation cannot rule out recurring plots or pretrained exposure. Answerable-only data do not establish unknown-question calibration.

Citations use Python character offsets in newline-normalized processed text; Docling offsets address converted text rather than PDF pages. Margins are not calibrated probabilities. Safe JSON/numeric-array persistence checks source and model identities. Raw text, QA, predictions, checkpoints and credentials remain private and ignored. Dataset/model licenses are separate from the software license.
