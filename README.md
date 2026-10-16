# Thematic Literature RAG

Trace themes across books with BM25 and dense retrieval, reciprocal-rank fusion, optional neural reranking, and verifiable passage citations. Candidate cross-book connections expose pairwise dense cosine similarity between retrieved passages. Character evidence is arranged by early, middle and late narrative position. Answers are evidence dossiers: the default deliberately does not fabricate psychological conclusions from snippets.

Measured novel-domain QA remains limited: the extractive reader scores 15.27% token F1/6.90% EM, while a bounded fresh-book generative experiment scores 0% after its strict support gate. Cross-book diagnostics miss one side of the annotated comparisons. This repository preserves those failures alongside the runnable pipeline and traceable evidence; it does not claim strong literary reasoning. Read the [executed evaluation](docs/evaluation.md).

## Run locally

Python 3.11+; a CPU is enough for the default model. Create a fresh virtual environment inside this repository. These portfolio repositories share a generic `src` module and may require incompatible optional dependencies; installing all five into one environment is unsupported.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
python -m pip install -e '.[dev]'
python -m src.cli fetch --acknowledge-local-rights
python -m src.cli index
python -m src.cli query "isolation rejection responsibility"
python -m src.cli arc Elizabeth pride
python -m src.cli arc Elizabeth pride --topic "vanity ignorance prejudice self knowledge"
python -m src.cli synthesize "resentment revenge love" --books pride frankenstein
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

SentenceTransformer produces real normalized dense embeddings. CrossEncoder scores query/passage pairs after hybrid candidate retrieval. Both default neural model revisions are pinned. Docling converts non-text documents into Markdown before indexing; actual HTML conversion was verified. These optional packages need a compatible PyTorch environment and may download weights. PDF OCR quality has not been evaluated. Only ingest documents you are authorized to process.

## Published benchmark and actual extractive answers

```bash
python -m src.cli fetch-qa --acknowledge-local-rights
python -m src.cli benchmark --output data/interim/reproduced-narrativeqa-test.json
python -m pip install -e '.[reader]'
python -m src.cli evaluate-reader --output data/interim/reproduced-reader-test.json
python -m src.cli index
python -m src.cli answer "Who persuaded Anne to reject Wentworth?" persuasion
python -m src.cli evaluate-themes
```

NarrativeQA metadata is pinned to a source commit and kept locally. The subset adds Gutenberg's *Dorian Gray* (29 official validation questions) and *Persuasion* (29 official test questions) to the existing books. Five retrieval policies are compared on validation only, then one selection is frozen before test retrieval. Gold-answer overlap measures are explicitly weak evidence proxies; they are not reading-comprehension accuracy. The separate pinned SQuAD2 pretrained MiniLM reader emits exact cited spans and a null-logit margin, with a threshold selected on validation. Its actual token-F1/abstention results and difficult cases are reported in [evaluation](docs/evaluation.md). These reader results measure extractive answers; the separate generative experiment is reported below.

Original locked outputs in docs are preserved; rerunning a locked evaluator requires a separately named output or a fresh checkout with results directed outside docs. Use a separate virtual environment for this repository. No novel text, benchmark CSV or fitted model is committed. See [benchmark data card](docs/narrativeqa-data-card.md).

The controlled [v2 reader round](docs/v2-improvements.md) adds fine passages,
expanded sliding contexts and deterministic citation assembly. Validation F1
increased modestly from 7.62% to 8.78%, with unchanged exact match. Its fresh
Journey test scores 8.33% F1 and 3.70% EM on 27 questions; all 22 emitted answers
match exact source offsets. Accuracy remains weak, and source matches do not prove
correct answers. The historical Qwen zero result is preserved. `answer-v2` checks
source/model/chunk contracts and supports rebuilding its selected index; see the
linked reproducibility instructions. Different test books prevent a direct gain
claim over the earlier Persuasion or Call of the Wild results.

The [v3 controlled reader comparison](docs/v3-reader-experiment.md) tested a pinned
larger RoBERTa reader on the same validation evidence. It lost (3.94% F1 versus
MiniLM's 8.78%), so production retains MiniLM, threshold 0 and rank penalty 0.8.
That frozen selection reached 19.84% F1 and 14.29% EM on 28 fresh questions about
a short Poe story, with 23/23 emitted spans matching source offsets. This is not
long-novel performance or proven entailment. A cached negative-threshold follow-up
raised validation F1 but regressed unknown controls; its diagnostic winner was
not deployed. Original failed experiments and all tested policies remain visible.

The [cached reliability study](docs/v4-cached-reliability.md) found no qualifying
span-selection improvement and retains those negatives. `answer-v2 --output`
now saves source/policy fingerprints in UTF-8; `audit-answer` rechecks a saved
answer against its source, chunk geometry and expected frozen policy without
neural inference. This verifies consistency, not authenticity or entailment.

## Optional grounded generation

```bash
python -m pip install -e '.[models,reader]'
python -m src.cli index --backend sentence-transformer
python -m src.cli generate "How does rejection change the desire for affection?" --books pride frankenstein --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --device cuda
```

The optional Qwen2.5-1.5B-Instruct backend emits JSON answers, supplied passage citations and exact supporting quotes. Invalid citations, invented quotes or malformed output force abstention. Valid quotes do **not** prove that the answer follows from them; inspect the evidence. The local CLI uses the index you built, reserves evidence per requested book, and accepts at most four books. A single private bounded Kaggle experiment tests this actual model on a fresh public NarrativeQA test book after validation-only prompt selection; see [generative experiment protocol](docs/generative-experiment.md). The local CPU wrapper has not been benchmarked and needs several GB of RAM; GPU is recommended. Downloading pretrained weights is required on first use, and no finetuning or autonomous character-coreference resolution is claimed.

## Traceability and limits

Each citation is `book:start-end`, indexing Python character offsets in newline-normalized processed UTF-8 text. Use `Path(...).read_text(encoding="utf-8")` and slice `[start:end]` to reproduce it. For Docling inputs these are converted-text offsets, not original PDF pages. Word windows overlap; progress is normalized token position, not chapter numbering. Cross-book retrieval reserves results per book so one long book cannot erase another. Ranking scores and reader margins are not calibrated probabilities. Out-of-vocabulary LSA queries abstain; in-vocabulary queries may still retrieve irrelevant passages. Source fingerprints and exact passage slices are checked before CLI inference; changed or missing sources require rebuilding the index.

The original four known-passage anchors remain historical smoke diagnostics. The expanded real QA subset and separately authored theme/arc evidence diagnostics provide broader, openly limited evaluation. Topic-aware arc retrieval can recover passages without an explicit name but marks their character attribution unverified. It does not solve speaker attribution or coreference. Extractive sentence synthesis links exact supporting sentences across books by shared query terms, while preserving their citations; thematic interpretation remains a human judgment.

Current index persistence uses checked JSON/numeric arrays instead of executable pickle. Older `index.pkl` files are rejected; rebuild them. Optional QA/reranking models are pretrained on other domains, and performance here should not be inferred from their original model-card scores.

Source code lives in `src/{data,features,models,visualization}`. The EDA notebook is supplemented by an [end-to-end narrative QA notebook](notebooks/02_narrative_training_and_audit.ipynb) with explicit optional training, saved-model answers, citation audits and plots; test scoring stays disabled by default. Root `train.py` and `evaluation_scripts` delegate to the source modules. Local data and model artifacts stay out of Git. CI runs offline behavioral tests. The software license does not relicense third-party books or model weights.
