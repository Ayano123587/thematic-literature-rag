# Narrative retrieval and reading

## Data

The pinned [FairytaleQA release](https://github.com/uci-soe/FairytaleQAData/tree/a24ddc17364666b7c13a425b9970c87368b03417) supplies 278 Gutenberg-sourced edited stories and 10,580 human QA under its Apache-2.0 license. Raw source files and annotations are kept private. This does not claim that every modern translation is public domain everywhere.

| Partition | Stories | QA |
|---|---:|---:|
| Custom training | 200 | 7,883 |
| Custom validation | 54 | 2,372 |
| Custom held-out test | 24 | 325 |

All Lang-labelled origins and Andersen are validation; Potter and Grimm are test. Every other source family is training. Evaluation uses every validation QA. The official split is 8,548/1,025/1,007; some official validation/test rows belong to custom training families, so these experiments are **not official-split benchmark results**. Prior validation exposure is disclosed in the protocol.

Normalized title/text and >=.8 five-token Jaccard source checks found no cross-story duplicate edges. A repeated adjacent section ID in remarkable-rocket is merged into one evidence unit, preserving every paragraph and character offset. All 7,883 training and 2,372 validation rows pass the loader. Preparation stages test QA as opaque bytes; gold parsing occurs only after the frozen one-shot release marker. See [audit](docs/full-coverage-audit.json) and [preflight](docs/full-coverage-preflight.json).

## Preparation

```powershell
python -m src.data.full_coverage_audit --fetch --output data/external/full-fairytale
python -m src.data.full_coverage_audit --source data/external/full-fairytale --audit-file docs/full-coverage-audit.json --stage-protocol docs/full-coverage-protocol.json --output data/interim/new-full-staging
```

Use a fresh repository virtual environment and install the reader extra. Preparation refuses to overwrite an existing cohort. Model weights, optimizer state, predictions and raw QA remain ignored.

## Models and objectives

The pinned T5-small generator uses 512 input tokens, 96 answer tokens, effective batch16, AdamW LR1e-4 and three epochs. Training uses human-linked sections; primary validation retrieves BM25 top4 sections without gold injection. Answer EM/token F1, evidence relevance and exact citation validity are separate measurements. Nonempty answer1/answer4 are evaluation references; ROUGE-L uses word LCS without stemming.

The pinned MiniLM cross-encoder uses listwise evidence supervision. The complete training cohort yields 8,848 groups/43,757 pairs. Three QA have no nongold negative and cannot form a contrastive group; they remain in the generator data and coverage count. Reranker selection uses all validation QA, followed by frozen-generator answer evaluation; evidence recall alone cannot establish answer quality.

## Training and validation

```powershell
python train.py full train --data data/interim/new-full-staging --protocol docs/full-coverage-protocol.json --output models/full-run
python train.py full train --data data/interim/new-full-staging --protocol docs/full-coverage-protocol.json --resume models/full-run --output models/full-continuation
python train.py full validate --data data/interim/new-full-staging --protocol docs/full-coverage-protocol.json --run models/full-continuation --epoch 1 --output data/interim/full-validation-epoch1
python train.py full-reranker train --data data/interim/new-full-staging --protocol docs/full-reranker-protocol.json --output models/full-ranking-run
```

Interrupted training saves optimizer/RNG state and source/model identities. Resume into a fresh output until `training-status.json` says `training_complete`. Validation compares complete epoch snapshots and resumes verified shards through `--resume`. This training workflow does not read test QA.

After complete validation for epochs 0–3, select the generator with the declared all-question F1 objective:

```powershell
python -m evaluation_scripts.select_full_validation generator --data data/interim/new-full-staging --protocol docs/full-coverage-protocol.json --run models/full-continuation --validation data/interim/val-e0 data/interim/val-e1 data/interim/val-e2 data/interim/val-e3 --output models/generator-validation-selection.json
```

Use the same command with `reranker`, its protocol/run and epochs 0–2 for evidence-recall selection; ties favor BM25, then the pretrained head, then earlier adapted epochs. Every checkpoint file, validation ID and shard hash is checked. Answer metrics are recomputed from current references and exact source citations. Missing, overlapping, changed or incomplete artifacts are rejected, and existing selections cannot be overwritten. The selection contains aggregates rather than raw QA or answers.

Checkpoint selection is not the final production policy. Compare selected retrieval with BM25 on complete validation using the frozen selected generator, then serialize the global policy before test release. No command in this workflow opens test QA.

## Context assembly

The context comparison uses one fixed question-matched window policy and the generator selected on all 2,372 validation questions. It preserves the same BM25 top four sections, 512-token input budget, tokenizer, weights and decoding. Exact source offsets identify every evidence slice. The [protocol](docs/context-assembly-protocol.json) requires at least 0.01 absolute answer F1 gain, no EM or source-citation regression, positive descriptive story-bootstrap lower bound and exact baseline replay. Section retention alone cannot select a policy.

```powershell
python -m src.context_validation --data data/interim/new-full-staging --generator-protocol docs/full-coverage-protocol.json --context-protocol docs/context-assembly-protocol.json --run models/full-continuation --selection models/generator-validation-selection.json --baseline-directory data/interim/selected-generator-validation --output data/interim/context-comparison
```

The command requires complete training and the audited complete-validation selection before loading weights. Interrupted comparisons resume only verified complete shards through `--previous`. The [tokenizer audit](docs/context-tokenizer-audit.json) checks every validation context against the pinned tokenizer and original source; it measures input and citation contracts, not neural answer quality.

## Results and interpretation

T5 completed three epochs on all 7,883 training QA. Mean training loss was 1.17869, 0.99408 and 0.90596; the [training aggregate](docs/full-generator-training.json) records checkpoint and source identities. All four checkpoints decoded all 2,372 validation questions. The declared token-F1 rule selected generator epoch 3. Complete evidence-recall validation selected reranker epoch 1; the final answer comparison then selected reranked question-matched windows.

| Complete cohort / policy | Questions | Token F1 | Exact match |
|---|---:|---:|---:|
| Validation: BM25, standard context | 2,372 | 0.26565 | 0.14165 |
| Validation: BM25, question windows | 2,372 | 0.33328 | 0.18128 |
| Validation: reranked, standard context | 2,372 | 0.27369 | 0.14334 |
| Validation: reranked, question windows | 2,372 | 0.35289 | 0.19309 |
| Frozen recipe, held-out test | 325 | 0.45696 | 0.22154 |

The [full-validation aggregate](docs/final-validation.json) binds the four-arm comparison. All 38 paired shards were independently replayed, including source windows, token budgets and citations. Validation selects the recipe; the separate test row evaluates that frozen recipe and is not an improvement over validation.

The [one-shot test aggregate](docs/final-test.json) covers all 325 questions across 24 Potter/Grimm stories: ROUGE-L 0.47001, evidence recall 0.96462 and citation validity 1.0. Every prediction hash, reference metric and source/section citation bound passed independent replay. A fresh package load reproduced all 325 prediction/citation dictionaries. [Package instructions](docs/production-package.md) identify the immutable artifacts and runtime.

Executed notebooks show training histories, complete validation comparisons and test results by source family and question type. Citation geometry is checked separately from answer accuracy; valid source coordinates do not establish entailment.

Question story identity and human section boundaries are given. Source-family splits do not prove plot or pretraining disjointness; repeated validation selection limits generalization claims. Free-form generation can be unsupported despite valid citation coordinates. Unknown-question calibration requires additional data.
