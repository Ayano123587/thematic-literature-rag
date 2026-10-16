# Narrative retrieval and reading

## Data

The pinned [FairytaleQA release](https://github.com/uci-soe/FairytaleQAData/tree/a24ddc17364666b7c13a425b9970c87368b03417) supplies 278 Gutenberg-sourced edited stories and 10,580 human QA under its Apache-2.0 license. Raw source files and annotations are kept private. This does not claim that every modern translation is public domain everywhere.

| Partition | Stories | QA |
|---|---:|---:|
| Custom training | 200 | 7,883 |
| Custom validation | 54 | 2,372 |
| Custom test, sealed | 24 | 325 |

All Lang-labelled origins and Andersen are validation; Potter and Grimm are test. Every other source family is training. All prior holdout roles are preserved, and all validation QA are used. The official split is 8,548/1,025/1,007; some official validation/test rows belong to custom training families, so these experiments are **not official-split benchmark results**. Prior validation exposure is disclosed in the protocol.

Normalized title/text and >=.8 five-token Jaccard source checks found no cross-story duplicate edges. A repeated adjacent section ID in remarkable-rocket is merged into one evidence unit, preserving every paragraph and character offset. All 7,883 training and 2,372 validation rows pass the loader; test QA are staged as bytes without parsing. See [audit](docs/full-coverage-audit.json) and [preflight](docs/full-coverage-preflight.json).

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
```

A normal cloud time ceiling triggers a checkpoint, not a smaller dataset. Two atomic progress slots retain optimizer/RNG state and exact source/model/runtime identities. Epoch snapshots remain available for all-validation comparison. Validation resumes complete hashed shards through `--resume`, decoding every required query. Test access is a separate globally frozen step; this runner never opens test QA.

## Results and interpretation

Full-cohort trained-model results are pending actual inference. The operational extractive baseline and its source/citation guards remain available. No full-data score or superiority claim is inferred from training loss, retrieval overlap or a small mechanics run.

Question story identity and human section boundaries are given. Source-family splits do not prove plot or pretraining disjointness; repeated validation selection limits generalization claims. Free-form generation can be unsupported despite valid citation coordinates. Unknown-question calibration requires additional data.
