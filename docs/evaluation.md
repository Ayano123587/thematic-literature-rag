# Evaluation

The full-data experiment compares the pinned initial T5 generator and every trained epoch on all 2,372 source-family validation QA. The generator selection objective is macro question answer token F1, with exact ties favoring the earlier checkpoint. The evidence reranker compares BM25, the pinned pretrained head and both adapted epochs by annotated section recall@4; ties favor BM25, then the pretrained head, then the earlier adapted epoch.

| Measurement | Meaning |
|---|---|
| Answer EM/token F1 | Maximum normalized score across nonempty answer1/answer4 references |
| ROUGE-L F1 | Word-level longest common subsequence, without stemming |
| Evidence recall/precision@4 | Returned section IDs compared with human-linked sections |
| Candidate recall@12 | Evidence available in the BM25 candidate pool |
| Retained evidence recall | Human-linked section IDs surviving tokenizer context assembly |
| Citation validity | Exact processed-source slice agreement |

Evidence relevance is distinct from answer accuracy. Source-valid citations do not establish entailment. Literal answer occurrence and answer-token overlap in retrieved passages are not QA scores.

Training uses annotated evidence; primary validation retrieves without gold injection. Complete validation reports must include every query exactly once. The [selection CLI](../evaluation_scripts/select_full_validation.py) checks the frozen cohort, checkpoint/config/tokenizer hashes, scientific source identities and complete hashed shards, then recomputes metrics from current references. Partial runs, missing/overlapping shards and changed artifacts are rejected. Hashes establish consistency rather than independent authenticity.

The complete four-arm comparison selects reranked question windows: token F1 0.35289 and exact match 0.19309 across all 2,372 validation questions. BM25 standard context scores 0.26565/0.14165 on that same cohort. The descriptive 54-story bootstrap interval for the selected-versus-baseline F1 gain is [0.07064, 0.10249] with 10,000 resamples. Reused validation and multiple comparisons limit confirmatory interpretation. See [validation measurements](final-validation.json).

The global production policy and package were frozen before the one-shot test release. All 325 custom test QA across 24 held-out stories were scored once: token F1 0.45696, exact match 0.22154 and ROUGE-L 0.47001. Full prediction hashes, reference/source replay and fresh-package reload parity were checked. These are different questions from validation; the scores are not a before/after comparison. No test result selects or refits a model. See [test measurements](final-test.json).

The operational extractive reader returns literal source spans under a frozen MiniLM policy. Saved answers can be audited without neural inference for source hashes, passage/answer/evidence agreement, chunk geometry and model revisions. Early missing-source rejection has no source-verified envelope. Citation guards and model loading are exercised by offline tests and actual saved-artifact checks; these do not measure semantic accuracy.

Important limits: story identity is supplied; source families may share plots; public stories may occur in pretraining; prior validation questions have been observed; annotations can omit alternative evidence. Answerable-only FairytaleQA does not establish unknown-question calibration. This experiment does not establish adult-novel interpretation or independently judged cross-book thematic reasoning.
