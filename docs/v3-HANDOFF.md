# RAG improvement handoff

Workspace: `D:\CodexData\portfolio-improvements\thematic-literature-rag`.
Branch: `codex/improve-v2`. The original checkout was not edited by this agent.
Root published v2 through its `8025102` commit and successful CI. V3 source is
in commits `2b0c890`, `35c28c7`, `4b3b6a7`, plus the final commit containing this
handoff, calibration evidence and production guards.

All compute is finished. The only v3 private free GPU kernel,
`aikyatansinha/portfolio-literary-reader-v3`, version 1, completed in 146.3 seconds
on a Tesla T4 under a 900-second server/process limit. No further model inference
or GPU run was launched after its completion. Local follow-up used cached
validation spans only. Parent owns publication and goal pausing.

Measured evidence:

- V2 fresh Journey, 27 questions: F1 8.33%, EM 3.70%, abstention 18.52%,
  22/22 accepted answers source-exact.
- V3 matched GPU validation: MiniLM F1 8.78%, EM 6.90%; RoBERTa F1 3.94%,
  EM 3.45%, abstention 79.31%. Both falsely answered zero of four unknown controls.
  The larger-reader hypothesis failed and all policies remain recorded.
- V3 fresh Poe short story, 28 questions: selected MiniLM F1 19.84%, EM 14.29%,
  abstention 17.86%, 23/23 accepted answers source-exact. Short-story scope limits
  comparison to long novels; no fresh unknown-control cohort or entailment proof.
- Final cached calibration: MiniLM threshold -2/rank 1.5 reaches F1 15.10%,
  EM 13.79%, but false-answers one of four controls. Objective gain is only
  0.000718391. RoBERTa is unchanged. This is validation-only diagnostic evidence.

Production remains pinned MiniLM, threshold 0, rank penalty 0.8, with source,
embedding/reader/reranker revision, reconstructed chunk-contract, finite-logit,
question-token and pretrained-loading guards. The diagnostic negative threshold
is not deployed. Original model/test outcomes were not rerun. Journey and Poe
are observed tests and must never be relabeled fresh in later research.

Validation: `python -m pytest -q` passed 32 tests in 21.72 seconds; compileall and
diff checks pass. Actual production CPU smoke after the v2 source/model/chunk
guards emitted an exact historical Dorian source span. Actual v3 GPU forward
passes exercised both pinned reader checkpoints. Later loading guards were
fixture-tested without another model run. Both official QA heads are present,
GPU weight hashes match pinned LFS metadata, and no uninitialized-head warning
appears. MiniLM's unused BERT pooler tensors are expected.

Private artifacts, all ignored:

- `data/interim/kaggle-reader-v3`: generated private bootstrap and metadata.
- `data/interim/v3-kaggle-output`: original results, selection, predictions,
  separate QA/gold/context files, logs and uploaded source.
- `models/v3-cask.lrag`, `data/raw/cask.txt`, `data/processed/cask.txt`:
  pinned source-only story index and source for reproduction/audit.
- Original v2 archives before chunk-contract migration are preserved under
  `data/interim/v2-*-pre-contract.lrag`; migrated vectors and passages are identical.

The exact v3 preparation uses locally retained frozen v2 retrieval caches; those
raw literary contexts are deliberately not public. Public production fetch/index/
answer instructions work without the private QA payload. Use a separate repository
venv. Exact fresh-story index/source can be copied to the original checkout by
root for `answer-v2 --index models/v3-cask.lrag`, without rescoring anything.

Preserve the original loading/source/model hashes. Version 1 captured Python,
Torch, Transformers and tokenizers versions and fixed sentence-transformers
installation; base-image NumPy/scikit-learn versions were not explicitly recorded.
Artifact download initially returned a Windows encoding error; a UTF-8 retry
succeeded. The experiment itself completed once. No large-model superiority,
domain finetuning, semantic entailment or broad no-answer calibration is claimed.

Next action: root publishes the final committed source/aggregate docs, verifies
Actions, finishes the existing five-repository batch and pauses the goal. No
additional search, training or threshold tuning is pending in this worktree.
