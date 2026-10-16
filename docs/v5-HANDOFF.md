# Narrative training checkpoint

The private free CPU job `aikyatansinha/portfolio-narrative-t5-cpu-train`, version 1,
is confirmed RUNNING. It executes source snapshot `6068261`, pinned T5-small and
the full prespecified 3,314 training QA / 400-question validation cohort. Its
payload source/data hashes are in `v5-train-preflight.json`. No test QA is mounted.
Estimated runtime is roughly 3–6 hours from the measured CPU smoke; this is an
estimate, not a completed result. There are no full-run answer scores yet.

Corrected CPU smoke version 2 actually loaded pretrained weights, trained on four
real QA, saved and reloaded the checkpoint, and reproduced two validation answers.
All loading diagnostics were empty, including missing/unexpected/mismatched keys.
`v5-cpu-smoke.json` records exact weights, environment, executed source and log
hashes. Smoke examples establish mechanics only, not QA performance. Version 1's
duplicate-official-ID failure occurred before model loading and is preserved.

The source-family split, hardware adaptation and canonical row-ID correction all
preceded model outcomes. Validation source identities/labels were available for
development; test QA was never parsed. The notebook's default cells also leave
test scoring, training, downloads, plots and inference disabled. Its 12 code cells
were compiled and actually executed with those defaults; nbformat validation
passed. No notebook model outputs are invented or committed.

Both cloud builders now use readable Python importing plain versioned files from
a separate private dataset. The historical v3 executed payloads/results remain
unchanged; its future reproduction builder is explicitly retrospective. There is
no encoded-source fallback. Standard ZIP upload is automatically extracted by
Kaggle into normal mounted Python/JSON/CSV files.

Next, recover completed full-run validation aggregates and loss curves, inspect
actual learning behavior and failure modes, and choose justified validation-only
follow-up hypotheses. Preserve all failures. Do not open or score held-out test
until the final model/retrieval policy is frozen. Earlier Dorian/Persuasion/Call of
the Wild/Journey/Cask tests remain observed. Whole-family fairy-tale separation
does not prove plot or pretraining disjointness, global routing, or cross-book
thematic reasoning. Answer F1, evidence relevance and exact citation location are
distinct measures.

Publication may include the clean source, notebook and current honest pending
status now; raw data, private predictions, logs and weights remain ignored. No
paid resource or GPU retry is authorized by this checkpoint; current job is CPU.
