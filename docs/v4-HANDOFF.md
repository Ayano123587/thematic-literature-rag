# Cached reliability checkpoint

The v4 protocol was committed before scoring (`21170e6`), followed by gold-free
implementation/tests (`d54e0b5`). Ten policies were evaluated once from the
original Dorian-only cached predictions/gold. None met the promotion rule.
Consensus was unchanged; ambiguity gaps lost correct answers; best cross-reader
agreement scored 9.11% F1 at only 37.93% coverage, below the precommitted 50% floor.
All negatives are in `docs/v4-cached-validation-results.json`. Historical report
metadata was used only to verify validation hashes, not for fresh-case analysis.

The useful production change is answer persistence/auditing. Source-verified
`answer-v2` results receive source/policy fingerprints, and `--output` writes UTF-8
without relying on shell redirection. `audit-answer` verifies exact source and
chunk geometry, policy equality, declared model pins, evidence fields and finite
eligible reader metadata. Early input/source rejections remain plain unaudited
responses. Fingerprints prove consistency, not authorship, entailment or ranking.
Malformed/missing structures, duplicate JSON keys, Boolean/oversized/nonfinite
numeric fields and oversized files fail clearly. Reads are bounded.

Validation: 41 tests passed in 72.55 seconds; compileall and diff check passed.
The real CLI audited a persisted actual v2 Dorian neural result with exit 0 and
rejected a fabricated-answer mutation with exit 2. This reused a saved neural
output; no new inference, GPU run or observed-test scoring occurred. Private
artifacts are `data/interim/v4-answer-artifact*.json` and
`data/interim/v4-cached-selections-private.json`; none is public.

Parent owns publication. The user subsequently revoked the earlier compute cap
and requested stronger actual training across the five repositories. Next RAG
work is primary-source research and a source-disjoint training/evaluation plan;
no training has yet been launched for that phase. Audio retains first GPU priority,
and any RAG GPU use must be coordinated with root. Existing Dorian validation
and all observed books must stay out of training labels. Journey/Cask are not fresh.
