# Cached reliability study and saved-answer auditing

The [protocol](v4-prespecified-protocol.json) was committed as `21170e6` before
scoring. Gold-free implementation/tests were committed as `d54e0b5` before the
one cached evaluation. Only the original Dorian validation predictions and gold
were loaded: 29 official questions and four authored unknown controls. No model
inference, GPU run or observed-test scoring occurred. The historical public v3
report supplied validation prediction/gold hashes; its fresh-test fields were
not used, and no observed-test prediction, gold or case files were loaded.
All ten policies remain in
[results](v4-cached-validation-results.json).

| Policy | Token F1 | EM | Answer coverage | Accepted gold-EM mismatches |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 8.78% | 6.90% | 93.10% | 25 |
| Source-consensus bonuses, all three weights | 8.78% | 6.90% | 93.10% | 25 |
| Minimum distinct-answer gap 1 | 5.33% | 3.45% | 68.97% | 19 |
| Minimum distinct-answer gap 2 | 4.52% | 3.45% | 58.62% | 16 |
| Minimum distinct-answer gap 4 | 1.07% | 0.00% | 37.93% | 11 |
| Exact cross-reader agreement, RoBERTa margin above -2 | 9.11% | 6.90% | 37.93% | 9 |
| Exact cross-reader agreement, margin above 0 | 3.94% | 3.45% | 17.24% | 4 |
| Exact cross-reader agreement, margin above 2 | 3.94% | 3.45% | 10.34% | 2 |

No policy met the prespecified promotion rule. The best cross-reader F1 gain was
only 0.34 percentage points and failed the 50% coverage floor. Ambiguity rejection
discarded correct answers; consensus changed no aggregate score. Overlapping
windows around the same source occurrence never count as independent support.
All policies falsely answered zero of four controls, but this follows the fixed
MiniLM positive-margin eligibility filter and is not independent calibration
evidence. Gold EM mismatch is not proof of semantic incorrectness or unsupported
interpretation. No refined grid or new threshold search followed these outcomes.
The current production selector remains unchanged pending separately justified
training/evaluation.

## Persisted answer integrity

Source-verified `answer-v2` results now include an answer-artifact schema, source
SHA256, frozen policy/model/configuration and canonical policy SHA256. The model's
answer span and evidence coordinates remain exact processed-text character
offsets. Early empty/oversized-question or absent-source rejections are plain
responses: they have not verified a source and do not receive an audited envelope.
Reader abstentions after source verification can carry an envelope with no answer.

```bash
python -m src.cli answer-v2 "Who painted Dorian's portrait?" dorian --output data/interim/answer.json
python -m src.cli audit-answer data/interim/answer.json
```

`--output` writes UTF-8 directly, avoiding shell redirection encoding differences.
The auditor needs no neural weights or GPU. By default it compares the declared
policy to `docs/v2-validation-selection.json`; `--selection` and `--sources` accept
explicit expected-policy and source-directory paths. It checks the source hash,
source slice, citation/answer/evidence agreement, reader eligibility, declared
chunk geometry, exact model revision declarations and frozen-policy equality.
Malformed/missing fields, nonfinite or Boolean numeric values, oversized numeric
values, duplicate JSON keys and oversized input are rejected clearly. Artifact
reads are bounded to one million bytes plus one byte for detecting overflow.

These checks establish consistency, **not authenticity, semantic correctness or
proof that the reader chose the highest-scoring candidate**. An attacker can
rewrite a self-consistent fingerprint; without the expected frozen selection the
audit cannot establish that policy equality either. Tests distinguish these
cases. A retained actual Dorian neural result was persisted and checked by the
real CLI with exit 0; a fabricated-answer mutation was rejected with exit 2.
No new inference was needed. See [CLI smoke](v4-artifact-cli-smoke.json).

Raw QA, answer strings, private artifacts and source books remain ignored. Journey,
Cask and all earlier tests are observed historical sets and were excluded from
policy selection. This study improves answer persistence/auditing, not measured
QA accuracy. Further actual training requires a new source-disjoint protocol.
