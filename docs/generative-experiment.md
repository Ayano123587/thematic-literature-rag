# Bounded grounded generation experiment

The extractive reader's low novel-domain F1 and missed paired thematic anchors motivate one actual generative experiment, not a claim that generation solves these failures.

The model is `Qwen/Qwen2.5-1.5B-Instruct`, Apache-2.0, pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Official model source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct. The generator is pretrained; no finetuning is claimed. The optional local CLI uses the same model revision, prompt contract and output validator as the private Kaggle experiment.

Fresh test source: *The Call of the Wild*, Gutenberg 215, Jack London (died 1916), official NarrativeQA document `1fa32a0798f79e13ca92ae84eda9ab3eb12c70a4`. The book was prespecified before inspecting or scoring its questions. All 30 matching official test questions are retained. Dorian's 29 validation questions compare two predefined prompts. Prompt selection is serialized before any fresh test generation. The previously seen Persuasion test is not reused as a fresh final test.

The fixed retrieval policy is BM25 + pinned MiniLM dense fusion + pinned MS-MARCO cross-encoder, chosen on Dorian validation. Four retrieved original-source passages enter each prompt; gold answers and summaries never enter the model context. The complete payload is private and ignored. [generative-input-manifest.json](generative-input-manifest.json) records its SHA-256, runner/template hashes, query IDs, model revision, source hash and budgets without distributing book/QA text.

Generation is deterministic (`do_sample=False`), at most 192 new tokens and 3,072 context tokens, with a 75-second per-generation limit. The one authorized free private Kaggle GPU job has a 3,600-second hard limit and a 3,300-second internal wall-clock budget, including setup. Canonical kernel: https://www.kaggle.com/code/aikyatansinha/portfolio-literary-grounded-qwen, version 1. Remote metadata was checked: private, GPU enabled, internet enabled. The code refuses to silently replace a missing GPU with an unbounded CPU run. No paid resources are used.

Every accepted generated answer must have valid supplied passage labels and exact source quotes for each citation. Invalid JSON, unknown citations or fabricated quotes force abstention and remain counted as protocol failures. This establishes source linkage, **not answer entailment**: a real quote can fail to support a generated conclusion. EM/token F1, schema failures, raw-answer citation/quote validity and abstention are reported separately. No failed questions are filtered out.

Raw model outputs, QA inputs, retrieved passages and weights remain private/ignored. Public experiment output excludes generated answer text and raw generations, retaining numerical measurements, IDs and source citation IDs. The runtime captures Python/Torch/Transformers/CUDA/GPU versions, exact weights hashes, model revision and input digest. Results must be read from the completed job before making a generative-performance claim.

## Completed version 1 results

The single private GPU job completed successfully. [generative-locked-test.json](generative-locked-test.json), [generative-validation-selection.json](generative-validation-selection.json) and [generative-run-manifest.json](generative-run-manifest.json) preserve the measured outcome and runtime/output hashes. Runtime: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1, CUDA 12.8. The safetensors weights SHA-256 is `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; the runtime input digest exactly matches the prepared manifest.

| Phase | Prompt | Token F1 | Exact match | Final abstention | Schema valid |
|---|---|---:|---:|---:|---:|
| Dorian validation, 29 questions | Direct | 0% | 0% | 100% | 0% |
| Dorian validation, 29 questions | Strict | 6.90% | 6.90% | 93.10% | 72.41% |
| Fresh Call of the Wild test, 30 questions | Validation-selected strict | **0%** | **0%** | 93.33% | 83.33% |

On the fresh test, the model attempted 24 non-abstained raw answers. Of those, 91.67% used valid citation labels, but only 8.33% satisfied exact supporting-quote checks. Only two answers passed the complete gate, and both were wrong. This experiment does not establish useful generative QA accuracy or complex literary reasoning. No alternative prompt, relaxed gate or model was selected from these outcomes, and no second GPU run was launched.

[generative-format-audit.json](generative-format-audit.json) gives retrospective failure diagnosis without changing scores: only two exact quote instances matched; 12 would match after whitespace normalization, and three fresh outputs failed JSON parsing. The gate was not relaxed to produce a more favorable test result. Neither valid labels nor normalized quotes would by themselves prove entailment. The current production validator additionally rejects non-list citation/quote fields; the audit found no such cases in the frozen run, so that hardening does not alter its reported scores.

The novel-domain extractive reader's 15.27% F1/6.90% EM used a different test novel (*Persuasion*), so it is not a direct head-to-head comparison with this fresh generative test. No independent generation no-answer-control cohort was included; final abstention rate is not a calibration claim.

An independent audit found a prewrite-versus-persisted newline hash mismatch in the first metadata fields. [provenance-audit.json](provenance-audit.json) corrects them transparently and preserves the original live payload field. All 116 validation and 120 fresh-test passage slices matched actual source text exactly. Corpus bytes, payload, prompts, policy and generation outputs were not changed. The current generator remains experimental; stronger passage selection, reliable structured output and independently judged entailment are unresolved.

Reproduction is explicit and never starts cloud compute automatically:

```bash
python -m src.cli fetch --acknowledge-local-rights
python -m src.cli fetch-qa --acknowledge-local-rights
python -m src.cli benchmark --output data/interim/reproduced-qa-test.json
python -m src.cli prepare-generation --acknowledge-local-rights --folder data/interim/generative-reproduction
# Inspect the prepared private metadata/payload locally before explicitly running cloud compute.
kaggle kernels push -p data/interim/generative-reproduction --accelerator NvidiaTeslaT4 -t 3600
```

This would create a new version of the same private kernel and must be treated as a reproduction, not another untouched test. Preserve the original payload and outputs. The one-shot fresh-test guarantee for version 1 is established by its orchestration and frozen selection order; a cloud rerun starts a new workspace and is not intrinsically prevented by a local file guard.
