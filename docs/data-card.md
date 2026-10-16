# Data and model provenance

FairytaleQA provides 278 edited Gutenberg-sourced stories and 10,580 human QA annotations. The official repository is [uci-soe/FairytaleQAData](https://github.com/uci-soe/FairytaleQAData), pinned at `a24ddc17364666b7c13a425b9970c87368b03417`; its provided files carry Apache-2.0. The downloader checks archive SHA256 `353ec962a911ea9b2a4fbba2fe9fdebe2a92c392ce4be36194f2062f0ac9a998` and retains attribution locally.

The custom source-family cohort uses all 7,883 training, 2,372 validation and 325 test QA. This is not the official benchmark partition: some official validation/test rows belong to custom training families. All Lang-labelled collections and Andersen are validation; Grimm and Potter are test. Test labels remain sealed. The [coverage audit](full-coverage-audit.json) records every source and duplicate check; the [protocol](full-coverage-protocol.json) specifies roles and evaluation.

The Gutenberg search adapter downloads *Pride and Prejudice* (1342) and *Frankenstein* (84). Gutenberg offers these editions as public domain in the USA; check [its license policy](https://www.gutenberg.org/policy/license.html) and local rights before reuse. The CLI requires explicit acknowledgement and preserves original license boundaries. This is not a worldwide copyright determination. Book text, QA, private predictions and weights are not redistributed.

The optional NarrativeQA adapter uses [google-deepmind/narrativeqa](https://github.com/google-deepmind/narrativeqa), revision `904246f6d1fe99a99a08a03501fe3e619af2cee5`, Apache-2.0 metadata/QA. It indexes original books rather than Wikipedia summaries or answer-bearing annotations. NarrativeQA questions may paraphrase summaries or refer to a different edition. Dorian, Persuasion, Call of the Wild, Journey and Cask are already observed evaluation sources and cannot supply a fresh test claim.

| Component | Model | Exact revision |
|---|---|---|
| Generator | google-t5/t5-small | df1b051c49625cf57a3d0d8d3863ed4d13564fe4 |
| Evidence reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a |
| Dense retrieval | sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| Extractive reader | deepset/minilm-uncased-squad2 | 934656cdda79824eabf503ed56e15c01ddbdbe3f |

T5, MiniLM embeddings and the MS-MARCO reranker carry Apache-2.0 model licenses. The [deepset reader](https://huggingface.co/deepset/minilm-uncased-squad2) carries CC BY4.0; attribution: Vaishali Pal, Branden Chan, Timo Möller, Malte Pietsch and Tanay Soni. Model-card benchmark scores are not project results. Software, model and dataset licenses are separate.

Processed-text hashes and Python character offsets describe the exact indexed representation, not page numbers. The Gutenberg adapter freezes its newline representation for citation replay; FairytaleQA preserves edited dataset section offsets. Saved manifests distinguish raw-download bytes from processed text. Source-family separation does not prove plot or pretraining disjointness.
