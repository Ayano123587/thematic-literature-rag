# Local model artifacts

Indexes, pretrained weights, adapted checkpoints, optimizer state and private predictions remain ignored. Do not add them to Git.

`.lrag` indexes use schema-checked JSON and numeric NPZ arrays with pickle disabled, array-size limits and source fingerprints. Neural indexes require the exact embedding-model commit revision. Production loading refuses revisionless artifacts; the forensic loader permits inspection only. Rebuild incompatible indexes with `python -m src.cli index`.

T5 generator and evidence-reranker training save immutable epoch snapshots and two atomic continuation slots. Each snapshot identifies source, protocol, tokenizer and weight hashes. Complete validation selects checkpoints; partial training does not authorize inference or test release. See [training instructions](../Report.md) and [data/model licenses](../docs/data-card.md).
