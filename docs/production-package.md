# Trained narrative reader

The selected recipe uses T5-small after three complete training epochs and a cross-encoder after one epoch. BM25 retrieves 12 sections; the cross-encoder keeps four. Question-matched windows supply at most 512 input tokens. Decoding uses four beams, at most 96 new tokens, no sampling and early stopping.

The [private Kaggle package](https://www.kaggle.com/code/aikyatansinha/audited-narrative-production-package) is version 1. Access requires the owner's account. Keep checkpoints and source data outside Git. The package contract SHA256 is:

```
a3148b684777e34db0c814f27d0f5d24f8437d9f17dbe1c9c466c64f83fd6ff1
```

Use the exact recorded runtime: Python 3.13.15, PyTorch 2.11.0+cpu, transformers 4.57.6, tokenizers 0.22.2, NumPy 2.1.3 and huggingface-hub 0.36.2 on x86_64. Authentication checks every checkpoint byte and frozen source before loading weights. A different runtime requires its own reproducibility check.

```python
from src.models.frozen_narrative_reader import authenticate, load, answer

contract_sha = "a3148b684777e34db0c814f27d0f5d24f8437d9f17dbe1c9c466c64f83fd6ff1"
authenticate(package_directory, contract_sha)
reader = load(package_directory, contract_sha, "cpu")
prediction = answer(question, processed_story, reader)
print(prediction["answer"])
print(prediction["retrieved_evidence"])
```

`processed_story` follows the source preparation contract: immutable full text, section identifiers and character offsets. Each returned citation points into that exact text. Story identity is supplied in the benchmark. The separate indexed-book CLI performs exploratory cross-book routing; these benchmark scores do not validate that routing or literary interpretation.

The full-validation notebook compares all four retrieval/context policies on the same 2,372 questions. The production package was frozen before the single 325-question test release. Test outcomes never choose a checkpoint, context policy or decoding setting.
