import pytest
from src.models.loading import validate_loading_info


def test_pretrained_qa_head_cannot_be_randomly_initialized():
    validate_loading_info({'unexpected_keys': ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']})
    validate_loading_info({})
    with pytest.raises(ValueError, match='missing'):
        validate_loading_info({'missing_keys': ['qa_outputs.weight']})
    with pytest.raises(ValueError, match='mismatched'):
        validate_loading_info({'mismatched_keys': [('qa_outputs.weight', [2, 384], [3, 384])]})
    with pytest.raises(ValueError, match='Unexpected'):
        validate_loading_info({'unexpected_keys': ['classifier.weight']})
