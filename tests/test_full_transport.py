import pytest

from src.prepare_full_generator import prepare


def test_oversized_kernel_title_is_rejected_before_creating_payload(tmp_path):
    output = tmp_path/"payload"
    with pytest.raises(ValueError, match="50 characters"):
        prepare("unused", "unused", output, "owner/source", "smoke", task="reranker", run_tag="rng-proof")
    assert not output.exists()


@pytest.mark.parametrize("kwargs, message", [
    ({"action": "validate"}, "immutable training"),
    ({"action": "train", "validation_source": "owner/shards"}, "only to validation"),
    ({"action": "train", "task": "unknown"}, "Unknown full-data task"),
])
def test_invalid_transport_contract_fails_before_source_packaging(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        prepare("unused", "unused", tmp_path/"payload", "owner/source", **kwargs)
