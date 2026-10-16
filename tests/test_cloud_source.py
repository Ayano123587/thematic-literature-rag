import hashlib
import json
from pathlib import Path

import pytest

from src.cloud import narrative_runner
from src.prepare_narrative_kernel import prepare
from src.prepare_reader_v3 import prepare as prepare_legacy


def test_future_builders_require_separate_private_dataset(tmp_path):
    with pytest.raises(ValueError, match="private Kaggle"):
        prepare(tmp_path, tmp_path / "out", True, None)
    with pytest.raises(ValueError, match="private Kaggle"):
        prepare_legacy(tmp_path / "out", None)


def test_plain_mounted_source_hash_and_test_exclusion(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.mkdir()
    code = payload / "train.py"
    code.write_text("print('readable fixture source')\n")
    manifest = payload / "payload-provenance.json"
    manifest.write_text(json.dumps({"file_sha256": {
        "train.py": hashlib.sha256(code.read_bytes()).hexdigest()}}))
    monkeypatch.setattr(narrative_runner, "EXPECTED_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setattr(narrative_runner, "Path",
                        lambda value: tmp_path if value == "/kaggle/input" else Path(value))
    assert narrative_runner.mounted_source()[0] == payload
    (payload / "private-data/test").mkdir(parents=True)
    with pytest.raises(ValueError, match="must not include test"):
        narrative_runner.mounted_source()
    code.write_text("print('modified')\n")
    with pytest.raises(ValueError, match="file changed"):
        narrative_runner.mounted_source()
