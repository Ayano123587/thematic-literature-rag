"""Software checks against the actual cancelled run's unchanged JSON receipts."""

import ast
import shutil
from pathlib import Path

import pytest

from src.reranker_cancelled_parent import freeze_metadata

ROOT = Path(__file__).resolve().parents[1]
OBS = (
    ROOT
    / "data/interim/reranker-auth-continuation-v3-reviewed/terminal-inspection-20261017T112513Z"
)


def actual_contract(root=OBS):
    return freeze_metadata(
        root / "small-output/full-reranker/run",
        root / "small-output/continuation-runtime.json",
        root / "post-download-version-proof.json",
        root / "status.json",
    )


@pytest.fixture
def actual_metadata(tmp_path):
    if not OBS.is_dir():
        pytest.skip("Private actual cancelled run metadata unavailable")
    shutil.copytree(OBS / "small-output", tmp_path / "small-output")
    for name in ("post-download-version-proof.json", "status.json"):
        shutil.copyfile(OBS / name, tmp_path / name)
    return tmp_path


def test_actual_durable_boundary_and_both_state_hashes(actual_metadata):
    contract = actual_contract(actual_metadata)
    assert contract["progress"]["next_microstep"] == 3840
    assert contract["scientifically_complete"] is False
    assert contract["parent"].endswith("/3")
    assert len(contract["files"]) == 20
    assert set(name for name in contract["files"] if name.endswith(".pt")) == {
        "progress/slot-0.pt",
        "progress/slot-1.pt",
    }


@pytest.mark.parametrize(
    "name", ["progress/latest.json", "progress/slot-1.json", "epoch-1/receipt.json"]
)
def test_changed_actual_metadata_refused(actual_metadata, name):
    path = actual_metadata / "small-output/full-reranker/run" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        actual_contract(actual_metadata)


def test_absent_final_receipt_not_invented(actual_metadata):
    path = actual_metadata / "small-output/full-reranker/run/epoch-2"
    path.mkdir()
    with pytest.raises(ValueError, match="claims final"):
        actual_contract(actual_metadata)


def test_actual_runner_authenticates_before_runtime_and_trainer():
    path = ROOT / "src/cloud/reranker_cancelled_runner.py"
    raw = path.read_text()
    rendered = raw.replace(
        "EXPECTED_MANIFEST_SHA256 = None", "EXPECTED_MANIFEST_SHA256 = 'review-only'", 1
    )
    tree = ast.parse(rendered)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "EXPECTED_MANIFEST_SHA256"
            for t in node.targets
        )
    ]
    assert len(assignments) == 1
    assert assignments[0].value.value == "review-only"
    assert raw.index("identity = authenticate(*auth_args)") < raw.index(
        "runtime_receipt = restore_runtime(identity)"
    )
    assert raw.index("identity = authenticate(*auth_args)") < raw.index(
        "from src.full_reranker_training import train_full"
    )
    assert (
        rendered.replace(
            "EXPECTED_MANIFEST_SHA256 = 'review-only'",
            "EXPECTED_MANIFEST_SHA256 = None",
            1,
        )
        == raw
    )
