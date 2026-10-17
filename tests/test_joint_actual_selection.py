"""Actual completed metadata proves selection/transport ordering, not native loads."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src import joint_answer_validation as joint
from src.prepare_joint_answer import cloud_contract

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/interim/full-reranker-validation-completed-v2"
PARENT = (
    ROOT
    / "data/interim/reranker-cancelled-continuation-v1-reviewed/checkpoint-input-update-v1/terminal-small-output-v2/full-reranker/run"
)


@pytest.fixture
def actual():
    path = AUDIT / "audit-derived-reranker-selection-v2.json"
    if not path.is_file():
        pytest.skip("Actual full2372 audited metadata absent")
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "478f46a8adf827cfce3bfe07df8099eeddb1eb2f52a15be212657d8fab56be3c"
    )
    return path, json.loads(path.read_bytes())


def test_actual_all_three_snapshot_auth_order_and_unselected_failure(
    actual, monkeypatch
):
    path, selection = actual
    seen = []

    def metadata_only_snapshot(run, epoch, lineage):
        # Explicit software substitute: no actual native file is loaded on host.
        seen.append(epoch)
        receipt = joint.read(Path(run) / f"epoch-{epoch}/receipt.json")
        assert receipt["identity"] == lineage
        return receipt

    monkeypatch.setattr(joint, "snapshot_receipt", metadata_only_snapshot)
    chosen, receipt, snapshots = joint.authenticate_reranker_selection(
        PARENT, path, selection["training_identity"]
    )
    assert seen == [0, 1, 2] and set(snapshots) == {0, 1, 2}
    assert (
        chosen["epoch"] == 1 and receipt["weights_sha256"] == chosen["weights_sha256"]
    )
    seen.clear()

    def fail_last(run, epoch, lineage):
        current = metadata_only_snapshot(run, epoch, lineage)
        if epoch == 2:
            raise ValueError(
                "Injected software failure authenticating actual unselected epoch2"
            )
        return current

    monkeypatch.setattr(joint, "snapshot_receipt", fail_last)
    with pytest.raises(ValueError, match="unselected epoch2"):
        joint.authenticate_reranker_selection(
            PARENT, path, selection["training_identity"]
        )
    assert seen == [0, 1, 2]


def test_actual_selection_rejects_objective_before_native_auth(
    actual, tmp_path, monkeypatch
):
    _, selection = actual
    selection = copy.deepcopy(selection)
    selection["objective"] = "precision_at_4"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    monkeypatch.setattr(
        joint, "snapshot_receipt", lambda *_: pytest.fail("native auth reached")
    )
    with pytest.raises(ValueError, match="predeclared"):
        joint.authenticate_reranker_selection(
            PARENT, path, selection["training_identity"]
        )


def test_actual_embedded_selection_contract_and_tamper(actual, tmp_path):
    path, _ = actual
    frozen = json.loads(
        (
            ROOT / "data/interim/context-comparison-frozen-inputs-v1/frozen-inputs.json"
        ).read_bytes()
    )
    # Actual prior generator/context identities supply all unchanged marker hashes.
    artifacts = {
        "generator_run": {
            "kernel": "aikyatansinha/portfolio-narrative-full-resume-train-v6/1",
            "marker": "receipt.json",
            "sha256": frozen["checkpoint_receipt_sha256"],
            "directory": True,
        },
        "generator_selection": {
            "kernel": "aikyatansinha/portfolio-rag-generator-selection-v6/1",
            "marker": "selection.json",
            "sha256": frozen["selection_sha256"],
            "directory": False,
        },
        "baseline_directory": {
            "kernel": "aikyatansinha/portfolio-rag-generator-validation-e3-v6/1",
            "marker": "validation-status.json",
            "sha256": frozen["baseline_status_sha256"],
            "directory": True,
        },
    }
    context = (
        ROOT
        / "data/interim/context-comparison-v1-reviewed/terminal-output-v1/context-comparison/identity.json"
    )
    artifacts["context_directory"] = {
        "kernel": "aikyatansinha/portfolio-rag-full-context-comparison-v1/1",
        "marker": "identity.json",
        "sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
        "directory": True,
    }
    parent = PARENT.parent / "continuation-execution.json"
    artifacts["reranker_run"] = {
        "kernel": "aikyatansinha/portfolio-reranker-cancelled-resume-v1/2",
        "marker": "continuation-execution.json",
        "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "directory": True,
    }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    relative = "private-artifacts/reranker-selection.json"
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_bytes(path.read_bytes())
    artifacts["reranker_selection"] = {
        "source": "payload",
        "path": relative,
        "sha256": digest,
        "directory": False,
    }
    runtime = json.loads(
        (
            AUDIT / "epoch-0/terminal-small-output-v2/continuation-runtime.json"
        ).read_bytes()
    )
    # Runtime structure is taken from actual provenance, not estimated versions.
    lineage = joint.read(PARENT / "provenance.json")["identity"]
    contract = {
        "cloud": {
            "runtime": {
                "python": lineage["python"],
                "torch": lineage["torch"],
                "machine": "x86_64",
                "environment": lineage["environment"],
            },
            "artifacts": artifacts,
            "resume": None,
        },
        "files": {relative: digest},
        "generator_selection_sha256": frozen["selection_sha256"],
        "reranker_selection_sha256": digest,
        "context_identity_sha256": artifacts["context_directory"]["sha256"],
    }
    assert runtime and cloud_contract(contract, tmp_path) == contract["cloud"]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="embedded"):
        cloud_contract(contract, tmp_path)


def test_actual_full_payload_crlf_source_lf_provider_roundtrip(tmp_path):
    original = ROOT / "data/interim/joint-full2372-stage-v1/dataset"
    if not original.is_dir():
        pytest.skip("Actual533-file reviewed source payload absent")
    dataset = tmp_path / "dataset"
    shutil.copytree(original, dataset)
    manifest_path = dataset / "joint-source-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for row in manifest["files"]:
        if row["path"] in (
            "src/cloud/joint_answer_runner.py",
            "src/prepare_joint_answer.py",
        ):
            raw = (
                (ROOT / row["path"])
                .read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"\n", b"\r\n")
            )
            (dataset / row["file"]).write_bytes(raw)
            row["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    raw_template = (dataset / "src__cloud__joint_answer_runner.py").read_bytes()
    assert b"\r\n" in raw_template
    rendered = raw_template.replace(b"\r\n", b"\n").replace(
        b"EXPECTED_MANIFEST_SHA256 = None",
        ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode(),
        1,
    )
    program = tmp_path / "run.py"
    program.write_bytes(rendered)
    namespace = {"__name__": "actual_lf_transport_fixture", "__file__": str(program)}
    exec(compile(rendered, str(program), "exec"), namespace)
    restored = namespace["restore"](dataset, tmp_path / "restored", program, digest)
    assert len(restored["files"]) == 533
    assert (
        tmp_path / "restored/src/cloud/joint_answer_runner.py"
    ).read_bytes() == raw_template
    # Raw source hashing stays strict even though only the runner line endings
    # are canonicalized for the provider's serialization.
    (dataset / "src__cloud__joint_answer_runner.py").write_bytes(raw_template + b"\n")
    with pytest.raises(ValueError, match="Payload SHA"):
        namespace["restore"](dataset, tmp_path / "refused", program, digest)
    assert not (tmp_path / "refused").exists()
