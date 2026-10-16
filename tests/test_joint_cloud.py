"""Pure software transport fixtures; no real QA or trained models are staged."""

import hashlib
import json
from pathlib import Path

import pytest

from src.prepare_joint_answer import SOURCES, cloud_contract
from src.cloud.joint_answer_runner import locate, restore


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def transport(tmp_path):
    root = Path(__file__).parents[1]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    payload = {name: (root / name).read_bytes() for name in SOURCES}
    payload.update(
        {
            "joint-frozen-inputs.json": b"{}",
            "joint-protocol.json": b"{}",
            "generator-protocol.json": b"{}",
            "reranker-protocol.json": b"{}",
            "private-data/manifest.json": b"{}",
            "NOTICE/FairytaleQA-LICENSE": b"fixture license",
        }
    )
    rows = []
    for name, raw in payload.items():
        flat = name.replace("/", "__")
        (dataset / flat).write_bytes(raw)
        rows.append({"path": name, "file": flat, "sha256": sha(raw)})
    manifest = {
        "schema": "joint_answer_transport_v1",
        "files": rows,
        "frozen_inputs_sha256": sha(b"{}"),
        "license_notice_sha256": sha(b"fixture license"),
        "source_commit": "fixture",
        "test_QA_packaged": False,
        "weights_refitted": False,
    }

    def render():
        path = dataset / "joint-source-manifest.json"
        path.write_text(json.dumps(manifest))
        expected = sha(path.read_bytes())
        program = tmp_path / "run.py"
        template = payload["src/cloud/joint_answer_runner.py"].decode()
        program.write_bytes(
            template.replace(
                "EXPECTED_MANIFEST_SHA256 = None",
                "EXPECTED_MANIFEST_SHA256 = " + repr(expected),
            ).encode()
        )
        return expected, program

    return dataset, manifest, render


def test_all_hashes_verified_before_restore_import(transport, tmp_path):
    dataset, manifest, render = transport
    expected, program = render()
    assert restore(dataset, tmp_path / "workspace", program, expected) == manifest


@pytest.mark.parametrize(
    "change",
    ["payload", "runner", "missing", "traversal", "test", "duplicate", "license"],
)
def test_transport_poison_before_any_copy(transport, tmp_path, change):
    dataset, manifest, render = transport
    if change == "missing":
        manifest["files"].pop()
    if change == "traversal":
        manifest["files"][0]["path"] = "../escape"
    if change == "test":
        manifest["files"][0]["path"] = "private-data/test/fixture.qa.csv"
    if change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    if change == "license":
        manifest["license_notice_sha256"] = "a" * 64
    expected, program = render()
    if change == "payload":
        (dataset / manifest["files"][0]["file"]).write_bytes(b"changed")
    if change == "runner":
        program.write_bytes(program.read_bytes() + b"# changed")
    with pytest.raises(ValueError):
        restore(dataset, tmp_path / "workspace", program, expected)
    assert not (tmp_path / "workspace").exists()


def test_actual_manifest_pin_required(tmp_path):
    with pytest.raises(ValueError):
        restore(tmp_path, tmp_path / "new", tmp_path / "run.py")


@pytest.mark.parametrize("matches", [0, 2])
def test_mount_marker_unique(tmp_path, matches):
    for i in range(matches):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / "identity.json").write_text("fixture")
    with pytest.raises(ValueError):
        locate(tmp_path, "identity.json", sha(b"fixture"))


def test_missing_actual_cloud_contract_refused(tmp_path):
    with pytest.raises(ValueError):
        cloud_contract({}, tmp_path)
