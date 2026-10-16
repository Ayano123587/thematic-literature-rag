"""Transport fixtures only: no private QA, model loading, training or inference."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from src import prepare_context_comparison as staging
from src.cloud import context_comparison_runner as cloud


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(staging.encoded(value))
    return path


def fixture_contract(tmp_path):
    sources = {name: staging.sha(name) for name in staging.SCIENCE}
    runtime = {
        "python": "software-fixture",
        "torch": "software-fixture",
        "machine": "software-fixture",
        "environment": {
            n: "software-fixture"
            for n in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        },
    }
    lineage = {
        **runtime,
        "source_manifest_sha256": staging.MANIFEST,
        "protocol_sha256": staging.GENERATOR_PROTOCOL,
        "smoke": False,
        "micro_batch": 2,
        "threads": 4,
        "generator": staging.read("docs/full-coverage-protocol.json")["generator"],
        "source_sha256": {
            n: sources["src/" + n]
            for n in (
                "full_generator_training.py",
                "resumable_training.py",
                "narrative_training.py",
                "data/prepare_fairytale.py",
            )
        },
    }
    candidates = [
        {
            "epoch": i,
            "weights_sha256": staging.WEIGHTS[i],
            "validation": {"n": 2372, "f1": 0.1 * (i + 1)},
        }
        for i in range(4)
    ]
    selected = {
        "status": "validation_checkpoint_selection",
        "task": "generator",
        "validation_questions": 2372,
        "validation_ids_sha256": staging.IDS,
        "manifest_sha256": staging.MANIFEST,
        "protocol_sha256": staging.GENERATOR_PROTOCOL,
        "audit_source_sha256": sources["src/full_validation_selection.py"],
        "test_accessed": False,
        "final_test_authorized": False,
        "candidates": candidates,
        "selected": candidates[3],
        "training_identity": lineage,
        "artifact_receipts": [
            {
                "epoch": i,
                "checkpoint_receipt_sha256": "b" * 64,
                "validation_status_sha256": "c" * 64,
            }
            for i in range(4)
        ],
    }
    selection = put(tmp_path / "selection.json", selected)
    contract = {
        "schema": "context_comparison_inputs_v1",
        "selection_sha256": staging.sha(selection),
        "selected_epoch": 3,
        "weights_sha256": staging.WEIGHTS[3],
        "checkpoint_receipt_sha256": "b" * 64,
        "baseline_status_sha256": "c" * 64,
        "baseline_identity_sha256": "d" * 64,
        "baseline_shards_sha256": {n: "e" * 64 for n in staging.SHARDS},
        "source_manifest_sha256": staging.MANIFEST,
        "validation_ids_sha256": staging.IDS,
        "generator_protocol_sha256": staging.GENERATOR_PROTOCOL,
        "context_protocol_sha256": sources["docs/context-assembly-protocol.json"],
        "source_sha256": sources,
        "runtime": runtime,
        "training_kernel": "aikyatansinha/portfolio-narrative-full-resume-train-v6/1",
        "selection_kernel": "owner/selection/1",
        "baseline_kernel": "owner/baseline/1",
        "training_status_sha256": staging.TRAINING_STATUS,
        "training_provenance_sha256": staging.TRAINING_PROVENANCE,
        "resume": None,
    }
    path = put(tmp_path / "frozen.json", contract)
    return path, selection, contract, selected


def test_complete_metadata_contract_authenticates_without_qa_or_model(tmp_path):
    path, selection, contract, _ = fixture_contract(tmp_path)
    assert staging.authenticate(path, staging.sha(path), selection)[0] == contract


@pytest.mark.parametrize(
    "defect",
    [
        "placeholder",
        "missing",
        "missing_shard",
        "source",
        "wrong_weight",
        "wrong_epoch",
        "unversioned",
        "runtime",
        "partial_selection",
        "wrong_winner",
        "changed_status",
        "training_identity",
    ],
)
def test_unfrozen_or_malformed_selection_never_releases_qa(tmp_path, defect):
    path, selection, contract, chosen = fixture_contract(tmp_path)
    if defect == "placeholder":
        contract["selection_sha256"] = "0" * 64
    elif defect == "missing":
        contract.pop("selection_sha256")
    elif defect == "missing_shard":
        contract["baseline_shards_sha256"].pop(staging.SHARDS[-1])
    elif defect == "source":
        contract["source_sha256"]["src/context_assembly.py"] = "a" * 64
    elif defect == "wrong_weight":
        contract["weights_sha256"] = "a" * 64
    elif defect == "wrong_epoch":
        contract["selected_epoch"] = True
    elif defect == "unversioned":
        contract["selection_kernel"] = "owner/selection"
    elif defect == "runtime":
        contract["runtime"] = {}
    elif defect == "partial_selection":
        chosen["candidates"].pop()
    elif defect == "wrong_winner":
        chosen["selected"] = chosen["candidates"][0]
    elif defect == "changed_status":
        chosen["artifact_receipts"][3]["validation_status_sha256"] = "a" * 64
    else:
        chosen["training_identity"]["threads"] = 2
    put(selection, chosen)
    if defect not in ("placeholder", "missing"):
        contract["selection_sha256"] = staging.sha(selection)
    put(path, contract)
    with pytest.raises(ValueError):
        staging.authenticate(path, staging.sha(path), selection)


def test_dirty_stager_stops_before_every_private_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **k: " M README.md"
    )
    monkeypatch.setattr(
        staging, "authenticate", lambda *a: pytest.fail("Private contract read")
    )
    with pytest.raises(ValueError, match="Commit reviewed"):
        staging.prepare(
            "never-read-QA",
            "never-read-contract",
            "a" * 64,
            "never-read-selection",
            tmp_path / "out",
            "owner/dataset",
            "owner/kernel",
        )
    assert not (tmp_path / "out").exists()


def test_missing_actual_selection_stops_before_qa(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "")

    def reject(*args):
        raise ValueError("Actual audited selection artifact required")

    monkeypatch.setattr(staging, "authenticate", reject)
    with pytest.raises(ValueError, match="Actual audited"):
        staging.prepare(
            "poison-QA",
            "missing-contract",
            "a" * 64,
            "missing-selection",
            tmp_path / "out",
            "owner/dataset",
            "owner/kernel",
        )
    assert not (tmp_path / "out").exists()


def resume_fixture(tmp_path, monkeypatch):
    ids = [f"fixture-{i:04d}" for i in range(2372)]
    monkeypatch.setattr(
        staging, "IDS", hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    )
    identity = {"software-fixture": True}
    queries = [{"id": i, "section_ids": ["section-1"]} for i in ids]
    put(tmp_path / "identity.json", identity)
    put(
        tmp_path / "comparison-status.json",
        {
            "identity": identity,
            "complete": False,
            "questions_completed": 64,
            "questions_required": 2372,
            "test_accessed": False,
            "final_test_authorized": False,
        },
    )
    put(
        tmp_path / "retrieval-plan.json",
        {
            "identity": identity,
            "queries": queries,
            "queries_sha256": hashlib.sha256(
                json.dumps(queries, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    pairs = [{"id": i} for i in ids[:64]]
    put(
        tmp_path / staging.SHARDS[0],
        {
            "identity": identity,
            "pairs": pairs,
            "pairs_sha256": hashlib.sha256(
                json.dumps(pairs, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        },
    )
    contract = {
        "kernel": "owner/partial/1",
        "identity_sha256": staging.sha(tmp_path / "identity.json"),
        "status_sha256": staging.sha(tmp_path / "comparison-status.json"),
        "retrieval_plan_sha256": staging.sha(tmp_path / "retrieval-plan.json"),
        "shards_sha256": {staging.SHARDS[0]: staging.sha(tmp_path / staging.SHARDS[0])},
    }
    return contract, identity


def test_external_resume_receipts_preserve_complete_partial_shards(
    tmp_path, monkeypatch
):
    contract, identity = resume_fixture(tmp_path, monkeypatch)
    staging.validate_resume_contract(contract)
    staging.authenticate_resume(tmp_path, contract, identity)


@pytest.mark.parametrize(
    "defect",
    [
        "valid_plan_change",
        "shard",
        "identity",
        "promotion",
        "extra",
        "wrong_pair",
        "wrong_count",
    ],
)
def test_changed_continuation_cannot_forward_model(tmp_path, monkeypatch, defect):
    contract, identity = resume_fixture(tmp_path, monkeypatch)
    if defect == "valid_plan_change":
        path = tmp_path / "retrieval-plan.json"
        value = staging.read(path)
        value["queries"][0]["section_ids"] = ["other-valid-section"]
        value["queries_sha256"] = hashlib.sha256(
            json.dumps(value["queries"], sort_keys=True).encode()
        ).hexdigest()
        put(path, value)
    elif defect == "extra":
        put(tmp_path / staging.SHARDS[1], {})
    else:
        name = (
            "identity.json"
            if defect == "identity"
            else (
                staging.SHARDS[0]
                if defect in ("shard", "wrong_pair")
                else "comparison-status.json"
            )
        )
        path = tmp_path / name
        value = staging.read(path)
        if defect == "identity":
            value["changed"] = True
        elif defect == "promotion":
            value["decision"] = {"promoted": True}
        elif defect == "wrong_count":
            value["questions_completed"] = 63
        else:
            value["pairs"][0]["id"] = "wrong"
        put(path, value)
        # A recomputed self-hash alone is never enough. For semantic tests,
        # authenticate the changed file but keep the required scientific identity.
        if defect in ("promotion", "wrong_count"):
            contract["status_sha256"] = staging.sha(path)
        if defect == "identity":
            contract["identity_sha256"] = staging.sha(path)
        if defect == "wrong_pair":
            contract["shards_sha256"][name] = staging.sha(path)
    with pytest.raises(ValueError):
        staging.authenticate_resume(tmp_path, contract, identity)


def transport_fixture(tmp_path, extra=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = {n: Path(n).read_bytes() for n in staging.SCIENCE | staging.NEW_SOURCE}
    files.update(
        {
            "private-data/manifest.json": b"software fixture",
            "frozen-comparison-inputs.json": b"software fixture",
            "selected-generator.json": b"software fixture",
            "NOTICE/FairytaleQA-LICENSE": b"upstream license software fixture",
        }
    )
    if extra:
        files["src/context_transport_future_helper.py"] = b"# reviewed owned helper\n"
    rows = []
    for name, raw in files.items():
        flat = name.replace("/", "__")
        (inputs / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "context_comparison_transport_v1",
        "dispatchable": True,
        "test_QA_packaged": False,
        "weights_refitted": False,
        "frozen_inputs_sha256": hashlib.sha256(
            files["frozen-comparison-inputs.json"]
        ).hexdigest(),
        "license_notice_sha256": hashlib.sha256(
            files["NOTICE/FairytaleQA-LICENSE"]
        ).hexdigest(),
        "files": rows,
    }
    path = put(inputs / "context-source-manifest.json", manifest)
    digest = staging.sha(path)
    program = tmp_path / "run.py"
    program.write_bytes(
        files["src/cloud/context_comparison_runner.py"].replace(
            b"EXPECTED_MANIFEST_SHA256 = None",
            ("EXPECTED_MANIFEST_SHA256 = " + repr(digest)).encode(),
        )
    )
    return inputs, program, digest


def test_exact_transport_allows_new_reviewed_helpers_without_model_import(tmp_path):
    inputs, program, digest = transport_fixture(tmp_path, extra=True)
    cloud.restore(inputs, tmp_path / "workspace", program, digest)
    assert (tmp_path / "workspace/src/context_transport_future_helper.py").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "manifest",
        "runner",
        "payload",
        "test",
        "duplicate",
        "escape",
        "missing",
        "notice",
    ],
)
def test_transport_tamper_fails_before_copy_or_import(tmp_path, defect):
    inputs, program, digest = transport_fixture(tmp_path)
    if defect == "runner":
        program.write_bytes(program.read_bytes() + b"# changed")
    elif defect == "payload":
        (inputs / "selected-generator.json").write_bytes(b"changed")
    else:
        path = inputs / "context-source-manifest.json"
        manifest = staging.read(path)
        if defect == "manifest":
            manifest["test_QA_packaged"] = True
        elif defect == "duplicate":
            manifest["files"].append(manifest["files"][0])
        elif defect == "missing":
            manifest["files"].pop(0)
        elif defect == "notice":
            manifest["license_notice_sha256"] = "a" * 64
        else:
            manifest["files"][0]["path"] = (
                "private-data/test/story.qa.csv" if defect == "test" else "../escape.py"
            )
        put(path, manifest)
        digest = staging.sha(path)
    with pytest.raises(ValueError):
        cloud.restore(inputs, tmp_path / "workspace", program, digest)
    assert not (tmp_path / "workspace").exists()


def test_runner_has_one_renderable_manifest_assignment():
    assert (
        Path(cloud.__file__).read_text().count("EXPECTED_MANIFEST_SHA256 = None") == 1
    )


def test_runtime_guard_does_not_load_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fixture"))
    monkeypatch.setattr(cloud.platform, "python_version", lambda: "fixture")
    monkeypatch.setattr(cloud.platform, "machine", lambda: "fixture")
    monkeypatch.setattr(cloud.importlib.metadata, "version", lambda *a: "fixture")
    with pytest.raises(ValueError, match="before model loading"):
        cloud.check_runtime(
            {
                "python": "wrong",
                "torch": "fixture",
                "machine": "fixture",
                "environment": {"numpy": "fixture"},
            }
        )


@pytest.mark.parametrize(
    "partial",
    [
        {"decision": {"promoted": True}},
        {"baseline": {}},
        {"questions_completed": 2372},
        {"questions_completed": 63},
    ],
)
def test_partial_result_never_contains_promotion_or_aggregates(partial):
    result = {
        "complete": False,
        "questions_completed": 64,
        "questions_required": 2372,
        "test_accessed": False,
        "final_test_authorized": False,
        **partial,
    }
    with pytest.raises(ValueError):
        cloud.completed_status(result)


def test_partial_artifact_prefix_matches_completed_status(tmp_path):
    (tmp_path / "shard-00000.json").write_text("software fixture")
    result = {
        "complete": False,
        "questions_completed": 64,
        "questions_required": 2372,
        "test_accessed": False,
        "final_test_authorized": False,
    }
    assert set(cloud.completed_shards(tmp_path, result)) == {"shard-00000.json"}


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "wrong_prefix", "truncated_prefix"]
)
def test_partial_missing_or_extra_artifacts_cannot_emit_receipt(tmp_path, defect):
    result = {
        "complete": False,
        "questions_completed": 64,
        "questions_required": 2372,
        "test_accessed": False,
        "final_test_authorized": False,
    }
    if defect != "missing":
        (tmp_path / "shard-00000.json").write_text("software fixture")
    if defect == "extra":
        (tmp_path / "shard-00064.json").write_text("extra")
    elif defect == "wrong_prefix":
        (tmp_path / "shard-00000.json").rename(tmp_path / "shard-00064.json")
    elif defect == "truncated_prefix":
        result["questions_completed"] = 128
    with pytest.raises(ValueError, match="Exact completed"):
        cloud.completed_shards(tmp_path, result)


@pytest.mark.parametrize("defect", [None, "missing_last_four", "extra"])
def test_complete_artifacts_require_all38_and_no_extra(tmp_path, defect):
    result = {
        "complete": True,
        "questions_completed": 2372,
        "questions_required": 2372,
        "test_accessed": False,
        "final_test_authorized": False,
        "baseline": {},
        "candidate": {},
        "decision": {},
        "story_paired_F1_interval": {},
        "baseline_replay_exact": False,
    }
    for name in staging.SHARDS:
        (tmp_path / name).write_text("software fixture")
    if defect == "missing_last_four":
        (tmp_path / staging.SHARDS[-1]).unlink()
    elif defect == "extra":
        (tmp_path / "shard-99999.json").write_text("extra")
    if defect is None:
        assert len(cloud.completed_shards(tmp_path, result)) == 38
    else:
        with pytest.raises(ValueError, match="Exact completed"):
            cloud.completed_shards(tmp_path, result)
