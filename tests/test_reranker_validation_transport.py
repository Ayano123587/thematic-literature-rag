"""Opaque software fixtures verify pre-model completion and prefix gates."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import src.reranker_validation_transport as transport

ACTUAL_COMPLETED = (
    Path(__file__).resolve().parents[1]
    / "data/interim/reranker-cancelled-continuation-v1-reviewed"
    / "checkpoint-input-update-v1/terminal-small-output-v2"
)
ACTUAL_EXECUTION = "856fef8e745600a74c009c50fd96f8ece4cf81f03dcd16c7d77a0e78b44ce8a7"


@pytest.fixture
def actual_completed_metadata(tmp_path):
    if not ACTUAL_COMPLETED.is_dir():
        pytest.skip("Private actual completed continuation receipts unavailable")
    shutil.copytree(ACTUAL_COMPLETED, tmp_path / "completed")
    return tmp_path / "completed"


def test_actual_completed_run_receipts_bind_all_three_snapshots(
    actual_completed_metadata,
):
    contract = transport.freeze_training_metadata(
        actual_completed_metadata, ACTUAL_EXECUTION
    )
    assert (
        contract["parent"] == "aikyatansinha/portfolio-reranker-cancelled-resume-v1/2"
    )
    assert contract["kernel_id"] == 139224383
    assert set(contract["snapshots"]) == {"0", "1", "2"}
    assert contract["snapshots"]["2"]["weights_sha256"] == (
        "61a77b42daa4b9c7bda4e9fee55623d7d0842da3f88ca0212d37743ad3a7da32"
    )
    # Local metadata staging never opens a native model file.
    assert not list(actual_completed_metadata.rglob("*.safetensors"))


@pytest.mark.parametrize(
    "field", ["parent", "initial_progress", "initial_state_sha256", "runner_sha256"]
)
def test_actual_completed_run_rejects_changed_resumption_identity(
    actual_completed_metadata, field
):
    path = actual_completed_metadata / "full-reranker/continuation-execution.json"
    value = transport.read(path)
    value[field] = None
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        transport.freeze_training_metadata(
            actual_completed_metadata, transport.digest(path)
        )


def test_actual_last_snapshot_receipt_drift_refused(actual_completed_metadata):
    path = actual_completed_metadata / "full-reranker/run/epoch-2/receipt.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="epoch receipt bytes"):
        transport.freeze_training_metadata(actual_completed_metadata, ACTUAL_EXECUTION)


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return transport.digest(path)


def complete_parent(tmp_path, identity=None):
    identity = identity or {
        "python": "fixture",
        "torch": "fixture",
        "environment": {"transformers": "fixture"},
    }
    status = {
        "status": "training_complete",
        "progress": transport.FINAL_PROGRESS,
        "epochs_required": 2,
        "train_QA": 7883,
        "smoke": False,
        "test_accessed": False,
    }
    execution = {
        "schema": "reranker_continuation_execution_v1",
        "complete": True,
        "manifest_sha256": transport.CONTINUATION_MANIFEST,
        "runner_sha256": transport.CONTINUATION_RUNNER,
        "parent": transport.SCIENTIFIC_PARENT,
        "initial_progress": transport.INITIAL_PROGRESS,
        "initial_state_sha256": transport.INITIAL_STATE,
        "frozen_inputs_sha256": transport.FROZEN_INPUTS,
        "final_progress": transport.FINAL_PROGRESS,
        "test_accessed": False,
        "snapshots": {},
    }
    execution["training_status_sha256"] = save(
        tmp_path, "full-reranker/run/training-status.json", status
    )
    execution["provenance_sha256"] = save(
        tmp_path, "full-reranker/run/provenance.json", {"identity": identity}
    )
    execution["runtime_receipt_sha256"] = save(
        tmp_path,
        "continuation-runtime.json",
        {
            "actual": {
                key: identity[key] for key in ("python", "torch", "environment")
            },
            "expected": {
                key: identity[key] for key in ("python", "torch", "environment")
            },
            "difference_keys": [],
            "bootstrap_error": None,
        },
    )
    for epoch in (0, 1, 2):
        files = {}
        for name in (
            "model.safetensors",
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ):
            path = tmp_path / f"full-reranker/run/epoch-{epoch}/{name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"opaque software fixture {epoch} {name}".encode())
            files[name] = transport.digest(path)
        receipt = {
            "identity": identity,
            "epoch": epoch,
            "files": files,
            "weights_sha256": files["model.safetensors"],
        }
        sha = save(tmp_path, f"full-reranker/run/epoch-{epoch}/receipt.json", receipt)
        execution["snapshots"][str(epoch)] = {
            "receipt_sha256": sha,
            "weights_sha256": files["model.safetensors"],
            "files": files,
        }
    return save(tmp_path, "full-reranker/continuation-execution.json", execution)


def test_actual_complete_metadata_then_all_cloud_bytes(tmp_path):
    sha = complete_parent(tmp_path)
    contract = transport.freeze_training_metadata(tmp_path, sha)
    assert transport.authenticate_training(tmp_path, contract) == contract["identity"]
    (tmp_path / "full-reranker/run/epoch-2/model.safetensors").unlink()
    # Local staging relies on actual metadata and never needs native weights.
    assert transport.freeze_training_metadata(tmp_path, sha) == contract
    with pytest.raises(ValueError):
        transport.authenticate_training(tmp_path, contract)


@pytest.mark.parametrize(
    "change",
    [
        "checkpointed",
        "epoch_missing",
        "runtime_diff",
        "wrong_manifest",
        "partial_train",
    ],
)
def test_no_future_partial_or_wrong_parent_contract(tmp_path, change):
    complete_parent(tmp_path)
    path = tmp_path / "full-reranker/continuation-execution.json"
    execution = json.loads(path.read_text())
    if change == "checkpointed":
        execution["complete"] = False
    elif change == "epoch_missing":
        execution["snapshots"].pop("2")
    elif change == "wrong_manifest":
        execution["manifest_sha256"] = "a" * 64
    elif change == "partial_train":
        name = "full-reranker/run/training-status.json"
        status = transport.read(tmp_path / name)
        status["train_QA"] = 20
        execution["training_status_sha256"] = save(tmp_path, name, status)
    else:
        name = "continuation-runtime.json"
        runtime = transport.read(tmp_path / name)
        runtime["difference_keys"] = ["torch"]
        execution["runtime_receipt_sha256"] = save(tmp_path, name, runtime)
    sha = save(tmp_path, "full-reranker/continuation-execution.json", execution)
    with pytest.raises(ValueError):
        transport.freeze_training_metadata(tmp_path, sha)


def prefix(tmp_path, monkeypatch):
    ids = [f"fixture-{i}" for i in range(2372)]
    monkeypatch.setattr(
        transport,
        "VALIDATION_IDS",
        hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    )
    identity = {
        "run": {},
        "epoch": 1,
        "weights_sha256": "a" * 64,
        "ids_sha256": transport.VALIDATION_IDS,
        "shard_size": 64,
    }
    rows = [
        {
            "id": name,
            "selected_sections": ["1"],
            "candidate_sections": ["1"],
            "recall": 1.0,
            "precision": 1.0,
            "candidate_recall": 1.0,
            "top4_ceiling": 1.0,
        }
        for name in ids[:64]
    ]
    save(tmp_path, "identity.json", identity)
    save(
        tmp_path,
        "validation-status.json",
        {
            "identity": identity,
            "smoke": False,
            "test_accessed": False,
            "questions_required": 2372,
            "questions_completed": 64,
            "complete": False,
        },
    )
    save(
        tmp_path,
        "shard-00000.json",
        {
            "identity": identity,
            "rows": rows,
            "rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        },
    )
    return identity, ids


def test_exact_prefix_authentication(tmp_path, monkeypatch):
    identity, ids = prefix(tmp_path, monkeypatch)
    frozen = transport.freeze_validation_prefix(tmp_path, identity, ids)
    assert transport.authenticate_prefix(tmp_path, frozen, ids) == 64
    (tmp_path / "identity.json").write_text("{}")
    with pytest.raises(ValueError):
        transport.authenticate_prefix(tmp_path, frozen, ids)


@pytest.mark.parametrize(
    "change", ["gap", "extra", "wrong_count", "nan", "overlap", "partial_metric"]
)
def test_invalid_prefix_before_model_load(tmp_path, monkeypatch, change):
    identity, ids = prefix(tmp_path, monkeypatch)
    if change == "gap":
        (tmp_path / "shard-00000.json").rename(tmp_path / "shard-00064.json")
    elif change == "extra":
        (tmp_path / "pending-cases.json").write_text("[]")
    elif change in ("wrong_count", "partial_metric"):
        status = transport.read(tmp_path / "validation-status.json")
        status["questions_completed"] = 63 if change == "wrong_count" else 64
        if change == "partial_metric":
            status["validation"] = {}
        save(tmp_path, "validation-status.json", status)
    else:
        shard = transport.read(tmp_path / "shard-00000.json")
        if change == "nan":
            shard["rows"][0]["recall"] = float("nan")
        else:
            shard["rows"][1]["id"] = shard["rows"][0]["id"]
        save(tmp_path, "shard-00000.json", shard)
    with pytest.raises(ValueError):
        transport.freeze_validation_prefix(tmp_path, identity, ids)


def test_validation_full_runner_render_roundtrip():
    import ast
    from pathlib import Path
    import types

    template = Path("src/cloud/reranker_validation_runner.py").read_text()
    value = "b" * 64
    rendered = template.replace(
        "EXPECTED_MANIFEST_SHA256 = None",
        "EXPECTED_MANIFEST_SHA256 = " + repr(value),
        1,
    )
    tree = ast.parse(rendered)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_MANIFEST_SHA256"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    module = types.ModuleType("rendered_validation_software_fixture")
    exec(compile(rendered, "fixture.py", "exec"), module.__dict__)
    assert module.EXPECTED_MANIFEST_SHA256 == value
    assert '"EXPECTED_MANIFEST_SHA256 = None"' in rendered


def test_stager_refuses_absent_future_completion_before_output(tmp_path, monkeypatch):
    from pathlib import Path
    import src.prepare_reranker_validation as stager

    monkeypatch.setattr(
        stager.subprocess,
        "check_output",
        lambda *args, **kwargs: str(Path.cwd()) + "\n",
    )
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        stager.prepare(
            tmp_path / "original",
            tmp_path / "future-parent",
            "a" * 64,
            1,
            output,
            "owner/private",
        )
    assert not output.exists()


@pytest.mark.parametrize("elapsed,expected", [(0, 36000), (2000, 35200), (36000, 1200)])
def test_budget_accounts_for_startup_and_setup_reserve(elapsed, expected):
    from src.cloud.reranker_validation_runner import validation_budget

    assert validation_budget(100, 100 + elapsed) == expected


def test_budget_refuses_exhausted_startup():
    from src.cloud.reranker_validation_runner import validation_budget

    with pytest.raises(ValueError):
        validation_budget(100, 100 + 37000)


@pytest.mark.parametrize(
    "failure", [None, "paused", "qa", "last_snapshot", "runtime", "prefix"]
)
def test_actual_rendered_validation_flow(tmp_path, monkeypatch, failure):
    from pathlib import Path
    import sys
    import types
    import src.cloud.reranker_validation_runner as template_module

    inputs = tmp_path / "inputs"
    payload = inputs / "payload"
    parent = inputs / "parent"
    working = tmp_path / "working"
    working.mkdir()
    payload.mkdir(parents=True)

    def payload_file(name, raw):
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return transport.digest(path)

    identity = {
        "python": "fixture",
        "torch": "fixture",
        "environment": {"transformers": "fixture"},
        "source_sha256": {"science.py": payload_file("src/science.py", b"science")},
        "protocol_sha256": payload_file("docs/full-coverage-protocol.json", b"{}"),
        "manifest_sha256": payload_file("private-data/manifest.json", b"data"),
    }
    payload_file("private-data/validation/story.qa.csv", b"validation")
    execution_sha = complete_parent(parent, identity)
    training = transport.freeze_training_metadata(parent, execution_sha)
    ids = [f"fixture-{i}" for i in range(2372)]
    monkeypatch.setattr(
        transport,
        "VALIDATION_IDS",
        hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    )
    expected_identity = transport.validation_identity(training, 1)
    previous = inputs / "previous"
    previous.mkdir()
    old_identity, old_ids = prefix(previous, monkeypatch)
    assert old_ids == ids
    for name in ("identity.json", "validation-status.json", "shard-00000.json"):
        value = transport.read(previous / name)
        value["identity"] = (
            expected_identity
            if name != "identity.json"
            else value.get("identity", expected_identity)
        )
        if name == "identity.json":
            value = expected_identity
        save(previous, name, value)
    resume = transport.freeze_validation_prefix(previous, expected_identity, ids)
    contract = {"training": training, "epoch": 1, "resume": resume}
    payload_file("frozen-validation-inputs.json", json.dumps(contract).encode())
    template = Path(template_module.__file__).read_text()
    payload_file("src/cloud/reranker_validation_runner.py", template.encode())
    files = {
        p.relative_to(payload).as_posix(): transport.digest(p)
        for p in payload.rglob("*")
        if p.is_file()
    }
    manifest_sha = payload_file(
        "reranker-validation-manifest.json", json.dumps({"files": files}).encode()
    )
    rendered = template.replace(
        "EXPECTED_MANIFEST_SHA256 = None",
        "EXPECTED_MANIFEST_SHA256 = " + repr(manifest_sha),
        1,
    )
    script = tmp_path / "run.py"
    script.write_text(rendered)
    runner = types.ModuleType("rendered_validation_flow")
    runner.__file__ = str(script)
    exec(compile(rendered, str(script), "exec"), runner.__dict__)
    real_path = Path
    monkeypatch.setattr(
        runner,
        "Path",
        lambda value: (
            inputs
            if str(value) == "/kaggle/input"
            else (
                working
                if str(value) == "/kaggle/working"
                else (
                    working / "full-reranker-validation"
                    if str(value) == "/kaggle/working/full-reranker-validation"
                    else real_path(value)
                )
            )
        ),
    )
    calls = []
    actual_auth = transport.authenticate_training

    def authenticate(folder, frozen):
        result = actual_auth(folder, frozen)
        calls.append("parent authenticated")
        return result

    monkeypatch.setattr(transport, "authenticate_training", authenticate)

    def audit(stories, questions, rows, kind):
        assert [row["id"] for row in rows] == [q["id"] for q in questions]
        calls.append("prefix source audited")
        return rows

    def metric_equal(a, b):
        assert a == b

    monkeypatch.setitem(
        sys.modules,
        "src.full_validation_selection",
        types.SimpleNamespace(
            validation_cohort=lambda *args: ({}, [{"id": i} for i in ids]),
            recompute_rows=audit,
            metric_equal=metric_equal,
        ),
    )
    runtime_path = working / "runtime.json"
    runtime_path.write_text("{}")

    def restore(frozen):
        assert calls[:2] == ["parent authenticated", "prefix source audited"]
        calls.append("runtime restored")
        if failure == "runtime":
            raise ValueError("runtime fixture drift")
        return runtime_path

    monkeypatch.setitem(
        sys.modules,
        "src.cloud.reranker_continuation_runner",
        types.SimpleNamespace(
            restore_runtime=restore,
            runtime_versions=lambda: {
                k: identity[k] for k in ("python", "torch", "environment")
            },
        ),
    )

    def validate(data, protocol, run, epoch, output, **kwargs):
        assert calls[:3] == [
            "parent authenticated",
            "prefix source audited",
            "runtime restored",
        ]
        calls.append("model validator invoked")
        assert (
            kwargs["smoke"] is False
            and kwargs["shard_size"] == 64
            and kwargs["previous"] == previous
        )
        output.mkdir()
        rows = [
            {
                "id": name,
                "selected_sections": ["1"],
                "candidate_sections": ["1"],
                "recall": 1.0,
                "precision": 1.0,
                "candidate_recall": 1.0,
                "top4_ceiling": 1.0,
            }
            for name in ids
        ]
        if failure == "paused":
            rows = rows[:64]
        save(output, "identity.json", expected_identity)
        for start in range(0, len(rows), 64):
            shard_rows = rows[start : start + 64]
            save(
                output,
                f"shard-{start:05d}.json",
                {
                    "identity": expected_identity,
                    "rows": shard_rows,
                    "rows_sha256": hashlib.sha256(
                        json.dumps(shard_rows, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest(),
                },
            )
        status = {
            "identity": expected_identity,
            "smoke": False,
            "test_accessed": False,
            "questions_required": 2372,
            "questions_completed": len(rows),
            "complete": len(rows) == 2372,
        }
        if len(rows) == 2372:
            status.update({"validation": {"n": 2372}, "bm25": {"n": 2372}})
        save(output, "validation-status.json", status)
        return status

    monkeypatch.setitem(
        sys.modules,
        "src.full_reranker_training",
        types.SimpleNamespace(
            validate_epoch=validate, summarize_rows=lambda rows: {"n": len(rows)}
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.reranker_training",
        types.SimpleNamespace(
            evaluate_sections=lambda stories, questions: {"n": len(questions)}
        ),
    )
    if failure == "qa":
        (payload / "private-data/validation/story.qa.csv").write_bytes(b"changed")
    elif failure == "last_snapshot":
        (parent / "full-reranker/run/epoch-2/model.safetensors").write_bytes(b"changed")
    elif failure == "prefix":
        (previous / "shard-00000.json").write_text("{}")
    if failure not in (None, "paused"):
        with pytest.raises((ValueError, KeyError)):
            runner.main()
        assert "model validator invoked" not in calls
    else:
        runner.main()
        proof = transport.read(working / "reranker-validation-execution.json")
        assert proof["complete"] is (failure != "paused")
        assert proof["questions_completed"] == (64 if failure == "paused" else 2372)
        assert len(proof["prefix"]["files"]) == (3 if failure == "paused" else 40)


@pytest.mark.parametrize("epoch", [0, 1, 2])
def test_stager_actual_archive_full_cohort_and_real_import_closure(
    tmp_path, monkeypatch, epoch, actual_completed_metadata
):
    import subprocess
    import sys
    import zipfile
    import src.prepare_reranker_validation as stager

    repository_root = Path(__file__).resolve().parents[1]
    original = repository_root / "data/interim/full-reranker-v6-train-ready"
    if not (original / "dataset-upload/payload.zip").is_file():
        pytest.skip("Actual immutable full training source archive unavailable")
    names = (
        "src/reranker_validation_transport.py",
        "src/cloud/reranker_validation_runner.py",
        "src/prepare_reranker_validation.py",
        "src/reranker_continuation.py",
        "src/cloud/reranker_continuation_runner.py",
        "src/full_validation_selection.py",
    )
    repository = tmp_path / "actual-source-checkout"
    repository.mkdir()
    for name in names:
        destination = repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository_root / name).read_bytes())
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "core.autocrlf=false", "add", "."],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Source Packaging Check",
            "-c",
            "user.email=source-check@example.invalid",
            "commit",
            "-qm",
            "Actual source byte check",
        ],
        check=True,
    )
    monkeypatch.chdir(repository)
    output = tmp_path / "actual-stage"
    manifest = stager.prepare(
        original,
        actual_completed_metadata,
        ACTUAL_EXECUTION,
        epoch,
        output,
        "owner/private-source-byte-check",
    )
    assert manifest["epoch"] == epoch
    assert manifest["training_parent"] == transport.TRAINING_PARENT
    closure = manifest["local_source_closure"]
    assert {
        "src/full_validation_selection.py",
        "src/full_reranker_training.py",
        "src/full_generator_training.py",
        "src/resumable_training.py",
        "src/reranker_training.py",
        "src/narrative_training.py",
        "src/data/prepare_fairytale.py",
        "src/__init__.py",
        "src/data/__init__.py",
    } <= set(closure)
    extracted = tmp_path / "actual-extracted-payload"
    with zipfile.ZipFile(output / "dataset-upload/payload.zip") as handle:
        files = {name: handle.read(name) for name in handle.namelist()}
    assert all(
        hashlib.sha256(files[name]).hexdigest() == digest
        for name, digest in manifest["files"].items()
    )
    assert closure == stager.local_source_closure(files)
    for name, raw in files.items():
        if name.startswith("src/") and name.endswith(".py"):
            destination = extracted / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    # The exact formerly missing pre-runtime imports execute from real extracted
    # payload source, with isolated sys.path and no mocked module objects.
    program = (
        "import sys; sys.path.insert(0, " + repr(str(extracted)) + "); "
        "import src.full_validation_selection as audit; "
        "import src.reranker_validation_transport; "
        "import src.cloud.reranker_validation_runner; "
        "import src.cloud.reranker_continuation_runner; "
        "assert audit.__file__.startswith(" + repr(str(extracted)) + "); "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", program], check=True, capture_output=True
    )
    # Audit imports inside post-runtime model entrypoints without importing
    # numerical libraries or reading any native model artifact on this host.
    del files["src/resumable_training.py"]
    with pytest.raises(ValueError, match="resumable_training"):
        stager.local_source_closure(files)


def test_actual_failed_upload_missing_audit_module_is_rejected():
    import zipfile
    import src.prepare_reranker_validation as stager

    root = Path(__file__).resolve().parents[1]
    archive = (
        root
        / "data/interim/full-reranker-validation-completed-v1/epoch-0/dataset-upload/payload.zip"
    )
    if not archive.is_file():
        pytest.skip("Actual failed uploaded source archive unavailable")
    with zipfile.ZipFile(archive) as handle:
        files = {name: handle.read(name) for name in handle.namelist()}
    assert "src/full_validation_selection.py" not in files
    with pytest.raises(ValueError, match="full_validation_selection"):
        stager.local_source_closure(files)
