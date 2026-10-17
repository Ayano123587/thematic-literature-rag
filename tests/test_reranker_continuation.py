"""Software fixtures exercise external authentication before state deserialization."""

import json

import pytest

import src.reranker_continuation as continuation


def fixture_parent(tmp_path, monkeypatch):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return continuation.digest(path)

    identity = {"scientific_source": "fixture"}
    for slot in (0, 1):
        path = tmp_path / f"progress/slot-{slot}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"opaque optimizer model and RNG {slot}".encode())
        receipt = {
            "schema": 1,
            "slot": slot,
            "identity": identity,
            "progress": continuation.PROGRESS,
            "state_sha256": continuation.digest(path),
        }
        write(f"progress/slot-{slot}.json", receipt)
        if slot == 0:
            monkeypatch.setattr(continuation, "ACTIVE_STATE", receipt["state_sha256"])
            monkeypatch.setattr(
                continuation, "POINTER", write("progress/latest.json", receipt)
            )
    monkeypatch.setattr(
        continuation,
        "STATUS",
        write(
            "training-status.json",
            {"status": "checkpointed", "progress": continuation.PROGRESS},
        ),
    )
    monkeypatch.setattr(
        continuation, "PROVENANCE", write("provenance.json", {"identity": identity})
    )
    for epoch in (0, 1):
        path = tmp_path / f"epoch-{epoch}/model.safetensors"
        path.parent.mkdir()
        path.write_bytes(f"opaque fixture {epoch}".encode())
        sha = continuation.digest(path)
        write(
            f"epoch-{epoch}/receipt.json",
            {
                "identity": identity,
                "epoch": epoch,
                "files": {"model.safetensors": sha},
                "weights_sha256": sha,
            },
        )
    return continuation.freeze_metadata(tmp_path)


def test_full_external_authentication(tmp_path, monkeypatch):
    contract = fixture_parent(tmp_path, monkeypatch)
    assert continuation.authenticate(tmp_path, contract) == contract["identity"]


@pytest.mark.parametrize(
    "name",
    [
        "progress/slot-0.pt",
        "progress/slot-1.pt",
        "progress/slot-1.json",
        "progress/latest.json",
        "epoch-0/model.safetensors",
        "epoch-1/model.safetensors",
        "training-status.json",
        "provenance.json",
    ],
)
def test_changed_parent_fails_before_any_state_load(tmp_path, monkeypatch, name):
    contract = fixture_parent(tmp_path, monkeypatch)
    with (tmp_path / name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises((ValueError, KeyError)):
        continuation.authenticate(tmp_path, contract)


def test_unlisted_state_rejected(tmp_path, monkeypatch):
    contract = fixture_parent(tmp_path, monkeypatch)
    (tmp_path / "progress/unlisted.pt").write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="Unlisted"):
        continuation.authenticate(tmp_path, contract)


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError):
        continuation.safe_file(tmp_path, name)


def test_external_parent_version_required(tmp_path, monkeypatch):
    contract = fixture_parent(tmp_path, monkeypatch)
    contract["parent"] = continuation.PARENT.removesuffix("/1")
    with pytest.raises(ValueError):
        continuation.authenticate(tmp_path, contract)


def test_full_runner_single_assignment_self_render_roundtrip():
    import ast
    from pathlib import Path

    template = Path("src/cloud/reranker_continuation_runner.py").read_text()
    frozen_hash = "a" * 64
    rendered = template.replace(
        "EXPECTED_MANIFEST_SHA256 = None",
        "EXPECTED_MANIFEST_SHA256 = " + repr(frozen_hash),
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
    assert ast.literal_eval(assignments[0].value) == frozen_hash
    checker = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
    )
    assert ast.literal_eval(checker.args[0]) == "EXPECTED_MANIFEST_SHA256 = None"
    assert ast.literal_eval(checker.args[2]) == 1
    assert rendered == template.replace(
        ast.literal_eval(checker.args[0]),
        "EXPECTED_MANIFEST_SHA256 = " + repr(frozen_hash),
        ast.literal_eval(checker.args[2]),
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "runtime",
        "torch_runtime",
        "package_runtime",
        "pip_failure",
        "source",
        "qa",
        "template",
        "final_epoch",
        "final_snapshot",
        "paused",
    ],
)
def test_runner_authentication_and_completion_flow(tmp_path, monkeypatch, failure):
    import hashlib
    import sys
    import types
    from pathlib import Path
    import src.cloud.reranker_continuation_runner as runner

    input_root = tmp_path / "input"
    payload = input_root / "payload"
    parent = input_root / "parent"
    working = tmp_path / "working"
    working.mkdir()
    payload.mkdir(parents=True)
    (parent / "progress").mkdir(parents=True)
    (parent / "training-status.json").write_text("{}")
    (parent / "progress/latest.json").write_text("{}")

    def write(name, raw):
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    identity = {
        "python": "fixture",
        "torch": "fixture",
        "environment": {
            name: "fixture"
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        },
        "source_sha256": {"science.py": write("src/science.py", b"science")},
        "protocol_sha256": write("docs/full-coverage-protocol.json", b"protocol"),
        "manifest_sha256": write("private-data/manifest.json", b"data"),
    }
    write("private-data/train/question.csv", b"full training fixture")
    write("private-data/validation/question.csv", b"full validation fixture")
    contract = {
        "parent": continuation.PARENT,
        "progress": continuation.PROGRESS,
        "files": {"progress/slot-0.pt": "fixture"},
    }
    write("frozen-continuation-inputs.json", json.dumps(contract).encode())
    template = Path(runner.__file__).read_text()
    write("src/cloud/reranker_continuation_runner.py", template.encode())
    files = {
        p.relative_to(payload).as_posix(): continuation.digest(p)
        for p in payload.rglob("*")
        if p.is_file()
    }
    raw = json.dumps({"files": files}).encode()
    manifest_sha = write("reranker-continuation-manifest.json", raw)
    script = tmp_path / "run.py"
    script.write_text(
        template.replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(manifest_sha),
            1,
        )
    )
    # Execute the rendered source itself: importing the unrendered module would
    # miss a substitution that accidentally changes the checker's own literal.
    rendered_module = types.ModuleType("actual_rendered_continuation_fixture")
    rendered_module.__file__ = str(script)
    exec(compile(script.read_text(), str(script), "exec"), rendered_module.__dict__)
    runner = rendered_module
    monkeypatch.setattr(runner, "__file__", str(script))
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", manifest_sha)
    original_path = Path
    monkeypatch.setattr(
        runner,
        "Path",
        lambda value: (
            input_root
            if str(value) == "/kaggle/input"
            else (
                working / "full-reranker/run"
                if str(value) == "/kaggle/working/full-reranker/run"
                else (
                    working if str(value) == "/kaggle/working" else original_path(value)
                )
            )
        ),
    )
    monkeypatch.setattr(
        runner.platform,
        "python_version",
        lambda: "wrong" if failure == "runtime" else "fixture",
    )
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: (
            "wrong"
            if (failure == "torch_runtime" and name == "torch")
            or (failure == "package_runtime" and name == "transformers")
            else "fixture"
        ),
    )
    calls = []
    monkeypatch.setattr(
        continuation,
        "authenticate",
        lambda folder, frozen: calls.append("authenticated") or identity,
    )

    def pip(command, check):
        assert calls == ["authenticated"]
        assert command[:5] == [sys.executable, "-m", "pip", "install", "-q"]
        assert command[5:] == [
            name + "==fixture"
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        ]
        assert check is True
        assert runner.os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert runner.os.environ["OMP_NUM_THREADS"] == "4"
        calls.append("pinned bootstrap")
        if failure == "pip_failure":
            raise runner.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", pip)

    def trainer(data, protocol, output, **kwargs):
        calls.append("trained")
        assert calls[0] == "authenticated"
        assert calls[1] == "pinned bootstrap"
        output.mkdir(parents=True)
        progress = {"epoch": 2 if failure in ("paused", "final_epoch") else 3}
        result = {
            "status": "checkpointed" if failure == "paused" else "training_complete",
            "progress": progress,
        }
        (output / "progress").mkdir()
        (output / "progress/slot-0.pt").write_bytes(b"opaque")
        final = {
            "slot": 0,
            "identity": identity,
            "progress": progress,
            "state_sha256": continuation.digest(output / "progress/slot-0.pt"),
        }
        (output / "progress/latest.json").write_text(json.dumps(final))
        (output / "training-status.json").write_text(json.dumps(result))
        (output / "provenance.json").write_text(json.dumps({"identity": identity}))
        for epoch in range(2 if failure == "paused" else 3):
            folder = output / f"epoch-{epoch}"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(b"opaque snapshot")
            weight = continuation.digest(folder / "model.safetensors")
            receipt = {
                "identity": identity,
                "epoch": epoch,
                "files": {"model.safetensors": weight},
                "weights_sha256": weight,
            }
            (folder / "receipt.json").write_text(json.dumps(receipt))
            if failure == "final_snapshot" and epoch == 2:
                (folder / "model.safetensors").write_bytes(b"tampered")
        return result

    monkeypatch.setitem(
        sys.modules,
        "src.full_reranker_training",
        types.SimpleNamespace(train_full=trainer),
    )
    if failure in ("source", "qa"):
        (
            payload
            / (
                "src/science.py"
                if failure == "source"
                else "private-data/validation/question.csv"
            )
        ).write_bytes(b"changed")
    if failure == "template":
        script.write_text(script.read_text() + "# changed")
    if failure in (
        "runtime",
        "torch_runtime",
        "package_runtime",
        "pip_failure",
        "source",
        "qa",
        "template",
        "final_epoch",
        "final_snapshot",
    ):
        with pytest.raises(ValueError):
            runner.main()
        if failure in (
            "runtime",
            "torch_runtime",
            "package_runtime",
            "pip_failure",
            "source",
            "qa",
            "template",
        ):
            assert "trained" not in calls
        if failure in ("runtime", "torch_runtime", "package_runtime", "pip_failure"):
            diagnostic = json.loads((working / "continuation-runtime.json").read_text())
            assert diagnostic["model_loaded"] is False
            assert diagnostic["expected"]["python"] == "fixture"
            if failure == "runtime":
                assert diagnostic["difference_keys"] == ["python"]
            if failure == "torch_runtime":
                assert diagnostic["difference_keys"] == ["torch"]
            if failure == "package_runtime":
                assert diagnostic["difference_keys"] == ["environment.transformers"]
            if failure == "pip_failure":
                assert diagnostic["bootstrap_error"] == "CalledProcessError"
    else:
        runner.main()
        proof = json.loads(
            (working / "full-reranker/continuation-execution.json").read_text()
        )
        assert proof["complete"] is (failure != "paused")
        assert len(proof["snapshots"]) == (2 if failure == "paused" else 3)


@pytest.mark.parametrize("failure", [None, "dirty_source", "archive", "cwd"])
def test_stager_exact_archive_head_and_versioned_mount(tmp_path, monkeypatch, failure):
    import hashlib
    from pathlib import Path
    import zipfile
    import src.prepare_reranker_continuation as stager

    source_root = Path.cwd()
    names = (
        "src/reranker_continuation.py",
        "src/cloud/reranker_continuation_runner.py",
        "src/prepare_reranker_continuation.py",
    )
    originals = {name: (source_root / name).read_bytes() for name in names}
    for name, raw in originals.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    original = tmp_path / "original/dataset-upload"
    original.mkdir(parents=True)
    files = {
        "src/science.py": b"original",
        "private-data/manifest.json": b"manifest",
        "private-data/train/story.qa.csv": b"train",
        "private-data/validation/story.qa.csv": b"validation",
        "docs/full-coverage-protocol.json": b"protocol",
    }
    science_sha = hashlib.sha256(files["src/science.py"]).hexdigest()
    manifest = {
        "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    }
    with zipfile.ZipFile(original / "payload.zip", "w") as handle:
        for name, raw in files.items():
            handle.writestr(name, raw)
        handle.writestr("full-payload-provenance.json", json.dumps(manifest))
    (original / "LICENSE").write_text("license fixture")
    monkeypatch.setattr(
        stager, "ORIGINAL_ARCHIVE_SHA256", continuation.digest(original / "payload.zip")
    )
    monkeypatch.setattr(
        stager,
        "freeze_metadata",
        lambda parent: {
            "identity": {"source_sha256": {"science.py": science_sha}},
            "parent": continuation.PARENT,
        },
    )

    def git(arguments, text=False):
        if arguments[-1] == "--show-toplevel":
            return str(tmp_path / "wrong" if failure == "cwd" else tmp_path) + "\n"
        if "show" in arguments:
            return originals[arguments[-1].removeprefix("HEAD:")]
        return "fixture-commit\n"

    monkeypatch.setattr(stager.subprocess, "check_output", git)
    monkeypatch.chdir(tmp_path)
    if failure == "dirty_source":
        (tmp_path / names[0]).write_bytes(b"changed")
    if failure == "archive":
        with (original / "payload.zip").open("ab") as handle:
            handle.write(b"changed")
    output = tmp_path / "stage"
    if failure:
        with pytest.raises(ValueError):
            stager.prepare(
                original.parent, tmp_path / "parent", output, "owner/private"
            )
        assert not output.exists()
    else:
        result = stager.prepare(
            original.parent, tmp_path / "parent", output, "owner/private"
        )
        metadata = json.loads((output / "kernel-metadata.json").read_text())
        assert metadata["kernel_sources"] == [continuation.PARENT]
        assert metadata["is_private"] and not metadata["enable_gpu"]
        assert result["source_commit"] == "fixture-commit"
