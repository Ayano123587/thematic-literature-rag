"""Guarded demo fixtures only; no real models, native arrays or held-out QA."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from src import indexed_demo_transport as guard
from src import prepare_indexed_demo as staging

REPOSITORY = Path(__file__).resolve().parents[1]
EXTRA = {
    "src/indexed_narrative_cli.py",
    "src/models/indexed_narrative_reader.py",
    "src/models/train_model.py",
    "src/features/__init__.py",
    "src/features/build_features.py",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo, meta, books = tmp_path / "repo", tmp_path / "metadata", tmp_path / "books"
    repo.mkdir()
    meta.mkdir()
    books.mkdir()
    for name in guard.BRIDGE | guard.TRANSPORT:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / name).read_bytes())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for args in [
        ["config", "user.name", "Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["config", "core.autocrlf", "false"],
        ["add", "."],
        ["commit", "-qm", "fixture"],
    ]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    monkeypatch.chdir(repo)
    source = {
        name: guard.digest(repo / name) for name in guard.BRIDGE | guard.TRANSPORT
    }
    passages, hashes = [], {}
    for book in guard.BOOKS:
        text = "Fixture theme in " + book
        (books / (book + ".txt")).write_text(text)
        hashes[book] = guard.sha(text.encode())
        passages.append(
            dict(
                id=book + ":2-" + str(len(text)),
                book=book,
                start=2,
                end=len(text),
                text=text[2:],
                progress=0.0,
            )
        )
    index = tmp_path / "index.lrag"
    with zipfile.ZipFile(index, "w") as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(
                dict(schema=1, backend="lsa", passages=passages, source_hashes=hashes)
            ),
        )
        archive.writestr("arrays.npz", b"OPAQUE SOFTWARE FIXTURE - never deserialize")
    runtime = {
        "python": "fixture-python",
        "torch": "fixture-torch",
        "environment": {name: "1.2.3" for name in guard.CORE_PACKAGES},
    }
    policy = dict(
        schema="joint_answer_selection_v1",
        selected_arm="bm25_windows",
        validation_questions=2372,
        validation_ids_sha256=guard.VALIDATION_IDS,
        test_accessed=False,
    )
    joint = dict(
        complete=True,
        questions_completed=2372,
        questions_required=2372,
        test_accessed=False,
        selection=policy,
    )
    write_json(meta / "joint-selection.json", policy)
    write_json(meta / "joint-status.json", joint)
    contract = dict(
        schema="frozen_narrative_inference_v1",
        joint_result_sha256=guard.digest(meta / "joint-status.json"),
        source_sha256={n.removeprefix("src/"): source[n] for n in guard.BRIDGE - EXTRA},
        files={
            "joint-selection.json": guard.digest(meta / "joint-selection.json"),
            "generator/model.safetensors": guard.sha(b"OPAQUE FIXTURE WEIGHTS"),
        },
    )
    write_json(meta / "inference-contract.json", contract)
    frozen = dict(
        schema="root_frozen_indexed_demo_v1",
        demo_authorized=True,
        joint_validation_questions=2372,
        joint_validation_ids_sha256=guard.VALIDATION_IDS,
        test_accessed=False,
        package_parent="fixture/package/1",
        package_contract_sha256=guard.digest(meta / "inference-contract.json"),
        joint_result_sha256=guard.digest(meta / "joint-status.json"),
        selected_arm="bm25_windows",
        source_sha256=source,
        runtime=runtime,
        index_environment={name: "1.2.3" for name in guard.INDEX_PACKAGES},
        index_runtime_scope="prospective_demo_dependencies_not_recorded_model_training_runtime",
        books=list(guard.BOOKS),
        prompts=list(guard.PROMPTS),
        assets={"assets/index.lrag": guard.digest(index)}
        | {
            "assets/" + b + ".txt": guard.digest(books / (b + ".txt"))
            for b in guard.BOOKS
        },
    )
    freeze = tmp_path / "freeze.json"
    write_json(freeze, frozen)

    def stage():
        output = tmp_path / "stage"
        manifest = staging.prepare(
            repo,
            meta,
            freeze,
            guard.digest(freeze),
            index,
            books,
            output,
            "fixture/source",
        )
        return output, manifest

    return SimpleNamespace(
        repo=repo,
        meta=meta,
        books=books,
        index=index,
        frozen=frozen,
        freeze=freeze,
        stage=stage,
        tmp=tmp_path,
        passages=passages,
    )


def extracted_fixture(output):
    payload = output / "inspected-payload"
    payload.mkdir()
    with zipfile.ZipFile(output / "upload/indexed-demo-payload.zip") as archive:
        archive.extractall(payload)
    (payload / "indexed-demo-manifest.json").write_bytes(
        (output / "upload/indexed-demo-manifest.json").read_bytes()
    )
    return payload


def test_successful_real_git_stager_exact_inventory(fixture):
    f = fixture
    output, manifest = f.stage()
    payload = extracted_fixture(output)
    guard.authenticate_payload(
        payload, manifest, guard.digest(payload / "indexed-demo-manifest.json")
    )
    assert manifest["package_parent"] == "fixture/package/1"
    assert all(
        guard.digest(payload / n) == v for n, v in f.frozen["source_sha256"].items()
    )
    assert not any("qa.csv" in n or "private-test" in n for n in manifest["files"])
    kernel = guard.read(output / "kernel-metadata.json")
    assert kernel["is_private"] and kernel["enable_gpu"] is False
    assert b"\r" not in (output / "run.py").read_bytes()
    with pytest.raises(ValueError, match="Preserve"):
        f.stage()


@pytest.mark.parametrize(
    "fault", ["incomplete", "selected", "source", "runtime", "prompts", "parent"]
)
def test_missing_actual_freeze_refuses_before_stage(fixture, fault):
    f = fixture
    if fault == "incomplete":
        joint = guard.read(f.meta / "joint-status.json")
        joint["complete"] = False
        write_json(f.meta / "joint-status.json", joint)
        f.frozen["joint_result_sha256"] = guard.digest(f.meta / "joint-status.json")
    elif fault == "selected":
        f.frozen["selected_arm"] = "reranked_standard"
    elif fault == "source":
        del f.frozen["source_sha256"][next(iter(f.frozen["source_sha256"]))]
    elif fault == "runtime":
        del f.frozen["index_environment"]["scipy"]
    elif fault == "prompts":
        f.frozen["prompts"] = ["new tuned prompt"]
    else:
        f.frozen["package_parent"] = "unversioned"
    write_json(f.freeze, f.frozen)
    with pytest.raises(ValueError):
        f.stage()
    assert not (f.tmp / "stage").exists()


@pytest.mark.parametrize("fault", ["source", "book", "index", "untracked"])
def test_source_assets_or_head_changed_before_output(fixture, fault):
    f = fixture
    if fault == "source":
        (f.repo / "src/indexed_demo_transport.py").write_bytes(b"changed")
    elif fault == "book":
        (f.books / "pride.txt").write_text("changed")
    elif fault == "index":
        f.index.write_bytes(b"changed")
    else:
        path = f.repo / "src/prepare_indexed_demo.py"
        path.write_bytes(path.read_bytes() + b"\n")
        f.frozen["source_sha256"]["src/prepare_indexed_demo.py"] = guard.digest(path)
        write_json(f.freeze, f.frozen)
    with pytest.raises(ValueError):
        f.stage()
    assert not (f.tmp / "stage").exists()


def synthetic_results(f, index_sha):
    outputs = []
    for question in guard.PROMPTS:
        books = []
        for p in f.passages:
            evidence = dict(
                citation=p["id"],
                start=p["start"],
                end=p["end"],
                text=p["text"],
                retrieval_score=0.1,
            )
            answer = dict(
                story=p["book"],
                source_sha256=guard.sha(
                    (f.books / (p["book"] + ".txt")).read_text().encode()
                ),
                selected_arm="bm25_windows",
                generator_weights_sha256=guard.sha(b"OPAQUE FIXTURE WEIGHTS"),
                answer="software fixture",
                retrieved_evidence=[
                    dict(
                        section=p["id"], start=p["start"], end=p["end"], text=p["text"]
                    )
                ],
            )
            books.append(
                dict(
                    book=p["book"],
                    abstained=False,
                    candidate_evidence=[evidence],
                    result=answer,
                )
            )
        outputs.append(
            dict(
                schema="exploratory-indexed-narrative-v1",
                question=question,
                scope="Exploratory software fixture",
                retrieval_mode="hybrid",
                candidate_limit_per_book=6,
                index_sha256=index_sha,
                books=books,
                cross_book_connections=[
                    dict(
                        left=f.passages[0]["id"],
                        right=f.passages[1]["id"],
                        cosine_similarity=0.2,
                    )
                ],
            )
        )
    return outputs


def rendered(f, monkeypatch):
    output, manifest = f.stage()
    inputs = f.tmp / "inputs"
    inputs.mkdir()
    mount = inputs / "source"
    mount.mkdir()
    # Installed SDK skips upload-root dataset-metadata.json; ZIP remains opaque.
    for path in (output / "upload").iterdir():
        assert path.is_file()
        if path.name != "dataset-metadata.json":
            (mount / path.name).write_bytes(path.read_bytes())
    payload = f.tmp / "work/indexed-demo-source"
    package = inputs / "package"
    package.mkdir()
    (package / "generator").mkdir()
    (package / "generator/model.safetensors").write_bytes(b"OPAQUE FIXTURE WEIGHTS")
    (package / "joint-selection.json").write_bytes(
        (f.meta / "joint-selection.json").read_bytes()
    )
    (package / "inference-contract.json").write_bytes(
        (f.meta / "inference-contract.json").read_bytes()
    )
    namespace = {
        "__file__": str(output / "run.py"),
        "__name__": "fixture_rendered_runner",
    }
    exec(
        compile((output / "run.py").read_bytes(), str(output / "run.py"), "exec"),
        namespace,
    )
    calls = []
    monkeypatch.setattr(
        namespace["subprocess"], "run", lambda args, check: calls.append(("pip", args))
    )
    namespace["runtime_state"] = lambda names: {
        "python": f.frozen["runtime"]["python"],
        "torch": f.frozen["runtime"]["torch"],
        "environment": f.frozen["runtime"]["environment"]
        | f.frozen["index_environment"],
    }
    fake_torch = SimpleNamespace(
        set_num_threads=lambda n: calls.append(("torch", n)),
        use_deterministic_algorithms=lambda x: calls.append(("deterministic", x)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    expected = synthetic_results(f, f.frozen["assets"]["assets/index.lrag"])

    def answer(question, *args, **kwargs):
        calls.append(("answer", question))
        return copy.deepcopy(expected[guard.PROMPTS.index(question)])

    monkeypatch.setitem(
        sys.modules,
        "src.models.indexed_narrative_reader",
        SimpleNamespace(indexed_answers=answer),
    )
    return namespace, inputs, f.tmp / "work", payload, package, calls


def test_actual_rendered_runner_happy_full_reload_and_receipts(fixture, monkeypatch):
    ns, inputs, work, payload, package, calls = rendered(fixture, monkeypatch)
    receipt = ns["execute"](inputs, work)
    assert (
        receipt["complete"]
        and receipt["reload_exact"]
        and receipt["test_accessed"] is False
    )
    assert receipt["backend"] == "lsa" and receipt["benchmark_scores"] is None
    assert (
        set(receipt["recorded_inference_runtime"]["environment"]) == guard.CORE_PACKAGES
    )
    assert set(receipt["prospective_index_environment"]) == guard.INDEX_PACKAGES
    assert (
        receipt["index_runtime_scope"]
        == "prospective_demo_dependencies_not_recorded_model_training_runtime"
    )
    assert sum(c[0] == "answer" for c in calls) == 4
    assert calls[0][0] == "pip"
    pins = calls[0][1][5:]
    assert set(pins) == {
        n + "==1.2.3" for n in guard.CORE_PACKAGES | guard.INDEX_PACKAGES
    }
    assert not any(pin.startswith("torch==") for pin in pins)
    assert len(guard.read(work / "indexed-demo-answers.json")) == 2


@pytest.mark.parametrize("fault", ["payload", "package", "runtime", "pip", "template"])
def test_rendered_preload_failure_order(fixture, monkeypatch, fault):
    ns, inputs, work, payload, package, calls = rendered(fixture, monkeypatch)
    if fault == "payload":
        (inputs / "source/indexed-demo-payload.zip").write_bytes(
            b"changed opaque archive"
        )
    elif fault == "package":
        (package / "generator/model.safetensors").write_bytes(b"changed")
    elif fault == "runtime":
        ns["runtime_state"] = lambda names: {"wrong": "runtime"}
    elif fault == "pip":

        def fail(*a, **kw):
            raise subprocess.CalledProcessError(1, "fixture")

        monkeypatch.setattr(ns["subprocess"], "run", fail)
    else:
        Path(ns["__file__"]).write_text("changed")
    with pytest.raises(ValueError):
        ns["execute"](inputs, work)
    assert not any(c[0] in {"torch", "answer"} for c in calls)
    if fault in {"payload", "package", "template"}:
        assert not calls
    else:
        assert (work / "indexed-demo-runtime.json").is_file()


@pytest.mark.parametrize("target", ["package", "archive"])
def test_rendered_postflight_tamper_never_complete(fixture, monkeypatch, target):
    ns, inputs, work, payload, package, calls = rendered(fixture, monkeypatch)
    fake = sys.modules["src.models.indexed_narrative_reader"]
    original = fake.indexed_answers
    count = []

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        count.append(1)
        if len(count) == 4:
            if target == "package":
                (package / "generator/model.safetensors").write_bytes(b"changed")
            else:
                (inputs / "source/indexed-demo-payload.zip").write_bytes(b"changed")
        return result

    fake.indexed_answers = change
    with pytest.raises(ValueError, match="package bytes|Original mounted"):
        ns["execute"](inputs, work)
    assert not (work / "indexed-demo-receipt.json").exists()


@pytest.mark.parametrize("fault", ["candidate", "citation", "arm", "link", "partial"])
def test_source_replay_rejects_outputs(fixture, fault):
    f = fixture
    meta, texts = guard.index_metadata(
        f.index, {b: f.books / (b + ".txt") for b in guard.BOOKS}
    )
    results = synthetic_results(f, f.frozen["assets"]["assets/index.lrag"])
    book = results[0]["books"][0]
    if fault == "candidate":
        book["candidate_evidence"][0]["start"] = 0
    elif fault == "citation":
        book["result"]["retrieved_evidence"][0]["section"] = f.passages[1]["id"]
    elif fault == "arm":
        book["result"]["selected_arm"] = "reranked_windows"
    elif fault == "link":
        results[0]["cross_book_connections"][0]["cosine_similarity"] = 7
    else:
        results.pop()
    with pytest.raises(ValueError):
        guard.audit_outputs(
            results,
            meta,
            texts,
            "bm25_windows",
            f.frozen["assets"]["assets/index.lrag"],
            guard.sha(b"OPAQUE FIXTURE WEIGHTS"),
        )


def test_last_source_tamper_before_any_project_import(fixture, monkeypatch):
    import builtins

    ns, inputs, work, payload, package, calls = rendered(fixture, monkeypatch)
    bundle = inputs / "source/indexed-demo-payload.zip"
    with zipfile.ZipFile(bundle) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    members["src/resumable_training.py"] = b"changed source"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
    manifest_path = inputs / "source/indexed-demo-manifest.json"
    manifest = guard.read(manifest_path)
    manifest["archive_sha256"] = guard.digest(bundle)
    write_json(manifest_path, manifest)
    ns["EXPECTED_MANIFEST_SHA256"] = guard.digest(manifest_path)
    imported = []
    original = builtins.__import__

    def watch(name, *args, **kwargs):
        if name == "src" or name.startswith("src."):
            imported.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", watch)
    with pytest.raises(ValueError, match="before source import"):
        ns["execute"](inputs, work)
    assert imported == [] and calls == []


def test_fresh_reload_answer_difference_refuses_completion(fixture, monkeypatch):
    ns, inputs, work, payload, package, calls = rendered(fixture, monkeypatch)
    fake = sys.modules["src.models.indexed_narrative_reader"]
    original = fake.indexed_answers
    count = []

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        count.append(1)
        if len(count) == 4:
            result["books"][0]["result"]["answer"] = "changed on fresh reload"
        return result

    fake.indexed_answers = changed
    with pytest.raises(ValueError, match="reload deterministic"):
        ns["execute"](inputs, work)
    assert not (work / "indexed-demo-receipt.json").exists()


def test_manifest_forbids_qa_extra_even_when_hash_bound(fixture):
    output, manifest = fixture.stage()
    payload = extracted_fixture(output)
    (payload / "forbidden.qa.csv").write_bytes(b"software fixture")
    manifest["files"]["forbidden.qa.csv"] = guard.sha(b"software fixture")
    write_json(payload / "indexed-demo-manifest.json", manifest)
    with pytest.raises(ValueError, match="Only exact demo"):
        guard.authenticate_payload(
            payload, manifest, guard.digest(payload / "indexed-demo-manifest.json")
        )


@pytest.mark.parametrize("fault", ["traversal", "symlink", "duplicate"])
def test_actual_runner_zip_safety_before_import_or_extraction(tmp_path, fault):
    source = REPOSITORY / "src/cloud/indexed_demo_runner.py"
    ns = {"__file__": str(source), "__name__": "zip_safety_fixture"}
    exec(compile(source.read_bytes(), str(source), "exec"), ns)
    mount = tmp_path / "inputs/source"
    mount.mkdir(parents=True)
    bundle = mount / "indexed-demo-payload.zip"
    name = "../escape.py" if fault == "traversal" else "src/fixture.py"
    with zipfile.ZipFile(bundle, "w") as archive:
        if fault == "symlink":
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, b"fixture")
        else:
            archive.writestr(name, b"fixture")
            if fault == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr(name, b"fixture")
    manifest_path = mount / "indexed-demo-manifest.json"
    write_json(
        manifest_path,
        {
            "archive_sha256": guard.digest(bundle),
            "files": {name: guard.sha(b"fixture")},
        },
    )
    ns["EXPECTED_MANIFEST_SHA256"] = guard.digest(manifest_path)
    with pytest.raises(
        ValueError, match="archive inventory|Unsafe demo archive member"
    ):
        ns["execute"](tmp_path / "inputs", tmp_path / "work")
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "escape.py").exists()
