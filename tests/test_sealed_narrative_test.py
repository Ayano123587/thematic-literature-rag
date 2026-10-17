"""Software fixtures only: no actual held-out questions or model tensors."""

import csv
import io
import json
import platform
from pathlib import Path

import pytest

import src.sealed_narrative_test as sealed


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"question_id,question,answer1\n1,a,b\n", ["1"]),
        (
            b'\xef\xbb\xbfquestion_id,question,answer1\r\n"1","a,\r\n""b""","x"\r\n',
            ["1"],
        ),
        (b"question_id,question,answer1\n1,\xff,\xfe\n", ["1"]),
        (b'question_id,answer1\n"id""quoted",x\n', ['id"quoted']),
        (b"question_id,question\n1,a\n1,b\n", ["1", "1"]),
        (b'question_id,question\n"  raw\nID  ",x\n,y\n', ["  raw\nID  ", ""]),
        (b"question_id,question\n" + b"x" * 201 + b",a\n", ["x" * 201]),
        (b"question_id,question\n\x01,a\n", ["\x01"]),
    ],
)
def test_id_scanner_decodes_only_metadata(raw, expected):
    assert sealed.question_ids_only(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"question_id,question\n\xff,a\n",
        b'question_id,question\n1,"truncated',
        b'question_id,question\n1,"quoted"junk\n',
        b'question_id,question\n1,x"y\n',
        b"question_id,question_id\n1,2\n",
        b"question_id,question\n1\n",
        b"question,answer\nx,y\n",
    ],
)
def test_id_scanner_rejects_structural_or_metadata_ambiguity(raw):
    with pytest.raises(ValueError):
        sealed.question_ids_only(raw)


def fixture_cohort(tmp_path, monkeypatch, count=325):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index, family in enumerate(sorted(sealed.TEST_FAMILIES)):
        name = "software-story-" + str(index)
        text = "fixture text"
        story = {
            "id": name,
            "text": text,
            "source_sha256": sealed.sha(text.encode()),
            "sections": [{"id": "1", "text": text, "start": 0, "end": len(text)}],
        }
        folder = data / "test"
        folder.mkdir(exist_ok=True)
        (folder / (name + ".story.json")).write_text(json.dumps(story))
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "question_id",
                "question",
                "answer1",
                "answer4",
                "cor_section",
                "attribute1",
                "ex-or-im1",
            ],
        )
        writer.writeheader()
        for row in range(count // 2 + (1 if index == 0 and count % 2 else 0)):
            writer.writerow(
                {
                    "question_id": str(row),
                    "question": "Software fixture question",
                    "answer1": "fixture",
                    "answer4": "",
                    "cor_section": "1",
                    "attribute1": "character",
                    "ex-or-im1": "explicit",
                }
            )
        raw = stream.getvalue().encode("utf-8-sig")
        (folder / (name + ".qa.csv")).write_bytes(raw)
        records.append(
            {
                "id": name,
                "origin": family,
                "split": "test",
                "source_sha256": story["source_sha256"],
                "qa_sha256": sealed.sha(raw),
            }
        )
    (data / "manifest.json").write_text(json.dumps({"stories": records}))
    monkeypatch.setattr(sealed, "FULL_MANIFEST", sealed.digest(data / "manifest.json"))
    frozen = {"package_contract_sha256": "a" * 64, "joint_result_sha256": "b" * 64}
    ledger = tmp_path / "ledger"
    reservation = sealed.reserve(ledger, frozen, "c" * 64)
    return data, ledger, reservation


def test_full325_metadata_reservation_and_tamper(tmp_path, monkeypatch):
    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch)
    cohort = sealed.cohort_metadata(data, reservation)
    assert cohort["questions"] == 325 and cohort["gold_values_decoded"] is False
    sealed.authenticate_cohort(data, cohort)
    path = next((data / "test").glob("*.qa.csv"))
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setattr(
        sealed,
        "question_ids_only",
        lambda raw: pytest.fail("Scanner called before source hash gate"),
    )
    with pytest.raises(ValueError, match="before metadata scan"):
        sealed.cohort_metadata(data, reservation)
    assert reservation.exists()


def test_repeated_raw_ids_keep_distinct_canonical_ordinals(tmp_path, monkeypatch):
    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch)
    path = sorted((data / "test").glob("*.qa.csv"))[0]
    rows = list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))))
    raw_id = '  shared\n"raw"  '
    rows[0]["question_id"] = raw_id
    rows[1]["question_id"] = raw_id
    rows[2]["question_id"] = ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode("utf-8-sig"))
    manifest = sealed.read(data / "manifest.json")
    name = path.name.removesuffix(".qa.csv")
    next(r for r in manifest["stories"] if r["id"] == name)["qa_sha256"] = (
        sealed.digest(path)
    )
    (data / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(sealed, "FULL_MANIFEST", sealed.digest(data / "manifest.json"))
    cohort = sealed.cohort_metadata(data, reservation)
    story = next(r for r in cohort["stories"] if r["id"] == name)
    assert story["row_ids"][:3] == [
        name + ":row1:" + raw_id,
        name + ":row2:" + raw_id,
        name + ":row3:",
    ]
    assert len(set(cohort["ids"])) == 325
    sealed.authenticate_cohort(data, cohort)
    story["row_ids"][1] = story["row_ids"][0]
    with pytest.raises(ValueError, match="question IDs differ"):
        sealed.authenticate_cohort(data, cohort)


def test_subset_cannot_be_released(tmp_path, monkeypatch):
    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch, 324)
    with pytest.raises(ValueError, match="325"):
        sealed.cohort_metadata(data, reservation)
    assert reservation.exists()


def test_persistent_dispatch_failure_consumes_once(tmp_path, monkeypatch):
    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch)
    cohort = sealed.cohort_metadata(data, reservation)
    ticket = {
        "reservation_sha256": sealed.digest(reservation),
        "package_contract_sha256": "a" * 64,
        "cohort_sha256": sealed.sha(json.dumps(cohort, sort_keys=True).encode()),
    }
    (ledger / "release-bound.json").write_text(json.dumps(ticket))
    calls = []

    def push():
        assert (ledger / "dispatch-consumed.json").exists()
        calls.append(1)
        raise RuntimeError("simulated submission transport failure")

    with pytest.raises(RuntimeError):
        sealed.consume_dispatch(ledger, ticket, push)
    with pytest.raises(FileExistsError):
        sealed.consume_dispatch(ledger, ticket, push)
    with pytest.raises(FileExistsError):
        sealed.reserve(
            ledger,
            {"package_contract_sha256": "a" * 64, "joint_result_sha256": "b" * 64},
            "c" * 64,
        )
    assert calls == [1]


@pytest.mark.parametrize("failure", [False, True])
def test_full325_selected_policy_reload_and_failure_marker(
    tmp_path, monkeypatch, failure
):
    import src.models.frozen_narrative_reader as reader

    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch)
    cohort = sealed.cohort_metadata(data, reservation)
    ticket = {
        "selected_arm": "bm25_windows",
        "schema": "sealed_narrative_test_ticket_v1",
        "questions": 325,
        "package_contract_sha256": "a" * 64,
        "root_freeze_sha256": "c" * 64,
        "cohort_sha256": sealed.sha(json.dumps(cohort, sort_keys=True).encode()),
    }
    output = tmp_path / "cloud-output"
    policy = {"selected_arm": "bm25_windows"}
    contract = {"files": {"generator/model.safetensors": "d" * 64}}
    calls = {"loads": 0, "answers": 0, "gold": 0}
    runtime = {
        "python": "fixture",
        "torch": "fixture",
        "machine": platform.machine(),
        "environment": {
            name: "fixture"
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        },
    }
    ticket["runtime"] = runtime
    ticket["source_sha256"] = {
        name: sealed.digest(Path(name))
        for name in (
            "src/sealed_narrative_test.py",
            "src/cloud/sealed_narrative_test_runner.py",
            "src/prepare_sealed_narrative_test.py",
        )
    }
    frozen = {
        "package_contract_sha256": "a" * 64,
        "joint_result_sha256": "b" * 64,
        "package_parent": "owner/actual-package/1",
        "selected_arm": "bm25_windows",
        "runtime": runtime,
        "inference_source_sha256": {},
        "release_source_sha256": dict(ticket["source_sha256"]),
    }
    ticket["joint_result_sha256"] = frozen["joint_result_sha256"]
    ticket["package_parent"] = frozen["package_parent"]
    monkeypatch.setattr(sealed, "authorize_freeze", lambda *args: frozen)
    monkeypatch.setattr(sealed, "runtime_now", lambda: runtime)
    monkeypatch.setattr(reader, "authenticate", lambda *args: (contract, policy))
    actual_parse = sealed.parse_gold

    def gold(*args):
        assert (output / "sealed-test-started.json").exists()
        calls["gold"] += 1
        return actual_parse(*args)

    monkeypatch.setattr(sealed, "parse_gold", gold)

    def load(*args):
        assert calls["gold"] == 1
        calls["loads"] += 1
        return None, None, None, policy, contract

    def answer(question, story, loaded):
        assert question == "Software fixture question"
        calls["answers"] += 1
        if failure and calls["answers"] == 4:
            raise RuntimeError("simulated model failure")
        return {
            "answer": "fixture",
            "story": story["id"],
            "source_sha256": story["source_sha256"],
            "selected_arm": "bm25_windows",
            "generator_weights_sha256": "d" * 64,
            "retrieved_evidence": [
                {
                    "section": "1",
                    "start": 0,
                    "end": len(story["text"]),
                    "text": story["text"],
                }
            ],
        }

    monkeypatch.setattr(reader, "load", load)
    monkeypatch.setattr(reader, "answer", answer)
    if failure:
        with pytest.raises(RuntimeError):
            sealed.evaluate(
                tmp_path / "package",
                data,
                cohort,
                ticket,
                output,
                root_freeze=tmp_path / "root-freeze.json",
                root_sha256="c" * 64,
            )
        with pytest.raises(FileExistsError):
            sealed.evaluate(
                tmp_path / "package",
                data,
                cohort,
                ticket,
                output,
                root_freeze=tmp_path / "root-freeze.json",
                root_sha256="c" * 64,
            )
        assert (
            calls["gold"] == 1
            and calls["loads"] == 1
            and not (output / "test-result.json").exists()
        )
    else:
        report = sealed.evaluate(
            tmp_path / "package",
            data,
            cohort,
            ticket,
            output,
            root_freeze=tmp_path / "root-freeze.json",
            root_sha256="c" * 64,
        )
        assert (
            report["questions"] == 325
            and report["reload_questions"] == 325
            and report["metrics"]["n"] == 325
            and report["metrics"]["f1"] == 1.0
        )
        assert calls == {"loads": 2, "answers": 650, "gold": 1}
        assert len(report["prediction_files"]) == 325


def test_gold_parser_requires_marker_before_any_gold_read(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sealed,
        "authenticate_cohort",
        lambda *args: pytest.fail("Inputs read before marker gate"),
    )
    with pytest.raises(FileNotFoundError):
        sealed.parse_gold(tmp_path, {}, tmp_path / "absent-marker", {})


@pytest.mark.parametrize(
    "change",
    ["root_release", "root_and_ticket_release", "ticket_source", "joint", "parent"],
)
def test_direct_api_root_release_and_ticket_closure_before_inputs(
    tmp_path, monkeypatch, change
):
    release = {name: sealed.digest(Path(name)) for name in sealed.RELEASE_SOURCES}
    runtime = {"fixture": "runtime"}
    frozen = {
        "package_contract_sha256": "a" * 64,
        "joint_result_sha256": "b" * 64,
        "package_parent": "owner/package/1",
        "selected_arm": "bm25_windows",
        "runtime": runtime,
        "inference_source_sha256": {},
        "release_source_sha256": dict(release),
    }
    ticket = {
        "schema": "sealed_narrative_test_ticket_v1",
        "questions": 325,
        "root_freeze_sha256": "c" * 64,
        "package_contract_sha256": "a" * 64,
        "joint_result_sha256": "b" * 64,
        "package_parent": "owner/package/1",
        "selected_arm": "bm25_windows",
        "runtime": runtime,
        "source_sha256": dict(release),
    }
    name = sorted(release)[0]
    if change in ("root_release", "root_and_ticket_release"):
        frozen["release_source_sha256"][name] = "e" * 64
    if change in ("ticket_source", "root_and_ticket_release"):
        ticket["source_sha256"][name] = "e" * 64
    if change == "joint":
        ticket["joint_result_sha256"] = "e" * 64
    if change == "parent":
        ticket["package_parent"] = "owner/package/2"
    monkeypatch.setattr(sealed, "authorize_freeze", lambda *args: frozen)
    monkeypatch.setattr(sealed, "runtime_now", lambda: runtime)
    monkeypatch.setattr(
        sealed,
        "authenticate_cohort",
        lambda *args: pytest.fail("Test inputs reached before root source/ticket gate"),
    )
    with pytest.raises(ValueError):
        sealed.evaluate(
            tmp_path,
            {},
            {},
            ticket,
            tmp_path / "output",
            root_freeze=tmp_path / "freeze",
            root_sha256="c" * 64,
        )


def package_metadata_fixture(tmp_path):
    metadata = tmp_path / "package-metadata"
    metadata.mkdir()
    source = tmp_path / "inference-src"
    source.mkdir()
    (source / "fixture.py").write_bytes(b"# Opaque inference source fixture\n")
    policy = {
        "schema": "joint_answer_selection_v1",
        "selected_arm": "bm25_windows",
        "validation_questions": 2372,
        "validation_ids_sha256": sealed.VALIDATION_IDS,
        "test_accessed": False,
    }
    (metadata / "joint-selection.json").write_text(json.dumps(policy))
    contract = {
        "schema": "frozen_narrative_inference_v1",
        "joint_result_sha256": "b" * 64,
        "source_sha256": {"fixture.py": sealed.digest(source / "fixture.py")},
        "files": {
            "joint-selection.json": sealed.digest(metadata / "joint-selection.json"),
            "generator/model.safetensors": "d" * 64,
        },
    }
    (metadata / "inference-contract.json").write_text(json.dumps(contract))
    runtime = {
        "python": "fixture",
        "torch": "fixture",
        "machine": platform.machine(),
        "environment": {
            name: "fixture"
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        },
    }
    frozen = {
        "schema": "root_frozen_narrative_test_release_v1",
        "joint_validation_questions": 2372,
        "joint_validation_ids_sha256": sealed.VALIDATION_IDS,
        "test_authorized": True,
        "test_previously_scored": False,
        "test_questions": 325,
        "test_families": sorted(sealed.TEST_FAMILIES),
        "test_manifest_sha256": sealed.FULL_MANIFEST,
        "package_parent": "owner/actual-package/1",
        "package_contract_sha256": sealed.digest(metadata / "inference-contract.json"),
        "joint_result_sha256": "b" * 64,
        "selected_arm": "bm25_windows",
        "inference_source_sha256": contract["source_sha256"],
        "release_source_sha256": {
            name: sealed.digest(Path(name)) for name in sealed.RELEASE_SOURCES
        },
        "runtime": runtime,
    }
    freeze = tmp_path / "root-freeze.json"
    freeze.write_text(json.dumps(frozen))
    return metadata, source, freeze, sealed.digest(freeze), frozen


@pytest.mark.parametrize(
    "change", ["none", "unfrozen", "subset", "wrong_policy", "wrong_hash"]
)
def test_actual_root_metadata_freeze_gate(tmp_path, change):
    metadata, source, freeze, root_sha, frozen = package_metadata_fixture(tmp_path)
    if change == "unfrozen":
        frozen["test_authorized"] = False
    elif change == "subset":
        frozen["joint_validation_questions"] = 64
    elif change == "wrong_policy":
        frozen["selected_arm"] = "reranked_windows"
    if change != "none":
        freeze.write_text(json.dumps(frozen))
        root_sha = sealed.digest(freeze)
    if change == "wrong_hash":
        root_sha = "e" * 64
    if change == "none":
        assert sealed.authorize_freeze(metadata, freeze, root_sha) == frozen
    else:
        with pytest.raises(ValueError):
            sealed.authorize_freeze(metadata, freeze, root_sha)


@pytest.mark.parametrize("license_failure", [False, True])
def test_stager_real_git_head_opaque_full325_archive(
    tmp_path, monkeypatch, license_failure
):
    import subprocess
    import zipfile
    import src.prepare_sealed_narrative_test as stager

    data, _, _ = fixture_cohort(tmp_path, monkeypatch)
    metadata, source, freeze, root_sha, frozen = package_metadata_fixture(tmp_path)
    license_bytes = b"Apache license software fixture\n"
    (metadata / "LICENSE").write_bytes(license_bytes)
    monkeypatch.setattr(stager, "LICENSE_SHA256", sealed.sha(license_bytes))
    if license_failure:
        (metadata / "LICENSE").write_bytes(b"changed license")
    repository = tmp_path / "repository"
    repository.mkdir()
    source_names = (
        "src/sealed_narrative_test.py",
        "src/cloud/sealed_narrative_test_runner.py",
        "src/prepare_sealed_narrative_test.py",
    )
    current = Path.cwd()
    for name in source_names:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((current / name).read_bytes())
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Software Fixture"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "Opaque transport software fixture"],
    ):
        subprocess.run(command, cwd=repository, check=True, capture_output=True)
    monkeypatch.chdir(repository)
    ledger = tmp_path / "dispatch-ledger"
    output = tmp_path / "release-stage"
    if license_failure:
        with pytest.raises(ValueError, match="license"):
            stager.prepare(
                source,
                metadata,
                freeze,
                root_sha,
                data,
                ledger,
                output,
                "owner/private",
            )
        assert not ledger.exists() and not output.exists()
        return
    ticket = stager.prepare(
        source, metadata, freeze, root_sha, data, ledger, output, "owner/private"
    )
    assert (
        ticket["questions"] == 325
        and (ledger / "release-reserved.json").exists()
        and (ledger / "release-bound.json").exists()
        and not (ledger / "dispatch-consumed.json").exists()
    )
    preflight = sealed.read(output / "preflight.json")
    assert preflight["gold_values_decoded"] is False and preflight[
        "runner_sha256"
    ] == sealed.digest(output / "run.py")
    kernel = sealed.read(output / "kernel-metadata.json")
    assert (
        kernel["kernel_sources"] == ["owner/actual-package/1"]
        and kernel["is_private"]
        and not kernel["enable_gpu"]
    )
    with zipfile.ZipFile(output / "dataset-upload/payload.zip") as archive:
        manifest = json.loads(archive.read("sealed-test-manifest.json"))
        assert (
            manifest["source_commit"]
            == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
        for name, expected in manifest["files"].items():
            assert sealed.sha(archive.read(name)) == expected
        assert archive.read("NOTICE/FairytaleQA-LICENSE") == license_bytes
        assert set(archive.namelist()) == set(manifest["files"]) | {
            "sealed-test-manifest.json"
        }
    assert b"\r" not in (output / "run.py").read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        stager.prepare(
            source,
            metadata,
            freeze,
            root_sha,
            data,
            ledger,
            tmp_path / "second-stage",
            "owner/second",
        )
    archive_path = output / "dataset-upload/payload.zip"
    reviewed_archive = archive_path.read_bytes()
    archive_path.write_bytes(reviewed_archive + b"changed")
    with pytest.raises(ValueError):
        stager.dispatch(
            output, ledger, lambda stage: pytest.fail("Unreviewed archive dispatched")
        )
    assert not (ledger / "dispatch-consumed.json").exists()
    archive_path.write_bytes(reviewed_archive)
    pushes = []

    def push(stage):
        assert (ledger / "dispatch-consumed.json").exists()
        pushes.append(stage)
        return "software-fixture-accepted"

    assert stager.dispatch(output, ledger, push) == "software-fixture-accepted"
    with pytest.raises(FileExistsError):
        stager.dispatch(output, ledger, push)
    assert pushes == [output]


@pytest.mark.parametrize(
    "failure", [None, "runtime", "pipfailure", "qa", "source", "postflight"]
)
def test_actual_rendered_cloud_runner_order_and_postflight(
    tmp_path, monkeypatch, failure
):
    import sys
    import types
    import shutil
    import src.cloud.sealed_narrative_test_runner as template_module

    data, ledger, reservation = fixture_cohort(tmp_path, monkeypatch)
    cohort = sealed.cohort_metadata(data, reservation)
    metadata, source, freeze, root_sha, frozen = package_metadata_fixture(tmp_path)
    inputs = tmp_path / "inputs"
    payload = inputs / "payload"
    package = inputs / "actual-package"
    working = tmp_path / "working"
    payload.mkdir(parents=True)
    package.mkdir()
    working.mkdir()
    shutil.copytree(data, payload / "private-test")
    shutil.copytree(metadata, payload / "package-metadata")
    shutil.copy2(
        metadata / "inference-contract.json", package / "inference-contract.json"
    )
    shutil.copy2(metadata / "joint-selection.json", package / "joint-selection.json")
    shutil.copy2(freeze, payload / "root-freeze.json")
    (payload / "cohort.json").write_text(json.dumps(cohort))
    ticket = {
        "schema": "sealed_narrative_test_ticket_v1",
        "root_freeze_sha256": root_sha,
        "package_contract_sha256": frozen["package_contract_sha256"],
        "selected_arm": frozen["selected_arm"],
        "package_parent": frozen["package_parent"],
        "cohort_sha256": sealed.sha(json.dumps(cohort, sort_keys=True).encode()),
        "runtime": frozen["runtime"],
        "questions": 325,
        "joint_result_sha256": frozen["joint_result_sha256"],
        "source_sha256": {
            "src/" + name: value
            for name, value in frozen["inference_source_sha256"].items()
        }
        | frozen["release_source_sha256"],
    }
    (payload / "release-ticket.json").write_text(json.dumps(ticket))
    template = Path(template_module.__file__).read_text()
    path = payload / "src/cloud/sealed_narrative_test_runner.py"
    path.parent.mkdir(parents=True)
    path.write_text(template)
    for name in sealed.RELEASE_SOURCES:
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    (payload / "src/fixture.py").write_bytes((source / "fixture.py").read_bytes())
    files = {
        p.relative_to(payload).as_posix(): sealed.digest(p)
        for p in payload.rglob("*")
        if p.is_file()
    }
    manifest = {"files": files, "root_freeze_sha256": root_sha}
    (payload / "sealed-test-manifest.json").write_text(json.dumps(manifest))
    manifest_sha = sealed.digest(payload / "sealed-test-manifest.json")
    rendered = template.replace(
        "EXPECTED_MANIFEST_SHA256 = None",
        "EXPECTED_MANIFEST_SHA256 = " + repr(manifest_sha),
        1,
    )
    script = tmp_path / "run.py"
    script.write_text(rendered)
    runner = types.ModuleType("actual_rendered_sealed_fixture")
    runner.__file__ = str(script)
    exec(compile(rendered, str(script), "exec"), runner.__dict__)
    real_path = Path

    def mapped(value):
        text = str(value)
        if text == "/kaggle/input":
            return inputs
        if text == "/kaggle/working":
            return working
        if text.startswith("/kaggle/working/"):
            return working / text.removeprefix("/kaggle/working/")
        return real_path(value)

    monkeypatch.setattr(runner, "Path", mapped)
    calls = []
    actual_authorize = sealed.authorize_freeze

    def authorize(*args):
        result = actual_authorize(*args)
        calls.append("root freeze authorized")
        return result

    def authenticate(folder, contract_sha):
        assert contract_sha == frozen["package_contract_sha256"]
        assert sealed.digest(Path(folder) / "inference-contract.json") == contract_sha
        calls.append("package authenticated")

    monkeypatch.setitem(
        sys.modules,
        "src.models.frozen_narrative_reader",
        types.SimpleNamespace(authenticate=authenticate),
    )
    actual_cohort = sealed.authenticate_cohort

    def check_cohort(*args):
        actual_cohort(*args)
        calls.append("cohort authenticated")

    def evaluate(package_arg, data_arg, cohort_arg, ticket_arg, output, **kwargs):
        assert calls[:5] == [
            "root freeze authorized",
            "cohort authenticated",
            "package authenticated",
            "pinned bootstrap",
            "torch configured",
        ]
        calls.append("evaluate called")
        output.mkdir()
        sealed.start_cloud_marker(output, ticket_arg)
        (output / "test-result.json").write_text(
            json.dumps({"questions": 325, "complete": True})
        )
        if failure == "postflight":
            target = next((payload / "private-test/test").glob("*.qa.csv"))
            target.write_bytes(target.read_bytes() + b"changed")
        return {"questions": 325, "complete": True}

    shim = types.SimpleNamespace(
        authorize_freeze=authorize,
        authenticate_cohort=check_cohort,
        evaluate=evaluate,
        sha=sealed.sha,
    )
    monkeypatch.setitem(sys.modules, "src.sealed_narrative_test", shim)

    def pip(command, check):
        assert calls == [
            "root freeze authorized",
            "cohort authenticated",
            "package authenticated",
        ]
        assert command[:5] == [sys.executable, "-m", "pip", "install", "-q"]
        assert command[5:] == [
            name + "==fixture"
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        ]
        assert check
        assert (
            runner.os.environ["TOKENIZERS_PARALLELISM"] == "false"
            and runner.os.environ["OMP_NUM_THREADS"] == "4"
        )
        calls.append("pinned bootstrap")
        if failure == "pipfailure":
            raise runner.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", pip)
    monkeypatch.setattr(runner.platform, "python_version", lambda: "fixture")
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda name: "wrong" if failure == "runtime" and name == "torch" else "fixture",
    )

    def set_threads(count):
        assert count == 4
        calls.append("torch configured")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            set_num_threads=set_threads,
            use_deterministic_algorithms=lambda enabled: enabled is True,
        ),
    )
    if failure == "qa":
        next((payload / "private-test/test").glob("*.qa.csv")).write_bytes(b"changed")
    if failure == "source":
        (payload / "src/prepare_sealed_narrative_test.py").write_bytes(
            b"# changed last source payload"
        )
    if failure:
        with pytest.raises(ValueError):
            runner.main()
        assert (working / "sealed-test-failure.json").exists() and not (
            working / "sealed-test-execution.json"
        ).exists()
        if failure in ("qa", "source", "runtime", "pipfailure"):
            assert "evaluate called" not in calls and "torch configured" not in calls
        if failure == "source":
            assert calls == []
        if failure == "pipfailure":
            diagnostic = sealed.read(working / "sealed-runtime.json")
            assert (
                diagnostic["bootstrap_error"] == "CalledProcessError"
                and diagnostic["model_loaded"] is False
                and diagnostic["expected"] == frozen["runtime"]
            )
            assert (ledger / "release-reserved.json").exists()
        if failure == "postflight":
            assert (working / "sealed-full325/sealed-test-started.json").exists()
    else:
        runner.main()
        proof = sealed.read(working / "sealed-test-execution.json")
        assert (
            proof["full_questions"] == 325
            and proof["reload_questions"] == 325
            and proof["runner_sha256"] == sealed.digest(script)
        )
