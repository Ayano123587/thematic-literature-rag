"""Authenticate an audited policy, package exact weights, and prove fresh reload parity."""

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tempfile

EXPECTED_MANIFEST_SHA256 = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def locate(inputs, name, expected):
    root = Path(inputs).resolve()
    matches = []
    for path in Path(inputs).rglob(name):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("Unsafe mounted artifact")
        if any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError("Symlink mount ancestry")
        if digest(path) == expected:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("One exact mounted artifact required")
    return matches[0]


def safe(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe payload path")
    return p


def restore(inputs, workspace, program, expected=None):
    expected = expected or EXPECTED_MANIFEST_SHA256
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("External manifest pin required")
    marker = locate(inputs, "narrative-package-manifest.json", expected)
    manifest = read(marker)
    if (
        manifest.get("schema") != "narrative_package_transport_v1"
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Test-free package transport required")
    raw, flats = {}, set()
    for row in manifest["files"]:
        name, flat = row["path"], row["file"]
        safe(name)
        if safe(flat).name != flat or name in raw or flat in flats:
            raise ValueError("Overlapping transport path")
        path = marker.parent / flat
        if path.is_symlink() or not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError("Payload authentication before imports failed")
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Payload byte count differs")
        raw[name] = path.read_bytes()
        flats.add(flat)
    if {p.name for p in marker.parent.iterdir() if p.is_file()} != flats | {
        marker.name,
        "LICENSE",
    }:
        raise ValueError("Unexpected source mount inventory")
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    template = (
        raw["src/cloud/narrative_package_runner.py"].decode().replace("\r\n", "\n")
    )
    if (
        template.count(token) != 1
        or Path(program).read_bytes().replace(b"\r\n", b"\n")
        != template.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(expected), 1
        ).encode()
    ):
        raise ValueError("Rendered package runner differs")
    if (
        hashlib.sha256(raw["root-review.json"]).hexdigest()
        != manifest["root_review_sha256"]
    ):
        raise ValueError("Root package-only review differs")
    review = json.loads(raw["root-review.json"])
    recipe = json.loads(raw["recipe.json"])
    if hashlib.sha256(raw["recipe.json"]).hexdigest() != manifest["recipe_sha256"]:
        raise ValueError("Recipe differs")
    if (
        recipe["test_authorized"] is not False
        or recipe["selected_arm"] != "reranked_windows"
    ):
        raise ValueError("Validated package-only recipe required")
    # The exact externally reviewed receipt is carried verbatim; its digest, not
    # a self-asserted Boolean, grants this package-only operation.
    if (
        review.get("schema") != "root_rag_package_only_review_v1"
        or review.get("pass") is not True
        or review.get("proposal_sha256") != manifest["recipe_sha256"]
        or review.get("test_authorized") is not False
        or review.get("training_authorized") is not False
    ):
        raise ValueError("Exact package-only review required")
    for name, expected_source in review["source_files_verified"].items():
        if hashlib.sha256(raw["src/" + name]).hexdigest() != expected_source:
            raise ValueError("Root-reviewed inference source differs")
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError("Fresh private workspace required")
    for name, value in raw.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return manifest


def authenticate_checkpoint(inputs, specification):
    receipt_path = locate(inputs, "receipt.json", specification["receipt_sha256"])
    directory = receipt_path.parent
    receipt = read(receipt_path)
    files = specification["all_checkpoint_files"]
    if (
        receipt["files"] != files
        or receipt["weights_sha256"] != specification["weights_sha256"]
    ):
        raise ValueError("External checkpoint receipt differs")
    if {p.name for p in directory.iterdir()} != set(files) | {"receipt.json"}:
        raise ValueError("Checkpoint inventory differs")
    for name, expected in files.items():
        if (
            safe(name).name != name
            or (directory / name).is_symlink()
            or digest(directory / name) != expected
        ):
            raise ValueError("Native checkpoint authentication before imports failed")
    return directory


def check_runtime(expected):
    import torch

    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "environment": {
            k: importlib.metadata.version(k) for k in expected["environment"]
        },
    }
    if actual != expected:
        raise ValueError("Frozen runtime differs before model loading")
    return actual


def validate_citations(result, story):
    if (
        result["story"] != story["id"]
        or result["source_sha256"] != story["source_sha256"]
    ):
        raise ValueError("Reload answer source identity differs")
    sections = {s["id"]: s for s in story["sections"]}
    if not result["retrieved_evidence"]:
        raise ValueError("Reload answer has no source citations")
    for citation in result["retrieved_evidence"]:
        section = sections[citation["section"]]
        start, end = citation["start"], citation["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not section["start"] <= start < end <= section["end"]
            or story["text"][start:end] != citation["text"]
            or section["text"][start - section["start"] : end - section["start"]]
            != citation["text"]
        ):
            raise ValueError(
                "Reload citation differs from exact supplied source section"
            )


def validate_saved_parity(demonstration, saved, selected_arm):
    if (
        demonstration["id"] != saved["id"]
        or demonstration["result"]["story"] != saved["story"]
        or demonstration["result"]["selected_arm"] != selected_arm
        or demonstration["result"]["answer"] != saved["arms"][selected_arm]["answer"]
        or demonstration["result"]["retrieved_evidence"]
        != saved["arms"][selected_arm]["citations"]
    ):
        raise ValueError(
            "Packaged inference differs from audited selected-arm answer/citations"
        )


def main(
    inputs=Path("/kaggle/input"), working=Path("/kaggle/working"), program=__file__
):
    output = Path(working) / "narrative-production"
    output.mkdir(exist_ok=False)
    try:
        workspace = Path(tempfile.mkdtemp(prefix="narrative-package-")) / "source"
        manifest = restore(inputs, workspace, program)
        recipe = read(workspace / "recipe.json")
        # Both complete native inventories are checked before numerical imports,
        # package copying, tokenizer construction, or any model deserialization.
        generator = authenticate_checkpoint(inputs, recipe["generator"])
        reranker = authenticate_checkpoint(inputs, recipe["reranker"])
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "4"
        pins = recipe["runtime"]["environment"]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                *[f"{k}=={v}" for k, v in sorted(pins.items())],
            ],
            check=True,
        )
        runtime = check_runtime(recipe["runtime"])
        sys.path.insert(0, str(workspace))
        from src.freeze_narrative_package import freeze
        from src.full_validation_selection import validation_cohort
        from src.joint_answer_validation import (
            authenticate_inputs,
            content_sha,
            exact_prefix,
        )
        from src.models.frozen_narrative_reader import answer, authenticate, load
        from src.narrative_training import retrieve
        from transformers import AutoTokenizer
        import torch

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)
        frozen = workspace / "joint-inputs"
        authenticate_inputs(
            frozen, recipe["training_and_source"]["frozen_joint_inputs_sha256"]
        )
        stories, questions = validation_cohort(
            frozen / "private-data", read(frozen / "generator-protocol.json")
        )
        bm25 = {
            q["id"]: [
                stories[q["story"]]["sections"][i]["id"]
                for i in retrieve(q["question"], stories[q["story"]]["sections"])
            ]
            for q in questions
        }
        joint = workspace / "joint-results"
        reranked = {
            q["id"]: q["section_ids"]
            for q in read(joint / "retrieval-plan.json")["queries"]
        }
        tokenizer = AutoTokenizer.from_pretrained(
            generator, local_files_only=True, use_fast=True
        )
        package = output / "package"
        receipt = freeze(
            package,
            joint,
            recipe["validation"]["status_sha256"],
            stories,
            questions,
            bm25,
            reranked,
            tokenizer,
            generator,
            reranker,
            workspace / "generator-selection.json",
            frozen / "private-artifacts/reranker-selection.json",
            frozen / "joint-protocol.json",
            recipe["training_and_source"]["inference_source_sha256"],
        )
        contract_sha = receipt["contract_sha256"]
        saved_rows = exact_prefix(
            joint,
            2372,
            read(joint / "joint-status.json")["identity"],
            [q["id"] for q in questions],
        )
        # Fixed first two prescribed validation IDs illustrate serialization
        # parity; no scores, test records, or post-hoc prompt selection.
        demonstrations = []
        for _ in range(2):
            loaded = load(package, contract_sha, device="cpu")
            demonstrations.append(
                [
                    {
                        "id": q["id"],
                        "result": answer(q["question"], stories[q["story"]], loaded),
                    }
                    for q in questions[:2]
                ]
            )
            del loaded
            gc.collect()
        if demonstrations[0] != demonstrations[1]:
            raise ValueError("Fresh model reload answer/citation parity failed")
        for demonstration, question in zip(demonstrations[0], questions[:2]):
            validate_citations(demonstration["result"], stories[question["story"]])
        for version in demonstrations:
            for demonstration, saved in zip(version, saved_rows[:2]):
                validate_saved_parity(demonstration, saved, receipt["selected_arm"])
        authenticate(package, contract_sha)
        authenticate_checkpoint(inputs, recipe["generator"])
        authenticate_checkpoint(inputs, recipe["reranker"])
        check_runtime(recipe["runtime"])
        for row in manifest["files"]:
            if digest(workspace / row["path"]) != row["sha256"]:
                raise ValueError("Postflight source/input drift")
        execution = {
            "schema": "narrative_package_execution_v1",
            "complete": True,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "recipe_sha256": manifest["recipe_sha256"],
            "root_review_sha256": manifest["root_review_sha256"],
            "runtime": runtime,
            "contract_sha256": contract_sha,
            "selected_arm": receipt["selected_arm"],
            "full_validation_questions": 2372,
            "fresh_reload_parity": True,
            "audited_selected_arm_answer_citation_replay_exact": True,
            "illustrative_validation_ids": [q["id"] for q in questions[:2]],
            "demonstration_sha256": content_sha(demonstrations[0]),
            "test_accessed": False,
            "weights_refitted": False,
        }
        write(output / "reload-demonstrations.json", demonstrations[0])
        write(output / "package-execution.json", execution)
        (output / "COMPLETE").write_text(
            digest(output / "package-execution.json") + "\n"
        )
    except Exception as error:
        write(
            output / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "reason": str(error)[:2000],
                "test_accessed": False,
                "complete": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
