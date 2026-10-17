"""Stage full ranking validation only from actual complete continuation receipts."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

from src.reranker_validation_transport import (
    freeze_training_metadata,
    freeze_validation_prefix,
    validation_identity,
    VALIDATION_IDS,
    TRAINING_PARENT,
)

ORIGINAL_ARCHIVE_SHA256 = (
    "2c54262221e95fe74b8a0b4fde15a23b9024509e51c764d12d36331af70a67df"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(
    original,
    parent,
    execution_sha256,
    epoch,
    output,
    dataset,
    previous=None,
    previous_parent=None,
):
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    if Path.cwd().resolve() != root:
        raise ValueError("Stage from repository root")
    training = freeze_training_metadata(parent, execution_sha256)
    expected_identity = validation_identity(training, epoch)
    original, output = Path(original), Path(output)
    if output.exists():
        raise ValueError("Preserve prior prepared validation source")
    archive = original / "dataset-upload/payload.zip"
    if sha(archive.read_bytes()) != ORIGINAL_ARCHIVE_SHA256:
        raise ValueError("Original executed scientific archive differs")
    with zipfile.ZipFile(archive) as handle:
        files = {name: handle.read(name) for name in handle.namelist()}
    old = json.loads(files.pop("full-payload-provenance.json"))
    if set(files) != set(old["files"]) or any(
        sha(raw) != old["files"][name] for name, raw in files.items()
    ):
        raise ValueError("Original science/data bytes differ")
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or "test" in Path(name).parts
        for name in files
    ):
        raise ValueError("Unsafe source or sealed test payload")
    ids = []
    for name, raw in files.items():
        if name.startswith("private-data/validation/") and name.endswith(".qa.csv"):
            story = Path(name).name.removesuffix(".qa.csv")
            for ordinal, row in enumerate(
                csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))), 1
            ):
                ids.append(story + ":row" + str(ordinal) + ":" + row["question_id"])
    ids.sort()
    if len(ids) != 2372 or sha(json.dumps(ids).encode()) != VALIDATION_IDS:
        raise ValueError("Entire original validation cohort required")
    resume = None
    if previous is not None:
        if (
            not isinstance(previous_parent, str)
            or len(previous_parent.split("/")) != 3
            or not previous_parent.split("/")[-1].isdigit()
        ):
            raise ValueError("Actual versioned validation prefix mount required")
        resume = freeze_validation_prefix(previous, expected_identity, ids)
        if resume["complete"]:
            raise ValueError("Do not resubmit a complete validation epoch")
    elif previous_parent is not None:
        raise ValueError("Unexpected previous validation mount")
    for name, expected in training["identity"]["source_sha256"].items():
        if sha(files["src/" + name]) != expected:
            raise ValueError("Original scientific checkpoint source differs")
    paths = (
        "src/reranker_validation_transport.py",
        "src/cloud/reranker_validation_runner.py",
        "src/prepare_reranker_validation.py",
        "src/reranker_continuation.py",
        "src/cloud/reranker_continuation_runner.py",
    )
    for name in paths:
        raw = Path(name).read_bytes()
        if subprocess.check_output(["git", "show", "HEAD:" + name]) != raw:
            raise ValueError("Validation transport source differs from committed HEAD")
        files[name] = raw
    contract = {
        "schema": "full_reranker_validation_inputs_v1",
        "training": training,
        "epoch": epoch,
        "resume": resume,
        "previous_parent": previous_parent,
    }
    files["frozen-validation-inputs.json"] = (
        json.dumps(contract, indent=2) + "\n"
    ).encode()
    files["NOTICE/FairytaleQA-LICENSE"] = (
        original / "dataset-upload/LICENSE"
    ).read_bytes()
    manifest = {
        "schema": "full_reranker_validation_manifest_v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {name: sha(raw) for name, raw in files.items()},
        "training_parent": TRAINING_PARENT,
        "epoch": epoch,
        "test_accessed": False,
    }
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    files["reranker-validation-manifest.json"] = raw
    output.mkdir(parents=True)
    upload = output / "dataset-upload"
    upload.mkdir()
    with zipfile.ZipFile(upload / "payload.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        for name, content in files.items():
            handle.writestr(name, content)
    (upload / "LICENSE").write_bytes(files["NOTICE/FairytaleQA-LICENSE"])
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": "Private complete reranker validation source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n"
    )
    runner = (
        files["src/cloud/reranker_validation_runner.py"]
        .decode()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(raw)),
            1,
        )
    )
    (output / "run.py").write_text(runner, encoding="utf-8", newline="\n")
    metadata = {
        "id": "aikyatansinha/portfolio-full-reranker-validation-e" + str(epoch) + "-v1",
        "title": "Portfolio Full Reranker Validation E" + str(epoch) + " V1",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "kernel_sources": [TRAINING_PARENT]
        + ([previous_parent] if previous_parent else []),
        "competition_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "preflight.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha(raw),
                "runner_sha256": sha((output / "run.py").read_bytes()),
                "archive_sha256": sha((upload / "payload.zip").read_bytes()),
                "epoch": epoch,
                "training_parent": TRAINING_PARENT,
                "actual_training_execution_sha256": execution_sha256,
                "test_accessed": False,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "parent", "execution-sha256", "output", "dataset"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--previous")
    parser.add_argument("--previous-parent")
    args = parser.parse_args()
    prepare(
        args.original,
        args.parent,
        args.execution_sha256,
        args.epoch,
        args.output,
        args.dataset,
        args.previous,
        args.previous_parent,
    )
