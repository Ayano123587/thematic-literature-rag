"""Stage a fresh continuation using the exact already-executed science archive."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from src.reranker_continuation import PARENT, freeze_metadata

ORIGINAL_ARCHIVE_SHA256 = (
    "2c54262221e95fe74b8a0b4fde15a23b9024509e51c764d12d36331af70a67df"
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(original, parent, output, dataset):
    repository = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    if Path.cwd().resolve() != repository:
        raise ValueError("Stage from repository root")
    original, parent, output = map(Path, (original, parent, output))
    if output.exists():
        raise ValueError("Preserve existing stage")
    contract = freeze_metadata(parent)
    archive = original / "dataset-upload/payload.zip"
    if sha(archive.read_bytes()) != ORIGINAL_ARCHIVE_SHA256:
        raise ValueError("Original executed archive differs")
    with zipfile.ZipFile(archive) as handle:
        files = {name: handle.read(name) for name in handle.namelist()}
    original_manifest = json.loads(files.pop("full-payload-provenance.json"))
    if set(files) != set(original_manifest["files"]) or any(
        sha(raw) != original_manifest["files"][name] for name, raw in files.items()
    ):
        raise ValueError("Original executed payload identity differs")
    if any(
        "test" in Path(name).parts
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        for name in files
    ):
        raise ValueError("Unsafe or sealed source payload")
    for name, expected in contract["identity"]["source_sha256"].items():
        if sha(files["src/" + name]) != expected:
            raise ValueError("Original scientific source differs from checkpoint")
    new = (
        "src/reranker_continuation.py",
        "src/cloud/reranker_continuation_runner.py",
        "src/prepare_reranker_continuation.py",
    )
    for name in new:
        raw_source = Path(name).read_bytes()
        committed = subprocess.check_output(["git", "show", "HEAD:" + name])
        if raw_source != committed:
            raise ValueError("Staged transport source differs from committed HEAD")
        files[name] = raw_source
    files["frozen-continuation-inputs.json"] = (
        json.dumps(contract, indent=2) + "\n"
    ).encode()
    files["NOTICE/FairytaleQA-LICENSE"] = (
        original / "dataset-upload/LICENSE"
    ).read_bytes()
    manifest = {
        "schema": "reranker_continuation_manifest_v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "parent": PARENT,
        "original_archive_sha256": sha(archive.read_bytes()),
        "frozen_inputs_sha256": sha(files["frozen-continuation-inputs.json"]),
        "files": {name: sha(raw) for name, raw in files.items()},
        "test_accessed": False,
    }
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    files["reranker-continuation-manifest.json"] = raw
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
                "title": "Private full reranker continuation source",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )
    runner = (
        files["src/cloud/reranker_continuation_runner.py"]
        .decode()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(sha(raw)),
        )
    )
    (output / "run.py").write_text(runner, encoding="utf-8", newline="\n")
    metadata = {
        "id": "aikyatansinha/portfolio-reranker-auth-continuation-v1",
        "title": "Portfolio Reranker Auth Continuation V1",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [PARENT],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "preflight.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha(raw),
                "runner_sha256": sha((output / "run.py").read_bytes()),
                "archive_sha256": sha((upload / "payload.zip").read_bytes()),
                "parent": PARENT,
                "files": len(files),
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "parent", "output", "dataset"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    prepare(args.original, args.parent, args.output, args.dataset)
