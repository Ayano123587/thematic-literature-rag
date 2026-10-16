"""Run only the frozen, selected-checkpoint paired context experiment."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time

EXPECTED_MANIFEST_SHA256 = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def restore(inputs, workspace, program, expected=None):
    expected = EXPECTED_MANIFEST_SHA256 if expected is None else expected
    if not isinstance(expected, str) or len(expected) != 64 or expected == "0" * 64:
        raise ValueError("Exact rendered context manifest pin required")
    matches = [
        p
        for p in Path(inputs).rglob("context-source-manifest.json")
        if sha(p) == expected
    ]
    if len(matches) != 1:
        raise ValueError("One exact context-comparison manifest required")
    path = matches[0]
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema") != "context_comparison_transport_v1"
        or manifest.get("dispatchable") is not True
        or manifest.get("test_QA_packaged") is not False
        or manifest.get("weights_refitted") is not False
    ):
        raise ValueError("Frozen test-free comparison transport required")
    mandatory = {
        "src/prepare_context_comparison.py",
        "src/cloud/context_comparison_runner.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/prepare_fairytale.py",
        "src/narrative_training.py",
        "src/full_generator_training.py",
        "src/resumable_training.py",
        "src/full_validation_selection.py",
        "src/reranker_training.py",
        "src/context_assembly.py",
        "src/context_validation.py",
        "docs/full-coverage-protocol.json",
        "docs/context-assembly-protocol.json",
        "frozen-comparison-inputs.json",
        "selected-generator.json",
        "private-data/manifest.json",
        "NOTICE/FairytaleQA-LICENSE",
    }
    files, flats = {}, set()
    for row in manifest["files"]:
        name, flat = row["path"], row["file"]
        relative = PurePosixPath(name)
        allowed = (
            name in mandatory
            or name.startswith("src/")
            and name.endswith(".py")
            or len(relative.parts) >= 3
            and relative.parts[:2]
            in (("private-data", "train"), ("private-data", "validation"))
            and name.endswith((".qa.csv", ".story.json"))
        )
        if (
            not allowed
            or name in files
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or Path(flat).name != flat
            or "\\" in flat
            or ":" in flat
            or flat in (".", "..")
            or flat in flats
        ):
            raise ValueError("Unsafe, repeated or test-containing comparison payload")
        source = (path.parent / flat).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError("Comparison source escaped its private dataset")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Frozen comparison source or QA changed")
        files[name] = raw
        flats.add(flat)
    if not mandatory <= files.keys():
        raise ValueError("Required comparison source or contract missing")
    if not files["NOTICE/FairytaleQA-LICENSE"].strip() or hashlib.sha256(
        files["NOTICE/FairytaleQA-LICENSE"]
    ).hexdigest() != manifest.get("license_notice_sha256"):
        raise ValueError("Upstream license notice changed")
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    template = files["src/cloud/context_comparison_runner.py"].decode()
    if (
        template.count(token) != 1
        or Path(program).read_bytes()
        != template.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(expected)
        ).encode()
    ):
        raise ValueError("Executed context runner differs from reviewed template")
    if (
        hashlib.sha256(files["frozen-comparison-inputs.json"]).hexdigest()
        != manifest["frozen_inputs_sha256"]
    ):
        raise ValueError("Frozen comparison contract changed")
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError("Fresh context workspace required")
    workspace.mkdir(parents=True)
    for name, raw in files.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return manifest


def locate(inputs, name, digest):
    matches = [p for p in Path(inputs).rglob(name) if sha(p) == digest]
    if len(matches) != 1:
        raise ValueError("One exact immutable backend artifact required")
    return matches[0]


def check_runtime(expected):
    import torch

    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "environment": {
            name: importlib.metadata.version(name) for name in expected["environment"]
        },
    }
    if actual != expected:
        raise ValueError("Selected baseline runtime differs before model loading")
    return actual


def completed_status(result):
    if (
        result.get("questions_required") != 2372
        or result.get("test_accessed") is not False
        or result.get("final_test_authorized") is not False
    ):
        raise ValueError("Full test-free comparison coverage required")
    if result.get("complete") is True:
        if result.get("questions_completed") != 2372 or not all(
            k in result
            for k in (
                "baseline",
                "candidate",
                "decision",
                "story_paired_F1_interval",
                "baseline_replay_exact",
            )
        ):
            raise ValueError("Incomplete paired comparison cannot promote")
    elif result.get("complete") is False:
        if (
            type(result.get("questions_completed")) is not int
            or not 0 < result["questions_completed"] < 2372
            or result["questions_completed"] % 64
            or any(
                k in result
                for k in (
                    "baseline",
                    "candidate",
                    "decision",
                    "story_paired_F1_interval",
                )
            )
        ):
            raise ValueError(
                "Partial comparison cannot contain promotion or aggregates"
            )
    else:
        raise ValueError("Explicit completed or partial comparison required")


def completed_shards(output, result):
    completed_status(result)
    expected = {
        f"shard-{start:05d}.json"
        for start in range(0, result["questions_completed"], 64)
    }
    shards = {p.name: sha(p) for p in sorted(Path(output).glob("shard-*.json"))}
    if set(shards) != expected:
        raise ValueError("Exact completed paired-shard prefix required")
    return shards


def main(
    inputs=Path("/kaggle/input"), working=Path("/kaggle/working"), program=__file__
):
    started = time.monotonic()
    root = Path(working) / "context-source"
    manifest = restore(inputs, root, program)
    os.chdir(root)
    sys.path.insert(0, str(root))
    from src.prepare_context_comparison import (
        authenticate,
        authenticate_resume,
        comparison_identity,
        SHARDS,
    )

    contract, chosen = authenticate(
        "frozen-comparison-inputs.json",
        manifest["frozen_inputs_sha256"],
        "selected-generator.json",
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            *[
                name + "==" + version
                for name, version in contract["runtime"]["environment"].items()
            ],
        ],
        check=True,
    )
    runtime = check_runtime(contract["runtime"])
    training = locate(
        inputs, "training-status.json", contract["training_status_sha256"]
    ).parent
    if sha(training / "provenance.json") != contract["training_provenance_sha256"]:
        raise ValueError("Actual complete training provenance changed")
    selection = locate(inputs, "selection.json", contract["selection_sha256"])
    if selection.read_bytes() != Path("selected-generator.json").read_bytes():
        raise ValueError("Mounted backend selection differs from frozen selection")
    baseline = locate(
        inputs, "validation-status.json", contract["baseline_status_sha256"]
    ).parent
    if (
        sha(baseline / "identity.json") != contract["baseline_identity_sha256"]
        or {p.name for p in baseline.glob("shard-*.json")} != set(SHARDS)
        or any(
            sha(baseline / n) != digest
            for n, digest in contract["baseline_shards_sha256"].items()
        )
    ):
        raise ValueError("Selected full-baseline artifact bytes changed")
    previous = None
    if contract["resume"]:
        previous = locate(
            inputs, "comparison-status.json", contract["resume"]["status_sha256"]
        ).parent
    authenticate_resume(
        previous, contract["resume"], comparison_identity(contract, chosen)
    )
    if time.monotonic() - started >= 7200:
        raise RuntimeError("Startup exhausted comparison service margin")
    from src.context_validation import run_comparison

    output = Path(working) / "context-comparison"
    result = run_comparison(
        "private-data",
        "docs/full-coverage-protocol.json",
        "docs/context-assembly-protocol.json",
        training,
        selection,
        baseline,
        output,
        previous=previous,
        max_seconds=min(37800, 39000 - (time.monotonic() - started)),
        shard_size=64,
    )
    completed_status(result)
    if json.loads((output / "identity.json").read_bytes()) != result[
        "identity"
    ] or result["identity"] != comparison_identity(contract, chosen):
        raise ValueError("Saved comparison identity differs")
    shards = completed_shards(output, result)
    proof = {
        "schema": "context_comparison_execution_v1",
        "source_commit": manifest["source_commit"],
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(program),
        "frozen_inputs_sha256": manifest["frozen_inputs_sha256"],
        "license_notice_sha256": manifest["license_notice_sha256"],
        "selection_sha256": contract["selection_sha256"],
        "weights_sha256": contract["weights_sha256"],
        "runtime": runtime,
        "complete": result["complete"],
        "questions_completed": result["questions_completed"],
        "questions_required": 2372,
        "identity_sha256": sha(output / "identity.json"),
        "retrieval_plan_sha256": sha(output / "retrieval-plan.json"),
        "status_sha256": sha(output / "comparison-status.json"),
        "shards_sha256": shards,
        "test_accessed": False,
        "final_test_authorized": False,
        "weights_refitted": False,
    }
    (Path(working) / "context-comparison-execution.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
