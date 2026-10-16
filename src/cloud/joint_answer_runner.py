"""Run only frozen joint answer validation; packaging and test evaluation are separate."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(inputs, marker, expected):
    matches = [
        p
        for p in Path(inputs).rglob(marker)
        if p.is_file() and not p.is_symlink() and sha(p) == expected
    ]
    if len(matches) != 1:
        raise ValueError("One exact mounted artifact required")
    return matches[0]


def restore(inputs, workspace, program, expected=None):
    expected = EXPECTED_MANIFEST_SHA256 if expected is None else expected
    if not isinstance(expected, str) or len(expected) != 64 or expected == "0" * 64:
        raise ValueError("Rendered manifest pin required")
    path = locate(inputs, "joint-source-manifest.json", expected)
    manifest = json.loads(path.read_bytes())
    if (
        manifest.get("schema") != "joint_answer_transport_v1"
        or manifest.get("test_QA_packaged") is not False
        or manifest.get("weights_refitted") is not False
    ):
        raise ValueError("Test-free frozen transport required")
    source_names = {
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/prepare_fairytale.py",
        "src/narrative_training.py",
        "src/full_generator_training.py",
        "src/full_reranker_training.py",
        "src/resumable_training.py",
        "src/full_validation_selection.py",
        "src/reranker_training.py",
        "src/context_assembly.py",
        "src/context_validation.py",
        "src/joint_answer_selection.py",
        "src/joint_answer_validation.py",
        "src/prepare_joint_answer.py",
        "src/cloud/joint_answer_runner.py",
    }
    mandatory = source_names | {
        "joint-frozen-inputs.json",
        "joint-protocol.json",
        "generator-protocol.json",
        "reranker-protocol.json",
        "private-data/manifest.json",
        "NOTICE/FairytaleQA-LICENSE",
    }
    files = {}
    flats = set()
    for row in manifest["files"]:
        name, flat = row["path"], row["file"]
        relative = PurePosixPath(name)
        allowed = (
            name in mandatory
            or name.startswith("private-artifacts/")
            and name.endswith(".json")
            or len(relative.parts) >= 3
            and relative.parts[:2]
            in (("private-data", "train"), ("private-data", "validation"))
            and name.endswith((".qa.csv", ".story.json"))
        )
        if (
            not allowed
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or name in files
            or Path(flat).name != flat
            or "\\" in flat
            or ":" in flat
            or flat in flats
            or flat in (".", "..")
        ):
            raise ValueError("Unsafe or overlapping payload")
        source = path.parent / flat
        if source.is_symlink() or not source.resolve().is_relative_to(
            path.parent.resolve()
        ):
            raise ValueError("Unsafe payload filesystem")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Payload SHA differs before imports")
        files[name] = raw
        flats.add(flat)
    if not mandatory <= files.keys():
        raise ValueError("Required transport files missing")
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    template = files["src/cloud/joint_answer_runner.py"].decode()
    if (
        template.count(token) != 1
        or Path(program).read_bytes()
        != template.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(expected)
        ).encode()
    ):
        raise ValueError("Rendered runner differs")
    if (
        hashlib.sha256(files["joint-frozen-inputs.json"]).hexdigest()
        != manifest["frozen_inputs_sha256"]
        or hashlib.sha256(files["NOTICE/FairytaleQA-LICENSE"]).hexdigest()
        != manifest["license_notice_sha256"]
        or not files["NOTICE/FairytaleQA-LICENSE"].strip()
    ):
        raise ValueError("Contract or upstream license differs")
    destination = Path(workspace)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Fresh source workspace required")
    destination.mkdir(parents=True)
    for name, raw in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    print(
        json.dumps(
            {
                "phase": "transport_verified_before_import",
                "manifest_sha256": expected,
                "payload_files_verified": len(files),
            }
        ),
        flush=True,
    )
    return manifest


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


def main(
    inputs=Path("/kaggle/input"), working=Path("/kaggle/working"), program=__file__
):
    workspace = Path(working) / "joint-source"
    manifest = restore(inputs, workspace, program)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.joint_answer_validation import authenticate_inputs, run
    from src.prepare_joint_answer import cloud_contract

    contract = authenticate_inputs(workspace, manifest["frozen_inputs_sha256"])
    cloud = cloud_contract(contract, workspace)
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
                for name, version in cloud["runtime"]["environment"].items()
            ],
        ],
        check=True,
    )
    runtime = check_runtime(cloud["runtime"])
    paths = {}
    for role, artifact in cloud["artifacts"].items():
        path = locate(inputs, artifact["marker"], artifact["sha256"])
        paths[role] = path.parent if artifact["directory"] else path
    previous = None
    plan = None
    plan_sha = None
    if cloud["resume"]:
        resume = cloud["resume"]
        artifact = resume["artifact"]
        marker = locate(inputs, artifact["marker"], artifact["sha256"])
        previous = marker.parent
        plan = workspace / resume["plan_file"]
        plan_sha = resume["plan_sha256"]
    output = Path(working) / "joint-answer"
    result = run(
        workspace,
        manifest["frozen_inputs_sha256"],
        workspace / "private-data",
        workspace / "generator-protocol.json",
        paths["generator_run"],
        paths["generator_selection"],
        paths["baseline_directory"],
        workspace / "reranker-protocol.json",
        paths["reranker_run"],
        paths["reranker_selection"],
        paths["context_directory"],
        output,
        previous=previous,
        resume_plan=plan,
        resume_plan_sha256=plan_sha,
    )
    if (
        result.get("questions_required") != 2372
        or result.get("test_accessed") is not False
        or result.get("final_test_authorized") is not False
        or result.get("complete") != (result.get("questions_completed") == 2372)
        or not result.get("complete")
        and "selection" in result
    ):
        raise ValueError("Incomplete result cannot promote")
    expected = {
        f"shard-{i:05d}.json" for i in range(0, result["questions_completed"], 64)
    }
    shards = {p.name: sha(p) for p in output.glob("shard-*.json")}
    if set(shards) != expected:
        raise ValueError("Exact paired prefix required")
    receipt = {
        "schema": "joint_answer_execution_v1",
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(program),
        "frozen_inputs_sha256": manifest["frozen_inputs_sha256"],
        "source_commit": manifest["source_commit"],
        "runtime": runtime,
        "payload_files_verified": len(manifest["files"]),
        "complete": result["complete"],
        "questions_completed": result["questions_completed"],
        "status_sha256": sha(output / "joint-status.json"),
        "shards_sha256": shards,
        "test_accessed": False,
        "package_frozen": False,
        "final_test_authorized": False,
    }
    (Path(working) / "joint-answer-execution.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
