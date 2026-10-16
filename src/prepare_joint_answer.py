"""Stage only an actual frozen joint-answer artifact contract; never submit jobs."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

SOURCES = {
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
ROLES = {
    "generator_run",
    "reranker_run",
    "generator_selection",
    "reranker_selection",
    "baseline_directory",
    "context_directory",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def actual_sha(value):
    return (
        isinstance(value, str)
        and re.fullmatch("[0-9a-f]{64}", value) is not None
        and value != "0" * 64
    )


def cloud_contract(contract, root):
    cloud = contract.get("cloud")
    if (
        not isinstance(cloud, dict)
        or set(cloud) != {"runtime", "artifacts", "resume"}
        or set(cloud["artifacts"]) != ROLES
    ):
        raise ValueError("Actual full artifact mount contract required")
    runtime = cloud["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"python", "torch", "machine", "environment"}
        or set(runtime["environment"])
        != {"transformers", "tokenizers", "numpy", "huggingface-hub"}
        or any(
            not isinstance(v, str) or not v.strip()
            for v in (
                runtime["python"],
                runtime["torch"],
                runtime["machine"],
                *runtime["environment"].values(),
            )
        )
    ):
        raise ValueError("Exact actual runtime required")
    artifacts = list(cloud["artifacts"].values())
    if cloud["resume"] is not None:
        resume = cloud["resume"]
        if (
            set(resume) != {"artifact", "plan_file", "plan_sha256"}
            or not actual_sha(resume["plan_sha256"])
            or contract["files"].get(resume["plan_file"]) != resume["plan_sha256"]
        ):
            raise ValueError("Externally frozen resume plan required")
        plan = json.loads((Path(root) / resume["plan_file"]).read_bytes())
        count = plan.get("questions_completed")
        if (
            type(count) is not int
            or not 0 < count < 2372
            or count % 64
            or set(plan.get("shards_sha256", {}))
            != {f"shard-{i:05d}.json" for i in range(0, count, 64)}
            or any(not actual_sha(v) for v in plan["shards_sha256"].values())
            or not {
                "identity.json",
                "retrieval-plan.json",
                "joint-status.json",
            }.issubset(plan.get("files", {}))
            or any(not actual_sha(v) for v in plan["files"].values())
        ):
            raise ValueError("Complete externally hashed resume prefix required")
        artifacts.append(resume["artifact"])
    for artifact in artifacts:
        if (
            set(artifact) != {"kernel", "marker", "sha256", "directory"}
            or not re.fullmatch(r"[\w-]+/[\w-]+/[1-9][0-9]*", artifact["kernel"])
            or Path(artifact["marker"]).name != artifact["marker"]
            or not actual_sha(artifact["sha256"])
            or type(artifact["directory"]) is not bool
        ):
            raise ValueError("Actual versioned mount and marker identity required")
    for role, key in (
        ("generator_selection", "generator_selection_sha256"),
        ("reranker_selection", "reranker_selection_sha256"),
        ("context_directory", "context_identity_sha256"),
    ):
        if cloud["artifacts"][role]["sha256"] != contract[key]:
            raise ValueError("Mounted artifact differs from scientific freeze")
    return cloud


def prepare(input_root, frozen_inputs_sha256, output, dataset, kernel_id, license_file):
    root = Path(input_root)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh staging required")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit reviewed source before staging")
    from src.joint_answer_validation import authenticate_inputs

    contract = authenticate_inputs(root, frozen_inputs_sha256)
    cloud = cloud_contract(contract, root)
    if any(not re.fullmatch(r"[\w-]+/[\w-]+", v) for v in (dataset, kernel_id)):
        raise ValueError("Explicit canonical identities required")
    from src.full_validation_selection import validation_cohort, read
    from src.narrative_training import load_split

    data = root / "private-data"
    validation_cohort(data, read(root / "generator-protocol.json"))
    if len(load_split(data, "train")[1]) != 7883:
        raise ValueError("Full training source required")
    payload = {name: Path(name).read_bytes() for name in SOURCES}
    for name in contract["files"]:
        relative = Path(name)
        allowed = (
            name
            in {
                "joint-protocol.json",
                "generator-protocol.json",
                "reranker-protocol.json",
            }
            or name.startswith("private-artifacts/")
            and name.endswith(".json")
            or name == "private-data/manifest.json"
            or len(relative.parts) >= 3
            and relative.parts[:2]
            in (("private-data", "train"), ("private-data", "validation"))
            and name.endswith((".story.json", ".qa.csv"))
        )
        if not allowed:
            raise ValueError("Unapproved or test-containing private payload")
        if name in payload:
            raise ValueError("Private payload overlaps source")
        payload[name] = (root / name).read_bytes()
    payload["joint-frozen-inputs.json"] = (
        root / "joint-frozen-inputs.json"
    ).read_bytes()
    notice = Path(license_file).read_bytes()
    if not notice.strip():
        raise ValueError("Upstream license required")
    payload["NOTICE/FairytaleQA-LICENSE"] = notice
    output.mkdir(parents=True)
    upload = output / "dataset"
    upload.mkdir()
    rows = []
    flat_names = set()
    for name, raw in sorted(payload.items()):
        flat = name.replace("/", "__")
        if flat in flat_names:
            raise ValueError("Flattened payload collision")
        flat_names.add(flat)
        (upload / flat).write_bytes(raw)
        rows.append(
            {"path": name, "file": flat, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {
        "schema": "joint_answer_transport_v1",
        "files": rows,
        "frozen_inputs_sha256": frozen_inputs_sha256,
        "license_notice_sha256": hashlib.sha256(notice).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "test_QA_packaged": False,
        "weights_refitted": False,
    }
    manifest_path = upload / "joint-source-manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    template = payload["src/cloud/joint_answer_runner.py"].decode()
    token = "EXPECTED_MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One rendered runner binding required")
    (output / "run.py").write_bytes(
        template.replace(
            token, "EXPECTED_MANIFEST_SHA256 = " + repr(sha(manifest_path))
        ).encode()
    )
    (upload / "LICENSE").write_bytes(notice)
    (upload / "dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": dataset,
                "title": dataset.split("/")[1].replace("-", " "),
                "licenses": [{"name": "other"}],
            }
        )
    )
    mounts = sorted(
        {a["kernel"] for a in cloud["artifacts"].values()}
        | ({cloud["resume"]["artifact"]["kernel"]} if cloud["resume"] else set())
    )
    (output / "kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": kernel_id.split("/")[1].replace("-", " "),
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "kernel_sources": mounts,
            }
        )
    )
    (output / "preflight.json").write_bytes(
        encoded(
            {
                "manifest_sha256": sha(manifest_path),
                "runner_sha256": sha(output / "run.py"),
                "source_commit": manifest["source_commit"],
                "staged_only": True,
                "test_accessed": False,
            }
        )
    )
    return manifest
