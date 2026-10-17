"""Stage only the actual audited full-validation recipe; never submit a job."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def source_closure(repo):
    repo = Path(repo)
    pending = {
        "src/cloud/narrative_package_runner.py",
        "src/prepare_narrative_package.py",
        "src/narrative_cli.py",
    }
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Missing regular production module")
        found.add(name)
        for parent in path.relative_to(repo).parents:
            init = repo / parent / "__init__.py"
            if init.is_file():
                pending.add(init.relative_to(repo).as_posix())
        for node in ast.walk(ast.parse(path.read_bytes())):
            modules = []
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src")
            ):
                modules = [node.module]
            if isinstance(node, ast.Import):
                modules = [
                    alias.name for alias in node.names if alias.name.startswith("src.")
                ]
            for module in modules:
                relative = Path(*module.split("."))
                candidate = repo / relative.with_suffix(".py")
                if not candidate.is_file():
                    candidate = repo / relative / "__init__.py"
                if not candidate.is_file():
                    raise ValueError("Unresolved local source import")
                pending.add(candidate.relative_to(repo).as_posix())
    return sorted(found)


def prepare(
    repo,
    joint_stage,
    frozen_inputs,
    generator_selection,
    license_file,
    output,
    *,
    dataset,
    kernel_id,
    expected_root_review_sha256,
    expected_recipe_sha256,
):
    repo, stage, root, output = map(Path, (repo, joint_stage, frozen_inputs, output))
    if output.exists() or output.is_symlink():
        raise ValueError("Fresh immutable package stage required")
    if any(not re.fullmatch(r"[\w-]+/[\w-]+", value) for value in (dataset, kernel_id)):
        raise ValueError("Explicit canonical private identities required")
    review_path = stage / "root-package-only-review-v1.json"
    recipe_path = stage / "final-joint-recipe-proposal-v1.json"
    if (
        sha(review_path) != expected_root_review_sha256
        or sha(recipe_path) != expected_recipe_sha256
    ):
        raise ValueError("Actual externally approved recipe/review required")
    review, recipe = json.loads(review_path.read_bytes()), json.loads(
        recipe_path.read_bytes()
    )
    if (
        review.get("pass") is not True
        or review.get("proposal_sha256") != expected_recipe_sha256
        or review.get("test_authorized") is not False
        or review.get("training_authorized") is not False
    ):
        raise ValueError("Actual package-only root authorization required")
    for name, expected in review["source_files_verified"].items():
        if sha(repo / "src" / name) != expected:
            raise ValueError("Reviewed scientific/inference source changed")
    contract_path = root / "joint-frozen-inputs.json"
    if (
        sha(contract_path)
        != recipe["training_and_source"]["frozen_joint_inputs_sha256"]
    ):
        raise ValueError("Frozen complete source cohort differs")
    contract = json.loads(contract_path.read_bytes())
    payload = {name: (repo / name).read_bytes() for name in source_closure(repo)}
    for name, digest in contract["files"].items():
        if "test" in Path(name).parts or sha(root / name) != digest:
            raise ValueError("No test QA or changed source artifact admitted")
        payload["joint-inputs/" + name] = (root / name).read_bytes()
    payload["joint-inputs/joint-frozen-inputs.json"] = contract_path.read_bytes()
    result = stage / "terminal-output-v2"
    execution = json.loads((result / "joint-answer-execution.json").read_bytes())
    if (
        sha(result / "joint-answer-execution.json")
        != recipe["validation"]["execution_sha256"]
        or execution.get("complete") is not True
        or execution.get("questions_completed") != 2372
        or execution.get("test_accessed") is not False
    ):
        raise ValueError("Actual full joint result required")
    shard_names = {f"shard-{i:05d}.json" for i in range(0, 2372, 64)}
    if set(execution["shards_sha256"]) != shard_names:
        raise ValueError("All 38 full-cohort shards required")
    for name, digest in execution["shards_sha256"].items():
        path = result / "joint-answer" / name
        if sha(path) != digest:
            raise ValueError("Changed paired validation artifact")
    for path in (result / "joint-answer").iterdir():
        if path.name in shard_names | {
            "identity.json",
            "joint-status.json",
            "retrieval-plan.json",
        }:
            payload["joint-results/" + path.name] = path.read_bytes()
    if (
        sha(result / "joint-answer/joint-status.json")
        != recipe["validation"]["status_sha256"]
    ):
        raise ValueError("Final selection status changed")
    for name, key in (
        ("independent-full2372-joint-audit-v2.json", "independent_full_audit_sha256"),
        (
            "independent-reranked-plan-audit-v2.json",
            "independent_reranked_plan_audit_sha256",
        ),
    ):
        path = result / name
        if sha(path) != recipe["validation"][key]:
            raise ValueError("Independent full-cohort audit changed")
        payload["provenance/" + name] = path.read_bytes()
    if sha(generator_selection) != recipe["generator"]["selection_sha256"]:
        raise ValueError("Original generator selection changed")
    payload["generator-selection.json"] = Path(generator_selection).read_bytes()
    payload["root-review.json"] = review_path.read_bytes()
    payload["recipe.json"] = recipe_path.read_bytes()
    payload["provenance/joint-answer-execution.json"] = (
        result / "joint-answer-execution.json"
    ).read_bytes()
    payload["NOTICE/FairytaleQA-LICENSE"] = Path(license_file).read_bytes()
    if not payload["NOTICE/FairytaleQA-LICENSE"].strip():
        raise ValueError("Actual upstream license required")
    rows = [
        {
            "path": name,
            "file": name.replace("/", "__"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for name, raw in sorted(payload.items())
    ]
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Flat source names overlap")
    manifest = {
        "schema": "narrative_package_transport_v1",
        "files": rows,
        "root_review_sha256": expected_root_review_sha256,
        "recipe_sha256": expected_recipe_sha256,
        "test_accessed": False,
        "weights_refitted": False,
        "source_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_bytes_authority": "exact manifest inventory; new transport reviewed separately before publication",
    }
    manifest_raw = encoded(manifest)
    digest = hashlib.sha256(manifest_raw).hexdigest()
    template = (
        payload["src/cloud/narrative_package_runner.py"].decode().replace("\r\n", "\n")
    )
    token = "EXPECTED_" + "MANIFEST_SHA256 = None"
    if template.count(token) != 1:
        raise ValueError("One manifest assignment required")
    runner = template.replace(
        token, "EXPECTED_MANIFEST_SHA256 = " + repr(digest), 1
    ).encode()
    output.mkdir(parents=True)
    for name in ("payload", "kernel", "upload"):
        (output / name).mkdir()
    for row in rows:
        (output / "payload" / row["file"]).write_bytes(payload[row["path"]])
    (output / "payload/narrative-package-manifest.json").write_bytes(manifest_raw)
    (output / "payload/LICENSE").write_bytes(payload["NOTICE/FairytaleQA-LICENSE"])
    (output / "kernel/run.py").write_bytes(runner)
    parents = [recipe["generator"]["parent"], recipe["reranker"]["parent"]]
    (output / "kernel/kernel-metadata.json").write_bytes(
        encoded(
            {
                "id": kernel_id,
                "title": "Audited Narrative Production Package",
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset],
                "kernel_sources": parents,
                "competition_sources": [],
            }
        )
    )
    with zipfile.ZipFile(
        output / "upload/package-source.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted((output / "payload").iterdir()):
            archive.writestr(path.name, path.read_bytes())
    (output / "upload/dataset-metadata.json").write_bytes(
        encoded(
            {
                "id": dataset,
                "title": "Audited Narrative Package Frozen Source",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            }
        )
    )
    receipt = {
        "status": "staged_not_uploaded",
        "manifest_sha256": digest,
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "payload_files": len(rows),
        "expanded_remote_files": len(rows) + 2,
        "source_paths": source_closure(repo),
        "parents": parents,
        "no_test_access_or_native_host_load": True,
    }
    (output / "stage-receipt.json").write_bytes(encoded(receipt))
    return receipt
