"""Authenticated exploratory indexed-book demo; no benchmark scoring or fitting."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

EXPECTED_MANIFEST_SHA256 = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_state(names):
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "environment": {name: version(name) for name in names},
    }


def execute(inputs, work):
    inputs, work = Path(inputs), Path(work)
    manifests = list(inputs.rglob("indexed-demo-manifest.json"))
    if len(manifests) != 1 or digest(manifests[0]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unexpected externally authenticated demo mount")
    mount = manifests[0].parent
    manifest = json.loads(manifests[0].read_bytes())
    bundle = mount / "indexed-demo-payload.zip"
    if bundle.is_symlink() or digest(bundle) != manifest.get("archive_sha256"):
        raise ValueError("Demo opaque archive differs before source import")
    # Authenticate every member and safe path before extraction or project imports.
    files = {}
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        if (
            len({e.filename for e in entries}) != len(entries)
            or {e.filename for e in entries} != set(manifest["files"])
            or sum(e.file_size for e in entries) > 100_000_000
        ):
            raise ValueError("Unexpected demo archive inventory")
        for entry in entries:
            name = entry.filename
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or "\\" in name
                or ":" in name
                or entry.is_dir()
                or ((entry.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("Unsafe demo archive member")
            raw = archive.read(entry)
            if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
                raise ValueError("Demo payload differs before source import")
            files[name] = raw
    if digest(bundle) != manifest["archive_sha256"]:
        raise ValueError("Demo archive changed during authentication")
    payload = work / "indexed-demo-source"
    if payload.exists() or payload.is_symlink():
        raise ValueError("Fresh demo source extraction required")
    payload.mkdir(parents=True)
    for name, raw in files.items():
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (payload / "indexed-demo-manifest.json").write_bytes(manifests[0].read_bytes())
    rendered = (
        (payload / "src/cloud/indexed_demo_runner.py")
        .read_text()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256),
            1,
        )
    )
    if Path(__file__).read_text() != rendered:
        raise ValueError("Actual runner source differs")
    sys.path.insert(0, str(payload))
    from src.indexed_demo_transport import (
        BOOKS,
        PROMPTS,
        authenticate_package,
        authenticate_payload,
        at,
        authorize,
        audit_outputs,
        index_metadata,
    )

    authenticate_payload(payload, manifest, EXPECTED_MANIFEST_SHA256)
    frozen = authorize(
        payload / "package-metadata",
        payload / "root-freeze.json",
        manifest["root_freeze_sha256"],
    )
    if manifest["package_parent"] != frozen["package_parent"]:
        raise ValueError("Immutable package parent differs")
    for name, expected in frozen["source_sha256"].items():
        if digest(at(payload, name)) != expected:
            raise ValueError("Expanded frozen source differs")
    for name, expected in frozen["assets"].items():
        if digest(at(payload, name)) != expected:
            raise ValueError("Existing frozen demo asset differs")
    packages = [
        p.parent
        for p in inputs.rglob("inference-contract.json")
        if not p.resolve().is_relative_to(payload.resolve())
    ]
    if len(packages) != 1:
        raise ValueError("Exactly one actual globally frozen package required")
    package = packages[0]
    authenticate_package(package, frozen, payload / "package-metadata")
    metadata, texts = index_metadata(
        payload / "assets/index.lrag",
        {b: payload / "assets" / (b + ".txt") for b in BOOKS},
    )
    expected = {
        "python": frozen["runtime"]["python"],
        "torch": frozen["runtime"]["torch"],
        "environment": frozen["runtime"]["environment"] | frozen["index_environment"],
    }
    names = sorted(expected["environment"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    before = runtime_state(names)
    command = [sys.executable, "-m", "pip", "install", "-q"] + [
        n + "==" + expected["environment"][n] for n in names
    ]
    error = None
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        error = type(exc).__name__
    actual = runtime_state(names)
    work.mkdir(parents=True, exist_ok=True)
    (work / "indexed-demo-runtime.json").write_text(
        json.dumps(
            {
                "expected": expected,
                "before": before,
                "actual": actual,
                "bootstrap_error": error,
                "installed_pins": command[5:],
                "model_loaded": False,
            },
            indent=2,
        )
    )
    if error or actual != expected:
        raise ValueError("Exact authenticated inference/index runtime differs")
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from src.models.indexed_narrative_reader import indexed_answers

    def run_all():
        return [
            indexed_answers(
                q,
                payload / "assets/index.lrag",
                frozen["assets"]["assets/index.lrag"],
                payload / "assets",
                list(BOOKS),
                package,
                frozen["package_contract_sha256"],
                candidates=6,
                mode="hybrid",
                device="cpu",
            )
            for q in PROMPTS
        ]

    results = run_all()
    generator_sha = json.loads(
        (payload / "package-metadata/inference-contract.json").read_bytes()
    )["files"]["generator/model.safetensors"]
    audit_outputs(
        results,
        metadata,
        texts,
        frozen["selected_arm"],
        frozen["assets"]["assets/index.lrag"],
        generator_sha,
    )
    # Every call loads a fresh index/reader, so this repeats full deserialization/inference.
    replay = run_all()
    audit_outputs(
        replay,
        metadata,
        texts,
        frozen["selected_arm"],
        frozen["assets"]["assets/index.lrag"],
        generator_sha,
    )
    if replay != results:
        raise ValueError("Fresh reload deterministic output parity differs")
    if (
        digest(manifests[0]) != EXPECTED_MANIFEST_SHA256
        or digest(bundle) != manifest["archive_sha256"]
    ):
        raise ValueError("Original mounted demo archive/manifest changed")
    authenticate_payload(payload, manifest, EXPECTED_MANIFEST_SHA256)
    authorize(
        payload / "package-metadata",
        payload / "root-freeze.json",
        manifest["root_freeze_sha256"],
    )
    authenticate_package(package, frozen, payload / "package-metadata")
    if runtime_state(names) != expected:
        raise ValueError("Postflight runtime changed")
    raw = (
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    (work / "indexed-demo-answers.json").write_bytes(raw)
    receipt = {
        "complete": True,
        "schema": "indexed_demo_receipt_v1",
        "exploratory_only": True,
        "backend": "lsa",
        "retrieval_mode": "hybrid",
        "books": list(BOOKS),
        "prompts": list(PROMPTS),
        "reload_exact": True,
        "source_offsets_replayed": True,
        "weights_refitted": False,
        "test_accessed": False,
        "benchmark_scores": None,
        "source_commit": manifest["source_commit"],
        "source_sha256": frozen["source_sha256"],
        "asset_sha256": frozen["assets"],
        "package_parent": frozen["package_parent"],
        "contract_sha256": frozen["package_contract_sha256"],
        "joint_result_sha256": frozen["joint_result_sha256"],
        "selected_arm": frozen["selected_arm"],
        "answers_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime": actual,
        "recorded_inference_runtime": frozen["runtime"],
        "prospective_index_environment": frozen["index_environment"],
        "index_runtime_scope": frozen["index_runtime_scope"],
        "runtime_receipt_sha256": digest(work / "indexed-demo-runtime.json"),
    }
    (work / "indexed-demo-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    work = Path("/kaggle/working")
    try:
        execute(Path("/kaggle/input"), work)
    except Exception as error:
        work.mkdir(parents=True, exist_ok=True)
        (work / "indexed-demo-failure.json").write_text(
            json.dumps(
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "message": str(error)[:500],
                }
            )
        )
        raise


if __name__ == "__main__":
    main()
