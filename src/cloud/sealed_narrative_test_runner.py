"""Execute only one authorized frozen full325 policy evaluation, with failure markers."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

EXPECTED_MANIFEST_SHA256 = None


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _execute():
    manifests = list(Path("/kaggle/input").rglob("sealed-test-manifest.json"))
    if len(manifests) != 1 or digest(manifests[0]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unexpected sealed test source mount")
    payload = manifests[0].parent
    manifest = json.loads(manifests[0].read_bytes())

    def source_auth():
        if digest(manifests[0]) != EXPECTED_MANIFEST_SHA256:
            raise ValueError("Sealed manifest changed")
        for name, expected in manifest["files"].items():
            relative = Path(name)
            path = payload / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or path.is_symlink()
                or not path.is_file()
                or digest(path) != expected
            ):
                raise ValueError("Sealed source/input bytes changed")

    source_auth()
    rendered = (
        (payload / "src/cloud/sealed_narrative_test_runner.py")
        .read_text()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256),
            1,
        )
    )
    if Path(__file__).read_text() != rendered:
        raise ValueError("Sealed runner template differs")
    sys.path.insert(0, str(payload))
    from src.sealed_narrative_test import (
        authorize_freeze,
        authenticate_cohort,
        evaluate,
        sha,
    )

    frozen = authorize_freeze(
        payload / "package-metadata",
        payload / "root-freeze.json",
        manifest["root_freeze_sha256"],
    )
    ticket = json.loads((payload / "release-ticket.json").read_bytes())
    cohort = json.loads((payload / "cohort.json").read_bytes())
    if (
        ticket.get("schema") != "sealed_narrative_test_ticket_v1"
        or ticket.get("root_freeze_sha256") != manifest["root_freeze_sha256"]
        or ticket.get("package_contract_sha256") != frozen["package_contract_sha256"]
        or ticket.get("selected_arm") != frozen["selected_arm"]
        or ticket.get("package_parent") != frozen["package_parent"]
        or ticket.get("joint_result_sha256") != frozen["joint_result_sha256"]
        or ticket.get("cohort_sha256")
        != sha(json.dumps(cohort, sort_keys=True).encode())
    ):
        raise ValueError("Frozen release ticket differs")
    expected_sources = {
        "src/" + name: value
        for name, value in frozen["inference_source_sha256"].items()
    } | frozen["release_source_sha256"]
    if ticket.get("source_sha256") != expected_sources:
        raise ValueError("Externally frozen complete source map differs")
    for name, expected_sha in expected_sources.items():
        if digest(payload / name) != expected_sha:
            raise ValueError("Externally frozen source bytes differ")
    authenticate_cohort(payload / "private-test", cohort)
    packages = [
        p.parent
        for p in Path("/kaggle/input").rglob("inference-contract.json")
        if not p.resolve().is_relative_to(payload.resolve())
    ]
    if len(packages) != 1:
        raise ValueError("Exactly one actual immutable production package required")
    package = packages[0]
    from src.models.frozen_narrative_reader import authenticate

    authenticate(package, ticket["package_contract_sha256"])
    expected = frozen["runtime"]
    names = ("transformers", "tokenizers", "numpy", "huggingface-hub")
    if set(expected["environment"]) != set(names):
        raise ValueError("Exact frozen runtime inventory required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"

    def runtime_state():
        def version(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None

        return {
            "python": platform.python_version(),
            "torch": version("torch"),
            "machine": platform.machine(),
            "environment": {name: version(name) for name in names},
        }

    before = runtime_state()
    bootstrap_error = None
    command = [sys.executable, "-m", "pip", "install", "-q"] + [
        name + "==" + expected["environment"][name] for name in names
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        bootstrap_error = type(error).__name__
    actual = runtime_state()
    runtime_path = Path("/kaggle/working/sealed-runtime.json")
    runtime_path.write_text(
        json.dumps(
            {
                "expected": expected,
                "before": before,
                "actual": actual,
                "model_loaded": False,
                "bootstrap_error": bootstrap_error,
                "installed_pins": command[5:],
            },
            indent=2,
        )
        + "\n"
    )
    if bootstrap_error is not None:
        raise ValueError("Frozen inference runtime bootstrap failed")
    if actual != expected:
        raise ValueError("Exact frozen inference runtime differs")
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = Path("/kaggle/working/sealed-full325")
    result = evaluate(
        package,
        payload / "private-test",
        cohort,
        ticket,
        output,
        root_freeze=payload / "root-freeze.json",
        root_sha256=manifest["root_freeze_sha256"],
    )
    source_auth()
    authenticate(package, ticket["package_contract_sha256"])
    authenticate_cohort(payload / "private-test", cohort)
    if Path(__file__).read_text() != rendered:
        raise ValueError("Sealed runner changed")
    if {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "machine": platform.machine(),
        "environment": {name: importlib.metadata.version(name) for name in names},
    } != actual:
        raise ValueError("Runtime changed during sealed evaluation")
    if result["questions"] != 325 or result["complete"] is not True:
        raise ValueError("Incomplete test cannot be finalized")
    proof = {
        "schema": "sealed_narrative_test_execution_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": digest(__file__),
        "package_contract_sha256": ticket["package_contract_sha256"],
        "root_freeze_sha256": ticket["root_freeze_sha256"],
        "cohort_sha256": ticket["cohort_sha256"],
        "ticket_sha256": digest(payload / "release-ticket.json"),
        "runtime_receipt_sha256": digest(runtime_path),
        "result_sha256": digest(output / "test-result.json"),
        "full_questions": 325,
        "selected_arm": ticket["selected_arm"],
        "reload_questions": 325,
        "test_used_for_selection": False,
        "weights_refitted": False,
    }
    with (Path("/kaggle/working") / "sealed-test-execution.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(proof, handle, indent=2, allow_nan=False)


def main():
    try:
        return _execute()
    except Exception as error:
        folder = Path("/kaggle/working")
        folder.mkdir(parents=True, exist_ok=True)
        failure = folder / "sealed-test-failure.json"
        if not failure.exists():
            with failure.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "sealed_narrative_execution_failure_v1",
                        "exception_type": type(error).__name__,
                        "message": str(error)[:300],
                        "cloud_marker_preserved": (
                            folder / "sealed-full325/sealed-test-started.json"
                        ).is_file(),
                        "retry_authorized": False,
                    },
                    handle,
                    indent=2,
                )
        raise


if __name__ == "__main__":
    main()
