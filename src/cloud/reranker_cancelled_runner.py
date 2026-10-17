"""Resume only the externally authenticated cancelled /3 full reranker run."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

EXPECTED_MANIFEST_SHA256 = None

PACKAGES = ("transformers", "tokenizers", "numpy", "huggingface-hub")


def runtime_versions():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "environment": {name: version(name) for name in PACKAGES},
    }


def restore_runtime(identity):
    """Restore authenticated package pins before any numerical training import."""
    if set(identity["environment"]) != set(PACKAGES):
        raise ValueError("Incomplete original runtime package inventory")
    expected = {key: identity[key] for key in ("python", "torch", "environment")}
    before = runtime_versions()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    command = [sys.executable, "-m", "pip", "install", "-q"] + [
        name + "==" + identity["environment"][name] for name in PACKAGES
    ]
    receipt_path = Path("/kaggle/working") / "continuation-runtime.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_error = None
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        bootstrap_error = type(error).__name__
    actual = runtime_versions()
    differences = []
    for name in ("python", "torch"):
        if actual[name] != expected[name]:
            differences.append(name)
    differences.extend(
        "environment." + name
        for name in PACKAGES
        if actual["environment"][name] != expected["environment"][name]
    )
    diagnostic = {
        "schema": "reranker_continuation_runtime_v1",
        "expected": expected,
        "before": before,
        "actual": actual,
        "difference_keys": differences,
        "bootstrap_error": bootstrap_error,
        "installed_pins": command[5:],
        "test_accessed": False,
        "model_loaded": False,
    }
    receipt_path.write_text(json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")
    print(json.dumps(diagnostic, allow_nan=False), flush=True)
    if bootstrap_error is not None:
        raise ValueError("Original runtime bootstrap failed")
    if differences:
        raise ValueError(
            "Original numerical runtime differs: " + ", ".join(differences)
        )
    return receipt_path


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    began = time.monotonic()
    matches = list(Path("/kaggle/input").rglob("reranker-continuation-manifest.json"))
    if len(matches) != 1 or sha(matches[0]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unexpected continuation source mount")
    payload = matches[0].parent
    manifest = json.loads(matches[0].read_bytes())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = payload / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError("Changed continuation source/data bytes")
    if (payload / "private-data/test").exists():
        raise ValueError("Sealed test must not be present")
    rendered = (
        (payload / "src/cloud/reranker_cancelled_runner.py")
        .read_text()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256),
            1,
        )
    )
    if Path(__file__).read_text() != rendered:
        raise ValueError("Continuation runner template differs")
    sys.path.insert(0, str(payload))
    from src.reranker_cancelled_parent import authenticate
    from src.reranker_continuation import read

    contract = read(payload / "frozen-continuation-inputs.json")
    parents = [
        p.parent
        for p in Path("/kaggle/input").rglob("provenance.json")
        if (p.parent / "progress/latest.json").is_file()
        and sha(p) == contract["files"]["provenance.json"]
    ]
    if len(parents) != 1:
        raise ValueError("Expected one exact cancelled parent mount")
    parent = parents[0]
    auth_args = (
        parent,
        payload / "parent-observation/runtime.json",
        payload / "parent-observation/version.json",
        payload / "parent-observation/cancellation.json",
        contract,
    )
    identity = authenticate(*auth_args)
    runtime_receipt = restore_runtime(identity)
    for name, expected in identity["source_sha256"].items():
        if sha(payload / "src" / name) != expected:
            raise ValueError("Original scientific training source differs")
    protocol = payload / "docs/full-coverage-protocol.json"
    if (
        sha(protocol) != identity["protocol_sha256"]
        or sha(payload / "private-data/manifest.json") != identity["manifest_sha256"]
    ):
        raise ValueError("Original scientific data/protocol differs")
    # Authentication precedes importing training code, parent copy and torch.load.
    from src.full_reranker_training import train_full

    output = Path("/kaggle/working/full-reranker/run")
    output.parent.mkdir(exist_ok=True)
    remaining = 39000 - (time.monotonic() - began)
    if remaining < 600:
        raise ValueError("Insufficient continuation checkpoint margin")
    result = train_full(
        payload / "private-data",
        protocol,
        output,
        resume=parent,
        max_seconds=min(37800, remaining),
    )
    authenticate(*auth_args)
    for name, expected in manifest["files"].items():
        if sha(payload / name) != expected:
            raise ValueError("Continuation inputs changed during execution")
    final = read(output / "progress/latest.json")
    if final["identity"] != identity or result["progress"] != final["progress"]:
        raise ValueError("Final optimizer boundary differs")
    if sha(output / "progress" / f"slot-{final['slot']}.pt") != final["state_sha256"]:
        raise ValueError("Final optimizer/RNG bytes differ")
    complete = result["status"] == "training_complete"
    if complete and final["progress"]["epoch"] != 3:
        raise ValueError("Full three-epoch training is incomplete")
    snapshots = {}
    for epoch in range(3 if complete else 2):
        folder = output / f"epoch-{epoch}"
        receipt = read(folder / "receipt.json")
        if receipt["identity"] != identity or receipt["epoch"] != epoch:
            raise ValueError("Final snapshot identity differs")
        for name, expected in receipt["files"].items():
            if sha(folder / name) != expected:
                raise ValueError("Final snapshot bytes differ")
        snapshots[str(epoch)] = {
            "receipt_sha256": sha(folder / "receipt.json"),
            "weights_sha256": receipt["weights_sha256"],
            "files": receipt["files"],
        }
    proof = {
        "schema": "reranker_continuation_execution_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(__file__),
        "runtime_receipt_sha256": sha(runtime_receipt),
        "parent": contract["parent"],
        "initial_progress": contract["progress"],
        "initial_state_sha256": contract["files"]["progress/slot-0.pt"],
        "frozen_inputs_sha256": sha(payload / "frozen-continuation-inputs.json"),
        "training_status_sha256": sha(output / "training-status.json"),
        "provenance_sha256": sha(output / "provenance.json"),
        "final_pointer_sha256": sha(output / "progress/latest.json"),
        "final_state_sha256": final["state_sha256"],
        "final_progress": final["progress"],
        "complete": complete,
        "snapshots": snapshots,
        "test_accessed": False,
        "seconds": time.monotonic() - began,
    }
    (output.parent / "continuation-execution.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
