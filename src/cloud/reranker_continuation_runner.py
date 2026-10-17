"""Resume only the externally authenticated original full reranker run."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

EXPECTED_MANIFEST_SHA256 = None


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
        (payload / "src/cloud/reranker_continuation_runner.py")
        .read_text()
        .replace(
            "EXPECTED_MANIFEST_SHA256 = None",
            "EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256),
        )
    )
    if Path(__file__).read_text() != rendered:
        raise ValueError("Continuation runner template differs")
    sys.path.insert(0, str(payload))
    from src.reranker_continuation import authenticate, digest, read

    contract = read(payload / "frozen-continuation-inputs.json")
    parents = [
        p.parent
        for p in Path("/kaggle/input").rglob("training-status.json")
        if (p.parent / "progress/latest.json").is_file()
    ]
    if len(parents) != 1:
        raise ValueError("Expected one exact immutable parent")
    parent = parents[0]
    identity = authenticate(parent, contract)
    if platform.python_version() != identity["python"] or any(
        importlib.metadata.version(name) != version
        for name, version in identity["environment"].items()
    ):
        raise ValueError("Original numerical runtime differs")
    if importlib.metadata.version("torch") != identity["torch"]:
        raise ValueError("Original torch runtime differs")
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
    authenticate(parent, contract)
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
        raise ValueError("Full two-epoch training is incomplete")
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
