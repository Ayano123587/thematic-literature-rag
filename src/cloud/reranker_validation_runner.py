"""Validate one frozen reranker epoch on the complete 2,372-question cohort."""

import hashlib
import json
from pathlib import Path
import sys
import time

EXPECTED_MANIFEST_SHA256 = None


def validation_budget(started, now):
    """Reserve model setup/postflight time; retain normal shard-boundary pauses."""
    remaining = 39000 - (now - started)
    if remaining < 2400:
        raise ValueError("Insufficient model setup and checkpoint service margin")
    return min(36000, remaining - 1800)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    began = time.monotonic()
    candidates = list(Path("/kaggle/input").rglob("reranker-validation-manifest.json"))
    if len(candidates) != 1 or sha(candidates[0]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unexpected full validation source mount")
    payload = candidates[0].parent
    manifest = json.loads(candidates[0].read_bytes())

    def source_auth():
        if sha(candidates[0]) != EXPECTED_MANIFEST_SHA256:
            raise ValueError("Validation manifest changed during execution")
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
                raise ValueError("Changed full validation source/data")
        if (payload / "private-data/test").exists():
            raise ValueError("Sealed test cannot be mounted")

    source_auth()
    template = (payload / "src/cloud/reranker_validation_runner.py").read_text()
    rendered = template.replace(
        "EXPECTED_MANIFEST_SHA256 = None",
        "EXPECTED_MANIFEST_SHA256 = " + repr(EXPECTED_MANIFEST_SHA256),
        1,
    )
    if Path(__file__).read_text() != rendered:
        raise ValueError("Validation rendered template differs")
    sys.path.insert(0, str(payload))
    from src.reranker_validation_transport import (
        authenticate_training,
        authenticate_prefix,
        freeze_validation_prefix,
        validation_identity,
        read,
    )

    contract = read(payload / "frozen-validation-inputs.json")
    parents = [
        p.parent.parent
        for p in Path("/kaggle/input").rglob("continuation-execution.json")
    ]
    if len(parents) != 1:
        raise ValueError("Exactly the actual completed /3 parent is required")
    parent = parents[0]
    identity = authenticate_training(parent, contract["training"])
    for name, expected in identity["source_sha256"].items():
        if sha(payload / "src" / name) != expected:
            raise ValueError("Original scientific source changed")
    data = payload / "private-data"
    protocol = payload / "docs/full-coverage-protocol.json"
    if (
        sha(protocol) != identity["protocol_sha256"]
        or sha(data / "manifest.json") != identity["manifest_sha256"]
    ):
        raise ValueError("Original data/protocol identity changed")
    # These modules use only Python metadata/strings; no numerical model import.
    from src.full_validation_selection import (
        validation_cohort,
        recompute_rows,
        metric_equal,
    )

    policy = read(protocol)
    stories, questions = validation_cohort(data, policy)
    ids = [q["id"] for q in questions]
    expected_identity = validation_identity(contract["training"], contract["epoch"])
    previous = None
    if contract["resume"] is not None:
        matches = list(Path("/kaggle/input").rglob("validation-status.json"))
        if len(matches) != 1:
            raise ValueError("Exactly one externally frozen validation prefix required")
        previous = matches[0].parent
        authenticate_prefix(previous, contract["resume"], ids)
        if contract["resume"]["identity"] != expected_identity:
            raise ValueError("Resume epoch or checkpoint changed")
        previous_rows = []
        for path in sorted(previous.glob("shard-*.json")):
            previous_rows.extend(read(path)["rows"])
        recompute_rows(
            stories, questions[: len(previous_rows)], previous_rows, "reranker"
        )
    elif list(Path("/kaggle/input").rglob("validation-status.json")):
        raise ValueError("Unrequested prior validation mount")
    # Full parent/source/QA/prefix authentication precedes numerical imports.
    from src.cloud.reranker_continuation_runner import restore_runtime, runtime_versions

    runtime_receipt = restore_runtime(identity)
    from src.full_reranker_training import validate_epoch, summarize_rows
    from src.reranker_training import evaluate_sections

    budget = validation_budget(began, time.monotonic())
    output = Path("/kaggle/working/full-reranker-validation")
    result = validate_epoch(
        data,
        protocol,
        parent / "full-reranker/run",
        contract["epoch"],
        output,
        previous=previous,
        smoke=False,
        threads=4,
        shard_size=64,
        max_seconds=budget,
    )
    source_auth()
    if Path(__file__).read_text() != rendered:
        raise ValueError("Validation runner changed during execution")
    authenticate_training(parent, contract["training"])
    if runtime_versions() != {
        key: identity[key] for key in ("python", "torch", "environment")
    }:
        raise ValueError("Numerical runtime changed during validation")
    prefix = freeze_validation_prefix(output, expected_identity, ids)
    rows = []
    for path in sorted(output.glob("shard-*.json")):
        rows.extend(read(path)["rows"])
    recompute_rows(stories, questions[: len(rows)], rows, "reranker")
    if result["complete"]:
        metric_equal(result["validation"], summarize_rows(rows))
        metric_equal(result["bm25"], evaluate_sections(stories, questions))
    proof = {
        "schema": "full_reranker_validation_execution_v1",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": sha(__file__),
        "training_parent": contract["training"]["parent"],
        "training_execution_sha256": contract["training"]["execution_sha256"],
        "frozen_inputs_sha256": sha(payload / "frozen-validation-inputs.json"),
        "runtime_receipt_sha256": sha(runtime_receipt),
        "epoch": contract["epoch"],
        "complete": prefix["complete"],
        "questions_completed": prefix["questions_completed"],
        "questions_required": 2372,
        "prefix": prefix,
        "test_accessed": False,
        "weights_refitted": False,
        "seconds": time.monotonic() - began,
    }
    (Path("/kaggle/working") / "reranker-validation-execution.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
