"""Package an audited joint answer policy and exact checkpoint inventories."""

import json
from pathlib import Path
import shutil

from src.full_validation_selection import digest, read
from src.joint_answer_selection import select_paired, sha256_value
from src.joint_answer_validation import (
    audit_joint,
    content_sha,
    exact_prefix,
    write,
    immutable_inputs,
    check_immutable,
)
from src.models.frozen_narrative_reader import authenticate


def freeze(
    output,
    joint_directory,
    expected_status_sha256,
    stories,
    questions,
    bm25_plan,
    reranked_plan,
    tokenizer,
    generator_checkpoint,
    reranker_checkpoint,
    generator_selection,
    reranker_selection,
    protocol,
    inference_source_sha256,
):
    """Cloud-only packaging after source rescoring; never train or reselect checkpoints."""
    directory = Path(joint_directory)
    if digest(directory / "joint-status.json") != sha256_value(expected_status_sha256):
        raise ValueError("Externally frozen complete joint result required")
    status = read(directory / "joint-status.json")
    if (
        status.get("complete") is not True
        or status.get("questions_completed") != 2372
        or status.get("questions_required") != 2372
        or status.get("test_accessed") is not False
    ):
        raise ValueError("Partial joint results cannot produce a production package")
    ids = [q["id"] for q in questions]
    rows = exact_prefix(directory, 2372, status["identity"], ids)
    audit_joint(stories, questions, rows, bm25_plan, reranked_plan, tokenizer)
    decision = select_paired(rows, ids, True)
    bindings = {
        "generator_selection_sha256": digest(generator_selection),
        "reranker_selection_sha256": digest(reranker_selection),
        "protocol_sha256": digest(protocol),
        "paired_validation_sha256": content_sha(rows),
    }
    decision["artifact_bindings"] = bindings
    if content_sha(status["selection"]) != content_sha(decision):
        raise ValueError("Recomputed joint policy differs")
    input_paths = [
        directory / "joint-status.json",
        Path(generator_selection),
        Path(reranker_selection),
        Path(protocol),
    ] + list(directory.glob("shard-*.json"))
    for folder in (Path(generator_checkpoint), Path(reranker_checkpoint)):
        if folder.exists():
            input_paths += [p for p in folder.iterdir() if p.is_file()]
    frozen_inputs = immutable_inputs(input_paths)
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Fresh package destination required")
    destination.mkdir(parents=True)
    for source, name in (
        (generator_selection, "generator-selection.json"),
        (reranker_selection, "reranker-selection.json"),
        (protocol, "joint-protocol.json"),
    ):
        shutil.copyfile(source, destination / name)
    write(destination / "joint-selection.json", decision)
    for kind, checkpoint in (
        ("generator", generator_checkpoint),
        ("reranker", reranker_checkpoint),
    ):
        if kind == "reranker" and not decision["selected_arm"].startswith("reranked"):
            continue
        checkpoint = Path(checkpoint)
        receipt = read(checkpoint / "receipt.json")
        expected = receipt.get("files")
        if (
            not isinstance(expected, dict)
            or {p.name for p in checkpoint.iterdir()}
            != set(expected) | {"receipt.json"}
            or any(p.is_symlink() or not p.is_file() for p in checkpoint.iterdir())
        ):
            raise ValueError("Checkpoint has unbound filesystem entries")
        for name, sha in expected.items():
            if Path(name).name != name or digest(checkpoint / name) != sha256_value(
                sha
            ):
                raise ValueError("Checkpoint receipt changed")
        target = destination / kind
        target.mkdir()
        for name in [*expected, "receipt.json"]:
            shutil.copyfile(checkpoint / name, target / name)
    check_immutable(frozen_inputs)
    if digest(directory / "joint-status.json") != expected_status_sha256:
        raise ValueError("Joint status drifted")
    replayed_rows = exact_prefix(directory, 2372, status["identity"], ids)
    if content_sha(replayed_rows) != bindings["paired_validation_sha256"]:
        raise ValueError("Paired artifacts drifted")
    audit_joint(stories, questions, replayed_rows, bm25_plan, reranked_plan, tokenizer)
    contract = {
        "schema": "frozen_narrative_inference_v1",
        "files": {
            p.relative_to(destination).as_posix(): digest(p)
            for p in destination.rglob("*")
            if p.is_file()
        },
        "source_sha256": inference_source_sha256,
        "artifact_bindings": bindings,
        "joint_result_sha256": expected_status_sha256,
    }
    write(destination / "inference-contract.json", contract)
    contract_sha = digest(destination / "inference-contract.json")
    authenticate(destination, contract_sha)
    return {
        "contract_sha256": contract_sha,
        "selected_arm": decision["selected_arm"],
        "weights_refitted": False,
        "test_accessed": False,
    }
