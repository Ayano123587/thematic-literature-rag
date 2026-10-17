"""Metadata-only contracts and pre-model authentication for full ranking validation."""

import hashlib
import json
import math
from pathlib import Path
import re

TRAINING_PARENT = "aikyatansinha/portfolio-reranker-cancelled-resume-v1/2"
TRAINING_KERNEL_ID = 139224383
SCIENTIFIC_PARENT = "aikyatansinha/portfolio-reranker-auth-continuation-v1/3"
INITIAL_PROGRESS = {"epoch": 2, "next_microstep": 3840, "loss_sum": 2230.544269273232}
INITIAL_STATE = "bbb5822b3edf03a9dd25ba0fd6b7060871b035f402df06b51d154f3314c80db1"
FROZEN_INPUTS = "372030aa8a1ba80a97908e5986c5a6a0cdf982cb53670d34a57c94b9493734aa"
CONTINUATION_MANIFEST = (
    "4ea7656ed556e84c5308461dd82734a8ec44dbee6775c03dcc90df0c90a112a2"
)
CONTINUATION_RUNNER = "678f6b2386bee01c1f54eb36d5abdd602171a41529dc403b20d103e028d4d455"
VALIDATION_IDS = "40d2ddd6b065469a8b947d84dbb630efcaac83acecc695606022116cc1a0f7cb"
FINAL_PROGRESS = {"epoch": 3, "next_microstep": 0, "loss_sum": 0.0}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def file_at(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe validation artifact path")
    root = Path(root).resolve()
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Missing or unsafe validation artifact")
    return path


def hash_value(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        or value == "0" * 64
    ):
        raise ValueError("Actual externally frozen SHA256 required")
    return value


def freeze_training_metadata(root, execution_sha256):
    """Requires actual complete resumed /2 receipts; never reads local model tensors."""
    hash_value(execution_sha256)
    root = Path(root)
    execution_name = "full-reranker/continuation-execution.json"
    if digest(file_at(root, execution_name)) != execution_sha256:
        raise ValueError("Actual completed continuation execution differs")
    execution = read(root / execution_name)
    if (
        execution.get("schema") != "reranker_continuation_execution_v1"
        or execution.get("complete") is not True
        or execution.get("manifest_sha256") != CONTINUATION_MANIFEST
        or execution.get("runner_sha256") != CONTINUATION_RUNNER
        or execution.get("parent") != SCIENTIFIC_PARENT
        or execution.get("initial_progress") != INITIAL_PROGRESS
        or execution.get("initial_state_sha256") != INITIAL_STATE
        or execution.get("frozen_inputs_sha256") != FROZEN_INPUTS
        or execution.get("final_progress") != FINAL_PROGRESS
        or execution.get("test_accessed") is not False
    ):
        raise ValueError("Actual full original two-epoch continuation is required")
    status_name = "full-reranker/run/training-status.json"
    provenance_name = "full-reranker/run/provenance.json"
    runtime_name = "continuation-runtime.json"
    required = {
        execution_name: execution_sha256,
        status_name: execution["training_status_sha256"],
        provenance_name: execution["provenance_sha256"],
        runtime_name: execution["runtime_receipt_sha256"],
    }
    for name, expected in required.items():
        if digest(file_at(root, name)) != hash_value(expected):
            raise ValueError("Actual completion metadata bytes differ")
    status = read(root / status_name)
    provenance = read(root / provenance_name)
    runtime = read(root / runtime_name)
    identity = provenance["identity"]
    expected_runtime = {
        key: identity[key] for key in ("python", "torch", "environment")
    }
    if (
        status.get("status") != "training_complete"
        or status.get("progress") != FINAL_PROGRESS
        or status.get("epochs_required") != 2
        or status.get("train_QA") != 7883
        or status.get("smoke") is not False
        or status.get("test_accessed") is not False
    ):
        raise ValueError("No partial, smoke or capped training parent")
    if (
        runtime.get("actual") != expected_runtime
        or runtime.get("expected") != expected_runtime
        or runtime.get("difference_keys") != []
        or runtime.get("bootstrap_error") is not None
    ):
        raise ValueError("Actual original numerical runtime was not restored")
    if set(execution.get("snapshots", {})) != {"0", "1", "2"}:
        raise ValueError("Every actual epoch snapshot receipt is required")
    snapshots = {}
    for epoch in (0, 1, 2):
        name = f"full-reranker/run/epoch-{epoch}/receipt.json"
        expected = execution["snapshots"][str(epoch)]
        if digest(file_at(root, name)) != hash_value(expected["receipt_sha256"]):
            raise ValueError("Actual epoch receipt bytes differ")
        receipt = read(root / name)
        if (
            receipt.get("identity") != identity
            or receipt.get("epoch") != epoch
            or receipt.get("files") != expected["files"]
            or receipt.get("weights_sha256") != expected["weights_sha256"]
            or receipt.get("files", {}).get("model.safetensors")
            != receipt.get("weights_sha256")
        ):
            raise ValueError("Actual snapshot inventory differs")
        if not {
            "model.safetensors",
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        }.issubset(receipt["files"]):
            raise ValueError("Incomplete actual model/tokenizer inventory")
        required[name] = expected["receipt_sha256"]
        for filename, value in receipt["files"].items():
            if Path(filename).name != filename or len(Path(filename).parts) != 1:
                raise ValueError("Unsafe epoch inventory")
            required[f"full-reranker/run/epoch-{epoch}/{filename}"] = hash_value(value)
        snapshots[str(epoch)] = expected
    return {
        "schema": "full_reranker_validation_parent_v1",
        "parent": TRAINING_PARENT,
        "kernel_id": TRAINING_KERNEL_ID,
        "execution_sha256": execution_sha256,
        "identity": identity,
        "snapshots": snapshots,
        "files": required,
        "test_accessed": False,
    }


def authenticate_training(root, contract):
    """Cloud-only full-byte authentication; caller has not imported a model."""
    if (
        contract.get("schema") != "full_reranker_validation_parent_v1"
        or contract.get("parent") != TRAINING_PARENT
        or contract.get("test_accessed") is not False
    ):
        raise ValueError("Immutable complete training contract required")
    if freeze_training_metadata(root, contract["execution_sha256"]) != contract:
        raise ValueError("Externally frozen complete training metadata changed")
    for name, expected in contract["files"].items():
        if digest(file_at(root, name)) != expected:
            raise ValueError("Actual model/tokenizer/completion bytes changed")
    for epoch in (0, 1, 2):
        folder = Path(root) / f"full-reranker/run/epoch-{epoch}"
        if {p.name for p in folder.iterdir()} != set(
            contract["snapshots"][str(epoch)]["files"]
        ) | {"receipt.json"}:
            raise ValueError("Unlisted epoch checkpoint files")
    return contract["identity"]


def validation_identity(contract, epoch):
    if type(epoch) is not int or epoch not in (0, 1, 2):
        raise ValueError("Only initial and both declared trained epochs are eligible")
    return {
        "run": contract["identity"],
        "epoch": epoch,
        "weights_sha256": contract["snapshots"][str(epoch)]["weights_sha256"],
        "ids_sha256": VALIDATION_IDS,
        "shard_size": 64,
    }


def freeze_validation_prefix(folder, identity, expected_ids):
    """Validate full ordered prefix before calling the model-loading validator."""
    folder = Path(folder)
    if (
        len(expected_ids) != 2372
        or len(set(expected_ids)) != 2372
        or hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
        != VALIDATION_IDS
    ):
        raise ValueError("All prescribed validation IDs are required")
    status = read(file_at(folder, "validation-status.json"))
    if (
        read(file_at(folder, "identity.json")) != identity
        or status.get("identity") != identity
        or status.get("smoke") is not False
        or status.get("test_accessed") is not False
        or status.get("questions_required") != 2372
    ):
        raise ValueError("Changed validation continuation identity")
    shards = sorted(folder.glob("shard-*.json"))
    rows = []
    required = {
        name: digest(folder / name)
        for name in ("validation-status.json", "identity.json")
    }
    for index, path in enumerate(shards):
        if path.name != f"shard-{index*64:05d}.json" or path.is_symlink():
            raise ValueError("Noncontiguous validation prefix")
        shard = read(path)
        expected = expected_ids[index * 64 : index * 64 + 64]
        if (
            not expected
            or shard.get("identity") != identity
            or [row["id"] for row in shard["rows"]] != expected
            or hashlib.sha256(
                json.dumps(shard["rows"], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            != shard.get("rows_sha256")
        ):
            raise ValueError("Validation prefix row/hash/cohort drift")
        rows.extend(shard["rows"])
        for row in shard["rows"]:
            for metric in ("recall", "precision", "candidate_recall", "top4_ceiling"):
                value = row.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Nonfinite or invalid ranking prefix metric")
            selected = row.get("selected_sections")
            candidates = row.get("candidate_sections")
            if (
                not isinstance(selected, list)
                or not isinstance(candidates, list)
                or len(selected) != min(4, len(candidates))
                or len(set(selected)) != len(selected)
                or len(set(candidates)) != len(candidates)
                or len(candidates) > 12
                or not set(selected) <= set(candidates)
            ):
                raise ValueError("Invalid ranking prefix section geometry")
        required[path.name] = digest(path)
    if (
        not rows
        or len(rows) != status.get("questions_completed")
        or status.get("complete") is not (len(rows) == 2372)
    ):
        raise ValueError("Partial validation status contradicts exact prefix")
    if not status["complete"] and ("validation" in status or "bm25" in status):
        raise ValueError("Partial cohorts cannot expose final metrics")
    if {p.name for p in folder.iterdir()} != set(required):
        raise ValueError("Unlisted validation prefix artifact")
    return {
        "schema": "full_reranker_validation_prefix_v1",
        "identity": identity,
        "questions_completed": len(rows),
        "complete": status["complete"],
        "files": required,
        "inventory_sha256": hashlib.sha256(
            json.dumps(required, sort_keys=True).encode()
        ).hexdigest(),
    }


def authenticate_prefix(folder, frozen, expected_ids):
    if (
        frozen.get("schema") != "full_reranker_validation_prefix_v1"
        or freeze_validation_prefix(folder, frozen["identity"], expected_ids) != frozen
    ):
        raise ValueError("Externally frozen validation prefix changed")
    return frozen["questions_completed"]
