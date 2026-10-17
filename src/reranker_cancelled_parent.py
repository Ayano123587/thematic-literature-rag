"""Metadata-only guard for the observed interrupted continuation /3.

This contract authenticates an incomplete optimizer boundary, never completion.
Native state bytes must be authenticated in the cloud before deserialization.
"""

from pathlib import Path

from src.reranker_continuation import digest, read, safe_file, PROVENANCE

PARENT = "aikyatansinha/portfolio-reranker-auth-continuation-v1/3"
KERNEL_ID = 139180918
SOURCE_SHA256 = "8486c79580612b57cb8ea9edd9ad099e0f9caf794e83e82e2b3259485b459206"
RUNTIME_SHA256 = "cf8a3e40aa20a6c97a6b9453d28fe9eda6f097120fe1ca45f0282f5ea2f80bf7"
POINTER_SHA256 = "b7bf0aaae7b191819f4aa0c5833154f9d53d6b38667957a14a3e800beca4e0b5"
BACKUP_SHA256 = "e9b90e2f91660c90f15699388a7ca5066a7502e179a86f60d8cb784c36228def"
STATE_SHA256 = (
    "bbb5822b3edf03a9dd25ba0fd6b7060871b035f402df06b51d154f3314c80db1",
    "3a03a1b2c4fabb4818e369c3de80d0b63e248f4ef904f401c1a44d46865279b1",
)
EPOCH_RECEIPTS = (
    "f91cc15947ebebc933fb5acbe3c6a803263dcfaefb0ff5e2a0a1d16f73be21be",
    "bf11c47848e8d767faf4a27e40e9e2c23995f95226e479987e4cb24bfbec0eb8",
)
PROGRESS = {"epoch": 2, "next_microstep": 3840, "loss_sum": 2230.544269273232}


def freeze_metadata(parent, runtime_path, version_proof, cancellation_proof):
    """Freeze actual small metadata; never read native model or optimizer files."""
    parent = Path(parent)
    proof = read(version_proof)
    meta = proof.get("metadata", {})
    if (
        meta.get("id") != KERNEL_ID
        or meta.get("currentVersionNumber") != 3
        or meta.get("ref") != PARENT.rsplit("/", 1)[0]
        or proof.get("source_sha256") != SOURCE_SHA256
        or read(cancellation_proof).get("result", {}).get("status")
        != "CANCEL_ACKNOWLEDGED"
    ):
        raise ValueError("Actual cancelled parent version/status proof differs")
    if digest(runtime_path) != RUNTIME_SHA256:
        raise ValueError("Actual parent runtime receipt differs")
    runtime = read(runtime_path)
    if runtime.get("actual") != runtime.get("expected") or runtime.get(
        "difference_keys"
    ):
        raise ValueError("Parent runtime was not restored exactly")
    inventory = {
        "provenance.json": PROVENANCE,
        "progress/latest.json": POINTER_SHA256,
        "progress/slot-0.json": POINTER_SHA256,
        "progress/slot-1.json": BACKUP_SHA256,
    }
    for name, expected in inventory.items():
        if digest(safe_file(parent, name)) != expected:
            raise ValueError("Actual interrupted parent metadata differs")
    identity = read(parent / "provenance.json")["identity"]
    pointer = read(parent / "progress/latest.json")
    if pointer.get("progress") != PROGRESS or pointer.get("slot") != 0:
        raise ValueError("Not the durable interrupted boundary")
    for slot, state_sha in enumerate(STATE_SHA256):
        receipt = read(parent / f"progress/slot-{slot}.json")
        if (
            receipt.get("identity") != identity
            or receipt.get("slot") != slot
            or receipt.get("schema") != 1
            or receipt.get("state_sha256") != state_sha
        ):
            raise ValueError("Interrupted optimizer identity differs")
        inventory[f"progress/slot-{slot}.pt"] = state_sha
    for epoch, expected in enumerate(EPOCH_RECEIPTS):
        name = f"epoch-{epoch}/receipt.json"
        if digest(safe_file(parent, name)) != expected:
            raise ValueError("Original epoch receipt differs")
        receipt = read(parent / name)
        if receipt.get("identity") != identity or receipt.get("epoch") != epoch:
            raise ValueError("Original epoch identity differs")
        inventory[name] = expected
        for filename, sha in receipt["files"].items():
            if Path(filename).name != filename or len(Path(filename).parts) != 1:
                raise ValueError("Unsafe snapshot name")
            inventory[f"epoch-{epoch}/{filename}"] = sha
    for forbidden in ("epoch-2", "training-status.json", "continuation-execution.json"):
        if (parent / forbidden).exists():
            raise ValueError("Interrupted parent unexpectedly claims final output")
    return {
        "schema": "reranker_cancelled_parent_v1",
        "parent": PARENT,
        "kernel_id": KERNEL_ID,
        "source_sha256": SOURCE_SHA256,
        "status": "CANCEL_ACKNOWLEDGED",
        "scientifically_complete": False,
        "identity": identity,
        "progress": PROGRESS,
        "files": inventory,
        "runtime_sha256": RUNTIME_SHA256,
        "version_proof_sha256": digest(version_proof),
        "cancellation_proof_sha256": digest(cancellation_proof),
        "test_accessed": False,
    }


def authenticate(parent, runtime_path, version_proof, cancellation_proof, contract):
    """Cloud-only gate: hash every native byte before caller loads/copies state."""
    if (
        freeze_metadata(parent, runtime_path, version_proof, cancellation_proof)
        != contract
    ):
        raise ValueError("Externally frozen interrupted-parent contract differs")
    for name, expected in contract["files"].items():
        if digest(safe_file(parent, name)) != expected:
            raise ValueError("Interrupted optimizer/model/tokenizer bytes differ")
    for folder in ("progress", "epoch-0", "epoch-1"):
        actual = {
            p.relative_to(parent).as_posix() for p in (Path(parent) / folder).iterdir()
        }
        expected = {name for name in contract["files"] if name.startswith(folder + "/")}
        if actual != expected:
            raise ValueError("Unlisted interrupted checkpoint entry")
    return contract["identity"]
