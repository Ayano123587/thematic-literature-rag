"""Authenticate an externally frozen reranker optimizer-boundary continuation."""

import hashlib
import json
from pathlib import Path

PARENT = "aikyatansinha/portfolio-narrative-full-reranker-train-v6/1"
STATUS = "3186a2e65033ab78b395f7362bae2d0c30982c8d22918d9b96b253980f63bc08"
PROVENANCE = "17f86227f6724d3682c981592e3005bb2b275b9638aad08f3dd62ee32c7ad3bd"
POINTER = "3e7b4fa473da1392b994935adbe4b913b87c5f89934336a24790fb9bef2e7742"
ACTIVE_STATE = "24bd059207f098b81364b7498ab1f363a91ac6fa762556343de41b172c961fae"
PROGRESS = {"epoch": 2, "next_microstep": 1824, "loss_sum": 1080.1274504770408}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Unsafe continuation path")
    path = Path(root) / relative
    if path.is_symlink() or any(
        p.is_symlink() for p in path.parents if p != Path(root).parent
    ):
        raise ValueError("Symlink continuation payload")
    if not path.is_file():
        raise ValueError("Missing continuation payload")
    return path


def freeze_metadata(parent):
    """Read JSON metadata only; called locally before any cloud model operation."""
    parent = Path(parent)
    for name, expected in (
        ("training-status.json", STATUS),
        ("provenance.json", PROVENANCE),
        ("progress/latest.json", POINTER),
    ):
        if digest(safe_file(parent, name)) != expected:
            raise ValueError("Unexpected actual parent receipt")
    pointer = read(parent / "progress/latest.json")
    status = read(parent / "training-status.json")
    identity = read(parent / "provenance.json")["identity"]
    if (
        pointer.get("identity") != identity
        or pointer.get("progress") != PROGRESS
        or status.get("progress") != PROGRESS
        or status.get("status") != "checkpointed"
        or pointer.get("slot") != 0
        or pointer.get("state_sha256") != ACTIVE_STATE
    ):
        raise ValueError("Actual parent checkpoint boundary differs")
    inventory = {
        name: expected
        for name, expected in (
            ("training-status.json", STATUS),
            ("provenance.json", PROVENANCE),
            ("progress/latest.json", POINTER),
        )
    }
    for slot in (0, 1):
        name = f"progress/slot-{slot}.json"
        receipt = read(safe_file(parent, name))
        if (
            receipt.get("identity") != identity
            or receipt.get("slot") != slot
            or receipt.get("schema") != 1
        ):
            raise ValueError("Parent optimizer slot receipt differs")
        if slot == 0 and receipt != pointer:
            raise ValueError("Active pointer differs from slot receipt")
        inventory[name] = digest(parent / name)
        inventory[f"progress/slot-{slot}.pt"] = receipt["state_sha256"]
    for epoch in (0, 1):
        name = f"epoch-{epoch}/receipt.json"
        receipt = read(safe_file(parent, name))
        if (
            receipt.get("identity") != identity
            or receipt.get("epoch") != epoch
            or receipt.get("files", {}).get("model.safetensors")
            != receipt.get("weights_sha256")
        ):
            raise ValueError("Parent epoch snapshot receipt differs")
        inventory[name] = digest(parent / name)
        for filename, expected in receipt["files"].items():
            if len(Path(filename).parts) != 1 or Path(filename).name != filename:
                raise ValueError("Unsafe snapshot inventory")
            inventory[f"epoch-{epoch}/{filename}"] = expected
    return {
        "schema": "reranker_continuation_inputs_v1",
        "parent": PARENT,
        "identity": identity,
        "progress": PROGRESS,
        "files": inventory,
        "test_accessed": False,
    }


def authenticate(parent, contract):
    """Verify all optimizer/RNG/model bytes before caller imports or copies state."""
    if (
        contract.get("schema") != "reranker_continuation_inputs_v1"
        or contract.get("parent") != PARENT
        or contract.get("test_accessed") is not False
    ):
        raise ValueError("Invalid immutable continuation contract")
    if freeze_metadata(parent) != contract:
        raise ValueError("Externally frozen continuation metadata changed")
    for name, expected in contract["files"].items():
        if digest(safe_file(parent, name)) != expected:
            raise ValueError("Externally frozen optimizer/RNG/snapshot bytes changed")
    for folder in ("progress", "epoch-0", "epoch-1"):
        actual = {
            p.relative_to(parent).as_posix()
            for p in (Path(parent) / folder).iterdir()
            if p.is_file()
        }
        expected = {name for name in contract["files"] if name.startswith(folder + "/")}
        if actual != expected:
            raise ValueError("Unlisted continuation state file")
    return contract["identity"]
