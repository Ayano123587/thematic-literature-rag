"""CPU optimizer-boundary checkpoints with exact dropout RNG continuation.

Two alternating slots retain the last committed checkpoint while writing its
successor. Only completed receipts update the pointer. Epoch model snapshots
and measured histories are retained separately by the full-data runner.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def rng_state():
    numpy = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])]}


def restore_rng(value):
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    name, state, position, gaussian, cached = value["numpy"]
    np.random.set_state((name, np.asarray(state, dtype=np.uint32), position, gaussian, cached))


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def save_progress(directory, model, optimizer, identity, progress):
    """Call only after optimizer.step/zero_grad, never mid accumulation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / "latest.json"
    previous = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None
    if previous is not None and (previous.get("identity") != identity or type(previous.get("slot")) is not int
                                 or previous["slot"] not in (0, 1)):
        raise ValueError("Refuse to overwrite a different/malformed checkpoint lineage")
    slot = 1 - previous["slot"] if previous else 0
    if any(parameter.grad is not None and bool(parameter.grad.ne(0).any()) for parameter in model.parameters()):
        raise ValueError("Checkpoint must be at an optimizer boundary")
    bundle = {"schema": 1, "identity": identity, "progress": progress,
              "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()}
    if not finite_tensors(bundle):
        raise ValueError("Nonfinite checkpoint state")
    path = directory / f"slot-{slot}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    receipt = {"schema": 1, "slot": slot, "state_sha256": file_sha(path),
               "identity": identity, "progress": progress}
    atomic_json(directory / f"slot-{slot}.json", receipt)
    atomic_json(pointer, receipt)
    return receipt


def load_progress(directory, model, optimizer, identity):
    directory = Path(directory)
    receipt = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != 1 or type(receipt.get("slot")) is not int or receipt["slot"] not in (0, 1):
        raise ValueError("Invalid checkpoint pointer")
    if receipt.get("identity") != identity:
        raise ValueError("Resume source/model/policy/runtime identity changed")
    path = directory / f"slot-{receipt['slot']}.pt"
    if file_sha(path) != receipt.get("state_sha256"):
        raise ValueError("Changed resume checkpoint bytes")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("schema") != 1 or bundle.get("identity") != identity or bundle.get(
            "progress") != receipt.get("progress") or not finite_tensors(bundle):
        raise ValueError("Malformed or mismatched resume checkpoint")
    current = optimizer.state_dict()["param_groups"]
    restored = bundle["optimizer"]["param_groups"]
    if len(current) != len(restored) or any({k: v for k, v in a.items() if k != "params"} != {
            k: v for k, v in b.items() if k != "params"} for a, b in zip(current, restored)):
        raise ValueError("Resume optimizer configuration changed")
    model.load_state_dict(bundle["model"], strict=True)
    optimizer.load_state_dict(bundle["optimizer"])
    restore_rng(bundle["rng"])
    return bundle["progress"]


def epoch_order(size, seed, epoch):
    """Isolated generator avoids perturbing model/dropout RNG on resume."""
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def optimize_epoch(model, optimizer, batches, progress, accumulation, checkpoint,
                   *, checkpoint_every=128, deadline=None):
    """Resume only at complete accumulation groups; return False at wall limit.

    Batches are deterministic callables so skipped batches never consume RNG.
    Deadlines stop at a safe boundary and do not remove training examples.
    """
    start = progress["next_microstep"]
    if type(accumulation) is not int or accumulation < 1 or type(start) is not int or not 0 <= start <= len(batches):
        raise ValueError("Invalid training accumulation/resume position")
    if start != len(batches) and start % accumulation:
        raise ValueError("Resume position is inside an accumulation group")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("Invalid checkpoint frequency")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for index in range(start, len(batches)):
        loss = model(**batches[index]()).loss
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Nonfinite/scalar training loss required")
        progress["loss_sum"] += float(loss.detach())
        group_start = (index // accumulation) * accumulation
        divisor = min(accumulation, len(batches) - group_start)
        (loss / divisor).backward()
        boundary = (index + 1) % accumulation == 0 or index + 1 == len(batches)
        if boundary:
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Nonfinite training gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            progress["next_microstep"] = index + 1
            expired = deadline is not None and time.monotonic() >= deadline
            if expired or (index + 1) % checkpoint_every == 0 or index + 1 == len(batches):
                checkpoint(dict(progress))
            if expired and index + 1 < len(batches):
                return False
        if (index + 1) % 100 == 0:
            print(json.dumps({"epoch": progress["epoch"], "microstep": index + 1,
                              "mean_train_loss": progress["loss_sum"] / (index + 1)}), flush=True)
    return True
