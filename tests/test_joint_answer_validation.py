"""Small software fixtures for joint validation transport; no benchmark scores."""

import hashlib
import json
from pathlib import Path

import pytest

from src.joint_answer_validation import (
    authenticate_inputs,
    bound_files,
    content_sha,
    exact_prefix,
)
from src.freeze_narrative_package import freeze


def shard(directory, start=0, count=64):
    rows = [{"id": str(i)} for i in range(start, start + count)]
    path = directory / f"shard-{start:05d}.json"
    path.write_text(
        json.dumps(
            {
                "identity": {"fixture": True},
                "pairs": rows,
                "pairs_sha256": content_sha(rows),
            }
        )
    )
    return path


def test_exact_resume_prefix(tmp_path):
    p = shard(tmp_path)
    inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}
    assert (
        len(
            exact_prefix(
                tmp_path,
                64,
                {"fixture": True},
                [str(i) for i in range(2372)],
                inventory,
            )
        )
        == 64
    )


@pytest.mark.parametrize(
    "change", ["gap", "extra", "id", "identity", "content", "external_sha", "count"]
)
def test_resume_prefix_poison(tmp_path, change):
    p = shard(tmp_path)
    inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}
    if change == "gap":
        p.rename(tmp_path / "shard-00064.json")
    if change == "extra":
        shard(tmp_path, 64)
    if change in ("id", "identity", "content"):
        value = json.loads(p.read_text())
        if change == "id":
            value["pairs"][0]["id"] = "wrong"
        if change == "identity":
            value["identity"] = {}
        if change == "content":
            value["pairs_sha256"] = "a" * 64
        p.write_text(json.dumps(value))
    if change == "external_sha":
        inventory[p.name] = "a" * 64
    with pytest.raises(ValueError):
        exact_prefix(
            tmp_path,
            63 if change == "count" else 64,
            {"fixture": True},
            [str(i) for i in range(2372)],
            inventory,
        )


def test_partial_package_refused_before_copy_or_model(tmp_path):
    p = tmp_path / "joint-status.json"
    p.write_text(
        json.dumps(
            {
                "complete": False,
                "questions_completed": 64,
                "questions_required": 2372,
                "test_accessed": False,
            }
        )
    )
    with pytest.raises(ValueError, match="Partial"):
        freeze(
            tmp_path / "package",
            tmp_path,
            hashlib.sha256(p.read_bytes()).hexdigest(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    assert not (tmp_path / "package").exists()


def test_missing_actual_freeze_refused(tmp_path):
    path = tmp_path / "joint-frozen-inputs.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        authenticate_inputs(tmp_path, "0" * 64)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_input_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        bound_files(tmp_path, {name: "a" * 64})


@pytest.mark.parametrize("change", ["mutated", "missing", "emptyfreeze"])
def test_postflight_immutable_inputs(tmp_path, change):
    from src.joint_answer_validation import immutable_inputs, check_immutable

    p = tmp_path / "receipt.json"
    p.write_text("fixture")
    expected = immutable_inputs([p])
    if change == "mutated":
        p.write_text("changed")
    if change == "missing":
        p.unlink()
    if change == "emptyfreeze":
        expected = {}
    with pytest.raises(ValueError):
        check_immutable(expected)


@pytest.mark.parametrize(
    "inventory", [None, {}, {"receipt.json": None}, {"receipt.json": "0" * 64}]
)
def test_missing_actual_hash_contract_refused(tmp_path, inventory):
    (tmp_path / "receipt.json").write_text("fixture")
    with pytest.raises(ValueError):
        bound_files(tmp_path, inventory)
