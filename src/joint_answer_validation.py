"""Four-arm answer validation with authenticated paired caches and prefix resumes."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from src.context_validation import audit_pairs, evaluate_pairs, selected_checkpoint
from src.full_validation_selection import (
    FULL_VALIDATION_IDS_SHA256,
    completed_run,
    digest,
    metric_equal,
    read,
    snapshot_receipt,
)
from src.joint_answer_selection import (
    ARMS,
    adapted_checkpoint,
    select_paired,
    sha256_value,
)
from src.narrative_training import retrieve

PROTOCOL_SHA256 = "00493d88b452eecfba78de564ec4d9f4a9a9382fd9e5c60d4672eb808460b0c9"


def content_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bound_files(root, inventory):
    root = Path(root)
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Externally frozen artifact inventory required")
    for name, expected in inventory.items():
        p = Path(name)
        full = root / p
        if (
            p.is_absolute()
            or ".." in p.parts
            or not full.resolve().is_relative_to(root.resolve())
            or full.is_symlink()
            or digest(full) != sha256_value(expected)
        ):
            raise ValueError("Changed frozen validation input")


def exact_prefix(directory, count, identity, ids, inventory=None):
    """Authenticate only complete64-question shards, except the final4-question shard."""
    if (
        type(count) is not int
        or not 0 <= count <= 2372
        or (count != 2372 and count % 64)
    ):
        raise ValueError("Invalid completed prefix")
    directory = Path(directory)
    names = [f"shard-{i:05d}.json" for i in range(0, count, 64)]
    if {p.name for p in directory.glob("shard-*.json")} != set(names):
        raise ValueError("Missing or extra prefix shards")
    if inventory is not None:
        if set(inventory) != set(names):
            raise ValueError("Externally bound shard prefix differs")
        bound_files(directory, inventory)
    rows = []
    for start, name in zip(range(0, count, 64), names):
        shard = read(directory / name)
        pairs = shard["pairs"]
        if (
            shard.get("identity") != identity
            or [r["id"] for r in pairs] != ids[start : min(start + 64, count)]
            or shard.get("pairs_sha256") != content_sha(pairs)
        ):
            raise ValueError("Changed paired shard identity or rows")
        rows.extend(pairs)
    return rows


def authenticate_inputs(root, contract_sha256):
    root = Path(root)
    path = root / "joint-frozen-inputs.json"
    if digest(path) != sha256_value(contract_sha256):
        raise ValueError("Changed joint input freeze")
    contract = read(path)
    if (
        contract.get("schema") != "joint_answer_inputs_v1"
        or contract.get("test_accessed") is not False
        or contract.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
    ):
        raise ValueError("Full validation-only artifact freeze required")
    bound_files(root, contract["files"])
    for name, expected in contract["source_sha256"].items():
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or digest(Path(__file__).parent / p) != sha256_value(expected)
        ):
            raise ValueError("Changed joint scientific implementation")
    required = {
        "joint_answer_validation.py",
        "joint_answer_selection.py",
        "context_validation.py",
        "context_assembly.py",
        "full_validation_selection.py",
        "full_generator_training.py",
        "full_reranker_training.py",
        "reranker_training.py",
        "resumable_training.py",
        "narrative_training.py",
        "data/prepare_fairytale.py",
        "__init__.py",
        "data/__init__.py",
    }
    if (
        set(contract["source_sha256"]) != required
        or contract["files"].get("joint-protocol.json") != PROTOCOL_SHA256
    ):
        raise ValueError("Incomplete scientific source or changed joint protocol")
    for key in (
        "context_identity_sha256",
        "context_status_sha256",
        "context_plan_sha256",
        "generator_selection_sha256",
        "reranker_selection_sha256",
    ):
        sha256_value(contract[key])
    return contract


def audit_joint(stories, questions, rows, bm25_plan, reranked_plan, tokenizer):
    if [r["id"] for r in rows] != [q["id"] for q in questions]:
        raise ValueError("Joint coverage differs")
    for family, plan in (("bm25", bm25_plan), ("reranked", reranked_plan)):
        pairs = []
        for r in rows:
            if set(r["arms"]) != set(ARMS):
                raise ValueError("Four paired arms required")
            pairs.append(
                {
                    "id": r["id"],
                    "story": r["story"],
                    "selected_sections": r["selected_sections"][family],
                    "baseline": r["arms"][family + "_standard"],
                    "candidate": r["arms"][family + "_windows"],
                }
            )
        audit_pairs(stories, questions, pairs, plan, tokenizer)


def immutable_inputs(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Immutable input missing or symlinked")
        result[str(path.resolve())] = digest(path)
    return result


def check_immutable(expected):
    if not expected or immutable_inputs(expected) != expected:
        raise ValueError("Execution inputs drifted")


def run(
    root,
    contract_sha256,
    data,
    generator_protocol,
    generator_run,
    generator_selection,
    baseline_directory,
    reranker_protocol,
    reranker_run,
    reranker_selection,
    context_directory,
    output,
    *,
    previous=None,
    resume_plan=None,
    resume_plan_sha256=None,
    max_seconds=37800,
):
    """Cloud execution only. Unknown or incomplete frozen artifacts fail before loading."""
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not 0 <= max_seconds <= 39600
    ):
        raise ValueError("Safe cloud runtime window required")
    began = time.monotonic()
    contract = authenticate_inputs(root, contract_sha256)
    if (
        digest(generator_selection) != contract["generator_selection_sha256"]
        or digest(reranker_selection) != contract["reranker_selection_sha256"]
    ):
        raise ValueError("Selection differs from frozen contract")
    _, lineage, epoch, receipt, stories, questions, baseline = selected_checkpoint(
        data, generator_protocol, generator_run, generator_selection, baseline_directory
    )
    if (
        epoch != 3
        or receipt["weights_sha256"]
        != read(Path(root) / "joint-protocol.json")["generator_weights_sha256"]
    ):
        raise ValueError("Frozen epoch3 generator required")
    ranking_policy, ranking_lineage, _ = completed_run(
        reranker_run, data, reranker_protocol, "reranker"
    )
    chosen = adapted_checkpoint(read(reranker_selection))
    ranking_receipt = snapshot_receipt(reranker_run, chosen["epoch"], ranking_lineage)
    if ranking_receipt["weights_sha256"] != chosen["weights_sha256"]:
        raise ValueError("Trained reranker checkpoint differs")
    context = Path(context_directory)
    for filename, key in (
        ("identity.json", "context_identity_sha256"),
        ("comparison-status.json", "context_status_sha256"),
        ("retrieval-plan.json", "context_plan_sha256"),
    ):
        if digest(context / filename) != contract[key]:
            raise ValueError("Changed complete BM25 paired cache")
    status = read(context / "comparison-status.json")
    cache_identity = read(context / "identity.json")
    if (
        status.get("complete") is not True
        or status.get("questions_completed") != 2372
        or status.get("questions_required") != 2372
        or status.get("baseline_replay_exact") is not True
        or status.get("test_accessed") is not False
        or status.get("identity") != cache_identity
        or cache_identity.get("selected_epoch") != 3
        or cache_identity.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or cache_identity.get("context_protocol_sha256")
        != "83b3e611039695ba8a3dc6d01ee3edd7b00df8e1815c985ef46b56833fa44502"
        or cache_identity.get("weights_sha256") != receipt["weights_sha256"]
        or cache_identity.get("training") != lineage
        or cache_identity.get("selection_sha256") != digest(generator_selection)
    ):
        raise ValueError("Complete same-generator exact BM25 replay required")
    ids = [q["id"] for q in questions]
    cache = exact_prefix(
        context, 2372, cache_identity, ids, contract["context_shards_sha256"]
    )
    saved_plan = read(context / "retrieval-plan.json")
    if (
        saved_plan.get("identity") != cache_identity
        or saved_plan.get("queries_sha256") != content_sha(saved_plan["queries"])
        or [q["id"] for q in saved_plan["queries"]] != ids
    ):
        raise ValueError("Changed BM25 cache plan")
    bm25 = {r["id"]: r["section_ids"] for r in saved_plan["queries"]}
    for q in questions:
        sections = stories[q["story"]]["sections"]
        if bm25[q["id"]] != [
            sections[i]["id"] for i in retrieve(q["question"], sections)
        ]:
            raise ValueError("BM25 source-only retrieval changed")
    identity = {
        "contract_sha256": contract_sha256,
        "generator_weights_sha256": receipt["weights_sha256"],
        "reranker_weights_sha256": chosen["weights_sha256"],
        "validation_ids_sha256": FULL_VALIDATION_IDS_SHA256,
        "shard_size": 64,
        "arms": list(ARMS),
    }
    prefix = []
    prior_plan = None
    if previous is not None:
        if resume_plan is None or digest(resume_plan) != sha256_value(
            resume_plan_sha256
        ):
            raise ValueError("Externally frozen resume plan required")
        rp = read(resume_plan)
        if rp.get("identity") != identity:
            raise ValueError("Changed resume identity")
        if not {"identity.json", "retrieval-plan.json", "joint-status.json"}.issubset(
            rp["files"]
        ):
            raise ValueError("Resume metadata inventory incomplete")
        bound_files(previous, rp["files"])
        old_status = read(Path(previous) / "joint-status.json")
        if (
            read(Path(previous) / "identity.json") != identity
            or old_status.get("identity") != identity
            or old_status.get("questions_completed") != rp["questions_completed"]
            or old_status.get("questions_required") != 2372
            or old_status.get("complete") != (rp["questions_completed"] == 2372)
            or (old_status.get("complete") is not True and "selection" in old_status)
            or old_status.get("test_accessed") is not False
        ):
            raise ValueError("Resume status differs")
        prior_plan = read(Path(previous) / "retrieval-plan.json")
        prefix = exact_prefix(
            previous, rp["questions_completed"], identity, ids, rp["shards_sha256"]
        )
    elif resume_plan is not None or resume_plan_sha256 is not None:
        raise ValueError("Resume metadata without previous output")
    input_paths = [
        Path(root) / "joint-frozen-inputs.json",
        Path(data) / "manifest.json",
        Path(generator_protocol),
        Path(reranker_protocol),
        Path(generator_selection),
        Path(reranker_selection),
    ]
    input_paths += [
        context / name
        for name in ("identity.json", "comparison-status.json", "retrieval-plan.json")
    ]
    input_paths += [context / name for name in contract["context_shards_sha256"]]
    for folder, checkpoint_receipt in (
        (Path(generator_run) / "epoch-3", receipt),
        (Path(reranker_run) / f"epoch-{chosen['epoch']}", ranking_receipt),
    ):
        input_paths += [folder / "receipt.json"] + [
            folder / name for name in checkpoint_receipt["files"]
        ]
    frozen_execution_inputs = immutable_inputs(input_paths)
    import torch

    if (
        str(torch.__version__) != lineage["torch"]
        or platform.python_version() != lineage["python"]
        or any(
            importlib.metadata.version(k) != v
            for k, v in lineage["environment"].items()
        )
    ):
        raise ValueError("Frozen generator replay runtime required")
    if (
        ranking_lineage["python"] != lineage["python"]
        or ranking_lineage["torch"] != lineage["torch"]
        or ranking_lineage["environment"] != lineage["environment"]
    ):
        raise ValueError("Reranker runtime differs")
    for folder, model_receipt in (
        (Path(generator_run) / "epoch-3", receipt),
        (Path(reranker_run) / f"epoch-{chosen['epoch']}", ranking_receipt),
    ):
        if (
            any(p.is_symlink() for p in folder.rglob("*"))
            or {p.name for p in folder.iterdir() if p.is_file()}
            != set(model_receipt["files"]) | {"receipt.json"}
            or any(p.is_dir() for p in folder.iterdir())
        ):
            raise ValueError("Unbound checkpoint filesystem")
    from transformers import AutoTokenizer
    from src.full_generator_training import strict_model
    from src.full_reranker_training import strict_model as strict_ranker
    from src.reranker_training import SectionReranker

    torch.set_num_threads(lineage["threads"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(generator_run) / "epoch-3", local_files_only=True, use_fast=True
    )
    tokenizer.model_max_length = 512
    standard, windows = audit_pairs(stories, questions, cache, bm25, tokenizer)
    if any(
        a["answer"] != b["answer"] or a["citations"] != b["citations"]
        for a, b in zip(standard, baseline)
    ):
        raise ValueError("Cached BM25 standard baseline replay differs")
    ranking_model, _ = strict_ranker(Path(reranker_run) / f"epoch-{chosen['epoch']}")
    ranking_tokenizer = AutoTokenizer.from_pretrained(
        Path(reranker_run) / f"epoch-{chosen['epoch']}",
        local_files_only=True,
        use_fast=True,
    )
    scorer = SectionReranker(
        ranking_model, ranking_tokenizer, ranking_policy["model"]["max_tokens"]
    )
    queries = []
    for q in questions:
        sections = stories[q["story"]]["sections"]
        queries.append(
            {
                "id": q["id"],
                "section_ids": [
                    sections[i]["id"] for i in retrieve(q["question"], sections, scorer)
                ],
            }
        )
    plan = {
        "identity": identity,
        "queries": queries,
        "queries_sha256": content_sha(queries),
    }
    if prior_plan is not None and prior_plan != plan:
        raise ValueError("Resumed retrieval replay differs")
    reranked = {r["id"]: r["section_ids"] for r in queries}
    audit_joint(stories, questions[: len(prefix)], prefix, bm25, reranked, tokenizer)
    destination = Path(output)
    if destination.exists():
        raise ValueError("Fresh output directory required")
    destination.mkdir(parents=True)
    write(destination / "identity.json", identity)
    write(destination / "retrieval-plan.json", plan)
    if previous:
        for file in Path(previous).glob("shard-*.json"):
            shutil.copy2(file, destination / file.name)
    del scorer, ranking_model
    model, loading = strict_model(Path(generator_run) / "epoch-3")
    combined = list(prefix)
    deadline = began + max_seconds
    for start in range(len(prefix), 2372, 64):
        if time.monotonic() >= deadline:
            break
        batch = questions[start : start + 64]
        ranked = evaluate_pairs(model, tokenizer, stories, batch, reranked)
        rows = []
        for offset, pair in enumerate(ranked):
            index = start + offset
            rows.append(
                {
                    "id": pair["id"],
                    "story": pair["story"],
                    "selected_sections": {
                        "bm25": bm25[pair["id"]],
                        "reranked": reranked[pair["id"]],
                    },
                    "arms": {
                        "bm25_standard": standard[index],
                        "bm25_windows": windows[index],
                        "reranked_standard": pair["baseline"],
                        "reranked_windows": pair["candidate"],
                    },
                }
            )
        audit_joint(stories, batch, rows, bm25, reranked, tokenizer)
        write(
            destination / f"shard-{start:05d}.json",
            {"identity": identity, "pairs": rows, "pairs_sha256": content_sha(rows)},
        )
        combined.extend(rows)
        if time.monotonic() >= deadline:
            break
    complete = len(combined) == 2372
    result = {
        "identity": identity,
        "complete": complete,
        "questions_completed": len(combined),
        "questions_required": 2372,
        "loading_info": loading,
        "test_accessed": False,
        "final_test_authorized": False,
    }
    exact_prefix(destination, len(combined), identity, ids)
    if complete:
        audit_joint(stories, questions, combined, bm25, reranked, tokenizer)
        result["selection"] = select_paired(combined, ids, True)
        result["selection"]["artifact_bindings"] = {
            "generator_selection_sha256": digest(generator_selection),
            "reranker_selection_sha256": digest(reranker_selection),
            "protocol_sha256": PROTOCOL_SHA256,
            "paired_validation_sha256": content_sha(combined),
        }
    authenticate_inputs(root, contract_sha256)
    if (
        digest(generator_selection) != contract["generator_selection_sha256"]
        or digest(reranker_selection) != contract["reranker_selection_sha256"]
    ):
        raise ValueError("Selection artifact drifted")
    for filename, key in (
        ("identity.json", "context_identity_sha256"),
        ("comparison-status.json", "context_status_sha256"),
        ("retrieval-plan.json", "context_plan_sha256"),
    ):
        if digest(context / filename) != contract[key]:
            raise ValueError("Context artifact drifted")
    exact_prefix(context, 2372, cache_identity, ids, contract["context_shards_sha256"])
    snapshot_receipt(generator_run, 3, lineage)
    snapshot_receipt(reranker_run, chosen["epoch"], ranking_lineage)
    check_immutable(frozen_execution_inputs)
    for folder, model_receipt in (
        (Path(generator_run) / "epoch-3", receipt),
        (Path(reranker_run) / f"epoch-{chosen['epoch']}", ranking_receipt),
    ):
        if {p.name for p in folder.iterdir()} != set(model_receipt["files"]) | {
            "receipt.json"
        } or any(p.is_symlink() or not p.is_file() for p in folder.iterdir()):
            raise ValueError("Checkpoint inventory drifted")
    if previous is not None:
        if digest(resume_plan) != resume_plan_sha256:
            raise ValueError("Resume plan drifted")
        bound_files(previous, rp["files"])
        exact_prefix(
            previous, rp["questions_completed"], identity, ids, rp["shards_sha256"]
        )
    write(destination / "joint-status.json", result)
    return result
