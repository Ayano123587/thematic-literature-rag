"""Inference from authenticated full-training and global answer-policy artifacts."""

import hashlib
import json
from pathlib import Path

from src.context_assembly import assemble_question_windows
from src.joint_answer_selection import ARMS, adapted_checkpoint, sha256_value
from src.full_validation_selection import FULL_VALIDATION_IDS_SHA256
from src.models.narrative_generator import verify_story
from src.narrative_training import assemble, retrieve


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticate(folder, contract_sha256):
    """The caller supplies an independently frozen contract digest, never a default."""
    root = Path(folder)
    path = root / "inference-contract.json"
    if digest(path) != sha256_value(contract_sha256):
        raise ValueError("Changed inference contract")
    contract = json.loads(path.read_bytes())
    if contract.get("schema") != "frozen_narrative_inference_v1":
        raise ValueError("Current inference contract required")
    files = contract.get("files")
    if not isinstance(files, dict) or not {
        "generator-selection.json",
        "joint-selection.json",
        "joint-protocol.json",
        "generator/receipt.json",
        "generator/model.safetensors",
        "generator/config.json",
        "generator/tokenizer_config.json",
    }.issubset(files):
        raise ValueError("Complete source/model/selection inventory required")
    if root.is_symlink() or path.is_symlink():
        raise ValueError("Symlink artifacts forbidden")
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("Symlink artifacts forbidden")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {"inference-contract.json"}:
        raise ValueError("Unlisted or missing inference files")
    for name, expected in files.items():
        p = Path(name)
        if (
            p.is_absolute()
            or ".." in p.parts
            or not (root / p).resolve().is_relative_to(root.resolve())
            or digest(root / p) != sha256_value(expected)
        ):
            raise ValueError("Changed inference artifact")
    sources = contract.get("source_sha256")
    required = {
        "models/frozen_narrative_reader.py",
        "models/narrative_generator.py",
        "narrative_training.py",
        "context_assembly.py",
        "joint_answer_selection.py",
        "context_validation.py",
        "full_validation_selection.py",
        "full_reranker_training.py",
        "full_generator_training.py",
        "reranker_training.py",
        "resumable_training.py",
        "data/prepare_fairytale.py",
        "__init__.py",
        "models/__init__.py",
        "data/__init__.py",
        "narrative_cli.py",
    }
    if not isinstance(sources, dict) or set(sources) != required:
        raise ValueError("Inference source identity required")
    for name, expected in sources.items():
        if digest(Path(__file__).parents[1] / name) != sha256_value(expected):
            raise ValueError("Changed inference implementation")
    if (
        files["joint-protocol.json"]
        != "00493d88b452eecfba78de564ec4d9f4a9a9382fd9e5c60d4672eb808460b0c9"
    ):
        raise ValueError("Changed fixed joint protocol")
    protocol = json.loads((root / "joint-protocol.json").read_bytes())
    if protocol["arms"] != list(ARMS) or protocol["validation_questions"] != 2372:
        raise ValueError("Changed joint protocol fields")
    generator = json.loads((root / "generator-selection.json").read_bytes())
    policy = json.loads((root / "joint-selection.json").read_bytes())
    receipt = json.loads((root / "generator/receipt.json").read_bytes())
    if (
        generator.get("status") != "validation_checkpoint_selection"
        or generator.get("task") != "generator"
        or generator.get("validation_questions") != 2372
        or generator.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or generator.get("test_accessed") is not False
        or generator["selected"]["epoch"] != 3
    ):
        raise ValueError("Frozen full generator selection required")
    weights = generator["selected"]["weights_sha256"]
    if (
        weights != "4a51a48953e6c02c2c8aa5325a02600ae7a65dc8080abb5fb45e2b7131c39538"
        or receipt.get("weights_sha256") != weights
        or files["generator/model.safetensors"] != weights
    ):
        raise ValueError("Selected generator mismatch")
    if (
        policy.get("schema") != "joint_answer_selection_v1"
        or policy.get("selected_arm") not in ARMS
        or policy.get("validation_questions") != 2372
        or policy.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or policy.get("test_accessed") is not False
    ):
        raise ValueError("Frozen global answer policy required")
    if policy.get("artifact_bindings") != contract.get(
        "artifact_bindings"
    ) or not isinstance(policy.get("artifact_bindings"), dict):
        raise ValueError("Joint selection artifact bindings required")
    for key in (
        "generator_selection_sha256",
        "protocol_sha256",
        "paired_validation_sha256",
    ):
        sha256_value(policy["artifact_bindings"][key])
    if (
        policy["artifact_bindings"]["generator_selection_sha256"]
        != files["generator-selection.json"]
        or policy["artifact_bindings"]["protocol_sha256"]
        != files["joint-protocol.json"]
    ):
        raise ValueError("Joint policy source differs")
    if policy["selected_arm"].startswith("reranked"):
        if not {
            "reranker-selection.json",
            "reranker/model.safetensors",
            "reranker/receipt.json",
            "reranker/config.json",
            "reranker/tokenizer_config.json",
        }.issubset(files):
            raise ValueError("Actual trained reranker artifacts required")
        if (
            policy["artifact_bindings"].get("reranker_selection_sha256")
            != files["reranker-selection.json"]
        ):
            raise ValueError("Joint policy reranker selection differs")
        chosen = adapted_checkpoint(
            json.loads((root / "reranker-selection.json").read_bytes())
        )
        rr = json.loads((root / "reranker/receipt.json").read_bytes())
        if (
            chosen["weights_sha256"] != files["reranker/model.safetensors"]
            or rr.get("weights_sha256") != chosen["weights_sha256"]
            or rr.get("epoch") != chosen["epoch"]
        ):
            raise ValueError("Adapted reranker mismatch")
    for kind in ("generator", "reranker"):
        if kind == "reranker" and not policy["selected_arm"].startswith("reranked"):
            continue
        checkpoint_receipt = json.loads((root / kind / "receipt.json").read_bytes())
        model_files = checkpoint_receipt.get("files")
        if (
            not isinstance(model_files, dict)
            or not model_files
            or not ({"tokenizer.json", "spiece.model"} & set(model_files))
        ):
            raise ValueError("Complete tokenizer receipt required")
        inventory = {
            name.removeprefix(kind + "/"): value
            for name, value in files.items()
            if name.startswith(kind + "/") and name != kind + "/receipt.json"
        }
        if inventory != model_files:
            raise ValueError("Checkpoint filesystem differs from receipt")
    return contract, policy


def load(folder, contract_sha256, device="cpu"):
    contract, policy = authenticate(folder, contract_sha256)
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit CPU or CUDA device required")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    root = Path(folder)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "generator", local_files_only=True, use_fast=True
    )
    model, info = AutoModelForSeq2SeqLM.from_pretrained(
        root / "generator",
        local_files_only=True,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Incomplete generator loading")
    reranker = None
    if policy["selected_arm"].startswith("reranked"):
        from src.full_reranker_training import strict_model
        from src.reranker_training import SectionReranker

        ranked, _ = strict_model(root / "reranker")
        rt = AutoTokenizer.from_pretrained(
            root / "reranker", local_files_only=True, use_fast=True
        )
        reranker = SectionReranker(ranked.to(device), rt, 384)
    return model.to(device).eval(), tokenizer, reranker, policy, contract


def answer(question, story, loaded):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Nonempty question of at most 2000 characters required")
    verify_story(story)
    model, tokenizer, reranker, policy, contract = loaded
    indices = retrieve(question, story["sections"], reranker)
    assembly = (
        assemble_question_windows
        if policy["selected_arm"].endswith("windows")
        else assemble
    )
    context, citations = assembly(question, story["sections"], indices, tokenizer, 512)
    if len(tokenizer.encode(context)) > 512:
        raise ValueError("Frozen input budget exceeded")
    import torch

    with torch.inference_mode():
        output = model.generate(
            **tokenizer(context, return_tensors="pt").to(model.device),
            max_new_tokens=96,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            use_cache=True,
        )
    return {
        "answer": tokenizer.decode(output[0], skip_special_tokens=True),
        "story": story["id"],
        "source_sha256": story["source_sha256"],
        "selected_arm": policy["selected_arm"],
        "generator_weights_sha256": contract["files"]["generator/model.safetensors"],
        "retrieved_evidence": citations,
        "citation_claim": "Source locations, not entailment or calibrated answer confidence",
    }
