"""Experimental trained narrative generator with exact retrieved-source citations.

Generated answer text is not an extractive span or an entailment certificate.
"""
import json
from pathlib import Path

from src.data.prepare_fairytale import sha
from src.narrative_training import assemble, retrieve


def verify_run(folder, allow_smoke=False):
    folder = Path(folder)
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    if selection.get("status") != "frozen" or (selection.get("smoke") and not allow_smoke):
        raise ValueError("A full frozen training selection is required")
    provenance_path = folder / "provenance.json"
    if sha(provenance_path.read_bytes()) != selection.get("provenance_sha256"):
        raise ValueError("Training provenance changed")
    checkpoint = folder / "selected-checkpoint"
    if sha((checkpoint / "model.safetensors").read_bytes()) != selection.get("weights_sha256"):
        raise ValueError("Selected weights changed")
    return checkpoint, selection


def load_selected(folder, allow_smoke=False):
    checkpoint, selection = verify_run(folder, allow_smoke)
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model, info = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, local_files_only=True, use_safetensors=True, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete trained generator checkpoint")
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tokenizer, selection


def answer(question, story, model, tokenizer, selection):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("A nonempty question of at most 2000 characters is required")
    if sha(story["text"].encode()) != story["source_sha256"]:
        raise ValueError("Story source hash mismatch")
    for section in story["sections"]:
        if story["text"][section["start"]:section["end"]] != section["text"]:
            raise ValueError("Story section source-offset mismatch")
    indices = retrieve(question, story["sections"])
    context, citations = assemble(question, story["sections"], indices, tokenizer,
                                  min(512, tokenizer.model_max_length))
    import torch
    encoded = tokenizer(context, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=96, num_beams=4,
                                do_sample=False, early_stopping=True, use_cache=True)
    return {"schema": "narrative-generated-answer-v1", "answer_kind": "generated",
            "question": question, "answer": tokenizer.decode(output[0], skip_special_tokens=True),
            "story": story["id"], "source_sha256": story["source_sha256"],
            "trained_weights_sha256": selection["weights_sha256"],
            "retrieved_evidence": citations,
            "citation_claim": "Exact source location only; no entailment proof or calibrated no-answer claim"}
