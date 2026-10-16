"""One-shot evaluation of the frozen narrative checkpoint on sealed test QA."""
import argparse
import json
from pathlib import Path

from src.data.prepare_fairytale import sha
from src.narrative_training import evaluate, load_split


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.run / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "frozen" or selection.get("smoke"):
        raise ValueError("A full frozen training selection is required")
    checkpoint = args.run / "selected-checkpoint"
    if sha((checkpoint / "model.safetensors").read_bytes()) != selection["weights_sha256"]:
        raise ValueError("Selected weights changed")
    # Exclusive marker is written BEFORE reading test questions/gold. Failed test
    # runs require a separately disclosed retrospective recovery, never overwrite.
    with (args.run / "test-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"selection_sha256": sha(selection_path.read_bytes())}, handle)
    stories, questions = load_split(args.data, "test", selection_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True,
                                                use_safetensors=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions, metrics = evaluate(model, tokenizer, stories, questions)
    (args.run / "test-private.json").write_text(json.dumps(predictions), encoding="utf-8")
    (args.run / "test-results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
