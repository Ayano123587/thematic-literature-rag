"""Experimental CLI for a saved trained narrative checkpoint."""
import argparse
import json
from pathlib import Path

from src.models.narrative_generator import answer, load_selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--story", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer, selection = load_selected(args.run)
    story = json.loads(args.story.read_text(encoding="utf-8"))
    result = answer(args.question, story, model, tokenizer, selection)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "answer_kind": "generated",
                      "citations": len(result["retrieved_evidence"])}))


if __name__ == "__main__":
    main()
