"""Run an authenticated trained narrative reader: python -m src.narrative_cli."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument(
        "--story", required=True, help="Source story JSON without question labels"
    )
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    from src.models.frozen_narrative_reader import answer, load

    try:
        from src.models.narrative_generator import verify_story

        story = json.loads(Path(args.story).read_bytes())
        if not args.question.strip() or len(args.question) > 2000:
            raise ValueError("Nonempty question of at most 2000 characters required")
        verify_story(story)
        loaded = load(args.artifacts, args.contract_sha256, args.device)
        result = answer(args.question, story, loaded)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, "Inference refused: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
