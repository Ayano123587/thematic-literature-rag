"""Authenticated indexed-book reader: python -m src.indexed_narrative_cli."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--mode", choices=("hybrid", "bm25", "dense"), default="hybrid")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    from src.models.indexed_narrative_reader import indexed_answers

    try:
        result = indexed_answers(
            args.question,
            args.index,
            args.index_sha256,
            args.sources,
            args.books,
            args.artifacts,
            args.contract_sha256,
            args.candidates,
            args.mode,
            args.device,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, "Inference refused: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
