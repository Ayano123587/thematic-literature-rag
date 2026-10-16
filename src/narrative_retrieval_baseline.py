"""Validation-only evidence relevance diagnostic; never answer QA accuracy."""
import argparse
import json
from pathlib import Path

from src.data.prepare_fairytale import sha
from src.narrative_training import (evidence_metrics, load_split, retrieve,
                                    validation_cohort)


def run(data, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a validation diagnostic")
    stories, all_questions = load_split(data, "validation")
    questions = validation_cohort(all_questions)
    rows = []
    for question in questions:
        sections = stories[question["story"]]["sections"]
        selected = retrieve(question["question"], sections)
        metrics = evidence_metrics([sections[i]["id"] for i in selected], question["gold_sections"])
        rows.append({"id": question["id"], "origin": question["origin"], **metrics})
    report = {"status": "Validation-only BM25 evidence diagnostic; not answer EM/F1",
              "n": len(rows), "full_validation_n": len(all_questions),
              "cohort_ids_sha256": sha(json.dumps([q["id"] for q in questions]).encode()),
              "manifest_sha256": sha((Path(data) / "manifest.json").read_bytes()),
              "source_sha256": sha(Path(__file__).read_bytes()),
              "recall_at_4": sum(r["recall"] for r in rows) / len(rows),
              "precision_at_4": sum(r["precision"] for r in rows) / len(rows),
              "per_origin": {origin: {"n": sum(r["origin"] == origin for r in rows),
                  "recall_at_4": sum(r["recall"] for r in rows if r["origin"] == origin) /
                                 sum(r["origin"] == origin for r in rows)}
                  for origin in sorted(set(r["origin"] for r in rows))},
              "test_accessed": False, "token_truncation_not_measured": True}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.data, args.output)
