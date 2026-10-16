"""Aggregate source/annotation statistics without publishing QA or story text."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from src.data.prepare_fairytale import sha
from src.narrative_training import load_split


def histogram(values, edges):
    return {f"{low}-{high - 1}" if high is not None else f"{low}+":
            sum(value >= low and (high is None or value < high) for value in values)
            for low, high in zip(edges, edges[1:] + [None])}


def aggregate(metadata_path, staged):
    metadata_path, staged = Path(metadata_path), Path(staged)
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    chosen = {r["id"]: r["split"] for r in manifest["stories"]}
    by_origin = {}
    for origin in sorted(set(r["origin"] for r in rows)):
        subset = [r for r in rows if r["origin"] == origin]
        by_origin[origin] = {"stories": len(subset), "metadata_qa": sum(int(r["questions"]) for r in subset),
            "official_split": dict(Counter(r["split"] for r in subset)),
            "study_split": dict(Counter(chosen.get(r["filename"], "excluded") for r in subset))}
    distributions = {}
    for split in ("train", "validation", "test", "excluded"):
        subset = [r for r in rows if chosen.get(r["filename"], "excluded") == split]
        lengths = [int(r["words"]) for r in subset]
        distributions[split] = {"stories": len(subset), "metadata_qa": sum(int(r["questions"]) for r in subset),
            "word_histogram": histogram(lengths, [0, 500, 1000, 2000, 4000, 8000, 16000]),
            "median_words": statistics.median(lengths) if lengths else None}
    annotation = {}
    for split in ("train", "validation"):
        stories, questions = load_split(staged, split)
        original = Counter((q["story"], q["source_question_id"]) for q in questions)
        annotation[split] = {"questions": len(questions), "stories": len(stories),
            "attribute": dict(Counter(q["attribute"] for q in questions)),
            "explicit": dict(Counter(q["explicit"] for q in questions)),
            "references": dict(Counter(len(q["references"]) for q in questions)),
            "evidence_sections": dict(Counter(len(q["gold_sections"]) for q in questions)),
            "question_word_histogram": histogram([len(q["question"].split()) for q in questions],
                                                  [0, 5, 10, 15, 20, 40, 80]),
            "answer_word_histogram": histogram([len(q["references"][0].split()) for q in questions],
                                                [0, 2, 5, 10, 20, 40, 80, 160]),
            "duplicate_original_ids": sum(n - 1 for n in original.values()),
            "canonical_ids_unique": len({q["id"] for q in questions}) == len(questions)}
    return {"dataset_revision": "a24ddc17364666b7c13a425b9970c87368b03417",
            "metadata_sha256": sha(metadata_path.read_bytes()),
            "staging_manifest_sha256": sha((staged / "manifest.json").read_bytes()),
            "by_origin": by_origin, "source_lengths": distributions, "annotations": annotation,
            "test_qa_parsed": False, "text_or_answers_published": False,
            "test_qa_count_source": "Official story metadata only, not test CSV parsing"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--staged", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = aggregate(args.metadata, args.staged)
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"origins": len(report["by_origin"]), "test_qa_parsed": False}))
