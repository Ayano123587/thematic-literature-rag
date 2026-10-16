"""Audit complete validation artifacts and select checkpoints without model/test access."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.data.prepare_fairytale import sha
from src.narrative_training import (evidence_metrics, load_split, retrieve, rouge_l,
                                    select_validation, summarize, token_f1)

FULL_MANIFEST_SHA256 = "721733169b0c1e7b7c1d07fe1880d3c871d72017be6fc61923be126feea73225"
FULL_VALIDATION_IDS_SHA256 = "40d2ddd6b065469a8b947d84dbb630efcaac83acecc695606022116cc1a0f7cb"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metric_equal(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Changed validation aggregate schema")
        for key in expected:
            metric_equal(actual[key], expected[key])
    elif isinstance(expected, (int, float)):
        try:
            valid = type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Changed or nonfinite validation metric")
    elif actual != expected:
        raise ValueError("Changed validation aggregate")


def ranking_summary(rows):
    def metrics(items):
        return {"n": len(items), **{key: sum(r[field] for r in items)/len(items) for key, field in
            (("recall_at_4", "recall"), ("precision_at_4", "precision"),
             ("candidate_recall_at_12", "candidate_recall"), ("annotation_top4_ceiling", "top4_ceiling"))}}
    result = metrics(rows)
    for key in ("origin", "attribute", "explicit"):
        result[key] = {v: metrics([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    return result


def completed_run(run, data, protocol, kind):
    """Only final complete training outputs can dispatch or enter selection."""
    run = Path(run); policy = read(protocol); provenance = read(run/"provenance.json")
    lineage = provenance["identity"]; status = read(run/"training-status.json")
    expected_epochs = policy["generator" if kind == "generator" else "training"]["epochs"]
    count = status.get("all_training_QA" if kind == "generator" else "train_QA")
    if status.get("status") != "training_complete" or status.get("smoke") is not False or status.get(
            "epochs_required") != expected_epochs or type(count) is not int or count != 7883 or status.get("test_accessed") is not False or status.get(
            "progress") != {"epoch": expected_epochs+1, "next_microstep": 0, "loss_sum": 0.0}:
        raise ValueError("Selection requires complete full-cohort training, not partial/smoke checkpoints")
    manifest_key = "source_manifest_sha256" if kind == "generator" else "manifest_sha256"
    if digest(Path(data)/"manifest.json") != FULL_MANIFEST_SHA256 or lineage.get("smoke") is not False or lineage.get(manifest_key) != digest(Path(data)/"manifest.json") or lineage.get(
            "protocol_sha256") != digest(protocol):
        raise ValueError("Changed training source/protocol identity")
    if lineage.get("generator" if kind == "generator" else "model") != policy["generator" if kind == "generator" else "model"]:
        raise ValueError("Changed pinned model identity")
    if kind == "reranker" and lineage.get("training") != policy["training"]:
        raise ValueError("Changed listwise training configuration")
    loading = provenance.get("loading_info")
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    expected_weights = ("bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc" if kind == "generator"
                        else "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae")
    if not isinstance(loading, dict) or not all(k in loading for k in keys) or any(loading[k] for k in keys) or provenance.get(
            "pretrained_weights_sha256") != expected_weights:
        raise ValueError("Unverified pretrained initialization/head provenance")
    source_files = {"full_generator_training.py", "resumable_training.py", "narrative_training.py", "data/prepare_fairytale.py"}
    if kind == "reranker":
        source_files |= {"full_reranker_training.py", "reranker_training.py"}
    if not isinstance(lineage.get("source_sha256"), dict) or set(lineage["source_sha256"]) != source_files:
        raise ValueError("Incomplete scientific execution source identity")
    for name, expected in lineage["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or digest(Path(__file__).parent/relative) != expected:
            raise ValueError("Changed scientific execution source")
    return policy, lineage, expected_epochs


def snapshot_receipt(run, epoch, lineage):
    checkpoint = Path(run)/f"epoch-{epoch}"
    receipt = read(checkpoint/"receipt.json")
    files = receipt.get("files")
    if receipt.get("identity") != lineage or receipt.get("epoch") != epoch or not isinstance(files, dict) or not {
            "model.safetensors", "config.json", "tokenizer_config.json"}.issubset(files):
        raise ValueError("Changed checkpoint receipt")
    for name, expected in files.items():
        if Path(name).name != name or digest(checkpoint/name) != expected:
            raise ValueError("Changed checkpoint model/config/tokenizer bytes")
    if receipt.get("weights_sha256") != files["model.safetensors"]:
        raise ValueError("Conflicting checkpoint weight identity")
    return receipt


def validation_cohort(data, policy):
    if digest(Path(data)/"manifest.json") != FULL_MANIFEST_SHA256:
        raise ValueError("Changed prespecified complete source cohort")
    manifest = read(Path(data)/"manifest.json")
    records = manifest["stories"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Overlapping source story IDs")
    for key in ("source_sha256", "origin"):
        roles = {}
        for record in records:
            previous = roles.setdefault(record[key], record["split"])
            if previous != record["split"]:
                raise ValueError("Cross-split source/family overlap")
    stories, questions = load_split(data, "validation")
    questions = select_validation(questions, policy)
    if len(questions) != 2372 or len({q["id"] for q in questions}) != 2372:
        raise ValueError("Expected exactly all 2372 unique validation questions")
    if sha(json.dumps([q["id"] for q in questions]).encode()) != FULL_VALIDATION_IDS_SHA256:
        raise ValueError("Changed prespecified full validation IDs")
    return stories, questions


def audited_rows(directory, identity, expected_ids, kind):
    directory = Path(directory); status = read(directory/"validation-status.json")
    if status.get("complete") is not True or status.get("questions_completed") != len(expected_ids) or status.get(
            "questions_required") != len(expected_ids) or status.get("identity") != identity or status.get("smoke") is not False or status.get("test_accessed") is not False:
        raise ValueError("Incomplete or changed validation identity/coverage")
    loading = status.get("loading_info")
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if read(directory/"identity.json") != identity or not isinstance(loading, dict) or not all(k in loading for k in keys) or any(loading[k] for k in keys):
        raise ValueError("Changed validation identity or incompatible loaded head")
    size = identity["shard_size"]
    if type(size) is not int or size < 1:
        raise ValueError("Invalid validation shard size")
    paths = [directory/f"shard-{start:05d}.json" for start in range(0, len(expected_ids), size)]
    if set(directory.glob("shard-*.json")) != set(paths):
        raise ValueError("Missing or unexpected validation shards")
    rows = []; key = "predictions" if kind == "generator" else "rows"
    for start, path in zip(range(0, len(expected_ids), size), paths):
        shard = read(path); batch = shard[key]
        if shard.get("identity") != identity or [r["id"] for r in batch] != expected_ids[start:start+size] or sha(
                json.dumps(batch, sort_keys=True, allow_nan=False).encode()) != shard.get(key+"_sha256"):
            raise ValueError("Changed, overlapping or reordered validation shard")
        rows.extend(batch)
    return rows, status


def recompute_rows(stories, questions, rows, kind):
    checked = []
    for question, row in zip(questions, rows):
        story = stories[question["story"]]
        sections = {s["id"]: s for s in story["sections"]}
        if kind == "generator":
            if any(row.get(k) != v for k, v in question.items()) or not isinstance(row.get("answer"), str):
                raise ValueError("Changed question/reference/source metadata")
            citations = row["citations"]
            if not isinstance(citations, list):
                raise ValueError("Invalid citation envelope")
            valid = True
            for citation in citations:
                section = sections.get(citation["section"])
                start, end = citation["start"], citation["end"]
                if section is None or type(start) is not int or type(end) is not int or not section["start"] <= start <= end <= section["end"]:
                    raise ValueError("Citation outside its source section")
                valid &= story["text"][start:end] == citation["text"]
            selected = [story["sections"][i]["id"] for i in retrieve(question["question"], story["sections"])]
            if any(c["section"] not in selected for c in citations):
                raise ValueError("Citation absent from declared BM25 context")
            relevance = evidence_metrics(selected, question["gold_sections"])
            retained = evidence_metrics([c["section"] for c in citations], question["gold_sections"])
            scores = [token_f1(row["answer"], ref) for ref in question["references"]]
            metrics = {"em": max(s[0] for s in scores), "f1": max(s[1] for s in scores),
                "rouge_l": max(rouge_l(row["answer"], ref) for ref in question["references"]),
                "evidence_recall": relevance["recall"], "evidence_precision": relevance["precision"],
                "retained_evidence_recall": retained["recall"], "citation_validity": float(valid)}
            for key, value in metrics.items():
                metric_equal(row.get(key), value)
            if row.get("reference_count") != len(question["references"]):
                raise ValueError("Changed reference count")
            checked.append({**row, **metrics})
        else:
            from src.narrative_training import lexical_ranking
            candidates = [story["sections"][i]["id"] for i in lexical_ranking(question["question"], story["sections"])[:12]]
            selected = row["selected_sections"]
            if row.get("candidate_sections") != candidates or len(selected) != min(4, len(candidates)) or len(set(selected)) != len(selected) or not set(selected) <= set(candidates) or any(
                    row.get(k) != question[k] for k in ("id", "story", "origin", "attribute", "explicit", "gold_sections")):
                raise ValueError("Changed ranking candidate/gold/source identity")
            metrics = {**evidence_metrics(selected, question["gold_sections"]),
                "candidate_recall": evidence_metrics(candidates, question["gold_sections"])["recall"],
                "top4_ceiling": min(4, len(question["gold_sections"]))/len(question["gold_sections"])}
            for key, value in metrics.items():
                metric_equal(row.get(key), value)
            checked.append({**row, **metrics})
    return checked


def select(data, protocol, run, validations, output, kind):
    if kind not in ("generator", "reranker"):
        raise ValueError("Unknown checkpoint selection task")
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve serialized selection; refuse overwrite")
    policy, lineage, epochs = completed_run(run, data, protocol, kind)
    if policy.get("validation_sampling") != {"mode": "all"} or len(validations) != epochs+1:
        raise ValueError("Require initial and every epoch on uncapped validation")
    stories, questions = validation_cohort(data, policy)
    ids = [q["id"] for q in questions]; ids_sha = sha(json.dumps(ids).encode())
    if kind == "reranker" and ids_sha != policy["source"]["validation_ids_sha256"]:
        raise ValueError("Changed prespecified full validation IDs")
    candidates = []; receipts = []
    for epoch, directory in enumerate(validations):
        receipt = snapshot_receipt(run, epoch, lineage)
        identity = read(Path(directory)/"identity.json")
        expected = {"run": lineage, "epoch": epoch, "weights_sha256": receipt["weights_sha256"],
                    "validation_ids_sha256" if kind == "generator" else "ids_sha256": ids_sha,
                    "shard_size": identity.get("shard_size")}
        rows, status = audited_rows(directory, expected, ids, kind)
        rows = recompute_rows(stories, questions, rows, kind)
        metrics = summarize(rows) if kind == "generator" else ranking_summary(rows)
        metric_equal(status["validation"], metrics)
        candidates.append({"kind": "generator" if kind == "generator" else ("pretrained" if epoch == 0 else "adapted"),
                           "epoch": epoch, "weights_sha256": receipt["weights_sha256"], "validation": metrics})
        receipts.append({"epoch": epoch, "checkpoint_receipt_sha256": digest(Path(run)/f"epoch-{epoch}/receipt.json"),
                         "validation_status_sha256": digest(Path(directory)/"validation-status.json")})
    if kind == "reranker":
        from src.reranker_training import evaluate_sections
        baseline = evaluate_sections(stories, questions)
        candidates.insert(0, {"kind": "bm25", "epoch": 0, "validation": baseline})
    objective = "f1" if kind == "generator" else "recall_at_4"
    selected = max(candidates, key=lambda candidate: candidate["validation"][objective])
    result = {"status": "validation_checkpoint_selection", "task": kind, "selected": selected,
              "candidates": candidates, "objective": objective, "tie_order": "earliest listed; BM25 precedes pretrained and adapted epochs",
              "validation_questions": len(ids), "validation_ids_sha256": ids_sha,
              "manifest_sha256": digest(Path(data)/"manifest.json"), "protocol_sha256": digest(protocol),
              "training_identity": lineage, "artifact_receipts": receipts, "test_accessed": False,
              "audit_source_sha256": digest(__file__),
              "final_test_authorized": False,
              "remaining_gate": "Compare selected retrieval with BM25 using frozen selected generator on complete validation, then serialize global production policy before any test access."}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("generator", "reranker"))
    for name in ("data", "protocol", "run", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--validation", nargs="+", required=True, help="Complete directories in epoch0,1,... order")
    args = parser.parse_args()
    try:
        result = select(args.data, args.protocol, args.run, args.validation, args.output, args.task)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, "Validation selection refused: "+str(error)+"\n")
    print(json.dumps({"task": args.task, "selected_epoch": result["selected"]["epoch"],
                      "selected_kind": result["selected"]["kind"], "test_accessed": False}))


if __name__ == "__main__":
    main()
