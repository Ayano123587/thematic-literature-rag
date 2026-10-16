"""Paired full-validation context comparison with a frozen selected generator."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import time

import src.full_validation_selection as selection_audit

from src.context_assembly import assemble_question_windows
from src.data.prepare_fairytale import sha
from src.full_validation_selection import (
    FULL_MANIFEST_SHA256,
    FULL_VALIDATION_IDS_SHA256,
    audited_rows,
    completed_run,
    digest,
    metric_equal,
    read,
    recompute_rows,
    snapshot_receipt,
    validation_cohort,
)
from src.narrative_training import (
    assemble,
    evidence_metrics,
    retrieve,
    rouge_l,
    summarize,
    token_f1,
)


def story_interval(pairs, resamples=10000, seed=20261016):
    """Descriptive cluster bootstrap; retain all question pairs within each story."""
    if not pairs or type(resamples) is not int or resamples < 1:
        raise ValueError(
            "Nonempty paired questions and positive resample count required"
        )
    grouped = defaultdict(list)
    ids = set()
    for pair in pairs:
        if pair["id"] in ids:
            raise ValueError("Duplicate paired question")
        ids.add(pair["id"])
        for variant in ("baseline", "candidate"):
            value = pair[variant]["f1"]
            metric_equal(value, value)
            if not 0 <= value <= 1:
                raise ValueError("Answer F1 outside its score range")
        grouped[pair["story"]].append(pair["candidate"]["f1"] - pair["baseline"]["f1"])
    stories = sorted(grouped)
    rng = random.Random(seed)
    totals = {s: sum(grouped[s]) for s in stories}
    draws = []
    for _ in range(resamples):
        selected = rng.choices(stories, k=len(stories))
        draws.append(
            sum(totals[s] for s in selected) / sum(len(grouped[s]) for s in selected)
        )
    draws.sort()

    def quantile(fraction):
        index = fraction * (len(draws) - 1)
        low = int(index)
        high = min(low + 1, len(draws) - 1)
        return draws[low] + (draws[high] - draws[low]) * (index - low)

    return {
        "low": quantile(0.025),
        "high": quantile(0.975),
        "stories": len(stories),
        "questions": len(pairs),
        "resamples": resamples,
        "seed": seed,
        "interpretation": "Descriptive validation story-cluster interval; not independent confirmatory significance",
    }


def promotion(baseline, candidate, interval, replay_exact):
    for metrics in (baseline, candidate):
        for key in ("f1", "em", "citation_validity"):
            metric_equal(metrics[key], metrics[key])
            if not 0 <= metrics[key] <= 1:
                raise ValueError("Invalid promotion score")
    for key in ("low", "high"):
        metric_equal(interval[key], interval[key])
    if interval["low"] > interval["high"]:
        raise ValueError("Reversed descriptive interval")
    gates = {
        "minimum_F1_gain": candidate["f1"] - baseline["f1"] >= 0.01 - 1e-12,
        "no_EM_regression": candidate["em"] >= baseline["em"] - 1e-12,
        "all_source_citations_valid": candidate["citation_validity"] == 1.0
        and candidate["citation_validity"] >= baseline["citation_validity"],
        "positive_story_interval": interval["low"] > 0,
        "baseline_replay_exact": replay_exact is True,
    }
    return {
        "promoted": all(gates.values()),
        "gates": gates,
        "selected_context": (
            "ranked_question_windows" if all(gates.values()) else "source_order"
        ),
    }


def selected_checkpoint(
    data, generator_protocol, run, selection_path, baseline_directory
):
    policy, lineage, epochs = completed_run(run, data, generator_protocol, "generator")
    selection = read(selection_path)
    if (
        not isinstance(selection, dict)
        or selection.get("status") != "validation_checkpoint_selection"
        or selection.get("task") != "generator"
        or selection.get("validation_questions") != 2372
        or selection.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or selection.get("training_identity") != lineage
        or selection.get("test_accessed") is not False
        or selection.get("final_test_authorized") is not False
    ):
        raise ValueError("Expected audited complete generator checkpoint selection")
    if (
        selection.get("manifest_sha256") != digest(Path(data) / "manifest.json")
        or selection.get("protocol_sha256") != digest(generator_protocol)
        or selection.get("audit_source_sha256") != digest(selection_audit.__file__)
    ):
        raise ValueError("Changed selection source, manifest or protocol identity")
    candidates = selection["candidates"]
    if not isinstance(candidates, list) or any(
        not isinstance(row, dict) or type(row.get("epoch")) is not int
        for row in candidates
    ):
        raise ValueError("Malformed generator epoch candidates")
    if [r["epoch"] for r in candidates] != list(range(epochs + 1)) or any(
        r["validation"]["n"] != 2372 for r in candidates
    ):
        raise ValueError("Generator selection omits full-validation epoch candidates")
    for row in candidates:
        metric_equal(row["validation"]["f1"], row["validation"]["f1"])
    if selection["selected"] != max(candidates, key=lambda r: r["validation"]["f1"]):
        raise ValueError("Generator checkpoint differs from declared F1/tie selection")
    epoch = selection["selected"]["epoch"]
    receipt = snapshot_receipt(run, epoch, lineage)
    if selection["selected"].get("weights_sha256") != receipt["weights_sha256"]:
        raise ValueError("Selected checkpoint weight identity changed")
    receipts = selection["artifact_receipts"]
    if (
        [r["epoch"] for r in receipts] != list(range(epochs + 1))
        or receipts[epoch]["checkpoint_receipt_sha256"]
        != digest(Path(run) / f"epoch-{epoch}/receipt.json")
        or receipts[epoch]["validation_status_sha256"]
        != digest(Path(baseline_directory) / "validation-status.json")
    ):
        raise ValueError("Selected baseline artifact receipt changed")
    stories, questions = validation_cohort(data, policy)
    identity = read(Path(baseline_directory) / "identity.json")
    expected = {
        "run": lineage,
        "epoch": epoch,
        "weights_sha256": receipt["weights_sha256"],
        "validation_ids_sha256": FULL_VALIDATION_IDS_SHA256,
        "shard_size": identity.get("shard_size"),
    }
    rows, status = audited_rows(
        baseline_directory, expected, [q["id"] for q in questions], "generator"
    )
    rows = recompute_rows(stories, questions, rows, "generator")
    metric_equal(status["validation"], summarize(rows))
    metric_equal(selection["selected"]["validation"], summarize(rows))
    return policy, lineage, epoch, receipt, stories, questions, rows


def check_protocol(path):
    protocol = read(path)
    if not isinstance(protocol, dict) or any(
        not isinstance(protocol.get(k), dict)
        for k in ("candidate", "promotion", "interval", "decoding")
    ):
        raise ValueError("Expected a complete context protocol JSON object")
    if (
        protocol.get("source_manifest_sha256") != FULL_MANIFEST_SHA256
        or protocol.get("validation_questions") != 2372
        or protocol.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or protocol.get("candidate", {}).get("name") != "ranked_question_windows"
        or type(protocol["candidate"].get("input_tokens")) is not int
        or protocol["candidate"]["input_tokens"] != 512
        or protocol.get("decoding")
        != {
            "num_beams": 4,
            "max_new_tokens": 96,
            "do_sample": False,
            "early_stopping": True,
        }
        or protocol.get("promotion", {}).get("minimum_absolute_macro_answer_F1_gain")
        != 0.01
        or protocol.get("interval", {}).get("resamples") != 10000
        or protocol["interval"].get("seed") != 20261016
    ):
        raise ValueError("Changed prespecified context comparison configuration")
    return protocol


def score_row(question, answer, citations, selected_ids, story):
    scores = [token_f1(answer, ref) for ref in question["references"]]
    relevance = evidence_metrics(selected_ids, question["gold_sections"])
    retained = evidence_metrics(
        [c["section"] for c in citations], question["gold_sections"]
    )
    valid = all(
        story["text"][c["start"] : c["end"]] == c["text"]
        and 0 <= c["start"] < c["end"] <= len(story["text"])
        for c in citations
    )
    return {
        **question,
        "answer": answer,
        "citations": citations,
        "reference_count": len(question["references"]),
        "em": max(s[0] for s in scores),
        "f1": max(s[1] for s in scores),
        "rouge_l": max(rouge_l(answer, ref) for ref in question["references"]),
        "evidence_recall": relevance["recall"],
        "evidence_precision": relevance["precision"],
        "retained_evidence_recall": retained["recall"],
        "citation_validity": float(valid),
    }


def evaluate_pairs(model, tokenizer, stories, questions, plan, batch_size=8):
    """Only question/source contexts reach the model; references score after decoding."""
    import torch

    model.eval()
    pairs = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        variants = {"baseline": [], "candidate": []}
        evidence = {k: [] for k in variants}
        selected = []
        for q in batch:
            story = stories[q["story"]]
            positions = {
                section["id"]: i for i, section in enumerate(story["sections"])
            }
            ids = plan[q["id"]]
            indices = [positions[section] for section in ids]
            selected.append(ids)
            for key, assembly in (
                ("baseline", assemble),
                ("candidate", assemble_question_windows),
            ):
                context, citations = assembly(
                    q["question"], story["sections"], indices, tokenizer, 512
                )
                if len(tokenizer.encode(context)) > 512:
                    raise ValueError("Context exceeds the frozen input budget")
                variants[key].append(context)
                evidence[key].append(citations)
        decoded = {}
        for key, contexts in variants.items():
            encoded = tokenizer(contexts, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=96,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True,
                )
            decoded[key] = tokenizer.batch_decode(output, skip_special_tokens=True)
        for i, q in enumerate(batch):
            pair = {
                "id": q["id"],
                "story": q["story"],
                "selected_sections": selected[i],
            }
            for key in variants:
                pair[key] = score_row(
                    q,
                    decoded[key][i],
                    evidence[key][i],
                    selected[i],
                    stories[q["story"]],
                )
            pairs.append(pair)
    return pairs


def audit_pairs(stories, questions, pairs, plan, tokenizer):
    if len(pairs) != len(questions):
        raise ValueError("Incomplete paired question coverage")
    variants = {"baseline": [], "candidate": []}
    for question, pair in zip(questions, pairs):
        selected = plan[question["id"]]
        if (
            pair.get("id") != question["id"]
            or pair.get("story") != question["story"]
            or pair.get("selected_sections") != selected
        ):
            raise ValueError("Changed or overlapping question/retrieval pair")
        story = stories[question["story"]]
        positions = {section["id"]: i for i, section in enumerate(story["sections"])}
        indices = [positions[section] for section in selected]
        for key, assembly in (
            ("baseline", assemble),
            ("candidate", assemble_question_windows),
        ):
            row = pair[key]
            if any(row.get(k) != v for k, v in question.items()) or not isinstance(
                row.get("answer"), str
            ):
                raise ValueError("Changed paired question/reference metadata")
            context, citations = assembly(
                question["question"], story["sections"], indices, tokenizer, 512
            )
            if (
                row.get("citations") != citations
                or len(tokenizer.encode(context)) > 512
            ):
                raise ValueError(
                    "Changed prescribed context/source citations or token budget"
                )
            expected = score_row(question, row["answer"], citations, selected, story)
            for metric in (
                "em",
                "f1",
                "rouge_l",
                "evidence_recall",
                "evidence_precision",
                "retained_evidence_recall",
                "citation_validity",
                "reference_count",
            ):
                metric_equal(row.get(metric), expected[metric])
            variants[key].append(expected)
    return variants["baseline"], variants["candidate"]


def run_comparison(
    data,
    generator_protocol,
    context_protocol,
    run,
    selection,
    baseline_directory,
    output,
    *,
    previous=None,
    max_seconds=37800,
    shard_size=64,
):
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not 0 <= max_seconds <= 39600
        or type(shard_size) is not int
        or shard_size < 8
        or shard_size % 8
    ):
        raise ValueError(
            "Safe cloud window and eight-query-aligned validation shards required"
        )
    protocol = check_protocol(context_protocol)
    # These guards precede any model loading or new forward inference.
    policy, lineage, epoch, receipt, stories, questions, cached = selected_checkpoint(
        data, generator_protocol, run, selection, baseline_directory
    )
    identity = {
        "training": lineage,
        "selected_epoch": epoch,
        "weights_sha256": receipt["weights_sha256"],
        "selection_sha256": digest(selection),
        "baseline_status_sha256": digest(
            Path(baseline_directory) / "validation-status.json"
        ),
        "context_protocol_sha256": digest(context_protocol),
        "validation_ids_sha256": FULL_VALIDATION_IDS_SHA256,
        "shard_size": shard_size,
        "batch_size": 8,
        "source_sha256": {
            n: digest(Path(__file__).parent / n)
            for n in (
                "context_assembly.py",
                "context_validation.py",
                "full_validation_selection.py",
            )
        },
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve comparison outputs; resume into a fresh directory")
    output.mkdir(parents=True)
    from src.resumable_training import atomic_json

    if previous is not None:
        previous = Path(previous)
        if read(previous / "identity.json") != identity:
            raise ValueError("Changed context comparison continuation identity")
        for path in list(previous.glob("shard-*.json")) + [
            previous / "retrieval-plan.json"
        ]:
            shutil.copy2(path, output / path.name)
    atomic_json(output / "identity.json", identity)
    plan_path = output / "retrieval-plan.json"
    if plan_path.exists():
        saved = read(plan_path)
        if saved.get("identity") != identity or sha(
            json.dumps(saved["queries"], sort_keys=True).encode()
        ) != saved.get("queries_sha256"):
            raise ValueError("Changed source-only retrieval plan")
    else:
        # Freeze retrieval once, before either assembly is forwarded. No gold input.
        queries = []
        for q in questions:
            sections = stories[q["story"]]["sections"]
            queries.append(
                {
                    "id": q["id"],
                    "story": q["story"],
                    "question_sha256": sha(q["question"].encode()),
                    "section_ids": [
                        sections[i]["id"] for i in retrieve(q["question"], sections)
                    ],
                }
            )
        saved = {
            "identity": identity,
            "queries": queries,
            "queries_sha256": sha(json.dumps(queries, sort_keys=True).encode()),
        }
        atomic_json(plan_path, saved)
    if [r["id"] for r in saved["queries"]] != [q["id"] for q in questions]:
        raise ValueError("Changed or overlapping retrieval-plan query coverage")
    for q, row in zip(questions, saved["queries"]):
        known = {s["id"] for s in stories[q["story"]]["sections"]}
        if (
            row["story"] != q["story"]
            or row["question_sha256"] != sha(q["question"].encode())
            or len(row["section_ids"]) != min(4, len(known))
            or len(set(row["section_ids"])) != len(row["section_ids"])
            or not set(row["section_ids"]) <= known
        ):
            raise ValueError("Changed retrieval-plan question/source geometry")
    plan = {r["id"]: r["section_ids"] for r in saved["queries"]}
    import torch
    import importlib.metadata
    import platform
    from transformers import AutoTokenizer
    from src.full_generator_training import strict_model

    if (
        str(torch.__version__) != lineage["torch"]
        or platform.python_version() != lineage["python"]
        or any(
            importlib.metadata.version(k) != v
            for k, v in lineage["environment"].items()
        )
    ):
        raise ValueError("Context comparison requires the frozen baseline runtime")
    torch.set_num_threads(lineage["threads"])
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(run) / f"epoch-{epoch}"
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    tokenizer.model_max_length = 512
    model, loading = strict_model(checkpoint)
    combined = []
    deadline = time.monotonic() + max_seconds
    expected_paths = {
        output / f"shard-{start:05d}.json"
        for start in range(0, len(questions), shard_size)
    }
    if not set(output.glob("shard-*.json")) <= expected_paths:
        raise ValueError("Unexpected comparison shards")
    for start in range(0, len(questions), shard_size):
        path = output / f"shard-{start:05d}.json"
        batch = questions[start : start + shard_size]
        if path.exists():
            shard = read(path)
            if (
                shard.get("identity") != identity
                or [p["id"] for p in shard["pairs"]] != [q["id"] for q in batch]
                or sha(
                    json.dumps(shard["pairs"], sort_keys=True, allow_nan=False).encode()
                )
                != shard.get("pairs_sha256")
            ):
                raise ValueError("Changed or overlapping paired comparison shard")
        else:
            pairs = evaluate_pairs(model, tokenizer, stories, batch, plan)
            shard = {
                "identity": identity,
                "pairs": pairs,
                "pairs_sha256": sha(
                    json.dumps(pairs, sort_keys=True, allow_nan=False).encode()
                ),
            }
            atomic_json(path, shard)
        combined.extend(shard["pairs"])
        print(
            json.dumps(
                {"comparison_QA_completed": len(combined), "required": len(questions)}
            ),
            flush=True,
        )
        if time.monotonic() >= deadline:
            break
    complete = len(combined) == len(questions)
    status = {
        "complete": complete,
        "questions_completed": len(combined),
        "questions_required": len(questions),
        "identity": identity,
        "loading_info": loading,
        "test_accessed": False,
        "final_test_authorized": False,
    }
    if complete:
        baseline_rows, candidate_rows = audit_pairs(
            stories, questions, combined, plan, tokenizer
        )
        replay = all(
            a["answer"] == b["answer"] and a["citations"] == b["citations"]
            for a, b in zip(baseline_rows, cached)
        )
        baseline = summarize(baseline_rows)
        candidate = summarize(candidate_rows)
        interval = story_interval(
            combined, protocol["interval"]["resamples"], protocol["interval"]["seed"]
        )
        status.update(
            {
                "baseline": baseline,
                "candidate": candidate,
                "baseline_replay_exact": replay,
                "story_paired_F1_interval": interval,
                "decision": promotion(baseline, candidate, interval, replay),
            }
        )
    atomic_json(output / "comparison-status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "generator-protocol",
        "context-protocol",
        "run",
        "selection",
        "baseline-directory",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--previous")
    parser.add_argument("--max-seconds", type=float, default=37800)
    args = parser.parse_args()
    try:
        result = run_comparison(
            args.data,
            args.generator_protocol,
            args.context_protocol,
            args.run,
            args.selection,
            args.baseline_directory,
            args.output,
            previous=args.previous,
            max_seconds=args.max_seconds,
        )
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, "Context comparison refused: " + str(error) + "\n")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "questions": result["questions_completed"],
                "test_accessed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
