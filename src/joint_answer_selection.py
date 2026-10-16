"""Select a global answer policy from complete, paired validation artifacts."""

import math

from src.context_validation import promotion, story_interval
from src.full_validation_selection import FULL_VALIDATION_IDS_SHA256

ARMS = ("bm25_standard", "bm25_windows", "reranked_standard", "reranked_windows")


def sha256_value(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
        or value == "0" * 64
    ):
        raise ValueError("Actual SHA256 identity required")
    return value


def adapted_checkpoint(selection):
    """Keep the trained arm distinct even when BM25 wins retrieval selection."""
    if (
        selection.get("status") != "validation_checkpoint_selection"
        or selection.get("task") != "reranker"
        or selection.get("validation_questions") != 2372
        or selection.get("validation_ids_sha256") != FULL_VALIDATION_IDS_SHA256
        or selection.get("test_accessed") is not False
    ):
        raise ValueError("Complete frozen reranker validation selection required")
    candidates = [c for c in selection["candidates"] if c.get("kind") == "adapted"]
    if len(candidates) != 2 or {c.get("epoch") for c in candidates} != {1, 2}:
        raise ValueError("Both adapted epochs required")
    for c in candidates:
        sha256_value(c["weights_sha256"])
        value = c["validation"]["recall_at_4"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Invalid adapted recall")
    return min(candidates, key=lambda c: (-c["validation"]["recall_at_4"], c["epoch"]))


def select_paired(rows, expected_ids, replay_exact):
    """Input rows must already be independently rescored against source references."""
    import hashlib
    import json

    if (
        len(expected_ids) != 2372
        or len(set(expected_ids)) != 2372
        or hashlib.sha256(json.dumps(expected_ids).encode()).hexdigest()
        != FULL_VALIDATION_IDS_SHA256
    ):
        raise ValueError("Exact full validation cohort required")
    if [r["id"] for r in rows] != expected_ids or replay_exact is not True:
        raise ValueError("Complete ordered paired replay required")
    for r in rows:
        if (
            set(r["arms"]) != set(ARMS)
            or not isinstance(r["story"], str)
            or not r["story"]
        ):
            raise ValueError("Four arms and source story required")
        for metrics in r["arms"].values():
            for key in ("f1", "em", "citation_validity"):
                value = metrics[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError("Invalid answer metric")
    metrics = {
        arm: {
            k: sum(r["arms"][arm][k] for r in rows) / 2372
            for k in ("f1", "em", "citation_validity")
        }
        for arm in ARMS
    }
    comparisons = {}
    eligible = [ARMS[0]]
    for arm in ARMS[1:]:
        pairs = [
            {
                "id": r["id"],
                "story": r["story"],
                "baseline": r["arms"][ARMS[0]],
                "candidate": r["arms"][arm],
            }
            for r in rows
        ]
        interval = story_interval(pairs)
        decision = promotion(metrics[ARMS[0]], metrics[arm], interval, True)
        comparisons[arm] = {
            "interval": interval,
            "gates": decision["gates"],
            "eligible": decision["promoted"],
        }
        if decision["promoted"]:
            eligible.append(arm)
    chosen = max(eligible, key=lambda arm: metrics[arm]["f1"])
    return {
        "schema": "joint_answer_selection_v1",
        "selected_arm": chosen,
        "metrics": metrics,
        "comparisons": comparisons,
        "validation_questions": 2372,
        "validation_ids_sha256": FULL_VALIDATION_IDS_SHA256,
        "tie_order": list(ARMS),
        "interval_interpretation": "Descriptive reused-validation intervals across three comparisons; not unbiased or multiplicity-adjusted evidence",
        "test_accessed": False,
        "final_test_authorized": False,
    }
