import json
from pathlib import Path

import pytest

from src import context_validation as context


def pairs(delta):
    return [
        {
            "id": str(i),
            "story": "one" if i < 2 else "two",
            "baseline": {"f1": 0.2},
            "candidate": {"f1": 0.2 + delta},
        }
        for i in range(5)
    ]


def test_story_paired_interval_preserves_pairs_and_zero_gain():
    result = context.story_interval(pairs(0.1), resamples=200, seed=42)
    assert result["stories"] == 2 and result["questions"] == 5
    assert result["low"] == pytest.approx(0.1) and result["high"] == pytest.approx(0.1)
    assert context.story_interval(pairs(0), resamples=200)["high"] == 0
    duplicate = pairs(0.1)
    duplicate[-1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        context.story_interval(duplicate)


@pytest.mark.parametrize(
    "change", ["small_gain", "em_regression", "source_regression", "interval", "replay"]
)
def test_each_prespecified_promotion_gate_can_block_selection(change):
    baseline = {"f1": 0.2, "em": 0.1, "citation_validity": 1.0}
    candidate = {"f1": 0.22, "em": 0.1, "citation_validity": 1.0}
    interval = {"low": 0.001, "high": 0.04}
    replay = True
    assert context.promotion(baseline, candidate, interval, replay)["promoted"]
    if change == "small_gain":
        candidate["f1"] = 0.209
    if change == "em_regression":
        candidate["em"] = 0.09
    if change == "source_regression":
        candidate["citation_validity"] = 0.999
    if change == "interval":
        interval["low"] = -0.001
    if change == "replay":
        replay = False
    decision = context.promotion(baseline, candidate, interval, replay)
    assert not decision["promoted"] and decision["selected_context"] == "source_order"


def test_malformed_protocol_and_changed_decoding_fail_before_inference(tmp_path):
    original = context.read(Path("docs/context-assembly-protocol.json"))
    assert (
        context.check_protocol("docs/context-assembly-protocol.json")[
            "validation_questions"
        ]
        == 2372
    )
    for malformed in (
        [],
        {**original, "candidate": []},
        {**original, "decoding": {"num_beams": 1}},
        {**original, "source_manifest_sha256": "changed"},
    ):
        path = tmp_path / "protocol.json"
        path.write_text(json.dumps(malformed))
        with pytest.raises(ValueError):
            context.check_protocol(path)


def test_partial_checkpoint_gate_precedes_any_model_forward(tmp_path, monkeypatch):
    def partial(*args):
        raise ValueError("Training is incomplete")

    monkeypatch.setattr(context, "selected_checkpoint", partial)
    monkeypatch.setattr(
        context,
        "evaluate_pairs",
        lambda *args: pytest.fail("No partial model may be forwarded"),
    )
    with pytest.raises(ValueError, match="incomplete"):
        context.run_comparison(
            tmp_path,
            tmp_path,
            "docs/context-assembly-protocol.json",
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 10**400])
def test_promotion_rejects_malformed_or_nonfinite_scores(value):
    with pytest.raises(ValueError):
        context.promotion(
            {"f1": 0.2, "em": 0.1, "citation_validity": 1.0},
            {"f1": value, "em": 0.1, "citation_validity": 1.0},
            {"low": 0.0, "high": 0.1},
            True,
        )


def test_interval_resamples_whole_stories_not_individual_questions():
    rows = [
        {
            "id": "positive",
            "story": "short",
            "baseline": {"f1": 0.0},
            "candidate": {"f1": 1.0},
        }
    ]
    rows += [
        {
            "id": str(i),
            "story": "long",
            "baseline": {"f1": 0.0},
            "candidate": {"f1": 0.0},
        }
        for i in range(9)
    ]
    result = context.story_interval(rows, resamples=1000, seed=42)
    assert result["low"] == 0.0
    assert result["high"] == 1.0
    assert result["stories"] == 2 and result["questions"] == 10
