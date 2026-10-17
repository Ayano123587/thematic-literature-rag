"""Actual approved validation artifacts exercise package transport without model loading."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest

from src.cloud.narrative_package_runner import (
    authenticate_checkpoint,
    digest,
    validate_citations,
    validate_saved_parity,
)
from src.prepare_narrative_package import prepare, source_closure

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data/interim/joint-full2372-stage-v2"
FROZEN = ROOT / "data/interim/joint-full2372-frozen-inputs-v1"
REVIEW = "b29a295dd140967f65cc5c41ca619058e1a92541fa7acbb033e9a1d6c6fafca2"
RECIPE = "39eb30aeb50c3182e127a22d15d77cf025a6c24a5afab417c7f799fbb2b30db5"


def actual_stage(output):
    if not (STAGE / "root-package-only-review-v1.json").is_file():
        pytest.skip("Actual root-reviewed full2372 result required")
    return prepare(
        ROOT,
        STAGE,
        FROZEN,
        ROOT
        / "data/interim/full-validation-generator-v6-byte-bound/selection/output-v1/full-generator/selection.json",
        STAGE / "dataset/NOTICE__FairytaleQA-LICENSE",
        output,
        dataset="aikyatansinha/portfolio-narrative-package-source-v1",
        kernel_id="aikyatansinha/portfolio-narrative-production-package-v1",
        expected_root_review_sha256=REVIEW,
        expected_recipe_sha256=RECIPE,
    )


def test_actual_full_metadata_stage_render_and_raw_byte_refusal(tmp_path):
    destination = tmp_path / "stage"
    receipt = actual_stage(destination)
    assert len(receipt["source_paths"]) >= 20
    assert "src/models/frozen_narrative_reader.py" in source_closure(ROOT)
    runner = destination / "kernel/run.py"
    assert b"\r" not in runner.read_bytes()
    ns = runpy.run_path(str(runner), run_name="actual_rendered_package_review")
    restored = tmp_path / "restored"
    manifest = ns["restore"](destination / "payload", restored, runner)
    assert len(manifest["files"]) == receipt["payload_files"]
    assert not (restored / "joint-inputs/private-data/test").exists()
    assert "torch" not in sys.modules
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.freeze_narrative_package; import src.models.frozen_narrative_reader; assert 'torch' not in sys.modules",
        ],
        cwd=restored,
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(destination / "upload/package-source.zip") as archive:
        assert len(archive.namelist()) == receipt["expanded_remote_files"]
        for name in archive.namelist():
            assert archive.read(name) == (destination / "payload" / name).read_bytes()
    row = manifest["files"][-1]
    changed = destination / "payload" / row["file"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="authentication before imports"):
        ns["restore"](destination / "payload", tmp_path / "refused", runner)
    assert not (tmp_path / "refused").exists()


def test_actual_review_digest_refuses_before_output(tmp_path):
    if not (STAGE / "root-package-only-review-v1.json").is_file():
        pytest.skip("Actual reviewed artifacts required")
    with pytest.raises(ValueError, match="externally approved"):
        prepare(
            ROOT,
            STAGE,
            FROZEN,
            "unused",
            "unused",
            tmp_path / "no-output",
            dataset="owner/source",
            kernel_id="owner/kernel",
            expected_root_review_sha256="0" * 64,
            expected_recipe_sha256=RECIPE,
        )
    assert not (tmp_path / "no-output").exists()


def test_actual_authenticated_opaque_reranker_checkpoint_no_native_load():
    recipe_path = STAGE / "final-joint-recipe-proposal-v1.json"
    native = (
        ROOT
        / "data/interim/reranker-cancelled-continuation-v1-reviewed/historical-parent3-opaque-cache/run/epoch-1"
    )
    if not recipe_path.is_file() or not (native / "model.safetensors").is_file():
        pytest.skip("Previously authenticated actual opaque artifact cache required")
    recipe = json.loads(recipe_path.read_bytes())
    assert authenticate_checkpoint(native, recipe["reranker"]) == native
    assert digest(native / "receipt.json") == recipe["reranker"]["receipt_sha256"]


def test_actual_joint_answer_citations_refuse_a_different_real_story():
    from src.full_validation_selection import validation_cohort

    if not (STAGE / "terminal-output-v2/joint-answer/shard-00000.json").is_file():
        pytest.skip("Actual complete validation outputs required")
    stories, _ = validation_cohort(
        FROZEN / "private-data",
        json.loads((FROZEN / "generator-protocol.json").read_bytes()),
    )
    row = json.loads(
        (STAGE / "terminal-output-v2/joint-answer/shard-00000.json").read_bytes()
    )["pairs"][0]
    actual = row["arms"]["reranked_windows"]
    story = stories[row["story"]]
    result = {
        "story": row["story"],
        "source_sha256": story["source_sha256"],
        "retrieved_evidence": actual["citations"],
        "answer": actual["answer"],
    }
    validate_citations(result, story)
    another = next(value for key, value in stories.items() if key != story["id"])
    with pytest.raises(ValueError, match="source identity"):
        validate_citations(result, another)


def test_actual_saved_selected_arm_parity_and_different_real_arm_refusal():
    path = STAGE / "terminal-output-v2/joint-answer/shard-00000.json"
    if not path.is_file():
        pytest.skip("Actual audited validation pairs required")
    rows = json.loads(path.read_bytes())["pairs"]
    for row in rows[:2]:
        actual = row["arms"]["reranked_windows"]
        demonstration = {
            "id": row["id"],
            "result": {
                "story": row["story"],
                "selected_arm": "reranked_windows",
                "answer": actual["answer"],
                "retrieved_evidence": actual["citations"],
            },
        }
        validate_saved_parity(demonstration, row, "reranked_windows")
    row = next(
        row
        for row in rows
        if (
            row["arms"]["bm25_standard"]["answer"],
            row["arms"]["bm25_standard"]["citations"],
        )
        != (
            row["arms"]["reranked_windows"]["answer"],
            row["arms"]["reranked_windows"]["citations"],
        )
    )
    other = row["arms"]["bm25_standard"]
    different_real_output = {
        "id": row["id"],
        "result": {
            "story": row["story"],
            "selected_arm": "reranked_windows",
            "answer": other["answer"],
            "retrieved_evidence": other["citations"],
        },
    }
    with pytest.raises(ValueError, match="audited selected-arm"):
        validate_saved_parity(different_real_output, row, "reranked_windows")
