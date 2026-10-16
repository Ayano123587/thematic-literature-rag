import csv
import json

import pytest

from src.data.full_coverage_audit import audit, source_sections, stage
from src.data.prepare_fairytale import sha
from src.narrative_training import select_validation


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_adjacent_source_id_merge_keeps_every_character_and_offsets(tmp_path):
    path = tmp_path / "story.csv"
    write_csv(path, ["section", "text"], [{"section": "5", "text": "First paragraph."},
              {"section": "5", "text": "Second paragraph."}, {"section": "6", "text": "Last."}])
    sections, merged = source_sections(path)
    assert merged == 1 and len(sections) == 2
    text = "\n\n".join(s["text"] for s in sections)
    assert text == "First paragraph.\n\nSecond paragraph.\n\nLast."
    for section in sections:
        assert text[section["start"]:section["end"]] == section["text"]


def test_nonadjacent_duplicate_source_id_refused(tmp_path):
    path = tmp_path / "story.csv"
    write_csv(path, ["section", "text"], [{"section": k, "text": "Text."} for k in ("1", "2", "1")])
    with pytest.raises(ValueError, match="Nonadjacent"):
        source_sections(path)


def test_source_group_migration_preserves_all_qa_counts_without_qa_files(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "LICENSE").write_text("Fixture license")
    rows = [{"filename": "a", "origin": "first-round", "split": "train", "questions": "3"},
            {"filename": "b", "origin": "grimm-fairybook", "split": "test", "questions": "4"}]
    write_csv(source / "story_meta.csv", list(rows[0]), rows)
    for row in rows:
        write_csv(source / "data-by-train-split/section-stories" / row["split"] / (row["filename"]+"-story.csv"),
                  ["section", "text"], [{"section": "1", "text": "Identical source text."}])
    old = tmp_path / "old.json"; old.write_text(json.dumps({"stories": [{"id": "b", "split": "test"}]}))
    result = audit(source, old)
    assert result["full_source_family_counts"]["test"] == {"stories": 2, "QA_metadata": 7}
    assert result["full_source_family_counts"]["train"]["QA_metadata"] == 0
    assert result["QA_files_parsed"] is False
    assert not (source / "data-by-train-split/questions").exists()
    audit_path = tmp_path / "audit.json"; audit_path.write_text(json.dumps(result))
    policy_path = tmp_path / "protocol.json"
    policy_path.write_text(json.dumps({"dataset": {
        "audit_normalized_utf8_sha256": sha(audit_path.read_text().encode()),
        "full_coverage": {"custom": result["full_source_family_counts"]}}}))
    raw = b"Not parsed as QA during byte staging."
    for row in rows:
        path = source / "data-by-train-split/questions" / row["split"] / (row["filename"]+"-questions.csv")
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    output = tmp_path / "staged"
    staged = stage(source, output, audit_path, policy_path)
    assert len(staged["stories"]) == 2 and staged["test_QA_copied_as_bytes_only"]
    assert (output / "test/a.qa.csv").read_bytes() == raw
    with pytest.raises(ValueError, match="Preserve"):
        stage(source, output, audit_path, policy_path)


def test_full_validation_has_no_inherited_per_origin_cap():
    rows = [{"id": str(i), "origin": "fixture-origin"} for i in range(2372)]
    assert len(select_validation(rows, {})) == 100  # Historical behavior preserved.
    assert len(select_validation(rows, {"validation_sampling": {"mode": "all"}})) == 2372
    with pytest.raises(ValueError, match="Unsupported"):
        select_validation(rows, {"validation_sampling": {"mode": "all", "cap": 400}})
