import json
from pathlib import Path

import pytest

from src.data.prepare_fairytale import duplicate_reason, read_sections, sha
from src.narrative_training import (assemble, evidence_metrics, lexical_ranking,
                                    load_split, rouge_l, section_ids, token_f1, validation_cohort)
from src.models.narrative_generator import answer, verify_run, verify_story


def test_test_gold_gate_before_any_file_access(tmp_path):
    with pytest.raises(ValueError, match="sealed"):
        load_split(tmp_path / "does-not-exist", "test")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "unfinished"}))
    with pytest.raises(ValueError, match="Invalid frozen"):
        load_split(tmp_path, "test", selection)


def test_duplicate_audit_uses_source_not_question_labels():
    heldout = [{"id": "closed-test", "text": "one two three four five six seven"}]
    assert duplicate_reason({"id": "different", "text": heldout[0]["text"]}, heldout)[1] == "text"
    assert duplicate_reason({"id": "CLOSED TEST", "text": "different"}, heldout)[1] == "title"
    assert duplicate_reason({"id": "new", "text": "completely distinct story"}, heldout) is None


def test_section_offsets_preserve_unicode_and_csv_quotes(tmp_path):
    path = tmp_path / "story.csv"
    path.write_text('section,text\n1,"Mira said, “”hello””."\n2,Another scene.\n', encoding="utf-8")
    sections = read_sections(path)
    text = "\n\n".join(s["text"] for s in sections)
    assert all(text[s["start"]:s["end"]] == s["text"] for s in sections)


def test_real_answer_metrics_not_retrieval_surrogate():
    assert token_f1("The fox", "fox") == (1.0, 1.0)
    assert token_f1("don't", "dont") == (1.0, 1.0)
    assert token_f1("fox ran", "fox slept") == (0.0, 0.5)
    assert rouge_l("one two three", "one three") == pytest.approx(0.8)
    assert evidence_metrics(["1", "2"], ["2", "3"]) == {"recall": 0.5, "precision": 0.5}


def test_lexical_reader_does_not_take_gold():
    sections = [{"text": "A wolf howled."}, {"text": "Mira carried a silver key."}]
    assert lexical_ranking("Who carried the key?", sections)[0] == 1
    assert section_ids("1, 3, 5") == ["1", "3", "5"]
    with pytest.raises(ValueError):
        section_ids("")
    with pytest.raises(ValueError):
        section_ids("1-3")


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text) + (["EOS"] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_truncated_generation_context_citations_are_exact_source():
    sections = [{"id": "1", "text": "A" * 200, "start": 4, "end": 204}]
    context, citations = assemble("Who?", sections, [0], CharacterTokenizer(), 80)
    assert len(CharacterTokenizer().encode(context)) <= 80
    assert len(citations[0]["text"]) < 200
    assert citations[0]["end"] - citations[0]["start"] == len(citations[0]["text"])
    assert ("xxxx" + sections[0]["text"])[citations[0]["start"]:citations[0]["end"]] == citations[0]["text"]
    with pytest.raises(ValueError, match="capacity"):
        assemble("?" * 100, sections, [0], CharacterTokenizer(), 80)


def test_validation_cohort_is_order_invariant_and_label_independent():
    questions = [{"id": str(i), "origin": str(i % 2), "references": ["unused"]} for i in range(20)]
    first = validation_cohort(questions, 3)
    assert len(first) == 6
    assert first == validation_cohort(list(reversed(questions)), 3)
    changed = [{**q, "references": ["changed"]} for q in questions]
    assert [q["id"] for q in first] == [q["id"] for q in validation_cohort(changed, 3)]


def test_generator_rejects_untrained_smoke_and_invalid_source_before_inference(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"status": "frozen", "smoke": True}))
    with pytest.raises(ValueError, match="full frozen"):
        verify_run(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        answer("", {}, None, None, {})
    with pytest.raises(ValueError, match="hash mismatch"):
        answer("Who?", {"text": "Forged source", "source_sha256": "0" * 64}, None, None, {})


def test_duplicate_official_ids_keep_distinct_stable_rows_and_hash_guards(tmp_path):
    folder = tmp_path / "train"
    folder.mkdir()
    story = {"id": "fixture", "text": "Mira slept.", "sections": [
        {"id": "1", "text": "Mira slept.", "start": 0, "end": 11}]}
    (folder / "fixture.story.json").write_text(json.dumps(story))
    qa = folder / "fixture.qa.csv"
    qa.write_text("question_id,question,cor_section,answer1,answer4,attribute1,ex-or-im1\n"
                  "7,Who slept?,1,Mira,,character,explicit\n"
                  "7,What did Mira do?,1,slept,,action,explicit\n")
    manifest = {"stories": [{"id": "fixture", "split": "train", "origin": "fixture-origin",
                              "source_sha256": sha(story["text"].encode()),
                              "qa_sha256": sha(qa.read_bytes())}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, rows = load_split(tmp_path, "train")
    assert [r["id"] for r in rows] == ["fixture:row1:7", "fixture:row2:7"]
    assert [r["source_question_id"] for r in rows] == ["7", "7"]
    qa.write_text(qa.read_text() + "\n")
    with pytest.raises(ValueError, match="QA hash"):
        load_split(tmp_path, "train")


def test_generated_evidence_cannot_use_negative_or_empty_source_coordinates():
    story = {"text": "Mira", "source_sha256": sha(b"Mira"), "sections": [
        {"id": "1", "text": "Mira", "start": -4, "end": 4}]}
    # Python's negative slice would match, but the declared citation is invalid.
    with pytest.raises(ValueError, match="coordinates"):
        verify_story(story)
    story["sections"] = []
    with pytest.raises(ValueError, match="source sections"):
        verify_story(story)
