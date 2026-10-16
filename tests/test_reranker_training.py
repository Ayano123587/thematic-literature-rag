from src.reranker_training import build_groups, evaluate_sections, SectionReranker
import pytest


def fixture():
    sections = [{"id": "1", "text": "Mira hid the silver key."},
                {"id": "2", "text": "She later returned the key."},
                {"id": "3", "text": "A wolf howled in the forest."},
                {"id": "4", "text": "A cook prepared soup."},
                {"id": "5", "text": "The sun rose."},
                {"id": "6", "text": "Rain fell."}]
    stories = {"story": {"sections": sections}}
    questions = [{"id": "story:row1:1", "story": "story", "question": "What happened to the key?",
                  "gold_sections": ["1", "2"], "origin": "fixture", "attribute": "action", "explicit": "explicit"}]
    return stories, questions


def test_all_positive_sections_supervised_and_other_gold_never_negative():
    stories, questions = fixture()
    groups, skipped = build_groups(stories, questions)
    assert skipped == 0
    assert {g["positive_section"] for g in groups} == {"1", "2"}
    assert all(not set(g["section_ids"][1:]) & {"1", "2"} for g in groups)
    assert all(len(g["section_ids"]) == 5 for g in groups)
    assert groups == build_groups(stories, questions)[0]


def test_no_negative_questions_are_counted_not_given_fake_negative():
    stories, questions = fixture()
    questions[0]["gold_sections"] = [s["id"] for s in stories["story"]["sections"]]
    assert build_groups(stories, questions) == ([], 1)


def test_validation_ranking_inputs_are_question_and_source_only():
    stories, questions = fixture()
    class SourceOnlyScorer:
        def predict(self, pairs, **kwargs):
            assert all(isinstance(q, str) and isinstance(text, str) for q, text in pairs)
            # Favor relevant source sections based on source content, never labels.
            return [10.0 if "key" in text else 0.0 for q, text in pairs]
    report = evaluate_sections(stories, questions, SourceOnlyScorer())
    assert report["recall_at_4"] == 1.0
    assert report["precision_at_4"] == 0.5
    assert report["candidate_recall_at_12"] == 1.0


def test_malformed_ranking_capacity_rejected_before_model_access():
    with pytest.raises(ValueError):
        SectionReranker(None, None, batch_size=0)
    with pytest.raises(ValueError):
        SectionReranker(None, None, max_tokens=True)
