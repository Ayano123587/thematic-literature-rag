import re

import pytest

from src.context_assembly import (
    allocate_budget,
    assemble_question_windows,
    choose_window,
)


class WordOffsets:
    """Transparent fixture tokenizer; real pinned-tokenizer audit is separate."""

    def __call__(self, text, **unused):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(spans))), "offset_mapping": spans}

    def encode(self, text, add_special_tokens=True):
        return list(
            range(len(re.findall(r"\S+", text)) + (1 if add_special_tokens else 0))
        )


def section(text, name="1", start=0):
    return {"id": name, "text": text, "start": start, "end": start + len(text)}


def test_late_question_window_uses_exact_unicode_source_characters():
    text = "quiet field " * 60 + "café lantern glows in darkness " + "quiet field " * 60
    tokenizer = WordOffsets()
    context, cites = assemble_question_windows(
        "Where does the café lantern glow?",
        [section(text, start=100)],
        [0],
        tokenizer,
        64,
    )
    assert "café lantern" in context and cites[0]["start"] > 100
    assert text[cites[0]["start"] - 100 : cites[0]["end"] - 100] == cites[0]["text"]
    assert len(tokenizer.encode(context)) <= 64


def test_rank_order_budget_sharing_and_source_geometry():
    texts = ["apple " * 100, "pear " * 100, "plum " * 100, "berry " * 100]
    sections = [section(text, str(i), i * 1000) for i, text in enumerate(texts)]
    tokenizer = WordOffsets()
    context, cites = assemble_question_windows(
        "Where is the pear?", sections, [1, 3, 0, 2], tokenizer, 64
    )
    assert [c["section"] for c in cites] == ["1", "3", "0", "2"]
    assert len(tokenizer.encode(context)) <= 64
    for citation in cites:
        source = sections[int(citation["section"])]
        assert (
            source["text"][
                citation["start"] - source["start"] : citation["end"] - source["start"]
            ]
            == citation["text"]
        )
    assert allocate_budget([1, 100, 100], 30) == [1, 15, 14]


def test_no_overlap_or_exact_score_tie_chooses_earliest_source_window():
    tokenizer = WordOffsets()
    text = "apple pear " * 20
    offsets = tokenizer(text)["offset_mapping"]
    assert choose_window("unseen constellation", text, offsets, 8)[0] == 0
    assert choose_window("apple", text, offsets, 8)[0] == 0


@pytest.mark.parametrize("indices", [[-1], [True], [0, 0], [1], []])
def test_invalid_retrieval_indices_rejected(indices):
    with pytest.raises(ValueError, match="indices"):
        assemble_question_windows(
            "A question?", [section("a source passage")], indices, WordOffsets()
        )


def test_question_capacity_and_negative_source_coordinates_fail_closed():
    with pytest.raises(ValueError, match="capacity"):
        assemble_question_windows(
            "word " * 100, [section("a source passage")], [0], WordOffsets(), 32
        )
    with pytest.raises(ValueError, match="geometry"):
        assemble_question_windows(
            "What happens?", [section("a source passage", start=-1)], [0], WordOffsets()
        )
    with pytest.raises(ValueError, match="geometry"):
        assemble_question_windows(
            "What happens?",
            [{**section("a source passage"), "end": 1}],
            [0],
            WordOffsets(),
        )
