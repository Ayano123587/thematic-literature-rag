"""Pure software fixtures; no model loads or held-out corpus access."""

import hashlib
import sys
from types import SimpleNamespace

import pytest

from src.models import indexed_narrative_reader as bridge


@pytest.fixture
def setup(tmp_path, monkeypatch):
    passages, hashes = [], {}
    for book, text in [
        ("alpha", "A shared theme here."),
        ("beta", "Another theme there."),
    ]:
        (tmp_path / (book + ".txt")).write_text(text, encoding="utf-8")
        hashes[book] = hashlib.sha256(text.encode()).hexdigest()
        passages.append(
            SimpleNamespace(
                book=book,
                id=f"{book}:2-{len(text)}",
                start=2,
                end=len(text),
                text=text[2:],
            )
        )
    calls = []
    index = SimpleNamespace(
        passages=passages, source_hashes=hashes, dense=[[1.0, 0.0], [0.0, 1.0]]
    )

    def search(query, k, book, mode):
        calls.append(("search", book, mode, k))
        return [(p, 0.1) for p in index.passages if p.book == book][:k]

    index.search = search
    monkeypatch.setattr(bridge, "load_index", lambda *a: index)
    loaded = (None, None, None, {"selected_arm": "reranked_windows"}, None)

    def load(*args):
        calls.append(("load",))
        return loaded

    def answer(question, story, actual):
        assert actual is loaded
        calls.append(("answer", story["id"]))
        section = story["sections"][0]
        return {
            "story": story["id"],
            "source_sha256": story["source_sha256"],
            "selected_arm": "reranked_windows",
            "answer": "fixture",
            "retrieved_evidence": [
                dict(
                    section=section["id"],
                    start=section["start"],
                    end=section["end"],
                    text=section["text"],
                )
            ],
        }

    monkeypatch.setattr(bridge, "load_reader", load)
    monkeypatch.setattr(bridge, "answer_story", answer)

    def run(**kwargs):
        args = dict(
            question="theme",
            index_path="fixture",
            index_sha256="a" * 64,
            sources=tmp_path,
            books=["alpha", "beta"],
            artifacts="fixture",
            contract_sha256="b" * 64,
        )
        args.update(kwargs)
        return bridge.indexed_answers(**args)

    return tmp_path, index, calls, run


def test_per_book_exact_offsets_and_one_unchanged_reader(setup):
    folder, index, calls, run = setup
    result = run(candidates=2, mode="bm25")
    assert calls.count(("load",)) == 1
    assert calls.index(("load",)) > calls.index(("search", "beta", "bm25", 2))
    for output, passage in zip(result["books"], index.passages):
        answer = output["result"]
        assert answer["selected_arm"] == "reranked_windows"
        citation = answer["retrieved_evidence"][0]
        assert citation["section"] == passage.id
        assert (
            citation["text"]
            == (folder / (passage.book + ".txt")).read_text()[
                citation["start"] : citation["end"]
            ]
        )
    assert len(result["cross_book_connections"]) == 1
    assert "do not measure" in result["scope"]


@pytest.mark.parametrize("fault", ["missing", "changed", "empty", "offset", "hash"])
def test_all_requested_sources_before_search_or_reader(setup, fault):
    folder, index, calls, run = setup
    path = folder / "beta.txt"
    if fault == "missing":
        path.unlink()
    elif fault == "changed":
        path.write_text("modified")
    elif fault == "empty":
        path.write_text("")
        index.source_hashes["beta"] = hashlib.sha256(b"").hexdigest()
    elif fault == "offset":
        index.passages[1].start = 0
    else:
        del index.source_hashes["beta"]
    with pytest.raises(ValueError):
        run()
    assert calls == []


def test_empty_retrieval_abstains_without_reader(setup):
    folder, index, calls, run = setup
    index.search = lambda *a, **kw: []
    assert all(b["abstained"] for b in run()["books"])
    assert calls == []


def test_cross_book_candidate_rejected_before_reader(setup):
    folder, index, calls, run = setup
    index.search = lambda *a, **kw: [(index.passages[1], 0.1)]
    with pytest.raises(ValueError, match="Cross-book"):
        run()
    assert not calls


@pytest.mark.parametrize(
    "fault", ["section", "offset", "text", "arm", "story", "empty"]
)
def test_reader_citation_or_policy_tamper_refused(setup, monkeypatch, fault):
    folder, index, calls, run = setup
    original = bridge.answer_story

    def corrupted(*args):
        result = original(*args)
        if fault == "section":
            result["retrieved_evidence"][0]["section"] = "beta:2-20"
        elif fault == "offset":
            result["retrieved_evidence"][0]["start"] = 0
        elif fault == "text":
            result["retrieved_evidence"][0]["text"] = "unsupported"
        elif fault == "arm":
            result["selected_arm"] = "bm25_standard"
        elif fault == "story":
            result["story"] = "other"
        else:
            result["retrieved_evidence"] = []
        return result

    monkeypatch.setattr(bridge, "answer_story", corrupted)
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidates": 0},
        {"candidates": 31},
        {"mode": "other"},
        {"books": []},
        {"books": ["alpha", "alpha"]},
        {"books": ["../alpha"]},
        {"question": ""},
    ],
)
def test_input_bounds_before_index(setup, monkeypatch, kwargs):
    monkeypatch.setattr(
        bridge, "load_index", lambda *a: pytest.fail("early index load")
    )
    with pytest.raises(ValueError):
        setup[3](**kwargs)


def test_archive_hash_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "index"
    path.write_bytes(b"fixture archive")
    calls = []
    fake = SimpleNamespace(
        HybridIndex=SimpleNamespace(
            load=lambda stream: calls.append(stream.read()) or "index"
        )
    )
    monkeypatch.setitem(sys.modules, "src.models.train_model", fake)
    with pytest.raises(ValueError, match="before deserialization"):
        bridge.load_index(path, "a" * 64)
    assert calls == []
    assert (
        bridge.load_index(path, hashlib.sha256(path.read_bytes()).hexdigest())
        == "index"
    )
    assert calls == [b"fixture archive"]


def test_cli_forwards_authenticated_inputs_and_json(monkeypatch, capsys):
    from src import indexed_narrative_cli

    received = []
    monkeypatch.setattr(
        bridge,
        "indexed_answers",
        lambda *args: received.append(args) or {"scope": bridge.SCOPE},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reader",
            "theme",
            "--index",
            "index.lrag",
            "--index-sha256",
            "a" * 64,
            "--sources",
            "processed",
            "--books",
            "alpha",
            "beta",
            "--artifacts",
            "package",
            "--contract-sha256",
            "b" * 64,
            "--mode",
            "dense",
            "--candidates",
            "3",
        ],
    )
    indexed_narrative_cli.main()
    assert received == [
        (
            "theme",
            "index.lrag",
            "a" * 64,
            "processed",
            ["alpha", "beta"],
            "package",
            "b" * 64,
            3,
            "dense",
            "cpu",
        )
    ]
    assert "Exploratory" in capsys.readouterr().out


def test_nonunit_vectors_report_actual_cosine(setup):
    folder, index, calls, run = setup
    index.dense = [[3.0, 4.0], [0.0, 8.0]]
    assert run()["cross_book_connections"][0]["cosine_similarity"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]]
)
def test_invalid_selected_vectors_refused_before_reader(setup, vector):
    folder, index, calls, run = setup
    index.dense[1] = vector
    with pytest.raises(ValueError, match="vector"):
        run()
    assert ("load",) not in calls


@pytest.mark.parametrize("books", ["alpha", {"alpha"}, {"alpha": 1}, 1, [None], [[]]])
def test_books_collection_type_before_index(setup, monkeypatch, books):
    monkeypatch.setattr(
        bridge, "load_index", lambda *a: pytest.fail("early index load")
    )
    with pytest.raises(ValueError):
        setup[3](books=books)
