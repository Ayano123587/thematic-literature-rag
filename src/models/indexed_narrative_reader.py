"""Exploratory per-book answers using an authenticated, frozen narrative reader."""

import hashlib
import math
import io
from pathlib import Path
import re

SCOPE = (
    "Exploratory indexed-book answers. FairytaleQA supplied-story scores do not "
    "measure book routing, adult-novel interpretation or comparative reasoning. "
    "Citations identify source locations, not entailment or confidence."
)


def load_index(path, expected_sha256):
    # Authenticate before importing the numerical index implementation.
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("Externally supplied index SHA256 required")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Index archive SHA256 differs before deserialization")
    from src.models.train_model import HybridIndex

    return HybridIndex.load(io.BytesIO(raw))


def load_reader(folder, contract_sha256, device):
    from src.models.frozen_narrative_reader import load

    return load(folder, contract_sha256, device)


def answer_story(question, story, loaded):
    from src.models.frozen_narrative_reader import answer

    return answer(question, story, loaded)


def indexed_answers(
    question,
    index_path,
    index_sha256,
    sources,
    books,
    artifacts,
    contract_sha256,
    candidates=6,
    mode="hybrid",
    device="cpu",
):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Nonempty question of at most 2000 characters required")
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Unsupported retrieval mode")
    if type(candidates) is not int or not 1 <= candidates <= 30:
        raise ValueError("Require 1 to 30 candidate passages per book")
    if not isinstance(books, (list, tuple)) or not books or len(books) > 20:
        raise ValueError("Require a list or tuple of 1 to 20 requested books")
    if any(not isinstance(book, str) for book in books) or len(set(books)) != len(
        books
    ):
        raise ValueError("Require 1 to 20 distinct requested books")
    if any(
        not isinstance(b, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", b)
        for b in books
    ):
        raise ValueError("Invalid book identifier")
    index = load_index(index_path, index_sha256)
    texts = {}
    # All sources and every indexed passage are checked before search can load an encoder.
    for book in books:
        path = Path(sources) / (book + ".txt")
        expected = getattr(index, "source_hashes", {}).get(book)
        if not expected or not path.is_file() or path.is_symlink():
            raise ValueError("Missing authenticated full source for " + book)
        text = path.read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError("Full source SHA256 differs for " + book)
        passages = [p for p in index.passages if p.book == book]
        if not text.strip() or not passages:
            raise ValueError("Empty source or indexed book " + book)
        for p in passages:
            if (
                type(p.start) is not int
                or type(p.end) is not int
                or not 0 <= p.start < p.end <= len(text)
                or text[p.start : p.end] != p.text
                or p.id != f"{book}:{p.start}-{p.end}"
            ):
                raise ValueError("Indexed source offset differs")
        texts[book] = text
    prepared, evidence, vectors = {}, {}, {}
    positions = {p.id: i for i, p in enumerate(index.passages)}
    if len(positions) != len(index.passages):
        raise ValueError("Duplicate indexed passage IDs")
    for book in books:
        hits = index.search(question, k=candidates, book=book, mode=mode)
        if len(hits) > candidates or len({p.id for p, _ in hits}) != len(hits):
            raise ValueError("Candidate count or identities differ")
        sections, rows = [], []
        for p, score in hits:
            if (
                p.book != book
                or p.id not in positions
                or index.passages[positions[p.id]] != p
                or not math.isfinite(float(score))
            ):
                raise ValueError("Cross-book or unauthenticated retrieval candidate")
            sections.append(
                {"id": p.id, "start": p.start, "end": p.end, "text": p.text}
            )
            rows.append(
                {
                    "citation": p.id,
                    "start": p.start,
                    "end": p.end,
                    "text": p.text,
                    "retrieval_score": float(score),
                }
            )
            vector = tuple(float(x) for x in index.dense[positions[p.id]])
            if not vector or not all(math.isfinite(x) for x in vector):
                raise ValueError("Empty or nonfinite selected evidence vector")
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Zero or nonfinite selected evidence vector norm")
            vectors[p.id] = tuple(x / norm for x in vector)
        evidence[book] = rows
        if sections:
            prepared[book] = {
                "id": book,
                "text": texts[book],
                "source_sha256": index.source_hashes[book],
                "sections": sections,
            }
    links = []
    for i, left in enumerate(books):
        for right in books[i + 1 :]:
            for a in evidence[left]:
                for b in evidence[right]:
                    va, vb = vectors[a["citation"]], vectors[b["citation"]]
                    if len(va) != len(vb):
                        raise ValueError("Embedding dimensions differ")
                    similarity = sum(float(x) * float(y) for x, y in zip(va, vb))
                    if not math.isfinite(similarity):
                        raise ValueError("Nonfinite evidence similarity")
                    links.append(
                        {
                            "left": a["citation"],
                            "right": b["citation"],
                            "cosine_similarity": similarity,
                        }
                    )
    # No reader load for empty retrieval. Selected global arm remains untouched.
    loaded = load_reader(artifacts, contract_sha256, device) if prepared else None
    results = []
    for book in books:
        if book not in prepared:
            results.append(
                {"book": book, "abstained": True, "reason": "No retrieved evidence"}
            )
            continue
        story = prepared[book]
        result = answer_story(question, story, loaded)
        if (
            result.get("story") != book
            or result.get("source_sha256") != story["source_sha256"]
            or result.get("selected_arm") != loaded[3]["selected_arm"]
        ):
            raise ValueError("Reader source or selected arm differs")
        if not result.get("retrieved_evidence"):
            raise ValueError("Reader returned no source citations")
        section_map = {s["id"]: s for s in story["sections"]}
        for citation in result.get("retrieved_evidence", []):
            section = section_map.get(citation.get("section"))
            start, end = citation.get("start"), citation.get("end")
            if (
                section is None
                or type(start) is not int
                or type(end) is not int
                or not section["start"] <= start < end <= section["end"]
                or texts[book][start:end] != citation.get("text")
            ):
                raise ValueError("Cross-book or invalid reader citation")
        results.append(
            {
                "book": book,
                "abstained": False,
                "result": result,
                "candidate_evidence": evidence[book],
            }
        )
    return {
        "schema": "exploratory-indexed-narrative-v1",
        "question": question,
        "scope": SCOPE,
        "index_sha256": index_sha256,
        "retrieval_mode": mode,
        "candidate_limit_per_book": candidates,
        "books": results,
        "cross_book_connections": sorted(links, key=lambda x: -x["cosine_similarity"]),
        "connection_note": "Embedding similarity suggests comparisons; it does not prove a shared interpretation.",
    }
