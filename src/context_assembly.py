"""Question-only source-window assembly; no answer or gold-evidence input."""

from collections import Counter
import math

from src.data.prepare_fairytale import normalized

STOPWORDS = frozenset(
    "a an the is are was were be been being do does did have has had can could will would shall should may might must to of in on at by for from with about into over under and or but if as than that this these those it its they them their he him his she her we our you your who whom whose what which when where why how".split()
)


def allocate_budget(lengths, capacity):
    if (
        type(capacity) is not int
        or capacity < len(lengths)
        or not lengths
        or any(type(n) is not int or n < 1 for n in lengths)
    ):
        raise ValueError(
            "Positive source lengths and sufficient context capacity required"
        )
    allocation = [min(n, capacity // len(lengths)) for n in lengths]
    remaining = capacity - sum(allocation)
    while remaining and any(a < n for a, n in zip(allocation, lengths)):
        for i, length in enumerate(lengths):
            if remaining and allocation[i] < length:
                allocation[i] += 1
                remaining -= 1
    return allocation


def choose_window(question, text, offsets, capacity):
    """BM25 over fixed token windows; ties choose the earliest source window."""
    count = len(offsets)
    if not count or type(capacity) is not int or capacity < 1:
        raise ValueError(
            "Nonempty tokenized source and positive window capacity required"
        )
    capacity = min(capacity, count)
    last = count - capacity
    starts = sorted(set(range(0, last + 1, max(1, capacity // 2))) | {last})
    windows = [
        (offsets[start][0], offsets[start + capacity - 1][1]) for start in starts
    ]
    tokens = [normalized(text[start:end]).split() for start, end in windows]
    query = sorted(set(normalized(question).split()) - STOPWORDS)
    average = sum(map(len, tokens)) / len(tokens)
    frequencies = {word: sum(word in row for row in tokens) for word in query}
    scores = []
    for row in tokens:
        counts = Counter(row)
        score = 0.0
        for word in query:
            frequency = counts[word]
            idf = math.log(
                1 + (len(tokens) - frequencies[word] + 0.5) / (frequencies[word] + 0.5)
            )
            score += (
                idf
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * len(row) / max(1, average)))
            )
        scores.append(score)
    return windows[max(range(len(windows)), key=lambda i: (scores[i], -windows[i][0]))]


def assemble_question_windows(question, sections, indices, tokenizer, max_tokens=512):
    """Return only exact source slices within a verified total token budget."""
    if (
        not isinstance(question, str)
        or not question.strip()
        or type(max_tokens) is not int
        or not 32 <= max_tokens <= 512
    ):
        raise ValueError(
            "Nonempty question and a context budget between 32 and 512 required"
        )
    if (
        not isinstance(indices, (list, tuple))
        or not 1 <= len(indices) <= 4
        or any(type(i) is not int or not 0 <= i < len(sections) for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("One to four distinct valid ranked section indices required")
    chosen = [sections[i] for i in indices]
    if len({s["id"] for s in chosen}) != len(chosen):
        raise ValueError("Retrieved section IDs must be distinct")
    encoded = []
    for section in chosen:
        text = section["text"]
        if (
            not isinstance(text, str)
            or not isinstance(section["id"], str)
            or not section["id"]
            or type(section["start"]) is not int
            or type(section["end"]) is not int
            or section["start"] < 0
            or section["end"] - section["start"] != len(text)
        ):
            raise ValueError("Invalid exact source-section geometry")
        row = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = row["offset_mapping"]
        if (
            len(row["input_ids"]) != len(offsets)
            or not offsets
            or any(
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(text)
                for start, end in offsets
            )
        ):
            raise ValueError(
                "Fast tokenizer must supply nonempty exact character offsets"
            )
        encoded.append(offsets)
    prefix = "question: " + question + " context: "
    remaining = (
        max_tokens
        - len(tokenizer.encode(prefix, add_special_tokens=False))
        - 3
        - 2 * (len(chosen) - 1)
    )
    if remaining < max(16, len(chosen)):
        raise ValueError("Question leaves insufficient context capacity")
    budgets = allocate_budget([len(row) for row in encoded], remaining)
    pieces, citations = [], []
    for section, offsets, capacity in zip(chosen, encoded, budgets):
        start, end = choose_window(question, section["text"], offsets, capacity)
        piece = section["text"][start:end]
        pieces.append(piece)
        citations.append(
            {
                "section": section["id"],
                "start": section["start"] + start,
                "end": section["start"] + end,
                "text": piece,
            }
        )
    context = prefix + "\n\n".join(pieces)
    while len(tokenizer.encode(context)) > max_tokens and pieces:
        pieces[-1] = pieces[-1][:-1]
        citations[-1]["end"] -= 1
        citations[-1]["text"] = pieces[-1]
        if not pieces[-1]:
            pieces.pop()
            citations.pop()
        context = prefix + "\n\n".join(pieces)
    if not citations or len(tokenizer.encode(context)) > max_tokens:
        raise ValueError("No source evidence fits the declared token budget")
    return context, citations
