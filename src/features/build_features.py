import re
from dataclasses import dataclass


@dataclass
class Passage:
    id: str
    book: str
    start: int
    end: int
    text: str
    progress: float


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def chunk_book(book, text, words=180, overlap=40):
    if words < 2 or not 0 <= overlap < words:
        raise ValueError("Require words >= 2 and 0 <= overlap < words")
    spans = list(re.finditer(r"\S+", text))
    result = []
    for i in range(0, len(spans), words-overlap):
        j = min(i+words, len(spans))
        start, end = spans[i].start(), spans[j-1].end()
        result.append(Passage(f"{book}:{start}-{end}", book, start, end, text[start:end], i/max(1,len(spans))))
        if j == len(spans):
            break
    return result
