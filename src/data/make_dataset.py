"""Opt-in downloads preserve raw source and included license."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from src.config import BOOKS, DATA


def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("Expected Gutenberg license boundary markers are missing")
    return text[start.end():end.start()].strip()


def fetch(acknowledge=False, root=DATA):
    if not acknowledge:
        raise ValueError("Check local copyright law and pass --acknowledge-local-rights")
    records = []
    for key, meta in BOOKS.items():
        url = f"https://www.gutenberg.org/ebooks/{meta['id']}.txt.utf-8"
        req = urllib.request.Request(url, headers={"User-Agent": "LiteraryResearch/0.1"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        body = strip_gutenberg(raw.decode("utf-8-sig"))
        for folder in ("raw", "processed", "external", "interim"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "raw" / f"{key}.txt").write_bytes(raw)
        (root / "processed" / f"{key}.txt").write_text(body, encoding="utf-8")
        records.append(dict(meta, key=key, url=url, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    (root / "external" / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def ingest_document(path):
    path = Path(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    from docling.document_converter import DocumentConverter
    return DocumentConverter().convert(str(path)).document.export_to_markdown()
