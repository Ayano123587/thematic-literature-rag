"""Build private, source-origin held out narrative training data.

Test QA is staged as bytes only. This module does not parse or display it.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def shingles(text, n=5):
    words = normalized(text).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def duplicate_reason(story, heldout):
    """Compare source text only, never answers or retrieval outcomes."""
    own = shingles(story["text"])
    for other in heldout:
        if normalized(story["id"]) == normalized(other["id"]):
            return other["id"], "title", 1.0
        if normalized(story["text"]) == normalized(other["text"]):
            return other["id"], "text", 1.0
        theirs = shingles(other["text"])
        overlap = len(own & theirs) / max(1, len(own | theirs))
        if overlap >= 0.8:
            return other["id"], "five-token Jaccard", overlap
    return None


def read_sections(path):
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    sections, offset = [], 0
    for row in rows:
        text = row["text"]
        if not text.strip():
            raise ValueError(f"Empty source section in {path.name}")
        sections.append({"id": row["section"], "text": text,
                         "start": offset, "end": offset + len(text)})
        offset += len(text) + 2
    if len({s["id"] for s in sections}) != len(sections):
        raise ValueError("Duplicate section IDs")
    return sections


def prepare(source, output, rights, protocol):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a staged study")
    policy = json.loads(Path(protocol).read_text(encoding="utf-8"))
    rights_path = Path(rights)
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    dataset = policy["dataset"]
    metadata = list(csv.DictReader((source / "story_meta.csv").open(encoding="utf-8")))
    val = set(dataset["validation_origins"])
    test = set(dataset["test_origins"])
    excluded = set(dataset["excluded_origins"])
    if val & test or (val | test) & excluded:
        raise ValueError("Overlapping source partitions")
    if not val | test <= set(rights["eligible_origins"]):
        raise ValueError("Held out origins require rights verification")
    stories, manifest = [], []
    for row in metadata:
        origin, official = row["origin"], row["split"]
        if origin in excluded or origin not in rights["eligible_origins"]:
            continue
        split = "validation" if origin in val else "test" if origin in test else "train"
        if split == "train" and official != "train":
            continue
        base = source / "data-by-train-split"
        section_path = base / "section-stories" / official / (row["filename"] + "-story.csv")
        qa_path = base / "questions" / official / (row["filename"] + "-questions.csv")
        sections = read_sections(section_path)
        text = "\n\n".join(s["text"] for s in sections)
        story = {"id": row["filename"], "origin": origin, "split": split,
                 "official_split": official, "sections": sections, "text": text,
                 "source_sha256": sha(text.encode()), "qa_path": qa_path,
                 "qa_sha256": sha(qa_path.read_bytes())}
        stories.append(story)
    heldout = [s for s in stories if s["split"] != "train"]
    removed = []
    retained = []
    for story in stories:
        match = duplicate_reason(story, heldout) if story["split"] == "train" else None
        if match:
            removed.append({"id": story["id"], "matches": match[0],
                            "reason": match[1], "overlap": match[2]})
        else:
            retained.append(story)
    output.mkdir(parents=True)
    for story in retained:
        # Store test CSV verbatim without a DictReader or gold inspection.
        split_dir = output / story["split"]
        split_dir.mkdir(exist_ok=True)
        (split_dir / (story["id"] + ".qa.csv")).write_bytes(story.pop("qa_path").read_bytes())
        (split_dir / (story["id"] + ".story.json")).write_text(
            json.dumps(story, ensure_ascii=False), encoding="utf-8", newline="\n")
        manifest.append({k: story[k] for k in ("id", "origin", "split", "official_split",
                                             "source_sha256", "qa_sha256")})
    report = {"protocol_sha256": sha(Path(protocol).read_bytes()),
              "metadata_sha256": sha((source / "story_meta.csv").read_bytes()),
              "rights_sha256": sha(rights_path.read_bytes()),
              "stories": manifest, "removed_train_duplicates": removed,
              "qa_parsed": False, "test_qa_read_as_bytes_only": True}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("docs/v5-training-protocol.json"))
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.rights, args.protocol)
    print(json.dumps({"stories": len(result["stories"]),
                      "removed_train_duplicates": len(result["removed_train_duplicates"])}))
