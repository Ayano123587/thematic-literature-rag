"""Audit all pinned FairytaleQA sources without parsing any question/answer CSV."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from src.data.prepare_fairytale import normalized, sha, shingles

REVISION = "a24ddc17364666b7c13a425b9970c87368b03417"
VALIDATION = {"andersen-fairybook", "blue-fairybook", "green-fairybook", "lilac-fairybook"}
TEST = {"grimm-fairybook", "beatrix-potter-fairybook"}


def source_sections(path):
    """Merge adjacent repeated source IDs, retaining every paragraph and offsets.

    The official remarkable-rocket CSV has consecutive, distinct paragraphs
    sharing ID 5. Gold references identify that ID, so its complete text is one
    evidence unit; no unsupported choice of a single paragraph is made.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = []
    for row in rows:
        identity, text = row["section"], row["text"]
        if not identity or not text.strip():
            raise ValueError("Empty source section")
        if grouped and grouped[-1]["id"] == identity:
            grouped[-1]["text"] += "\n\n" + text
        else:
            grouped.append({"id": identity, "text": text})
    if len({r["id"] for r in grouped}) != len(grouped):
        raise ValueError("Nonadjacent repeated section ID is ambiguous")
    offset = 0
    for row in grouped:
        row.update(start=offset, end=offset + len(row["text"]))
        offset = row["end"] + 2
    return grouped, len(rows) - len(grouped)


def audit(source, previous_manifest):
    source = Path(source)
    with (source / "story_meta.csv").open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    old = json.loads(Path(previous_manifest).read_text(encoding="utf-8"))
    old_roles = {r["id"]: r["split"] for r in old["stories"]}
    items = []
    for row in metadata:
        path = source / "data-by-train-split/section-stories" / row["split"] / (row["filename"] + "-story.csv")
        sections, merged = source_sections(path)
        text = "\n\n".join(r["text"] for r in sections)
        role = "validation" if row["origin"] in VALIDATION else "test" if row["origin"] in TEST else "train"
        items.append({"id": row["filename"], "origin": row["origin"], "official": row["split"],
                      "QA_metadata": int(row["questions"]), "role": role,
                      "norm": normalized(text), "grams": shingles(text),
                      "source_sha256": sha(text.encode()), "source_file_sha256": sha(path.read_bytes()),
                      "adjacent_duplicate_rows_merged": merged})
    parents = list(range(len(items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = []
    for i, left in enumerate(items):
        for j in range(i):
            right = items[j]
            reason = None
            if normalized(left["id"]) == normalized(right["id"]):
                reason, overlap = "normalized title", 1.0
            elif left["norm"] == right["norm"]:
                reason, overlap = "normalized text", 1.0
            else:
                overlap = len(left["grams"] & right["grams"]) / max(1, len(left["grams"] | right["grams"]))
                if overlap >= .8:
                    reason = "five-token Jaccard >= .8"
            if reason:
                parents[find(i)] = find(j)
                edges.append({"left": left["id"], "right": right["id"], "reason": reason, "overlap": overlap})
    groups = defaultdict(list)
    for i, row in enumerate(items):
        groups[find(i)].append(row)
    priority = {"train": 0, "validation": 1, "test": 2}
    migrations = []
    for group in groups.values():
        role = max((r["role"] for r in group), key=priority.get)
        for row in group:
            if row["role"] != role:
                migrations.append({"id": row["id"], "previous_role": row["role"], "new_role": role})
            row["role"] = role
    inventory = [{k: v for k, v in row.items() if k not in ("norm", "grams")} for row in items]

    def counts(key, values):
        return {value: {"stories": sum(r[key] == value for r in items),
                        "QA_metadata": sum(r["QA_metadata"] for r in items if r[key] == value)} for value in values}

    return {"source_revision": REVISION, "metadata_sha256": sha((source / "story_meta.csv").read_bytes()),
            "license_sha256": sha((source / "LICENSE").read_bytes()),
            "official_counts": counts("official", ("train", "val", "test")),
            "full_source_family_counts": counts("role", ("train", "validation", "test")),
            "by_origin": counts("origin", sorted({r["origin"] for r in items})),
            "duplicate_edges": edges, "duplicate_group_migrations": migrations,
            "prior_observed_role_migrations": [{"id": r["id"], "old": old_roles[r["id"]], "new": r["role"]}
                                               for r in items if r["id"] in old_roles and old_roles[r["id"]] != r["role"]],
            "old_study_counts": {"train_QA": 3314, "validation_QA_total": 2372,
                                 "validation_QA_scored": 400, "sealed_test_QA_metadata": 325},
            "source_inventory": inventory, "QA_files_parsed": False, "test_answers_inspected": False,
            "count_basis": "Pinned official story metadata; no test CSV parsing",
            "raw_text_or_answers_published": False}


def stage(source, output, audit_path, protocol_path):
    """Stage every audited story and QA CSV; test CSV is copied as bytes only."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Preserve previous staging; use a new output directory")
    report = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    if dataset["audit_normalized_utf8_sha256"] != sha(Path(audit_path).read_text(encoding="utf-8").encode()):
        raise ValueError("Changed full coverage audit")
    if sha((source / "story_meta.csv").read_bytes()) != report["metadata_sha256"] or sha(
            (source / "LICENSE").read_bytes()) != report["license_sha256"]:
        raise ValueError("Changed pinned metadata or dataset license")
    if dataset["full_coverage"]["custom"] != report["full_source_family_counts"]:
        raise ValueError("Protocol/audit coverage mismatch")
    output.mkdir(parents=True)
    stories = []
    for item in report["source_inventory"]:
        name, official, role = item["id"], item["official"], item["role"]
        source_path = source / "data-by-train-split/section-stories" / official / (name + "-story.csv")
        if sha(source_path.read_bytes()) != item["source_file_sha256"]:
            raise ValueError("Changed audited story")
        sections, _ = source_sections(source_path)
        text = "\n\n".join(row["text"] for row in sections)
        if sha(text.encode()) != item["source_sha256"]:
            raise ValueError("Changed processed source")
        qa_path = source / "data-by-train-split/questions" / official / (name + "-questions.csv")
        raw_qa = qa_path.read_bytes()  # No DictReader, test questions/answers remain sealed.
        story = {"id": name, "origin": item["origin"], "split": role, "official_split": official,
                 "sections": sections, "text": text, "source_sha256": item["source_sha256"],
                 "qa_sha256": sha(raw_qa)}
        directory = output / role
        directory.mkdir(exist_ok=True)
        (directory / (name + ".qa.csv")).write_bytes(raw_qa)
        (directory / (name + ".story.json")).write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8", newline="\n")
        stories.append({k: story[k] for k in ("id", "origin", "split", "official_split", "source_sha256", "qa_sha256")})
    manifest = {"stories": stories, "source_revision": REVISION,
                "protocol_sha256": sha(Path(protocol_path).read_bytes()),
                "audit_normalized_utf8_sha256": dataset["audit_normalized_utf8_sha256"],
                "counts_metadata": report["full_source_family_counts"], "QA_parsed": False,
                "test_QA_copied_as_bytes_only": True, "source_family_split_not_official_benchmark": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--previous-manifest")
    parser.add_argument("--audit-file")
    parser.add_argument("--stage-protocol")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage_protocol:
        if not args.audit_file:
            parser.error("--stage-protocol requires --audit-file")
        result = stage(args.source, args.output, args.audit_file, args.stage_protocol)
        print(json.dumps({"stories": len(result["stories"]), "QA_parsed": False}))
    else:
        if not args.previous_manifest:
            parser.error("audit requires --previous-manifest")
        result = audit(args.source, args.previous_manifest)
        with Path(args.output).open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({"official": result["official_counts"], "full_custom": result["full_source_family_counts"],
                          "duplicate_edges": result["duplicate_edges"], "migrations": result["prior_observed_role_migrations"]}))
