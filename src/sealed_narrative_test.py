"""One-shot full FairytaleQA test release after an externally frozen joint policy."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re

FULL_MANIFEST = "721733169b0c1e7b7c1d07fe1880d3c871d72017be6fc61923be126feea73225"
TEST_FAMILIES = {"beatrix-potter-fairybook", "grimm-fairybook"}
ARMS = {"bm25_standard", "bm25_windows", "reranked_standard", "reranked_windows"}
VALIDATION_IDS = "40d2ddd6b065469a8b947d84dbb630efcaac83acecc695606022116cc1a0f7cb"
RELEASE_SOURCES = {
    "src/sealed_narrative_test.py",
    "src/cloud/sealed_narrative_test_runner.py",
    "src/prepare_sealed_narrative_test.py",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def required_sha(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        or value == "0" * 64
    ):
        raise ValueError("Actual externally frozen SHA256 required")
    return value


def at(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Unsafe sealed release artifact")
    return path


def question_ids_only(raw):
    """CSV byte scanner: retain header and ID bytes, skip all other values."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    position = 0
    width = None
    target = None
    ids = []
    while position < len(raw):
        fields = []
        column = 0
        while True:
            retain = width is None or column == target
            value = bytearray() if retain else None
            quoted = position < len(raw) and raw[position] == 34
            if quoted:
                position += 1
                closed = False
                while position < len(raw):
                    byte = raw[position]
                    position += 1
                    if byte == 34:
                        if position < len(raw) and raw[position] == 34:
                            position += 1
                            if retain:
                                value.append(34)
                        else:
                            closed = True
                            break
                    elif retain:
                        value.append(byte)
                if not closed:
                    raise ValueError("Truncated quoted CSV metadata")
                if position < len(raw) and raw[position] not in (44, 10, 13):
                    raise ValueError("Unexpected bytes after CSV quote")
            else:
                while position < len(raw) and raw[position] not in (44, 10, 13):
                    byte = raw[position]
                    position += 1
                    if byte == 34:
                        raise ValueError("Quote inside unquoted CSV field")
                    if retain:
                        value.append(byte)
            fields.append(bytes(value) if retain else None)
            column += 1
            if position >= len(raw):
                break
            delimiter = raw[position]
            position += 1
            if delimiter == 44:
                continue
            if delimiter == 13 and position < len(raw) and raw[position] == 10:
                position += 1
            break
        if width is None:
            header = [value.decode("utf-8") for value in fields]
            if len(set(header)) != len(header) or header.count("question_id") != 1:
                raise ValueError("Unambiguous question_id CSV column required")
            width = len(header)
            target = header.index("question_id")
        else:
            if len(fields) != width:
                raise ValueError("CSV metadata column count differs")
            identifier = fields[target].decode("utf-8").strip()
            if (
                not identifier
                or len(identifier) > 200
                or any(ord(c) < 32 for c in identifier)
                or identifier in ids
            ):
                raise ValueError("Invalid or duplicate source question ID")
            ids.append(identifier)
    if width is None or not ids:
        raise ValueError("Nonempty CSV metadata required")
    return ids


def authorize_freeze(metadata, root_freeze, expected_root_sha256):
    """Host metadata-only: require actual root-audited package/joint result."""
    if digest(root_freeze) != required_sha(expected_root_sha256):
        raise ValueError("Actual root freeze differs")
    frozen = read(root_freeze)
    metadata = Path(metadata)
    if (
        frozen.get("schema") != "root_frozen_narrative_test_release_v1"
        or frozen.get("joint_validation_questions") != 2372
        or frozen.get("test_authorized") is not True
        or frozen.get("test_previously_scored") is not False
        or frozen.get("test_questions") != 325
        or frozen.get("test_families") != sorted(TEST_FAMILIES)
        or frozen.get("test_manifest_sha256") != FULL_MANIFEST
        or frozen.get("joint_validation_ids_sha256") != VALIDATION_IDS
    ):
        raise ValueError("Actual complete joint policy/package authorization required")
    parent = frozen.get("package_parent")
    if (
        not isinstance(parent, str)
        or len(parent.split("/")) != 3
        or not parent.split("/")[-1].isdigit()
    ):
        raise ValueError("Actual immutable production package parent required")
    contract_path = at(metadata, "inference-contract.json")
    contract = read(contract_path)
    if (
        digest(contract_path) != required_sha(frozen["package_contract_sha256"])
        or contract.get("schema") != "frozen_narrative_inference_v1"
        or contract.get("joint_result_sha256")
        != required_sha(frozen["joint_result_sha256"])
    ):
        raise ValueError("Actual frozen package contract required")
    policy = read(at(metadata, "joint-selection.json"))
    if (
        digest(metadata / "joint-selection.json")
        != contract["files"].get("joint-selection.json")
        or policy.get("schema") != "joint_answer_selection_v1"
        or policy.get("selected_arm") not in ARMS
        or policy.get("selected_arm") != frozen.get("selected_arm")
        or policy.get("validation_questions") != 2372
        or policy.get("validation_ids_sha256") != VALIDATION_IDS
        or policy.get("test_accessed") is not False
    ):
        raise ValueError("Full joint selected arm differs")
    if contract.get("source_sha256") != frozen.get(
        "inference_source_sha256"
    ) or not isinstance(frozen.get("runtime"), dict):
        raise ValueError("Frozen actual source/runtime inventory required")
    if set(frozen.get("release_source_sha256", {})) != RELEASE_SOURCES:
        raise ValueError("Externally frozen release source inventory required")
    for value in frozen["release_source_sha256"].values():
        required_sha(value)
    return frozen


def reserve(ledger, frozen, root_freeze_sha256):
    """Persistent local reservation survives every staging/dispatch/cloud failure."""
    ledger = Path(ledger)
    ledger.mkdir(parents=True, exist_ok=True)
    marker = ledger / "release-reserved.json"
    value = {
        "schema": "sealed_narrative_reservation_v1",
        "package_contract_sha256": frozen["package_contract_sha256"],
        "joint_result_sha256": frozen["joint_result_sha256"],
        "root_freeze_sha256": required_sha(root_freeze_sha256),
        "test_questions": 325,
        "test_families": sorted(TEST_FAMILIES),
    }
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    return marker


def cohort_metadata(data, reservation):
    """After authorization/reservation: hash opaque QA BEFORE ID-only scan."""
    reserved = read(reservation)
    if (
        reserved.get("schema") != "sealed_narrative_reservation_v1"
        or reserved.get("test_questions") != 325
        or reserved.get("test_families") != sorted(TEST_FAMILIES)
    ):
        raise ValueError(
            "Persistent authorized reservation required before test metadata scan"
        )
    required_sha(reserved["package_contract_sha256"])
    required_sha(reserved["root_freeze_sha256"])
    data = Path(data)
    if digest(at(data, "manifest.json")) != FULL_MANIFEST:
        raise ValueError("Original full source manifest required")
    records = [
        r for r in read(data / "manifest.json")["stories"] if r["split"] == "test"
    ]
    if (
        not records
        or {r["origin"] for r in records} != TEST_FAMILIES
        or len({r["id"] for r in records}) != len(records)
    ):
        raise ValueError("Exact held-out source families required")
    files = {"manifest.json": FULL_MANIFEST}
    stories = []
    ids = []
    for record in sorted(records, key=lambda r: r["id"]):
        name = record["id"]
        if Path(name).name != name or not name:
            raise ValueError("Unsafe story ID")
        qa_name = "test/" + name + ".qa.csv"
        story_name = "test/" + name + ".story.json"
        qa = at(data, qa_name)
        raw = qa.read_bytes()
        if sha(raw) != record["qa_sha256"]:
            raise ValueError("QA bytes differ before metadata scan")
        source_ids = question_ids_only(raw)
        row_ids = [
            name + ":row" + str(i) + ":" + q for i, q in enumerate(source_ids, 1)
        ]
        ids.extend(row_ids)
        files[qa_name] = sha(raw)
        files[story_name] = digest(at(data, story_name))
        stories.append(
            {
                "id": name,
                "origin": record["origin"],
                "source_sha256": record["source_sha256"],
                "qa_sha256": record["qa_sha256"],
                "story_file_sha256": files[story_name],
                "questions": len(row_ids),
                "row_ids": row_ids,
            }
        )
    ids.sort()
    if len(ids) != 325 or len(set(ids)) != 325:
        raise ValueError("Exactly all325 eligible unique questions required")
    return {
        "schema": "sealed_narrative_cohort_v1",
        "questions": 325,
        "families": sorted(TEST_FAMILIES),
        "ids": ids,
        "ids_sha256": sha(json.dumps(ids).encode()),
        "stories": stories,
        "files": files,
        "gold_values_decoded": False,
    }


def authenticate_cohort(data, cohort):
    if (
        cohort.get("schema") != "sealed_narrative_cohort_v1"
        or cohort.get("questions") != 325
        or cohort.get("families") != sorted(TEST_FAMILIES)
        or cohort.get("gold_values_decoded") is not False
    ):
        raise ValueError("Frozen full test cohort required")
    ids = cohort["ids"]
    if (
        len(ids) != 325
        or len(set(ids)) != 325
        or ids != sorted(ids)
        or sha(json.dumps(ids).encode()) != cohort["ids_sha256"]
        or sorted(i for story in cohort["stories"] for i in story["row_ids"]) != ids
    ):
        raise ValueError("Frozen full test question IDs differ")
    if (
        digest(at(data, "manifest.json")) != FULL_MANIFEST
        or cohort["files"].get("manifest.json") != FULL_MANIFEST
    ):
        raise ValueError("Original full source manifest differs")
    actual_records = {
        r["id"]: r
        for r in read(at(data, "manifest.json"))["stories"]
        if r["split"] == "test"
    }
    if {s["id"] for s in cohort["stories"]} != set(actual_records):
        raise ValueError("Frozen source family coverage differs")
    expected_files = {"manifest.json"}
    for story in cohort["stories"]:
        record = actual_records[story["id"]]
        if (
            story["origin"] != record["origin"]
            or story["qa_sha256"] != record["qa_sha256"]
            or story["source_sha256"] != record["source_sha256"]
            or story["questions"] != len(story["row_ids"])
        ):
            raise ValueError("Frozen story metadata differs")
        expected_files.update(
            {"test/" + story["id"] + ".qa.csv", "test/" + story["id"] + ".story.json"}
        )
    if set(cohort["files"]) != expected_files:
        raise ValueError("Unlisted or missing sealed source files")
    for name, expected in cohort["files"].items():
        if digest(at(data, name)) != required_sha(expected):
            raise ValueError("Sealed test input bytes changed")


def consume_dispatch(ledger, ticket, push):
    """Explicit caller-provided push; consume first, never automatically retry."""
    ledger = Path(ledger)
    reservation = read(ledger / "release-reserved.json")
    if (
        ticket.get("reservation_sha256") != digest(ledger / "release-reserved.json")
        or ticket.get("package_contract_sha256")
        != reservation["package_contract_sha256"]
    ):
        raise ValueError("Wrong persistent release reservation")
    if read(ledger / "release-bound.json") != ticket:
        raise ValueError("Persistent package/cohort/source binding differs")
    with (ledger / "dispatch-consumed.json").open("x", encoding="utf-8") as handle:
        json.dump(ticket, handle, indent=2)
    return push()


def start_cloud_marker(output, ticket):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "sealed-test-started.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(ticket, handle, indent=2)
    return marker


def parse_gold(data, cohort, marker, ticket):
    """First gold parser: exclusive marker must already exist and match ticket."""
    if read(marker) != ticket:
        raise ValueError("Exclusive cloud release marker required before gold parse")
    authenticate_cohort(data, cohort)
    from src.models.narrative_generator import verify_story
    from src.narrative_training import section_ids

    stories = {}
    questions = []
    for record in cohort["stories"]:
        name = record["id"]
        story = read(at(data, "test/" + name + ".story.json"))
        verify_story(story)
        if story["id"] != name or story["source_sha256"] != record["source_sha256"]:
            raise ValueError("Original test source differs")
        stories[name] = story
        known = {s["id"] for s in story["sections"]}
        raw = at(data, "test/" + name + ".qa.csv").read_bytes()
        for ordinal, row in enumerate(
            csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))), 1
        ):
            refs = list(
                dict.fromkeys(
                    r
                    for r in (
                        row.get("answer1", "").strip(),
                        row.get("answer4", "").strip(),
                    )
                    if r
                )
            )
            gold = section_ids(row["cor_section"])
            identifier = name + ":row" + str(ordinal) + ":" + row["question_id"]
            if (
                not refs
                or not row["question"].strip()
                or not set(gold) <= known
                or identifier != record["row_ids"][ordinal - 1]
            ):
                raise ValueError("Test references/source identity differs")
            questions.append(
                {
                    "id": identifier,
                    "source_question_id": row["question_id"],
                    "source_row_ordinal": ordinal,
                    "story": name,
                    "question": row["question"],
                    "references": refs,
                    "gold_sections": gold,
                    "origin": record["origin"],
                    "attribute": row["attribute1"],
                    "explicit": row["ex-or-im1"],
                }
            )
    questions.sort(key=lambda q: q["id"])
    if [q["id"] for q in questions] != cohort["ids"]:
        raise ValueError("Full test cohort differs after gold parse")
    return stories, questions


def runtime_now():
    import importlib.metadata
    import platform

    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "numpy", "huggingface-hub")
        },
    }


def _evaluate(package, data, cohort, ticket, output, *, root_freeze, root_sha256):
    """Cloud-only selected-policy evaluation; no training, tuning or arm comparison."""
    frozen = authorize_freeze(package, root_freeze, root_sha256)
    if (
        ticket.get("schema") != "sealed_narrative_test_ticket_v1"
        or ticket.get("questions") != 325
        or ticket.get("root_freeze_sha256") != root_sha256
        or ticket.get("package_contract_sha256") != frozen["package_contract_sha256"]
        or ticket.get("selected_arm") != frozen["selected_arm"]
        or ticket.get("runtime") != frozen["runtime"]
        or ticket.get("joint_result_sha256") != frozen["joint_result_sha256"]
        or ticket.get("package_parent") != frozen["package_parent"]
    ):
        raise ValueError("Actual complete package/root/cohort release ticket required")
    if runtime_now() != frozen["runtime"]:
        raise ValueError("Exact frozen inference runtime required")
    expected_sources = {
        "src/" + name: value
        for name, value in frozen["inference_source_sha256"].items()
    } | frozen["release_source_sha256"]
    if ticket.get("source_sha256", {}) != expected_sources:
        raise ValueError("Complete sealed inference/source inventory required")
    for name, expected in ticket["source_sha256"].items():
        if digest(at(Path(__file__).parents[1], name)) != required_sha(expected):
            raise ValueError("Sealed inference/source implementation differs")
    from src.models.frozen_narrative_reader import authenticate, load, answer
    from src.context_validation import score_row
    from src.narrative_training import retrieve, summarize
    import gc

    contract, policy = authenticate(package, ticket["package_contract_sha256"])
    if (
        policy["selected_arm"] != ticket["selected_arm"]
        or sha(json.dumps(cohort, sort_keys=True).encode()) != ticket["cohort_sha256"]
    ):
        raise ValueError("Externally frozen selected policy/cohort required")
    authenticate_cohort(data, cohort)
    marker = start_cloud_marker(output, ticket)
    stories, questions = parse_gold(data, cohort, marker, ticket)
    loaded = load(package, ticket["package_contract_sha256"], "cpu")
    rows = []
    predictions = []
    output = Path(output)
    for ordinal, question in enumerate(questions):
        story = stories[question["story"]]
        prediction = answer(question["question"], story, loaded)
        if (
            prediction["selected_arm"] != ticket["selected_arm"]
            or prediction["generator_weights_sha256"]
            != contract["files"]["generator/model.safetensors"]
        ):
            raise ValueError("Selected policy/checkpoint changed")
        selected = [
            story["sections"][i]["id"]
            for i in retrieve(question["question"], story["sections"], loaded[2])
        ]
        row = score_row(
            question,
            prediction["answer"],
            prediction["retrieved_evidence"],
            selected,
            story,
        )
        row["selected_sections"] = selected
        rows.append(row)
        predictions.append(prediction)
        with (output / f"prediction-{ordinal:03d}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(
                {"id": question["id"], "prediction": prediction, "scored_row": row},
                handle,
                indent=2,
                allow_nan=False,
            )
    if len(rows) != 325 or [row["id"] for row in rows] != cohort["ids"]:
        raise ValueError("Incomplete test cohort cannot produce metrics")
    del loaded
    gc.collect()
    reloaded = load(package, ticket["package_contract_sha256"], "cpu")
    for question, expected in zip(questions, predictions):
        actual = answer(question["question"], stories[question["story"]], reloaded)
        if actual != expected:
            raise ValueError("Saved-package full325 reload prediction/citation drift")
    del reloaded
    gc.collect()
    authenticate(package, ticket["package_contract_sha256"])
    authenticate_cohort(data, cohort)
    authorize_freeze(package, root_freeze, root_sha256)
    if runtime_now() != frozen["runtime"]:
        raise ValueError("Frozen runtime changed during evaluation")
    for name, expected in ticket["source_sha256"].items():
        if digest(at(Path(__file__).parents[1], name)) != expected:
            raise ValueError("Sealed source changed during evaluation")
    rescored = []
    for question, row in zip(questions, rows):
        story = stories[question["story"]]
        scored = score_row(
            question, row["answer"], row["citations"], row["selected_sections"], story
        )
        scored["selected_sections"] = row["selected_sections"]
        if scored != row:
            raise ValueError("Postflight source/reference metric replay differs")
        rescored.append(scored)
    report = {
        "schema": "sealed_narrative_test_result_v1",
        "complete": True,
        "questions": 325,
        "ids_sha256": cohort["ids_sha256"],
        "families": cohort["families"],
        "selected_arm": ticket["selected_arm"],
        "package_contract_sha256": ticket["package_contract_sha256"],
        "root_freeze_sha256": ticket["root_freeze_sha256"],
        "cohort_sha256": ticket["cohort_sha256"],
        "metrics": summarize(rescored),
        "reload_questions": 325,
        "reload_predictions_and_citations_exact": True,
        "weights_refitted": False,
        "selection_performed_on_test": False,
        "marker_sha256": digest(marker),
        "prediction_files": {
            path.name: digest(path) for path in sorted(output.glob("prediction-*.json"))
        },
    }
    with (output / "test-result.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def evaluate(package, data, cohort, ticket, output, *, root_freeze, root_sha256):
    """Preserve a bounded failure receipt; never clear markers or retry."""
    try:
        return _evaluate(
            package,
            data,
            cohort,
            ticket,
            output,
            root_freeze=root_freeze,
            root_sha256=root_sha256,
        )
    except Exception as error:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        failure = output / "test-failure.json"
        if not failure.exists():
            with failure.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "sealed_narrative_failure_v1",
                        "exception_type": type(error).__name__,
                        "message": str(error)[:300],
                        "marker_preserved": (
                            output / "sealed-test-started.json"
                        ).is_file(),
                        "retry_authorized": False,
                    },
                    handle,
                    indent=2,
                )
        raise
