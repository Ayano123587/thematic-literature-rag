"""Software contract fixtures only; no benchmark results are manufactured."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src import joint_answer_selection as selection
from src.models.frozen_narrative_reader import authenticate


def ranking():
    return {
        "status": "validation_checkpoint_selection",
        "task": "reranker",
        "validation_questions": 2372,
        "validation_ids_sha256": selection.FULL_VALIDATION_IDS_SHA256,
        "test_accessed": False,
        "candidates": [
            {"kind": "bm25", "epoch": 0, "validation": {"recall_at_4": 1}},
            *[
                {
                    "kind": "adapted",
                    "epoch": e,
                    "weights_sha256": str(e) * 64,
                    "validation": {"recall_at_4": 0.5},
                }
                for e in (1, 2)
            ],
        ],
    }


def test_adapted_arm_distinct_from_bm25_and_earliest_tie():
    assert selection.adapted_checkpoint(ranking())["epoch"] == 1


@pytest.mark.parametrize("change", ["missing", "nan", "placeholder", "test"])
def test_adapted_contract_rejects(change):
    value = ranking()
    if change == "missing":
        value["candidates"].pop()
    if change == "nan":
        value["candidates"][1]["validation"]["recall_at_4"] = float("nan")
    if change == "placeholder":
        value["candidates"][1]["weights_sha256"] = "0" * 64
    if change == "test":
        value["test_accessed"] = True
    with pytest.raises(ValueError):
        selection.adapted_checkpoint(value)


def paired(monkeypatch):
    ids = [str(i) for i in range(2372)]
    monkeypatch.setattr(
        selection,
        "FULL_VALIDATION_IDS_SHA256",
        hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    )
    rows = [
        {
            "id": i,
            "story": "software-fixture",
            "arms": {
                a: {
                    "f1": 0.2 if a == selection.ARMS[0] else 0.3,
                    "em": 0.1,
                    "citation_validity": 1.0,
                }
                for a in selection.ARMS
            },
        }
        for i in ids
    ]
    return ids, rows


def test_joint_tie_uses_fixed_lower_complexity_order(monkeypatch):
    ids, rows = paired(monkeypatch)
    assert selection.select_paired(rows, ids, True)["selected_arm"] == "bm25_windows"


def test_regression_and_no_gain_keep_baseline(monkeypatch):
    ids, rows = paired(monkeypatch)
    for row in rows:
        row["arms"]["bm25_windows"]["em"] = 0
        row["arms"]["reranked_standard"]["citation_validity"] = 0
        row["arms"]["reranked_windows"]["f1"] = 0.201
    assert selection.select_paired(rows, ids, True)["selected_arm"] == "bm25_standard"


@pytest.mark.parametrize(
    "change", ["partial", "reordered", "replay", "missing_arm", "nonfinite"]
)
def test_joint_fail_closed(monkeypatch, change):
    ids, rows = paired(monkeypatch)
    if change == "partial":
        rows.pop()
    if change == "reordered":
        rows.reverse()
    if change == "missing_arm":
        rows[0]["arms"].pop("reranked_windows")
    if change == "nonfinite":
        rows[0]["arms"]["bm25_standard"]["f1"] = float("nan")
    with pytest.raises(ValueError):
        selection.select_paired(rows, ids, change != "replay")


def test_inference_refuses_unfrozen_contract_before_model_import(tmp_path):
    (tmp_path / "inference-contract.json").write_text("{}")
    with pytest.raises(ValueError):
        authenticate(tmp_path, "0" * 64)


def test_inference_requires_full_artifact_inventory(tmp_path):
    path = tmp_path / "inference-contract.json"
    path.write_text(
        json.dumps({"schema": "frozen_narrative_inference_v1", "files": {}})
    )
    with pytest.raises(ValueError):
        authenticate(tmp_path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    "extra",
    [
        "generator/tokenizer.json",
        "generator/special_tokens_map.json",
        "generator/hidden.bin",
    ],
)
def test_unlisted_model_files_rejected_before_loading(tmp_path, extra):
    files = {
        n: "1" * 64
        for n in (
            "generator-selection.json",
            "joint-selection.json",
            "joint-protocol.json",
            "generator/receipt.json",
            "generator/model.safetensors",
            "generator/config.json",
            "generator/tokenizer_config.json",
        )
    }
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture")
    (tmp_path / extra).write_text("unbound")
    path = tmp_path / "inference-contract.json"
    path.write_text(
        json.dumps({"schema": "frozen_narrative_inference_v1", "files": files})
    )
    with pytest.raises(ValueError, match="Unlisted"):
        authenticate(tmp_path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("arm", selection.ARMS)
def test_production_output_and_source_citations_match_existing_helpers(arm):
    import re
    from src.models.frozen_narrative_reader import answer
    from src.narrative_training import retrieve, assemble
    from src.context_assembly import assemble_question_windows

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping"):
                offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
                return {
                    "input_ids": list(range(len(offsets))),
                    "offset_mapping": offsets,
                }
            return Encoded(context=text)

        def decode(self, value, **kwargs):
            return value

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            self.actual = kwargs
            return ["fixture answer"]

    class Ranker:
        def predict(self, pairs, **kwargs):
            return list(range(len(pairs)))

    text = "Moon rises. Sun shines."
    story = {
        "id": "software-fixture",
        "text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "sections": [
            {"id": "a", "start": 0, "end": 11, "text": text[:11]},
            {"id": "b", "start": 12, "end": len(text), "text": text[12:]},
        ],
    }
    question = "What shines?"
    tokenizer, model = Tokenizer(), Model()
    ranker = Ranker() if arm.startswith("reranked") else None
    indices = retrieve(question, story["sections"], ranker)
    helper = assemble_question_windows if arm.endswith("windows") else assemble
    expected_context, expected_citations = helper(
        question, story["sections"], indices, tokenizer, 512
    )
    output = answer(
        question,
        story,
        (
            model,
            tokenizer,
            ranker,
            {"selected_arm": arm},
            {"files": {"generator/model.safetensors": "4" * 64}},
        ),
    )
    assert model.actual["context"] == expected_context
    assert model.actual["num_beams"] == 4 and model.actual["do_sample"] is False
    assert output["answer"] == "fixture answer"
    assert output["retrieved_evidence"] == expected_citations
    assert all(text[c["start"] : c["end"]] == c["text"] for c in expected_citations)


@pytest.fixture
def inference_package(tmp_path, monkeypatch):
    import ast
    from src.models import frozen_narrative_reader as reader

    root = Path(reader.__file__).parents[1]
    tree = ast.parse(Path(reader.__file__).read_text())
    names = next(
        ast.literal_eval(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "required" for t in n.targets)
    )
    weights = "4a51a48953e6c02c2c8aa5325a02600ae7a65dc8080abb5fb45e2b7131c39538"

    def put(name, value):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(value) if isinstance(value, dict) else value)

    put("generator/model.safetensors", "software-fixture-not-model-weights")
    put("generator/config.json", {})
    put("generator/tokenizer_config.json", {})
    put("generator/tokenizer.json", {})
    protocol = root.parent / "docs/joint-answer-policy-protocol.json"
    put("joint-protocol.json", protocol.read_text())
    put(
        "generator-selection.json",
        {
            "status": "validation_checkpoint_selection",
            "task": "generator",
            "validation_questions": 2372,
            "validation_ids_sha256": selection.FULL_VALIDATION_IDS_SHA256,
            "test_accessed": False,
            "selected": {"epoch": 3, "weights_sha256": weights},
        },
    )
    original_digest = reader.digest

    def digest(path):
        return (
            weights
            if Path(path) == tmp_path / "generator/model.safetensors"
            else original_digest(path)
        )

    monkeypatch.setattr(reader, "digest", digest)
    model_files = {p.name: digest(p) for p in (tmp_path / "generator").iterdir()}
    put("generator/receipt.json", {"weights_sha256": weights, "files": model_files})
    bindings = {
        "generator_selection_sha256": digest(tmp_path / "generator-selection.json"),
        "protocol_sha256": digest(tmp_path / "joint-protocol.json"),
        "paired_validation_sha256": "a" * 64,
    }
    put(
        "joint-selection.json",
        {
            "schema": "joint_answer_selection_v1",
            "selected_arm": "bm25_standard",
            "validation_questions": 2372,
            "validation_ids_sha256": selection.FULL_VALIDATION_IDS_SHA256,
            "test_accessed": False,
            "artifact_bindings": bindings,
        },
    )

    def freeze():
        policy = json.loads((tmp_path / "joint-selection.json").read_text())
        contract = {
            "schema": "frozen_narrative_inference_v1",
            "files": {
                p.relative_to(tmp_path).as_posix(): digest(p)
                for p in tmp_path.rglob("*")
                if p.is_file() and p.name != "inference-contract.json"
            },
            "source_sha256": {n: original_digest(root / n) for n in names},
            "artifact_bindings": policy["artifact_bindings"],
        }
        put("inference-contract.json", contract)
        return original_digest(tmp_path / "inference-contract.json")

    return tmp_path, freeze


def test_complete_software_inference_contract(inference_package):
    root, freeze = inference_package
    assert authenticate(root, freeze())[1]["selected_arm"] == "bm25_standard"


@pytest.mark.parametrize("name", ["generator-selection.json", "joint-selection.json"])
def test_changed_fullvalidation_ids_rejected(inference_package, name):
    root, freeze = inference_package
    path = root / name
    value = json.loads(path.read_text())
    value["validation_ids_sha256"] = "a" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="selection|policy"):
        authenticate(root, freeze())


def test_protocol_semantics_cannot_be_redefined_by_contract(inference_package):
    root, freeze = inference_package
    path = root / "joint-protocol.json"
    value = json.loads(path.read_text())
    value["selection"]["minimum_f1_gain"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protocol"):
        authenticate(root, freeze())


def test_transitive_source_tamper_rejected(inference_package):
    root, freeze = inference_package
    sha = freeze()
    path = root / "inference-contract.json"
    value = json.loads(path.read_text())
    value["source_sha256"]["reranker_training.py"] = "a" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="implementation"):
        authenticate(root, hashlib.sha256(path.read_bytes()).hexdigest())


def test_joint_policy_binds_reranker_selection(inference_package):
    root, freeze = inference_package
    folder = root / "reranker"
    folder.mkdir()
    for name in (
        "model.safetensors",
        "receipt.json",
        "config.json",
        "tokenizer_config.json",
    ):
        (folder / name).write_text("{}")
    (root / "reranker-selection.json").write_text(json.dumps(ranking()))
    path = root / "joint-selection.json"
    value = json.loads(path.read_text())
    value["selected_arm"] = "reranked_standard"
    value["artifact_bindings"]["reranker_selection_sha256"] = "a" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reranker selection differs"):
        authenticate(root, freeze())


@pytest.mark.parametrize("scenario", ["normal", "copy_drift", "broken_output"])
def test_full_four_arm_package_pipeline_software_fixture(
    inference_package, monkeypatch, scenario
):
    import re
    from src import freeze_narrative_package as freezer
    from src import joint_answer_validation as validation
    from src.models import frozen_narrative_reader as reader
    from src.context_validation import score_row
    from src.narrative_training import assemble, retrieve
    from src.context_assembly import assemble_question_windows

    root, initial_freeze = inference_package
    initial_freeze()
    sources = json.loads((root / "inference-contract.json").read_text())[
        "source_sha256"
    ]
    ids = [str(i) for i in range(2372)]
    ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    monkeypatch.setattr(selection, "FULL_VALIDATION_IDS_SHA256", ids_sha)
    monkeypatch.setattr(reader, "FULL_VALIDATION_IDS_SHA256", ids_sha)
    gen = root / "generator-selection.json"
    value = json.loads(gen.read_text())
    value["validation_ids_sha256"] = ids_sha
    gen.write_text(json.dumps(value))
    (root / "reranker-selection.json").write_text(json.dumps(ranking()))
    original_digest = reader.digest
    monkeypatch.setattr(
        reader,
        "digest",
        lambda p: (
            "4a51a48953e6c02c2c8aa5325a02600ae7a65dc8080abb5fb45e2b7131c39538"
            if Path(p).name == "model.safetensors"
            and Path(p).parent.name == "generator"
            else original_digest(p)
        ),
    )
    monkeypatch.setattr(freezer, "digest", reader.digest)

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

        def __call__(self, text, **kwargs):
            offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}

    tokenizer = Tokenizer()
    text = "Moon shines."
    story = {
        "id": "fixture",
        "text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "sections": [{"id": "section", "start": 0, "end": len(text), "text": text}],
    }
    questions = [
        {
            "id": i,
            "story": "fixture",
            "question": "What shines?",
            "references": ["Moon"],
            "gold_sections": ["section"],
        }
        for i in ids
    ]
    plan = {i: ["section"] for i in ids}
    rows = []
    for q in questions:
        arms = {}
        indices = retrieve(q["question"], story["sections"])
        for arm in selection.ARMS:
            helper = assemble_question_windows if arm.endswith("windows") else assemble
            _, citations = helper(
                q["question"], story["sections"], indices, tokenizer, 512
            )
            arms[arm] = score_row(q, "Moon", citations, ["section"], story)
        rows.append(
            {
                "id": q["id"],
                "story": "fixture",
                "selected_sections": {"bm25": ["section"], "reranked": ["section"]},
                "arms": arms,
            }
        )
    joint = root / "joint"
    joint.mkdir()
    identity = {"fixture": True}
    for start in range(0, 2372, 64):
        pairs = rows[start : start + 64]
        validation.write(
            joint / f"shard-{start:05d}.json",
            {
                "identity": identity,
                "pairs": pairs,
                "pairs_sha256": validation.content_sha(pairs),
            },
        )
    decision = selection.select_paired(rows, ids, True)
    decision["artifact_bindings"] = {
        "generator_selection_sha256": reader.digest(gen),
        "reranker_selection_sha256": reader.digest(root / "reranker-selection.json"),
        "protocol_sha256": reader.digest(root / "joint-protocol.json"),
        "paired_validation_sha256": validation.content_sha(rows),
    }
    validation.write(
        joint / "joint-status.json",
        {
            "identity": identity,
            "complete": True,
            "questions_completed": 2372,
            "questions_required": 2372,
            "test_accessed": False,
            "selection": decision,
        },
    )
    output = root / "production"
    if scenario == "copy_drift":
        original_copy = freezer.shutil.copyfile

        def drifting_copy(source, target):
            result = original_copy(source, target)
            gen.write_text(gen.read_text() + " ")
            return result

        monkeypatch.setattr(freezer.shutil, "copyfile", drifting_copy)
    if scenario == "broken_output":
        original_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path, "is_symlink", lambda p: p == output or original_symlink(p)
        )

    def perform():
        return freezer.freeze(
            output,
            joint,
            reader.digest(joint / "joint-status.json"),
            {"fixture": story},
            questions,
            plan,
            plan,
            tokenizer,
            root / "generator",
            root / "absent-reranker",
            gen,
            root / "reranker-selection.json",
            root / "joint-protocol.json",
            sources,
        )

    if scenario != "normal":
        with pytest.raises(ValueError):
            perform()
        return
    result = perform()
    assert result["selected_arm"] == "bm25_standard"
    assert (
        reader.authenticate(output, result["contract_sha256"])[1]["selected_arm"]
        == "bm25_standard"
    )
