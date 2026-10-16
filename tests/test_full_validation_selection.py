import json
from pathlib import Path

import pytest

from src import full_validation_selection as audit


LOADING = {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def shards(directory, identity, rows, size=2):
    identity = {**identity, "shard_size": size}
    write(directory/"identity.json", identity)
    for start in range(0, len(rows), size):
        batch = rows[start:start+size]
        write(directory/f"shard-{start:05d}.json", {"identity": identity, "predictions": batch,
              "predictions_sha256": audit.sha(json.dumps(batch, sort_keys=True, allow_nan=False).encode())})
    status = {"complete": True, "questions_completed": len(rows), "questions_required": len(rows),
              "identity": identity, "smoke": False, "test_accessed": False, "loading_info": LOADING}
    write(directory/"validation-status.json", status)
    return identity, status


def test_all_query_shards_reject_overlap_extra_missing_and_changed_bytes(tmp_path):
    rows = [{"id": str(i)} for i in range(5)]
    identity, status = shards(tmp_path, {"fixture": True}, rows)
    assert audit.audited_rows(tmp_path, identity, [r["id"] for r in rows], "generator")[0] == rows
    extra = tmp_path/"shard-99999.json"; write(extra, {})
    with pytest.raises(ValueError, match="unexpected"):
        audit.audited_rows(tmp_path, identity, [r["id"] for r in rows], "generator")
    extra.unlink()
    path = tmp_path/"shard-00002.json"; changed = audit.read(path)
    changed["predictions"][0]["id"] = "0"
    changed["predictions_sha256"] = audit.sha(json.dumps(changed["predictions"], sort_keys=True).encode())
    write(path, changed)
    with pytest.raises(ValueError, match="overlapping"):
        audit.audited_rows(tmp_path, identity, [r["id"] for r in rows], "generator")
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        audit.audited_rows(tmp_path, identity, [r["id"] for r in rows], "generator")


def test_partial_training_and_changed_model_bytes_fail_before_inference(tmp_path):
    policy = tmp_path/"policy.json"; write(policy, {"generator": {"epochs": 3}})
    run = tmp_path/"run"; write(run/"provenance.json", {"identity": {}})
    write(run/"training-status.json", {"status": "checkpointed", "smoke": False})
    with pytest.raises(ValueError, match="complete full-cohort"):
        audit.completed_run(run, tmp_path, policy, "generator")
    checkpoint = run/"epoch-0"; checkpoint.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer_config.json"):
        (checkpoint/name).write_bytes(b"fixture")
    files = {p.name: audit.digest(p) for p in checkpoint.iterdir()}
    write(checkpoint/"receipt.json", {"identity": {}, "epoch": 0, "files": files,
          "weights_sha256": files["model.safetensors"]})
    assert audit.snapshot_receipt(run, 0, {})["weights_sha256"] == files["model.safetensors"]
    (checkpoint/"config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        audit.snapshot_receipt(run, 0, {})


def test_completed_run_requires_exact_source_training_progress_and_pretrained_receipt(tmp_path, monkeypatch):
    data = tmp_path/"data"; write(data/"manifest.json", {"fixture": "no QA"})
    monkeypatch.setattr(audit, "FULL_MANIFEST_SHA256", audit.digest(data/"manifest.json"))
    policy = tmp_path/"policy.json"; spec = {"epochs": 3, "seed": 42}
    write(policy, {"generator": spec})
    lineage = {"smoke": False, "generator": spec, "source_manifest_sha256": audit.digest(data/"manifest.json"),
               "protocol_sha256": audit.digest(policy), "source_sha256": {name: audit.digest(Path(audit.__file__).parent/name)
                    for name in ("full_generator_training.py", "resumable_training.py", "narrative_training.py", "data/prepare_fairytale.py")}}
    run = tmp_path/"run"
    provenance = {"identity": lineage, "loading_info": LOADING,
        "pretrained_weights_sha256": "bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc"}
    write(run/"provenance.json", provenance)
    status = {"status": "training_complete", "smoke": False, "epochs_required": 3,
              "all_training_QA": 7883, "test_accessed": False,
              "progress": {"epoch": 4, "next_microstep": 0, "loss_sum": 0.0}}
    write(run/"training-status.json", status)
    assert audit.completed_run(run, data, policy, "generator")[2] == 3
    write(run/"training-status.json", {**status, "progress": {"epoch": 3, "next_microstep": 0, "loss_sum": 0.0}})
    with pytest.raises(ValueError, match="complete full-cohort"):
        audit.completed_run(run, data, policy, "generator")
    write(run/"training-status.json", status)
    write(run/"provenance.json", {**provenance, "loading_info": {**LOADING, "missing_keys": ["qa_head"]}})
    with pytest.raises(ValueError, match="initialization"):
        audit.completed_run(run, data, policy, "generator")


def test_source_family_overlap_refused_before_qa_loading(tmp_path, monkeypatch):
    write(tmp_path/"manifest.json", {"stories": [
        {"id": "one", "split": "train", "origin": "shared", "source_sha256": "a"},
        {"id": "two", "split": "validation", "origin": "shared", "source_sha256": "b"}]})
    monkeypatch.setattr(audit, "FULL_MANIFEST_SHA256", audit.digest(tmp_path/"manifest.json"))
    monkeypatch.setattr(audit, "load_split", lambda *a: pytest.fail("QA must not be parsed after source overlap"))
    with pytest.raises(ValueError, match="overlap"):
        audit.validation_cohort(tmp_path, {})


def fixture_question():
    story = {"text": "A fox runs.", "sections": [{"id": "1", "text": "A fox runs.", "start": 0, "end": 11}]}
    q = {"id": "s:1", "story": "s", "question": "Who runs?", "references": ["fox"], "gold_sections": ["1"],
         "origin": "fixture", "attribute": "character", "explicit": "explicit"}
    row = {**q, "answer": "fox", "reference_count": 1,
           "citations": [{"section": "1", "start": 0, "end": 11, "text": "A fox runs."}],
           **{k: 1.0 for k in ("em", "f1", "rouge_l", "evidence_recall", "evidence_precision",
                               "retained_evidence_recall", "citation_validity")}}
    return {"s": story}, q, row


def test_metrics_are_recomputed_against_current_sources_and_references():
    stories, q, row = fixture_question()
    assert audit.recompute_rows(stories, [q], [row], "generator")[0]["f1"] == 1
    forged = {**row, "f1": .9}
    with pytest.raises(ValueError, match="metric"):
        audit.recompute_rows(stories, [q], [forged], "generator")
    with pytest.raises(ValueError, match="metadata"):
        audit.recompute_rows(stories, [q], [{**row, "references": ["forged"]}], "generator")
    bad = {**row, "citations": [{**row["citations"][0], "end": 12}]}
    with pytest.raises(ValueError, match="outside"):
        audit.recompute_rows(stories, [q], [bad], "generator")


def test_selection_requires_every_epoch_uses_declared_ties_and_never_authorizes_test(tmp_path, monkeypatch):
    stories, q, row = fixture_question()
    data = tmp_path/"data"; write(data/"manifest.json", {"fixture": True})
    protocol = tmp_path/"protocol.json"; write(protocol, {"fixture": True})
    policy = {"validation_sampling": {"mode": "all"}}
    lineage = {"fixture": "No actual model/test data"}
    monkeypatch.setattr(audit, "completed_run", lambda *a: (policy, lineage, 1))
    monkeypatch.setattr(audit, "validation_cohort", lambda *a: (stories, [q]))
    monkeypatch.setattr(audit, "snapshot_receipt", lambda run, epoch, identity: {"weights_sha256": str(epoch)})
    run = tmp_path/"run"; directories = []
    for epoch in range(2):
        write(run/f"epoch-{epoch}/receipt.json", {})
        directory = tmp_path/f"validation-{epoch}"
        identity = {"run": lineage, "epoch": epoch, "weights_sha256": str(epoch),
                    "validation_ids_sha256": audit.sha(json.dumps([q["id"]]).encode())}
        identity, status = shards(directory, identity, [row])
        write(directory/"validation-status.json", {**status, "validation": audit.summarize([row])})
        directories.append(directory)
    with pytest.raises(ValueError, match="every epoch"):
        audit.select(data, protocol, run, directories[:1], tmp_path/"bad.json", "generator")
    output = tmp_path/"selection.json"
    selected = audit.select(data, protocol, run, directories, output, "generator")
    assert selected["selected"]["epoch"] == 0
    assert selected["final_test_authorized"] is False and selected["status"] != "frozen"
    with pytest.raises(ValueError, match="overwrite"):
        audit.select(data, protocol, run, directories, output, "generator")


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 10**400])
def test_nonfinite_or_boolean_metrics_fail_closed(value):
    with pytest.raises(ValueError):
        audit.metric_equal(value, .5)
