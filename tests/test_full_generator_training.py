import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="Full-data training mechanics require the optional reader runtime")

from src import full_generator_training as full


def test_cloud_budget_and_policy_guards_before_model_access(tmp_path):
    for value in (True, float("nan"), float("inf"), -1., 40000.):
        with pytest.raises(ValueError, match="safe cloud time"):
            full.train_full(tmp_path, tmp_path/"missing", tmp_path/"out", max_seconds=value)
    policy=tmp_path/"policy.json";policy.write_text(json.dumps({"validation_sampling":{"mode":"sample"}}))
    with pytest.raises(ValueError, match="uncapped"):
        full.identity(tmp_path,policy,2,False)


def test_validation_shard_resume_decodes_every_row_once_and_rejects_changes(tmp_path,monkeypatch):
    identity={"fixture":"No pretrained model or real QA used"}
    policy={"validation_sampling":{"mode":"all"},"generator":{"input_tokens":512},
            "dataset":{"full_coverage":{"custom":{"validation":{"QA_metadata":5}}}}}
    monkeypatch.setattr(full,"identity",lambda *a,**k:(policy,identity))
    questions=[{"id":str(i)} for i in range(5)]
    def load_split(data,split):
        assert split=="validation"  # A fixture failure would catch any test access.
        return {},questions
    monkeypatch.setattr(full,"load_split",load_split)
    tokenizer=SimpleNamespace(model_max_length=512)
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer)))
    monkeypatch.setattr(full,"strict_model",lambda *a,**k:(object(),{}))
    decoded=[]
    def evaluate(model,tokenizer,stories,rows):
        decoded.extend(q["id"] for q in rows)
        return [{"id":q["id"],"origin":"fixture","explicit":"fixture","attribute":"fixture",
                 "reference_count":1,**{key:.4 for key in ("em","f1","rouge_l","evidence_recall",
                 "evidence_precision","retained_evidence_recall","citation_validity")}} for q in rows],{}
    monkeypatch.setattr(full,"evaluate",evaluate)
    run=tmp_path/"run";snapshot=run/"epoch-1";snapshot.mkdir(parents=True)
    (snapshot/"model.safetensors").write_bytes(b"Fixture bytes, never loaded as a model")
    (snapshot/"config.json").write_text("{}");(snapshot/"tokenizer_config.json").write_text("{}")
    (snapshot/"receipt.json").write_text(json.dumps({"identity":identity,"epoch":1,
        "files":{p.name:full.file_sha(p) for p in snapshot.iterdir()},
        "weights_sha256":full.file_sha(snapshot/"model.safetensors")}))
    first=tmp_path/"first"
    status=full.validate_epoch(tmp_path,tmp_path,run,1,first,threads=1,max_seconds=0,shard_size=2)
    assert not status["complete"] and status["questions_completed"]==2
    second=tmp_path/"second"
    status=full.validate_epoch(tmp_path,tmp_path,run,1,second,previous=first,threads=1,max_seconds=100,shard_size=2)
    assert status["complete"] and status["validation"]["n"]==5 and decoded==["0","1","2","3","4"]
    path=first/"shard-00000.json";shard=json.loads(path.read_text());shard["predictions"][0]["f1"]=.9
    path.write_text(json.dumps(shard))
    with pytest.raises(ValueError,match="Changed validation shard"):
        full.validate_epoch(tmp_path,tmp_path,run,1,tmp_path/"bad",previous=first,threads=1,max_seconds=100,shard_size=2)
    (snapshot/"tokenizer_config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError,match="weights/config/tokenizer"):
        full.verify_snapshot(snapshot,identity,1)
