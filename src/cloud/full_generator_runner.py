"""Readable private CPU runner for full-data training/resume validation."""
import hashlib
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

EXPECTED_MANIFEST_SHA256 = None
ACTION = "smoke"
USE_PREVIOUS_RUN = False
VALIDATION_EPOCH = 0
USE_PREVIOUS_VALIDATION = False


def main():
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["OMP_NUM_THREADS"]="4"
    subprocess.run([sys.executable,"-m","pip","install","-q","transformers==4.57.6"],check=True)
    manifests=list(Path("/kaggle/input").rglob("full-payload-provenance.json"))
    if len(manifests)!=1:raise ValueError("Expected one versioned private source payload")
    path=manifests[0]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_MANIFEST_SHA256:
        raise ValueError("Full-data payload identity changed")
    source=path.parent;manifest=json.loads(path.read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe payload path")
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError("Changed mounted source/data")
    if (source/"private-data/test").exists():raise ValueError("Test QA must not be mounted")
    sys.path.insert(0,str(source))
    from src.full_generator_training import train_full,validate_epoch
    from src.resumable_training import atomic_json,file_sha
    output=Path("/kaggle/working/full-generator");output.mkdir()
    atomic_json(output/"payload-provenance.json",manifest)
    data=source/"private-data";protocol=source/"docs/full-coverage-protocol.json"
    previous_run=None
    if USE_PREVIOUS_RUN:
        runs=[p.parent for p in Path("/kaggle/input").rglob("training-status.json")
              if (p.parent/"provenance.json").is_file()]
        if len(runs)!=1:raise ValueError("Expected one immutable prior full training output")
        previous_run=runs[0]
    began=time.monotonic();success=False
    try:
        if ACTION=="smoke":
            # Actual pretrained-model comparison: uninterrupted versus checkpoint
            # continuation, one model resident at a time. No test is opened.
            direct=train_full(data,protocol,output/"direct",smoke=True,max_seconds=1800)
            gc.collect()
            interrupted=train_full(data,protocol,output/"interrupted",smoke=True,max_seconds=0)
            gc.collect()
            if interrupted["status"]!="checkpointed":raise ValueError("Smoke did not interrupt inside the epoch")
            resumed=train_full(data,protocol,output/"resumed",resume=output/"interrupted",smoke=True,max_seconds=1800)
            gc.collect()
            from safetensors.torch import load_file
            import torch
            left=load_file(output/"direct/epoch-1/model.safetensors")
            right=load_file(output/"resumed/epoch-1/model.safetensors")
            if left.keys()!=right.keys():raise ValueError("Resumed tensor schema changed")
            difference=max(float((left[key]-right[key]).abs().max()) for key in left)
            if difference!=0:raise ValueError("Checkpoint continuation differs from uninterrupted real training")
            del left,right
            validation=validate_epoch(data,protocol,output/"resumed",1,output/"smoke-validation",smoke=True)
            if not validation["complete"]:raise ValueError("Smoke validation incomplete")
            receipt={"mechanics_only":True,"train_QA":20,"validation_QA":2,"direct":direct,
                     "interrupted":interrupted,"resumed":resumed,"tensor_replay_max_abs_difference":difference,
                     "resumed_epoch_weights_sha256":file_sha(output/"resumed/epoch-1/model.safetensors"),
                     "actual_validation":validation,"test_accessed":False}
            atomic_json(output/"resume-smoke-receipt.json",receipt)
        elif ACTION=="train":
            train_full(data,protocol,output/"run",resume=previous_run)
        elif ACTION=="validate":
            if previous_run is None:raise ValueError("Validation requires a completed training kernel source")
            from src.full_validation_selection import completed_run
            completed_run(previous_run,data,protocol,"generator")
            previous_validation=None
            if USE_PREVIOUS_VALIDATION:
                shards=[p.parent for p in Path("/kaggle/input").rglob("validation-status.json")]
                if len(shards)!=1:raise ValueError("Expected one prior immutable validation output")
                previous_validation=shards[0]
            validate_epoch(data,protocol,previous_run,VALIDATION_EPOCH,output/"validation",previous=previous_validation)
        elif ACTION=="select":
            from src.full_validation_selection import select
            if previous_run is None:raise ValueError("Selection requires a completed training kernel source")
            reports=[p.parent for p in Path("/kaggle/input").rglob("validation-status.json")]
            reports.sort(key=lambda p:json.loads((p/"identity.json").read_text())["epoch"])
            select(data,protocol,previous_run,reports,output/"selection.json","generator")
        else:
            raise ValueError("Unsupported execution action")
        success=True
    finally:
        atomic_json(output/"runtime.json",{"success":success,"action":ACTION,"seconds":time.monotonic()-began,
                    "test_accessed":False,"free_private_CPU":True})


if __name__=="__main__":main()
