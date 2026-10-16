"""Full-cohort T5 training and resumable validation, without test access.

Training keeps every epoch snapshot. Validation decoding is a separate,
shard-resumable phase; a normal cloud time ceiling never becomes a data cap.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import torch

from src.data.prepare_fairytale import sha
from src.narrative_training import (assemble, evaluate, load_split, seed_all,
                                    select_validation, summarize)
from src.resumable_training import (atomic_json, epoch_order, file_sha,
                                    load_progress, optimize_epoch, save_progress)


def identity(data, protocol, micro_batch, smoke, threads=4):
    policy = json.loads(Path(protocol).read_text(encoding="utf-8"))
    if policy.get("validation_sampling") != {"mode": "all"}:
        raise ValueError("Full-data runner requires uncapped validation")
    if type(micro_batch) is not int or micro_batch < 1 or policy["generator"]["effective_batch_size"] % micro_batch:
        raise ValueError("Microbatch must divide the declared effective batch")
    value = {"source_manifest_sha256": file_sha(Path(data)/"manifest.json"),
             "protocol_sha256": file_sha(protocol), "micro_batch": micro_batch, "smoke": smoke,
             "threads": threads, "generator": policy["generator"], "torch": str(torch.__version__),
             "python": platform.python_version(), "machine": platform.machine(),
             "environment": {name: importlib.metadata.version(name) for name in
                             ("transformers", "tokenizers", "numpy", "huggingface-hub")},
             "source_sha256": {name: file_sha(Path(__file__).parent/name) for name in
                               ("full_generator_training.py", "resumable_training.py", "narrative_training.py",
                                "data/prepare_fairytale.py")}}
    return policy, value


def strict_model(checkpoint, revision=None):
    from transformers import AutoModelForSeq2SeqLM
    kwargs = {"use_safetensors": True, "output_loading_info": True}
    if revision is not None:
        kwargs["revision"] = revision
    else:
        kwargs["local_files_only"] = True
    model, loading = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, **kwargs)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete or mismatched full-generator weights")
    return model.to("cpu"), loading


def verify_snapshot(directory, run_identity, epoch):
    directory = Path(directory)
    receipt = json.loads((directory/"receipt.json").read_text(encoding="utf-8"))
    files = receipt.get("files")
    if receipt.get("identity") != run_identity or receipt.get("epoch") != epoch or not isinstance(files,dict) or not {
            "model.safetensors","config.json","tokenizer_config.json"}.issubset(files):
        raise ValueError("Changed preserved epoch identity/files")
    for name,expected in files.items():
        if Path(name).name!=name or file_sha(directory/name)!=expected:
            raise ValueError("Changed preserved checkpoint weights/config/tokenizer")
    if receipt["weights_sha256"]!=files["model.safetensors"]:
        raise ValueError("Conflicting preserved checkpoint weight hash")
    return receipt


def snapshot(output, epoch, model, tokenizer, run_identity, mean_loss=None):
    directory = Path(output)/f"epoch-{epoch}"
    if directory.exists():
        return verify_snapshot(directory,run_identity,epoch)
    directory.mkdir()
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    receipt = {"identity": run_identity, "epoch": epoch, "mean_train_loss": mean_loss,
               "weights_sha256": file_sha(directory/"model.safetensors"),
               "files":{p.name:file_sha(p) for p in directory.iterdir() if p.is_file()}}
    atomic_json(directory/"receipt.json", receipt)
    return receipt


def train_full(data, protocol, output, *, resume=None, micro_batch=2, threads=4,
               max_seconds=37800, smoke=False):
    """Train all declared rows, checkpointing safely before the service ceiling."""
    if type(threads) is not int or threads < 1 or isinstance(max_seconds, bool) or not isinstance(
            max_seconds, (int,float)) or not 0 <= max_seconds <= 39600:
        raise ValueError("Positive threads and a safe cloud time window required")
    began = time.monotonic(); deadline = began + max_seconds
    policy, run_identity = identity(data, protocol, micro_batch, smoke, threads)
    spec = policy["generator"]
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True); seed_all(spec["seed"])
    stories, questions = load_split(data, "train")
    if len(questions) != policy["dataset"]["full_coverage"]["custom"]["train"]["QA_metadata"]:
        raise ValueError("Training coverage differs from the full-data protocol")
    if smoke:
        questions = questions[:20]
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh run output; resume inputs stay immutable")
    output.mkdir(parents=True)
    if resume is not None:
        resume = Path(resume)
        previous = json.loads((resume/"provenance.json").read_text(encoding="utf-8"))
        if previous["identity"] != run_identity:
            raise ValueError("Resume provenance changed")
        shutil.copytree(resume/"progress", output/"progress")
        for directory in resume.glob("epoch-*"):
            if directory.is_dir():
                verify_snapshot(directory,run_identity,int(directory.name.split("-")[1]))
                shutil.copytree(directory, output/directory.name)
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq
    from huggingface_hub import hf_hub_download
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"], use_fast=True)
    tokenizer.model_max_length = spec["input_tokens"]
    model, loading = strict_model(spec["name"], spec["revision"])
    weights = hf_hub_download(spec["name"], "model.safetensors", revision=spec["revision"])
    if file_sha(weights) != "bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc":
        raise ValueError("Full-data protocol's pinned T5-small weight bytes changed")
    atomic_json(output/"provenance.json", {"identity": run_identity, "loading_info": loading,
                "pretrained_weights_sha256": file_sha(weights), "train_QA": len(questions),
                "train_QA_full": policy["dataset"]["full_coverage"]["custom"]["train"]["QA_metadata"], "threads": threads,
                "test_accessed": False, "smoke": smoke})
    rows = []
    for question in questions:
        story = stories[question["story"]]
        indices = [i for i,s in enumerate(story["sections"]) if s["id"] in question["gold_sections"]]
        context, _ = assemble(question["question"], story["sections"], indices, tokenizer, spec["input_tokens"])
        row = tokenizer(context)
        row["labels"] = tokenizer(text_target=question["references"][0], max_length=spec["answer_tokens"],
                                  truncation=True)["input_ids"]
        rows.append(row)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=.01)
    model.gradient_checkpointing_enable(); model.config.use_cache=False
    progress = {"epoch":1,"next_microstep":0,"loss_sum":0.}
    if resume is not None:
        progress = load_progress(output/"progress", model, optimizer, run_identity)
    snapshot(output,0,model,tokenizer,run_identity) if resume is None else None
    epochs = 1 if smoke else spec["epochs"]
    accumulation = spec["effective_batch_size"]//micro_batch
    while progress["epoch"] <= epochs:
        epoch = progress["epoch"]
        order = epoch_order(len(rows), spec["seed"], epoch)
        batches = [lambda indices=order[i:i+micro_batch]: collator([rows[j] for j in indices])
                   for i in range(0,len(order),micro_batch)]
        def checkpoint(value):
            save_progress(output/"progress",model,optimizer,run_identity,value)
        complete = optimize_epoch(model,optimizer,batches,progress,accumulation,checkpoint,deadline=deadline)
        if not complete:
            break
        snapshot(output,epoch,model,tokenizer,run_identity,progress["loss_sum"]/len(batches))
        progress = {"epoch":epoch+1,"next_microstep":0,"loss_sum":0.}
        checkpoint(progress)
        if time.monotonic() >= deadline:
            break
    status = {"status":"training_complete" if progress["epoch"]>epochs else "checkpointed",
              "progress":progress,"epochs_required":epochs,"all_training_QA":len(questions),
              "seconds":time.monotonic()-began,"test_accessed":False,"smoke":smoke}
    atomic_json(output/"training-status.json",status)
    print(json.dumps(status),flush=True)
    return status


def validate_epoch(data, protocol, run, epoch, output, *, previous=None, micro_batch=2,
                   threads=4, max_seconds=37800, smoke=False, shard_size=64):
    """Decode every validation query, resuming only complete hashed shards."""
    if type(epoch) is not int or epoch<0 or type(shard_size) is not int or shard_size<1 or type(threads) is not int or threads<1 or isinstance(
            max_seconds,bool) or not isinstance(max_seconds,(int,float)) or not 0<=max_seconds<=39600:
        raise ValueError("Valid epoch and positive shard size required")
    policy, run_identity = identity(data,protocol,micro_batch,smoke,threads)
    run,output = Path(run),Path(output)
    snapshot_path = run/f"epoch-{epoch}"
    receipt = verify_snapshot(snapshot_path,run_identity,epoch)
    stories,questions = load_split(data,"validation")
    questions = select_validation(questions,policy)
    if len(questions)!=policy["dataset"]["full_coverage"]["custom"]["validation"]["QA_metadata"]:
        raise ValueError("Validation coverage differs from full-data protocol")
    if smoke:
        questions=questions[:2]
    validation_identity={"run":run_identity,"epoch":epoch,"weights_sha256":receipt["weights_sha256"],
                         "validation_ids_sha256":sha(json.dumps([q["id"] for q in questions]).encode()),
                         "shard_size":shard_size}
    if output.exists():
        raise ValueError("Use a fresh validation output; preserve prior shards")
    output.mkdir(parents=True)
    if previous is not None:
        previous=Path(previous)
        if json.loads((previous/"identity.json").read_text(encoding="utf-8"))!=validation_identity:
            raise ValueError("Changed validation resume identity")
        for path in previous.glob("shard-*.json"):
            shutil.copy2(path,output/path.name)
    atomic_json(output/"identity.json",validation_identity)
    from transformers import AutoTokenizer
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    tokenizer=AutoTokenizer.from_pretrained(snapshot_path,local_files_only=True,use_fast=True)
    tokenizer.model_max_length=policy["generator"]["input_tokens"]
    model,loading=strict_model(snapshot_path)
    deadline=time.monotonic()+max_seconds
    combined=[]
    for start in range(0,len(questions),shard_size):
        path=output/f"shard-{start:05d}.json"
        expected_ids=[q["id"] for q in questions[start:start+shard_size]]
        if path.exists():
            shard=json.loads(path.read_text(encoding="utf-8"))
            if shard.get("identity")!=validation_identity or [r["id"] for r in shard["predictions"]]!=expected_ids or sha(
                    json.dumps(shard["predictions"],sort_keys=True,allow_nan=False).encode())!=shard.get("predictions_sha256"):
                raise ValueError("Changed validation shard")
        else:
            predictions,_=evaluate(model,tokenizer,stories,questions[start:start+shard_size])
            shard={"identity":validation_identity,"predictions":predictions,
                   "predictions_sha256":sha(json.dumps(predictions,sort_keys=True,allow_nan=False).encode())}
            atomic_json(path,shard)
        combined.extend(shard["predictions"])
        print(json.dumps({"epoch":epoch,"validation_QA_completed":len(combined),"required":len(questions)}),flush=True)
        if time.monotonic()>=deadline:
            break
    complete=len(combined)==len(questions)
    status={"complete":complete,"questions_completed":len(combined),"questions_required":len(questions),
            "identity":validation_identity,"loading_info":loading,"test_accessed":False,"smoke":smoke}
    if complete:
        status["validation"]=summarize(combined)
    atomic_json(output/"validation-status.json",status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("train","validate"))
    for name in ("data","protocol","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--resume");parser.add_argument("--run");parser.add_argument("--epoch",type=int)
    parser.add_argument("--micro-batch",type=int,default=2);parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--max-seconds",type=float,default=37800);parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.action=="train":
        train_full(args.data,args.protocol,args.output,resume=args.resume,micro_batch=args.micro_batch,
                   threads=args.threads,max_seconds=args.max_seconds,smoke=args.smoke)
    else:
        if args.run is None or args.epoch is None:parser.error("validate requires --run and --epoch")
        print(json.dumps(validate_epoch(args.data,args.protocol,args.run,args.epoch,args.output,
                         previous=args.resume,micro_batch=args.micro_batch,threads=args.threads,
                         max_seconds=args.max_seconds,smoke=args.smoke),indent=2))


if __name__=="__main__":
    main()
