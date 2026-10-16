"""Uncapped evidence ranking with optimizer continuation and hashed validation shards."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
from types import SimpleNamespace

import torch

from src.data.prepare_fairytale import sha
from src.full_generator_training import snapshot, verify_snapshot
from src.narrative_training import load_split, seed_all, select_validation
from src.reranker_training import SectionReranker, build_groups, evaluate_sections
from src.resumable_training import (atomic_json, epoch_order, file_sha, load_progress,
                                    optimize_epoch, save_progress)


class ListwiseRanker(torch.nn.Module):
    """Train the pretrained scalar head; every group starts with its positive."""
    def __init__(self, ranker):
        super().__init__()
        self.ranker = ranker

    def forward(self, counts, **encoded):
        if not counts or any(type(count) is not int or count < 2 for count in counts):
            raise ValueError("Each listwise group needs a positive and negative")
        logits = self.ranker(**encoded).logits[:, 0]
        if not torch.isfinite(logits).all() or sum(counts) != len(logits):
            raise ValueError("Invalid listwise logits/group geometry")
        losses, offset = [], 0
        for count in counts:
            if type(count) is not int or count < 2:
                raise ValueError("Each listwise group needs a positive and negative")
            losses.append(-torch.log_softmax(logits[offset:offset + count], dim=0)[0])
            offset += count
        return SimpleNamespace(loss=torch.stack(losses).mean())


def identity(data, protocol, smoke, threads):
    policy = json.loads(Path(protocol).read_text(encoding="utf-8"))
    training = policy["training"]
    if policy.get("validation_sampling") != {"mode": "all"}:
        raise ValueError("Full ranking requires all validation questions")
    if file_sha(Path(data)/"manifest.json") != policy["source"]["dataset_manifest_sha256"]:
        raise ValueError("Changed full source manifest")
    if type(threads) is not int or threads < 1 or any(type(training[k]) is not int or training[k] < 1
            for k in ("micro_groups", "effective_groups", "epochs")) or training["effective_groups"] % training["micro_groups"]:
        raise ValueError("Invalid group accumulation/thread configuration")
    value = {"manifest_sha256": file_sha(Path(data)/"manifest.json"), "protocol_sha256": file_sha(protocol),
             "model": policy["model"], "training": training, "smoke": smoke, "threads": threads,
             "torch": str(torch.__version__), "python": platform.python_version(),
             "environment": {n: importlib.metadata.version(n) for n in
                             ("transformers", "tokenizers", "numpy", "huggingface-hub")},
             "source_sha256": {n: file_sha(Path(__file__).parent/n) for n in
                 ("full_reranker_training.py", "reranker_training.py", "resumable_training.py",
                  "full_generator_training.py", "narrative_training.py", "data/prepare_fairytale.py")}}
    return policy, value


def strict_model(path, revision=None):
    from transformers import AutoModelForSequenceClassification
    kwargs = {"use_safetensors": True, "output_loading_info": True}
    kwargs.update({"revision": revision} if revision else {"local_files_only": True})
    model, info = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")) or model.config.num_labels != 1:
        raise ValueError("Incomplete pretrained scalar ranking head")
    return model.to("cpu"), info


def train_full(data, protocol, output, *, resume=None, smoke=False, threads=4, max_seconds=37800):
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not 0 <= max_seconds <= 39600:
        raise ValueError("Safe cloud checkpoint window required")
    began = time.monotonic(); deadline = began + max_seconds
    policy, lineage = identity(data, protocol, smoke, threads)
    spec, training = policy["model"], policy["training"]
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True); seed_all(spec["seed"])
    stories, questions = load_split(data, "train")
    if len(questions) != 7883:
        raise ValueError("Expected all 7883 source-family training questions")
    if smoke:
        questions = questions[:20]
    groups, skipped = build_groups(stories, questions, spec["seed"])
    if not groups:
        raise ValueError("No supervised ranking groups")
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve prior outputs; resume into a fresh directory")
    output.mkdir(parents=True)
    if resume is not None:
        resume = Path(resume)
        if json.loads((resume/"provenance.json").read_text())["identity"] != lineage:
            raise ValueError("Changed ranking resume identity")
        shutil.copytree(resume/"progress", output/"progress")
        for p in resume.glob("epoch-*"):
            verify_snapshot(p, lineage, int(p.name.split("-")[1]))
            shutil.copytree(p, output/p.name)
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"], use_fast=True)
    model, loading = strict_model(spec["name"], spec["revision"])
    weights = hf_hub_download(spec["name"], "model.safetensors", revision=spec["revision"])
    if file_sha(weights) != "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae":
        raise ValueError("Pinned pretrained ranking weight bytes changed")
    encoded_groups, truncated = [], 0
    print(json.dumps({"phase": "tokenizing", "groups": len(groups)}), flush=True)
    for group in groups:
        pairs = group["pairs"]
        if any(len(tokenizer.encode(q, add_special_tokens=False)) > spec["max_tokens"]-8 for q, _ in pairs):
            raise ValueError("Training question exceeds pair token capacity")
        lengths = tokenizer([q for q, _ in pairs], [s for _, s in pairs], truncation=False)["input_ids"]
        truncated += sum(len(row) > spec["max_tokens"] for row in lengths)
        encoded_groups.append(tokenizer([q for q, _ in pairs], [s for _, s in pairs],
                                       truncation="only_second", max_length=spec["max_tokens"]))
    atomic_json(output/"provenance.json", {"identity": lineage, "loading_info": loading,
                "pretrained_weights_sha256": file_sha(weights), "train_QA": len(questions),
                "groups": len(groups), "pairs": sum(len(g["pairs"]) for g in groups),
                "questions_without_negatives": skipped, "truncated_pairs": truncated, "test_accessed": False})
    wrapper = ListwiseRanker(model)
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    progress = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.0}
    if resume is not None:
        progress = load_progress(output/"progress", wrapper, optimizer, lineage)
    else:
        snapshot(output, 0, model, tokenizer, lineage)
    micro = training["micro_groups"]
    def batch(indices):
        rows = []
        for i in indices:
            encoded = encoded_groups[i]
            rows.extend({k: v[j] for k, v in encoded.items()} for j in range(len(encoded["input_ids"])))
        return {**tokenizer.pad(rows, padding=True, return_tensors="pt"),
                "counts": [len(groups[i]["pairs"]) for i in indices]}
    epochs = 1 if smoke else training["epochs"]
    while progress["epoch"] <= epochs:
        epoch = progress["epoch"]
        order = epoch_order(len(groups), spec["seed"], epoch)
        batches = [lambda ids=order[i:i+micro]: batch(ids) for i in range(0, len(order), micro)]
        def checkpoint(value):
            save_progress(output/"progress", wrapper, optimizer, lineage, value)
        if not optimize_epoch(wrapper, optimizer, batches, progress, training["effective_groups"]//micro,
                              checkpoint, deadline=deadline):
            break
        snapshot(output, epoch, model, tokenizer, lineage, progress["loss_sum"]/len(batches))
        progress = {"epoch": epoch+1, "next_microstep": 0, "loss_sum": 0.0}
        checkpoint(progress)
        if time.monotonic() >= deadline:
            break
    status = {"status": "training_complete" if progress["epoch"] > epochs else "checkpointed",
              "progress": progress, "epochs_required": epochs, "train_QA": len(questions),
              "seconds": time.monotonic()-began, "smoke": smoke, "test_accessed": False}
    atomic_json(output/"training-status.json", status)
    return status


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize empty validation")
    def metrics(items):
        return {"n": len(items), **{key: sum(r[field] for r in items)/len(items) for key, field in
                (("recall_at_4", "recall"), ("precision_at_4", "precision"),
                 ("candidate_recall_at_12", "candidate_recall"), ("annotation_top4_ceiling", "top4_ceiling"))}}
    result = metrics(rows)
    for key in ("origin", "attribute", "explicit"):
        result[key] = {v: metrics([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    return result


def validate_epoch(data, protocol, run, epoch, output, *, previous=None, smoke=False,
                   threads=4, max_seconds=37800, shard_size=64):
    if type(epoch) is not int or not 0 <= epoch <= 2 or type(shard_size) is not int or shard_size < 1 or isinstance(
            max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not 0 <= max_seconds <= 39600:
        raise ValueError("Valid snapshot, shard size and cloud window required")
    policy, lineage = identity(data, protocol, smoke, threads)
    stories, questions = load_split(data, "validation")
    questions = select_validation(questions, policy)
    if len(questions) != 2372 or sha(json.dumps([q["id"] for q in questions]).encode()) != policy["source"]["validation_ids_sha256"]:
        raise ValueError("Changed full validation coverage")
    if smoke:
        questions = questions[:2]
    checkpoint = Path(run)/f"epoch-{epoch}"
    receipt = verify_snapshot(checkpoint, lineage, epoch)
    cohort = {"run": lineage, "epoch": epoch, "weights_sha256": receipt["weights_sha256"],
              "ids_sha256": sha(json.dumps([q["id"] for q in questions]).encode()), "shard_size": shard_size}
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve prior validation shards")
    output.mkdir(parents=True)
    if previous is not None:
        previous = Path(previous)
        if json.loads((previous/"identity.json").read_text()) != cohort:
            raise ValueError("Changed validation continuation identity")
        for path in previous.glob("shard-*.json"):
            shutil.copy2(path, output/path.name)
    atomic_json(output/"identity.json", cohort)
    from transformers import AutoTokenizer
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model, loading = strict_model(checkpoint)
    scorer = SectionReranker(model, tokenizer, policy["model"]["max_tokens"])
    combined = []; deadline = time.monotonic()+max_seconds
    for start in range(0, len(questions), shard_size):
        path = output/f"shard-{start:05d}.json"
        expected = [q["id"] for q in questions[start:start+shard_size]]
        if path.exists():
            shard = json.loads(path.read_text())
            if shard.get("identity") != cohort or [r["id"] for r in shard["rows"]] != expected or sha(
                    json.dumps(shard["rows"], sort_keys=True, allow_nan=False).encode()) != shard.get("rows_sha256"):
                raise ValueError("Changed validation shard")
        else:
            private = output/"pending-cases.json"
            evaluate_sections(stories, questions[start:start+shard_size], scorer, private)
            rows = json.loads(private.read_text()); private.unlink()
            shard = {"identity": cohort, "rows": rows,
                     "rows_sha256": sha(json.dumps(rows, sort_keys=True, allow_nan=False).encode())}
            atomic_json(path, shard)
        combined.extend(shard["rows"])
        print(json.dumps({"validation_QA_completed": len(combined), "required": len(questions)}), flush=True)
        if time.monotonic() >= deadline:
            break
    complete = len(combined) == len(questions)
    status = {"complete": complete, "questions_completed": len(combined), "questions_required": len(questions),
              "identity": cohort, "loading_info": loading, "test_accessed": False, "smoke": smoke}
    if complete:
        status["validation"] = summarize_rows(combined)
        status["bm25"] = evaluate_sections(stories, questions)
    atomic_json(output/"validation-status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "validate"))
    for name in ("data", "protocol", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--resume"); parser.add_argument("--run"); parser.add_argument("--epoch", type=int)
    parser.add_argument("--max-seconds", type=float, default=37800); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "train":
        train_full(args.data, args.protocol, args.output, resume=args.resume, smoke=args.smoke, max_seconds=args.max_seconds)
    else:
        if args.run is None or args.epoch is None:
            parser.error("validate requires --run and --epoch")
        validate_epoch(args.data, args.protocol, args.run, args.epoch, args.output,
                       previous=args.resume, smoke=args.smoke, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
