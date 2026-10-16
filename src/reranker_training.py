"""Evidence-supervised narrative section ranking with real training annotations."""
import argparse
import json
from pathlib import Path
import random
import time

from src.data.prepare_fairytale import sha
from src.narrative_training import (evidence_metrics, lexical_ranking, load_split,
                                    retrieve, seed_all, select_validation)


def build_groups(stories, questions, seed=42):
    groups = []
    skipped = 0
    for question in questions:
        sections = stories[question["story"]]["sections"]
        gold = set(question["gold_sections"])
        order = lexical_ranking(question["question"], sections)
        negatives = [i for i in order if sections[i]["id"] not in gold]
        if not negatives:
            skipped += 1
            continue
        chosen = negatives[:3]
        rest = negatives[3:]
        if rest:
            rng = random.Random(seed + int(sha(question["id"].encode())[:12], 16))
            chosen.append(rng.choice(rest))
        for positive, section in enumerate(sections):
            if section["id"] in gold:
                indices = [positive] + chosen
                groups.append({"id": question["id"], "positive_section": section["id"],
                               "section_ids": [sections[i]["id"] for i in indices],
                               "pairs": [(question["question"], sections[i]["text"]) for i in indices]})
    return groups, skipped


class SectionReranker:
    def __init__(self, model, tokenizer, max_tokens=384, batch_size=8):
        if type(max_tokens) is not int or max_tokens < 16 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive ranking batch size and context capacity required")
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def predict(self, pairs, **unused):
        import numpy as np
        import torch
        self.model.eval()
        scores = []
        if any(len(self.tokenizer.encode(p[0], add_special_tokens=False)) > self.max_tokens - 8 for p in pairs):
            raise ValueError("Question exceeds ranking token capacity")
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            encoded = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                     max_length=self.max_tokens, padding=True,
                                     truncation="only_second", return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, 0]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite section-reranker logits")
            scores.extend(logits.cpu().tolist())
        return np.asarray(scores)


def evaluate_sections(stories, questions, reranker=None, cases_path=None):
    rows = []
    for question in questions:
        sections = stories[question["story"]]["sections"]
        indices = retrieve(question["question"], sections, reranker)
        selected = [sections[i]["id"] for i in indices]
        candidates = [sections[i]["id"] for i in lexical_ranking(question["question"], sections)[:12]]
        metrics = evidence_metrics(selected, question["gold_sections"])
        rows.append({"id": question["id"], "story": question["story"],
                     "selected_sections": selected, "gold_sections": question["gold_sections"],
                     "candidate_sections": candidates,
                     "origin": question["origin"], "attribute": question["attribute"],
                     "explicit": question["explicit"], **metrics,
                     "candidate_recall": evidence_metrics(candidates, question["gold_sections"])["recall"],
                     "top4_ceiling": min(4, len(question["gold_sections"])) / len(question["gold_sections"])})
    report = {"n": len(rows), "recall_at_4": sum(r["recall"] for r in rows) / len(rows),
              "precision_at_4": sum(r["precision"] for r in rows) / len(rows),
              "candidate_recall_at_12": sum(r["candidate_recall"] for r in rows) / len(rows),
              "annotation_top4_ceiling": sum(r["top4_ceiling"] for r in rows) / len(rows)}
    for group in ("origin", "attribute", "explicit"):
        report[group] = {value: {"n": sum(r[group] == value for r in rows),
            "recall_at_4": sum(r["recall"] for r in rows if r[group] == value) /
                           sum(r[group] == value for r in rows)}
            for value in sorted(set(r[group] for r in rows))}
    if cases_path is not None:
        with Path(cases_path).open("x", encoding="utf-8") as handle:
            json.dump(rows, handle)
    return report


def run(args):
    import importlib.metadata
    import platform
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite reranker evidence")
    policy = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    spec, training = policy["model"], policy["training"]
    manifest_path = Path(args.data) / "manifest.json"
    if sha(manifest_path.read_bytes()) != policy["source"]["dataset_manifest_sha256"]:
        raise ValueError("Prespecified dataset manifest changed")
    stories, train = load_split(args.data, "train")
    validation_stories, validation = load_split(args.data, "validation")
    validation = select_validation(validation, policy)
    if sha(json.dumps([q["id"] for q in validation]).encode()) != policy["source"]["validation_ids_sha256"]:
        raise ValueError("Prespecified validation cohort changed")
    if args.smoke:
        train, validation = train[:4], validation[:2]
    groups, skipped = build_groups(stories, train, spec["seed"])
    if not groups:
        raise ValueError("No real supervised ranking groups")
    torch.set_num_threads(4)
    seed_all(spec["seed"])
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"], use_fast=True)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        spec["name"], revision=spec["revision"], use_safetensors=True, output_loading_info=True)
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("Incomplete or incompatible pretrained ranking head: " + str(info))
    if model.config.num_labels != 1:
        raise ValueError("Expected one pretrained ranking logit")
    model.to("cpu")
    model.gradient_checkpointing_enable()
    scorer = SectionReranker(model, tokenizer, spec["max_tokens"])
    weights = Path(hf_hub_download(spec["name"], "model.safetensors", revision=spec["revision"]))
    target_lengths = [len(tokenizer(p[0], p[1], add_special_tokens=True)["input_ids"])
                      for group in groups for p in group["pairs"]]
    output.mkdir(parents=True)
    provenance = {"model": spec, "training": training, "loading_info": info,
                  "pretrained_weights_sha256": sha(weights.read_bytes()),
                  "protocol_sha256": sha(Path(args.protocol).read_bytes()),
                  "manifest_sha256": sha(manifest_path.read_bytes()),
                  "train_questions": len(train), "training_groups": len(groups),
                  "training_pairs": len(target_lengths), "questions_without_negatives": skipped,
                  "pair_truncations": sum(n > spec["max_tokens"] for n in target_lengths),
                  "validation_questions": len(validation), "smoke": args.smoke,
                  "python": platform.python_version(), "torch": torch.__version__,
                  "environment": {name: importlib.metadata.version(name)
                                  for name in ("transformers", "numpy", "tokenizers", "huggingface-hub")},
                  "source_sha256": sha(Path(__file__).read_bytes()), "test_accessed": False}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    results = [{"kind": "bm25", "epoch": 0, "validation": evaluate_sections(validation_stories, validation,
                    cases_path=output / "bm25-validation-private.json")},
               {"kind": "pretrained", "epoch": 0,
                "validation": evaluate_sections(validation_stories, validation, scorer,
                    output / "pretrained-validation-private.json")}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    accumulation = training["effective_groups"] // training["micro_groups"]
    best_adapted = None
    began = time.monotonic()
    for epoch in range(1, (1 if args.smoke else training["epochs"]) + 1):
        order = list(groups)
        random.Random(spec["seed"] + epoch).shuffle(order)
        batches = [order[i:i + training["micro_groups"]] for i in range(0, len(order), training["micro_groups"])]
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        for step, batch in enumerate(batches, 1):
            pairs = [p for group in batch for p in group["pairs"]]
            encoded = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                                truncation="only_second", max_length=spec["max_tokens"], return_tensors="pt")
            logits = model(**encoded).logits[:, 0]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite ranking training logits")
            losses, offset = [], 0
            for group in batch:
                count = len(group["pairs"])
                losses.append(-torch.log_softmax(logits[offset:offset + count], dim=0)[0])
                offset += count
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ranking training loss")
            loss_sum += float(loss.detach())
            group_start = ((step - 1) // accumulation) * accumulation
            divisor = min(accumulation, len(batches) - group_start)
            (loss / divisor).backward()
            if step % accumulation == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "microstep": step, "loss": loss_sum / step,
                                  "seconds": time.monotonic() - began}), flush=True)
        metrics = evaluate_sections(validation_stories, validation, scorer,
                                    output / f"adapted-epoch-{epoch}-validation-private.json")
        row = {"kind": "adapted", "epoch": epoch, "validation": metrics,
               "train_loss": loss_sum / len(batches), "seconds": time.monotonic() - began}
        results.append(row)
        if best_adapted is None or metrics["recall_at_4"] > best_adapted["validation"]["recall_at_4"]:
            best_adapted = row
            model.save_pretrained(output / "adapted-checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "adapted-checkpoint")
        (output / "validation-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Stable max preserves declared simplicity/earlier-epoch tie order.
    selected = max(results, key=lambda r: r["validation"]["recall_at_4"])
    selection = {"status": "frozen", "selected": selected, "best_adapted": best_adapted,
                 "model": spec, "smoke": args.smoke, "test_accessed": False,
                 "adapted_weights_sha256": sha((output / "adapted-checkpoint/model.safetensors").read_bytes()),
                 "provenance_sha256": sha((output / "provenance.json").read_bytes())}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if args.smoke:
        import gc
        reference_scores = scorer.predict(groups[0]["pairs"])
        del scorer, optimizer, model
        gc.collect()
        reloaded = AutoModelForSequenceClassification.from_pretrained(output / "adapted-checkpoint",
            local_files_only=True, use_safetensors=True)
        repeated = SectionReranker(reloaded, tokenizer).predict(groups[0]["pairs"])
        import numpy as np
        difference = float(np.max(np.abs(reference_scores - repeated)))
        (output / "reload-check.json").write_text(json.dumps({"max_logit_difference": difference}))
        if difference > 1e-6:
            raise ValueError("Saved ranking checkpoint changed logits")
    print(json.dumps({"selection": selected, "test_accessed": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="docs/v5-reranker-protocol.json")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
