"""Real narrative supervision, source-section retrieval and auditable generation.

Gold evidence is used for training and evaluation only. Runtime retrieval accepts
questions and sections, with no answer or gold-section argument.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import string
import time
from pathlib import Path

from src.data.prepare_fairytale import normalized, sha


def section_ids(value):
    if not isinstance(value, str) or not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", value):
        raise ValueError("Expected comma-separated evidence section IDs")
    ids = re.findall(r"\d+", value)
    if not ids:
        raise ValueError("Question has no annotated evidence sections")
    return list(dict.fromkeys(ids))


def load_split(folder, split, selection=None):
    folder = Path(folder)
    if split == "test":
        if selection is None or not Path(selection).is_file():
            raise ValueError("Test QA is sealed until serialized selection exists")
        frozen = json.loads(Path(selection).read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen" or not frozen.get("checkpoint"):
            raise ValueError("Invalid frozen selection")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    expected = {r["id"]: r for r in manifest["stories"] if r["split"] == split}
    stories, questions = {}, []
    for name, record in sorted(expected.items()):
        path = folder / split / (name + ".story.json")
        story = json.loads(path.read_text(encoding="utf-8"))
        if sha(story["text"].encode()) != record["source_sha256"]:
            raise ValueError("Source text hash changed")
        qa = folder / split / (name + ".qa.csv")
        if sha(qa.read_bytes()) != record["qa_sha256"]:
            raise ValueError("QA hash changed")
        for section in story["sections"]:
            if story["text"][section["start"]:section["end"]] != section["text"]:
                raise ValueError("Section source offsets changed")
        stories[name] = story
        for ordinal, row in enumerate(csv.DictReader(qa.open(encoding="utf-8-sig")), 1):
            refs = [row.get(key, "").strip() for key in ("answer1", "answer4")]
            refs = list(dict.fromkeys(r for r in refs if r))
            gold = section_ids(row["cor_section"])
            known = {s["id"] for s in story["sections"]}
            if not refs or not row["question"].strip() or not set(gold) <= known:
                raise ValueError(f"Malformed QA in {name}")
            questions.append({"id": name + ":row" + str(ordinal) + ":" + row["question_id"],
                              "source_question_id": row["question_id"], "source_row_ordinal": ordinal,
                              "story": name,
                              "question": row["question"], "references": refs,
                              "gold_sections": gold, "origin": record["origin"],
                              "attribute": row["attribute1"], "explicit": row["ex-or-im1"]})
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    return stories, questions


def lexical_ranking(question, sections):
    """BM25 section retrieval with deterministic source-order ties."""
    tokens = [normalized(s["text"]).split() for s in sections]
    avg = sum(map(len, tokens)) / max(1, len(tokens))
    query = set(normalized(question).split())
    df = {word: sum(word in t for t in tokens) for word in query}
    scores = []
    for index, words in enumerate(tokens):
        counts = collections.Counter(words)
        score = 0.0
        for word in query:
            frequency = counts[word]
            idf = math.log(1 + (len(tokens) - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(words) / max(1, avg)))
        scores.append((score, index))
    return [index for _, index in sorted(scores, key=lambda x: (-x[0], x[1]))]


def retrieve(question, sections, reranker=None):
    order = lexical_ranking(question, sections)
    if reranker is not None:
        first = order[:12]
        scores = reranker.predict([(question, sections[i]["text"]) for i in first],
                                  show_progress_bar=False, convert_to_numpy=True)
        if not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("Nonfinite reranker output")
        order = [i for _, i in sorted(zip(scores, first), key=lambda p: (-float(p[0]), p[1]))]
    return order[:4]


def evidence_metrics(selected, gold):
    selected, gold = set(selected), set(gold)
    return {"recall": len(selected & gold) / max(1, len(gold)),
            "precision": len(selected & gold) / max(1, len(selected))}


def token_f1(prediction, reference):
    def words(value):
        value = value.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", value).split()
    pred, gold = words(prediction), words(reference)
    same = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    f1 = 2 * same / (len(pred) + len(gold)) if pred or gold else 1.0
    return float(pred == gold), f1


def rouge_l(prediction, reference):
    left, right = normalized(prediction).split(), normalized(reference).split()
    row = [0] * (len(right) + 1)
    for token in left:
        previous = row[:]
        for j, other in enumerate(right, 1):
            row[j] = previous[j - 1] + 1 if token == other else max(previous[j], row[j - 1])
    return 2 * row[-1] / (len(left) + len(right)) if left or right else 1.0


def assemble(question, sections, indices, tokenizer, max_tokens=1024):
    """Truncate sections individually so retained citation text is exact source."""
    prefix = "question: " + question + " context: "
    remaining = max_tokens - len(tokenizer.encode(prefix, add_special_tokens=False)) - 3
    if remaining < 16:
        raise ValueError("Question leaves insufficient context capacity")
    pieces, citations = [], []
    for index in sorted(indices):
        section = sections[index]
        encoded = tokenizer(section["text"], add_special_tokens=False,
                            return_offsets_mapping=True)
        capacity = min(remaining, len(encoded["input_ids"]))
        if capacity <= 0:
            break
        end = encoded["offset_mapping"][capacity - 1][1]
        passage = section["text"][:end]
        pieces.append(passage)
        citations.append({"section": section["id"], "start": section["start"],
                          "end": section["start"] + end, "text": passage})
        remaining -= capacity + 2
    text = prefix + "\n\n".join(pieces)
    # BPE merges across boundary whitespace may change counts; never silently truncate
    # after declaring citations. Shrink by exact source character offsets if needed.
    while len(tokenizer.encode(text)) > max_tokens and pieces:
        pieces[-1] = pieces[-1][:-1]
        citations[-1]["end"] -= 1
        citations[-1]["text"] = pieces[-1]
        text = prefix + "\n\n".join(pieces)
    return text, citations


def summarize(predictions):
    def metrics(rows):
        return {key: sum(r[key] for r in rows) / max(1, len(rows))
                for key in ("em", "f1", "rouge_l", "evidence_recall", "evidence_precision",
                            "retained_evidence_recall", "citation_validity")}
    result = {"n": len(predictions), **metrics(predictions)}
    for group in ("origin", "explicit", "attribute", "reference_count"):
        values = sorted(set(str(r[group]) for r in predictions))
        result[group] = {value: {"n": sum(str(r[group]) == value for r in predictions),
                                **metrics([r for r in predictions if str(r[group]) == value])}
                         for value in values}
    return result


def evaluate(model, tokenizer, stories, questions, reranker=None, batch_size=8):
    import torch
    model.eval()
    predictions = []
    for start in range(0, len(questions), batch_size):
        rows = questions[start:start + batch_size]
        inputs, evidence, ranks = [], [], []
        for q in rows:
            story = stories[q["story"]]
            indices = retrieve(q["question"], story["sections"], reranker)
            context, citations = assemble(q["question"], story["sections"], indices, tokenizer,
                                          min(1024, tokenizer.model_max_length))
            inputs.append(context)
            evidence.append(citations)
            ranks.append(indices)
        encoded = tokenizer(inputs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=96, num_beams=4,
                                    do_sample=False, early_stopping=True, use_cache=True)
        answers = tokenizer.batch_decode(output, skip_special_tokens=True)
        for q, answer, citations, indices in zip(rows, answers, evidence, ranks):
            story = stories[q["story"]]
            scores = [token_f1(answer, ref) for ref in q["references"]]
            selected = [story["sections"][i]["id"] for i in indices]
            relevance = evidence_metrics(selected, q["gold_sections"])
            retained = evidence_metrics([c["section"] for c in citations], q["gold_sections"])
            validity = all(story["text"][c["start"]:c["end"]] == c["text"] for c in citations)
            predictions.append({**q, "answer": answer, "citations": citations,
                                "reference_count": len(q["references"]),
                                "em": max(s[0] for s in scores), "f1": max(s[1] for s in scores),
                                "rouge_l": max(rouge_l(answer, r) for r in q["references"]),
                                "evidence_recall": relevance["recall"],
                                "evidence_precision": relevance["precision"],
                                "retained_evidence_recall": retained["recall"],
                                "citation_validity": float(validity)})
    return predictions, summarize(predictions)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validation_cohort(questions, per_origin=100):
    """Predefined, label-independent sampling; never sample by difficulty/score."""
    origins = sorted(set(q["origin"] for q in questions))
    return [q for origin in origins for q in sorted(
        (q for q in questions if q["origin"] == origin),
        key=lambda q: (sha(q["id"].encode()), q["id"]))[:per_origin]]


def run(args):
    if args.micro_batch < 1 or args.threads < 1:
        raise ValueError("Microbatch and thread counts must be positive")
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
    from torch.utils.data import DataLoader
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite training evidence")
    output.mkdir(parents=True)
    policy = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    spec = policy["generator"]
    torch.set_num_threads(args.threads)
    seed_all(spec["seed"])
    train_stories, train = load_split(args.data, "train")
    valid_stories, valid = load_split(args.data, "validation")
    valid = validation_cohort(valid)
    if args.smoke:
        train, valid = train[:4], valid[:2]
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"], use_fast=True)
    model, loading = AutoModelForSeq2SeqLM.from_pretrained(
        spec["name"], revision=spec["revision"], output_loading_info=True,
        use_safetensors=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete generator checkpoint: {loading}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    import importlib.metadata
    import platform
    from huggingface_hub import hf_hub_download
    weights = Path(hf_hub_download(spec["name"], "model.safetensors", revision=spec["revision"]))
    provenance = {"protocol_sha256": sha(Path(args.protocol).read_bytes()),
                  "manifest_sha256": sha((Path(args.data) / "manifest.json").read_bytes()),
                  "model": spec, "loading_info": loading, "device": device,
                  "train_questions": len(train), "validation_questions": len(valid),
                  "validation_ids": [q["id"] for q in valid],
                  "torch": torch.__version__, "smoke": args.smoke,
                  "pretrained_weights_sha256": sha(weights.read_bytes()),
                  "python": platform.python_version(),
                  "environment": {package: importlib.metadata.version(package)
                                  for package in ("transformers", "tokenizers", "numpy", "huggingface-hub")},
                  "source_sha256": sha(Path(__file__).read_bytes())}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    baseline_predictions, baseline = evaluate(model, tokenizer, valid_stories, valid)
    (output / "baseline-private.json").write_text(json.dumps(baseline_predictions), encoding="utf-8")
    results = [{"epoch": 0, "validation": baseline}]
    rows = []
    for q in train:
        story = train_stories[q["story"]]
        # Real human-linked evidence only for supervised reader training. Primary
        # validation/test retrieval has no gold injection. Mixed-context ablation
        # is a separate optional hypothesis, not hidden here.
        indices = [i for i, s in enumerate(story["sections"]) if s["id"] in q["gold_sections"]]
        context, _ = assemble(q["question"], story["sections"], indices, tokenizer,
                              spec["input_tokens"])
        row = tokenizer(context)
        row["labels"] = tokenizer(text_target=q["references"][0], max_length=96,
                                   truncation=True)["input_ids"]
        rows.append(row)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    loader = DataLoader(rows, batch_size=args.micro_batch, shuffle=True, collate_fn=collator)
    accumulation = spec["effective_batch_size"] // args.micro_batch
    if accumulation < 1 or accumulation * args.micro_batch != spec["effective_batch_size"]:
        raise ValueError("Microbatch must divide effective batch size")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    epochs = 1 if args.smoke else spec["epochs"]
    best = None
    began = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss_sum += float(loss.detach())
            # Correct partial final accumulation group, rather than underweighting it.
            group_start = ((step - 1) // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / divisor).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "microstep": step, "loss": loss_sum / step,
                                  "seconds": time.monotonic() - began}), flush=True)
        predictions, metrics = evaluate(model, tokenizer, valid_stories, valid)
        results.append({"epoch": epoch, "loss": loss_sum / len(loader), "validation": metrics,
                        "seconds": time.monotonic() - began})
        (output / f"validation-epoch-{epoch}-private.json").write_text(json.dumps(predictions), encoding="utf-8")
        if best is None or metrics["f1"] > best["validation"]["f1"]:
            best = results[-1]
            model.save_pretrained(output / "selected-checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "selected-checkpoint")
        (output / "validation-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    checkpoint = output / "selected-checkpoint"
    selection = {"status": "frozen", "checkpoint": str(checkpoint), "selected": best,
                 "weights_sha256": sha((checkpoint / "model.safetensors").read_bytes()),
                 "retrieval": "BM25 top4", "smoke": args.smoke,
                 "provenance_sha256": sha((output / "provenance.json").read_bytes())}
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if args.smoke:
        # Exercise real persisted-weight loading and deterministic answer equality.
        # The optimizer/collator retain parameter references, so release them too.
        import gc
        del optimizer, scaler, loader, collator, model
        gc.collect()
        reloaded = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True,
                                                       use_safetensors=True).to(device)
        repeated, _ = evaluate(reloaded, tokenizer, valid_stories, valid)
        expected = json.loads((output / "validation-epoch-1-private.json").read_text())
        equal = [r["answer"] for r in repeated] == [r["answer"] for r in expected]
        (output / "reload-check.json").write_text(json.dumps({"answers_identical": equal,
                                                              "questions": len(repeated)}))
        if not equal:
            raise ValueError("Saved checkpoint reload changed deterministic smoke answers")
    # Smoke never opens test QA. Real test is a separate explicit command and
    # exclusive output marker, so a failed test is preserved rather than rerun.
    print(json.dumps({"selected_epoch": best["epoch"], "validation": best["validation"],
                      "test_accessed": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="docs/v5-training-protocol.json")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
