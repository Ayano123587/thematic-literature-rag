"""Portable, bounded, sequential extractive-reader comparison on frozen contexts."""
import collections
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import string
import time


def normalized(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text.lower().translate(
        str.maketrans('', '', string.punctuation))).split()


def token_f1(prediction, answer):
    left, right = collections.Counter(normalized(prediction)), collections.Counter(normalized(answer))
    overlap = sum((left & right).values())
    return 2 * overlap / (sum(left.values()) + sum(right.values())) if overlap else 0.0


def choose(spans, threshold, penalty):
    if not math.isfinite(threshold) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid finite policy')
    eligible = [s for s in spans if math.isfinite(s['null_margin']) and s['null_margin'] > threshold
                and isinstance(s['retrieval_rank'], int) and s['retrieval_rank'] >= 1]
    return max(eligible, key=lambda s: s['null_margin'] - penalty * (s['retrieval_rank'] - 1), default=None)


def extract(question, passages, tokenizer, model, device, batch_size=8):
    import torch
    if batch_size < 1:
        raise ValueError('Reader batch size must be positive')
    if not passages or not isinstance(question, str) or not question.strip():
        return []
    if len(tokenizer.encode(question, add_special_tokens=False)) > 352:
        return []
    encoded = tokenizer([question] * len(passages), [p['text'] for p in passages],
                        padding=True, truncation='only_second', max_length=384,
                        return_overflowing_tokens=True, stride=96,
                        return_offsets_mapping=True, return_tensors='pt')
    mapping = encoded.pop('overflow_to_sample_mapping').tolist()
    offsets = encoded.pop('offset_mapping').tolist()
    result = []
    for lo in range(0, len(mapping), batch_size):
        batch = {key: value[lo:lo + batch_size].to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**batch)
        starts, ends = output.start_logits.cpu(), output.end_logits.cpu()
        for local in range(len(starts)):
            row = lo + local
            if not torch.isfinite(starts[local]).all() or not torch.isfinite(ends[local]).all():
                continue
            context = [i for i, sequence in enumerate(encoded.sequence_ids(row))
                       if sequence == 1 and offsets[row][i][1] > offsets[row][i][0]]
            top_start = sorted(context, key=lambda i: -float(starts[local][i]))[:20]
            top_end = sorted(context, key=lambda i: -float(ends[local][i]))[:20]
            valid = [(float(starts[local][a] + ends[local][b]), a, b)
                     for a in top_start for b in top_end if a <= b < a + 30]
            if not valid:
                continue
            score, a, b = max(valid)
            cls = (encoded['input_ids'][row] == tokenizer.cls_token_id).nonzero()[0].item()
            passage = passages[mapping[row]]
            start, end = offsets[row][a][0], offsets[row][b][1]
            answer = passage['text'][start:end]
            result.append({'answer': answer, 'citation': passage['id'], 'book': passage['book'],
                           'answer_start': passage['start'] + start,
                           'answer_end': passage['start'] + end,
                           'null_margin': score - float(starts[local][cls] + ends[local][cls]),
                           'retrieval_rank': mapping[row] + 1, 'reader_window': row})
    return result


def evaluate(predictions, gold, thresholds, penalties):
    positives = [r for r in gold if r['kind'] == 'validation']
    controls = [r for r in gold if r['kind'] == 'control']
    if not positives or not controls:
        raise ValueError('Comparison requires validation answers and unknown controls')
    if len({r['id'] for r in gold}) != len(gold) or set(predictions) != {r['id'] for r in gold}:
        raise ValueError('Duplicate or misaligned comparison IDs')
    policies = []
    for threshold in thresholds:
        for penalty in penalties:
            selected = {key: choose(value, threshold, penalty) for key, value in predictions.items()}
            f1 = sum(max(token_f1(selected[r['id']]['answer'] if selected[r['id']] else '', a)
                         for a in r['answers']) for r in positives) / len(positives)
            exact = sum(any(normalized(selected[r['id']]['answer'] if selected[r['id']] else '')
                            == normalized(a) for a in r['answers']) for r in positives) / len(positives)
            false_rate = sum(selected[r['id']] is not None for r in controls) / len(controls)
            policies.append({'threshold': threshold, 'rank_penalty': penalty, 'token_f1': f1,
                             'exact_match': exact, 'control_false_answer_rate': false_rate,
                             'abstention_rate': sum(selected[r['id']] is None for r in positives) / len(positives),
                             'objective': f1 - .25 * false_rate})
    return {'policies': policies, 'selected': max(policies, key=lambda p: (
        p['objective'], p['exact_match'], p['threshold'], -p['rank_penalty']))}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(folder, output, budget_seconds=900):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from src.models.loading import validate_loading_info
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'reader-comparison.json').exists():
        raise ValueError('Completed comparison exists; preserve original evidence')
    started = float(os.environ.get('READER_STARTED_MONOTONIC', time.monotonic()))
    deadline = started + budget_seconds
    protocol = json.loads((folder / 'protocol.json').read_text())
    raw = (folder / 'inputs.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol['input_sha256']:
        raise ValueError('Input manifest mismatch')
    rows = json.loads(raw)['rows']
    if not torch.cuda.is_available():
        raise RuntimeError('Prespecified GPU unavailable; no silent CPU fallback')
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictions, weights = {}, {}
    for specification in protocol['models']:
        if time.monotonic() >= deadline:
            raise TimeoutError('15-minute run budget expired')
        name, revision = specification['name'], specification['revision']
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
            name, revision=revision, trust_remote_code=False, use_safetensors=True,
            output_loading_info=True)
        validate_loading_info(loading_info)
        model = model.to('cuda').eval()
        predictions[name] = {}
        for row in rows:
            if time.monotonic() >= deadline:
                raise TimeoutError('15-minute run budget expired')
            spans = extract(row['question'], row['passages'], tokenizer, model, 'cuda')
            known = {p['id']: p for p in row['passages']}
            for span in spans:
                passage = known[span['citation']]
                start, end = span['answer_start'] - passage['start'], span['answer_end'] - passage['start']
                if not 0 <= start < end <= len(passage['text']) or passage['text'][start:end] != span['answer']:
                    raise ValueError('Reader output does not match exact evidence coordinates')
            predictions[name][row['id']] = spans
            (output / 'predictions-private.json').write_text(json.dumps(predictions), encoding='utf-8')
            print('inferred', name, row['id'], flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        weights[name] = file_sha(hf_hub_download(name, 'model.safetensors', revision=revision))
    # Gold is read only after both forward-inference cohorts are complete.
    gold_raw = (folder / 'gold.json').read_bytes()
    if hashlib.sha256(gold_raw).hexdigest() != protocol['gold_sha256']:
        raise ValueError('Gold manifest mismatch')
    gold = json.loads(gold_raw)
    if {r['id'] for r in rows} != {r['id'] for r in gold}:
        raise ValueError('Input/gold alignment mismatch')
    reports = {name: evaluate(values, gold, protocol['thresholds'], protocol['rank_penalties'])
               for name, values in predictions.items()}
    selected_name = max(reports, key=lambda name: (
        reports[name]['selected']['objective'], reports[name]['selected']['exact_match'],
        reports[name]['selected']['threshold'], -reports[name]['selected']['rank_penalty']))
    frozen = {'models': reports, 'selected_model': selected_name,
              'selected_policy': reports[selected_name]['selected'],
              'frozen_at_unix': time.time(), 'protocol': protocol}
    (output / 'reader-validation-selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    fresh_report = None
    if protocol.get('fresh_test_document'):
        # Fresh question and gold files are first parsed after validation selection is serialized.
        from src.models.train_model import HybridIndex, encoder, cross_encoder
        from src.models.grounded_spans import expand_passage
        if file_sha(folder / 'fresh-inputs.json') != protocol['fresh_inputs_sha256']:
            raise ValueError('Fresh input manifest mismatch')
        fresh_rows = json.loads((folder / 'fresh-inputs.json').read_text())
        source_folder = folder / 'processed'
        text = (source_folder / 'cask.txt').read_text(encoding='utf-8')
        if file_sha(folder / 'fresh.lrag') != protocol['fresh_index_sha256']:
            raise ValueError('Fresh index manifest mismatch')
        index = HybridIndex.load(folder / 'fresh.lrag')
        index.verify_sources(source_folder)
        index.verify_chunk_contract(source_folder, 96, 48)
        contexts = {}
        for row in fresh_rows:
            if time.monotonic() >= deadline:
                raise TimeoutError('Budget expired during fresh retrieval; no restart')
            hits = index.search(row['question'], book='cask', k=12,
                                reranker='cross-encoder/ms-marco-MiniLM-L-6-v2')
            contexts[row['id']] = [vars(expand_passage(p, text, 700)) for p, _ in hits]
        encoder.cache_clear()
        cross_encoder.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
        specification = next(m for m in protocol['models'] if m['name'] == selected_name)
        tokenizer = AutoTokenizer.from_pretrained(selected_name, revision=specification['revision'], trust_remote_code=False)
        model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
            selected_name, revision=specification['revision'], trust_remote_code=False,
            use_safetensors=True, output_loading_info=True)
        validate_loading_info(loading_info)
        model = model.to('cuda').eval()
        fresh_predictions = {}
        for row in fresh_rows:
            if time.monotonic() >= deadline:
                raise TimeoutError('Budget expired during fresh reader; no restart')
            fresh_predictions[row['id']] = extract(row['question'], contexts[row['id']], tokenizer, model, 'cuda')
            (output / 'fresh-predictions-private.json').write_text(json.dumps(fresh_predictions), encoding='utf-8')
            print('fresh inferred', row['id'], flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if file_sha(folder / 'fresh-gold.json') != protocol['fresh_gold_sha256']:
            raise ValueError('Fresh gold manifest mismatch')
        fresh_gold = json.loads((folder / 'fresh-gold.json').read_text())
        if {r['id'] for r in fresh_rows} != {r['id'] for r in fresh_gold}:
            raise ValueError('Fresh input/gold alignment mismatch')
        policy = frozen['selected_policy']
        selected_spans = [choose(fresh_predictions[r['id']], policy['threshold'], policy['rank_penalty'])
                          for r in fresh_gold]
        count = len(fresh_gold)
        accepted = [span for span in selected_spans if span is not None]
        exact_source = sum(text[s['answer_start']:s['answer_end']] == s['answer'] for s in accepted)
        if exact_source != len(accepted):
            raise ValueError('Fresh answer/source mismatch')
        fresh_report = {'queries': count,
                        'token_f1': sum(max(token_f1(s['answer'] if s else '', a) for a in r['answers'])
                                        for s, r in zip(selected_spans, fresh_gold)) / count,
                        'exact_match': sum(any(normalized(s['answer'] if s else '') == normalized(a)
                                               for a in r['answers']) for s, r in zip(selected_spans, fresh_gold)) / count,
                        'abstention_rate': (count - len(accepted)) / count,
                        'accepted_answers': len(accepted), 'accepted_exact_source_checks': exact_source,
                        'no_answer_controls': 0, 'semantic_entailment_proven': False}
    report = {'scope': 'Reused Dorian validation; prespecified fresh Cask test only after serialized selection',
              'protocol': protocol, 'models': reports, 'weight_sha256': weights,
              'selection': frozen, 'fresh_test': fresh_report,
              'elapsed_seconds': time.monotonic() - started, 'python': platform.python_version(),
              'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'tokenizers']},
              'gpu': torch.cuda.get_device_name(0), 'all_emitted_spans_source_exact': True,
              'semantic_entailment_proven': False,
              'private_predictions_sha256': file_sha(output / 'predictions-private.json')}
    (output / 'reader-comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: value['selected'] for name, value in reports.items()}), flush=True)
    return report
