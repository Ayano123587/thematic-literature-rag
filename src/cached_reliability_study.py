"""Evaluate only prespecified cached validation policies; never load test outputs."""
import hashlib
import json
from pathlib import Path
from src.models.reliability import select_reliable
from src.reader_cloud import token_f1, normalized


def run():
    output = Path('docs/v4-cached-validation-results.json')
    if output.exists():
        raise ValueError('Study already completed; preserve original outcomes')
    protocol_path = Path('docs/v4-prespecified-protocol.json')
    protocol = json.loads(protocol_path.read_text())
    folder = Path('data/interim/v3-kaggle-output')
    predictions = json.loads((folder / 'predictions-private.json').read_text())
    gold = json.loads((folder / 'private-input/gold.json').read_text())
    previous = json.loads(Path('docs/v3-reader-locked-test.json').read_text())
    if hashlib.sha256((folder / 'predictions-private.json').read_bytes()).hexdigest() != previous['private_predictions_sha256']:
        raise ValueError('Cached validation predictions changed')
    if hashlib.sha256((folder / 'private-input/gold.json').read_bytes()).hexdigest() != previous['protocol']['gold_sha256']:
        raise ValueError('Cached validation gold changed')
    mini = predictions['deepset/minilm-uncased-squad2']
    roberta = predictions['deepset/roberta-base-squad2']
    if set(mini) != set(roberta) or set(mini) != {r['id'] for r in gold}:
        raise ValueError('Validation IDs are misaligned')
    text = Path('data/processed/dorian.txt').read_text(encoding='utf-8')
    for rows in (mini, roberta):
        for spans in rows.values():
            if any(s['book'] != 'dorian' or text[s['answer_start']:s['answer_end']] != s['answer'] for s in spans):
                raise ValueError('Cached source span mismatch')
    positives = [r for r in gold if r['kind'] == 'validation']
    controls = [r for r in gold if r['kind'] == 'control']
    if len(positives) != 29 or len(controls) != 4:
        raise ValueError('Expected fixed Dorian cohort')
    policies = [{'method': 'baseline', 'magnitude': 0}]
    for hypothesis in protocol['hypotheses']:
        field = {'source_consensus': 'weights', 'ambiguity_gap': 'minimum_gaps',
                 'cross_reader_agreement': 'roberta_minimum_margins'}[hypothesis['name']]
        policies.extend({'method': hypothesis['name'], 'magnitude': value} for value in hypothesis[field])
    results, private = [], {}
    for policy in policies:
        selections = {identifier: select_reliable(spans, policy, roberta[identifier])
                      for identifier, spans in mini.items()}
        f1s, exact, accepted = [], [], []
        for row in positives:
            span = selections[row['id']]['span']
            answer = span['answer'] if span else ''
            f1s.append(max(token_f1(answer, a) for a in row['answers']))
            exact.append(any(normalized(answer) == normalized(a) for a in row['answers']))
            accepted.append(span is not None)
        false = sum(selections[row['id']]['span'] is not None for row in controls)
        summary = {'policy': policy, 'token_f1': sum(f1s) / 29, 'exact_match': sum(exact) / 29,
                   'accepted_answers': sum(accepted), 'answer_coverage': sum(accepted) / 29,
                   'accepted_gold_em_mismatches': sum(a and not e for a, e in zip(accepted, exact)),
                   'control_false_answers': false, 'control_false_answer_rate': false / 4,
                   'objective': sum(f1s) / 29 - .25 * false / 4}
        results.append(summary)
        private[f"{policy['method']}:{policy['magnitude']}"] = selections
    baseline = results[0]
    for result in results:
        enough_coverage = result['answer_coverage'] >= .5
        noninferior = result['token_f1'] >= baseline['token_f1'] - 1e-12
        exact_ok = result['exact_match'] >= baseline['exact_match'] - 1e-12
        controls_ok = result['control_false_answers'] <= baseline['control_false_answers']
        gain = result['token_f1'] >= baseline['token_f1'] + .01 - 1e-12
        fewer = baseline['accepted_gold_em_mismatches'] - result['accepted_gold_em_mismatches'] >= 3
        result['meets_prespecified_promotion'] = enough_coverage and exact_ok and controls_ok and (gain or (noninferior and fewer))
    qualified = [r for r in results[1:] if r['meets_prespecified_promotion']]
    winner = max(qualified, key=lambda r: (r['objective'], r['exact_match'],
                 -r['accepted_gold_em_mismatches'], -r['policy']['magnitude'])) if qualified else None
    private_path = Path('data/interim/v4-cached-selections-private.json')
    private_path.write_text(json.dumps(private, allow_nan=False), encoding='utf-8')
    report = {'scope': 'Reused Dorian validation only; no observed-test predictions, gold, case analysis or inference',
              'historical_metadata_use': 'Only validation prediction/gold hash fields read from the historical public v3 report; its fresh-test fields are not used.',
              'protocol': protocol, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'results': results,
              'qualified_winner': winner, 'independent_heldout_score_available': False,
              'private_selections_sha256': hashlib.sha256(private_path.read_bytes()).hexdigest(),
              'annotation_limit': 'Gold EM mismatch is not proof of incorrectness or lack of semantic support.'}
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'baseline': baseline, 'winner': winner, 'policies': len(results)}, indent=2))
    return report
