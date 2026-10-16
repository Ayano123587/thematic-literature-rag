"""Gold-free cached span selection with explicit source-overlap handling."""
import math
from src.reader_cloud import normalized


def valid_span(span):
    return (isinstance(span.get('null_margin'), (int, float))
            and math.isfinite(span['null_margin'])
            and type(span.get('retrieval_rank')) is int and span['retrieval_rank'] >= 1
            and type(span.get('answer_start')) is int and type(span.get('answer_end')) is int
            and 0 <= span['answer_start'] < span['answer_end']
            and isinstance(span.get('answer'), str) and bool(span['answer'])
            and isinstance(span.get('book'), str))


def distinct_occurrences(spans):
    """Overlapping windows around one source occurrence never add support."""
    intervals = sorted({(s['answer_start'], s['answer_end']) for s in spans},
                       key=lambda pair: (pair[1], pair[0]))
    count, last_end = 0, -1
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    return count


def select_reliable(spans, policy, other_spans=None):
    method, magnitude = policy['method'], policy['magnitude']
    if method not in {'baseline', 'source_consensus', 'ambiguity_gap', 'cross_reader_agreement'}:
        raise ValueError('Unknown reliability method')
    if not isinstance(magnitude, (int, float)) or not math.isfinite(magnitude):
        raise ValueError('Policy magnitude must be finite')
    if method != 'cross_reader_agreement' and magnitude < 0:
        raise ValueError('Policy magnitude must be nonnegative')
    eligible = [s for s in spans if valid_span(s) and s['null_margin'] > 0]
    if method == 'cross_reader_agreement':
        agreement = {(s['book'], s['answer_start'], s['answer_end'])
                     for s in other_spans or [] if valid_span(s) and s['null_margin'] > magnitude}
        eligible = [s for s in eligible if (s['book'], s['answer_start'], s['answer_end']) in agreement]
    groups = {}
    for span in eligible:
        key = (span['book'], tuple(normalized(span['answer'])))
        groups.setdefault(key, []).append(span)
    ranked = []
    for members in groups.values():
        representative = max(members, key=lambda s: s['null_margin'] - .8 * (s['retrieval_rank'] - 1))
        score = representative['null_margin'] - .8 * (representative['retrieval_rank'] - 1)
        support = distinct_occurrences(members)
        if method == 'source_consensus':
            score += magnitude * math.log1p(min(2, max(0, support - 1)))
        ranked.append((score, representative, support))
    ranked.sort(key=lambda row: -row[0])
    gap = ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None
    accepted = bool(ranked) and not (method == 'ambiguity_gap' and gap is not None and gap < magnitude)
    return {'span': ranked[0][1] if accepted else None,
            'diagnostics': {'method': method, 'magnitude': magnitude,
                            'distinct_answer_groups': len(groups), 'adjusted_score_gap': gap,
                            'source_occurrences': ranked[0][2] if ranked else 0,
                            'agreement_is_entailment': False,
                            'reason': 'Accepted best eligible span' if accepted else 'No eligible or sufficiently separated span'}}
