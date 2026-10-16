import pytest
from src.models.reliability import distinct_occurrences, select_reliable


def span(answer='Mira', start=0, margin=4, rank=1):
    return {'answer': answer, 'answer_start': start, 'answer_end': start + len(answer),
            'null_margin': margin, 'retrieval_rank': rank, 'book': 'story',
            'citation': f'story:{start}-{start + len(answer)}'}


def test_consensus_never_counts_duplicate_overlapping_windows_as_new_evidence():
    same = [span(), span(), span(start=1)]
    assert distinct_occurrences(same) == 1
    other = span('Jonas', start=20, margin=4.4)
    policy = {'method': 'source_consensus', 'magnitude': 1}
    assert select_reliable(same[:2] + [other], policy)['span'] == other
    result = select_reliable(same[:2] + [span(start=100), other], policy)
    assert result['span']['answer'] == 'Mira'
    assert result['diagnostics']['source_occurrences'] == 2


def test_gap_uses_distinct_answers_not_overlapping_reader_windows():
    policy = {'method': 'ambiguity_gap', 'magnitude': 1}
    assert select_reliable([span(), span()], policy)['span'] is not None
    result = select_reliable([span(margin=5), span('Jonas', 20, margin=4.7)], policy)
    assert result['span'] is None
    assert result['diagnostics']['adjusted_score_gap'] == pytest.approx(.3)


def test_reader_agreement_requires_exact_source_offsets_and_finite_margins():
    policy = {'method': 'cross_reader_agreement', 'magnitude': 0}
    assert select_reliable([span()], policy, [span(start=50)])['span'] is None
    assert select_reliable([span()], policy, [span(margin=-1)])['span'] is None
    assert select_reliable([span()], policy, [span(margin=1)])['span'] is not None
    assert select_reliable([span(margin=float('nan'))], policy, [span()])['span'] is None
    with pytest.raises(ValueError, match='finite'):
        select_reliable([span()], {'method': 'source_consensus', 'magnitude': float('inf')})
