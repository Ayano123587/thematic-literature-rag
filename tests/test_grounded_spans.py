import pytest
from src.features.build_features import chunk_book
from src.models.grounded_spans import expand_passage,select_span,assemble


def test_expansion_keeps_exact_offsets_and_word_boundaries():
    text='Mira tended the lantern. Jonas repaired the boat. Mira returned before dawn.'
    passage=chunk_book('story',text,words=6,overlap=1)[1]
    expanded=expand_passage(passage,text,characters=9)
    assert text[expanded.start:expanded.end]==expanded.text
    assert expanded.start<=passage.start and expanded.end>=passage.end
    with pytest.raises(ValueError,match='mismatches'):expand_passage(passage,'changed source')


def test_rank_penalty_and_null_threshold_are_separate():
    spans=[{'null_margin':6,'retrieval_rank':1},{'null_margin':7,'retrieval_rank':8}]
    assert select_span(spans,threshold=2,rank_penalty=0)==spans[1]
    assert select_span(spans,threshold=2,rank_penalty=.3)==spans[0]
    assert select_span(spans,threshold=8) is None


def test_citation_assembly_fails_closed_on_fabrication_and_wrong_book():
    passage=chunk_book('story','Mira tended the lantern.',words=20,overlap=0)[0]
    span={'answer':'Mira','citation':passage.id,'book':'story','answer_start':0,'answer_end':4,'null_margin':10}
    assert assemble(span,[passage],'story')['evidence']['answer_end']==4
    assert assemble(span,[passage],'other')['abstained']
    assert assemble(dict(span,answer='Jonas'),[passage],'story')['abstained']
    assert assemble(dict(span,answer_start=-1),[passage],'story')['abstained']


def test_sliding_window_configuration_rejected_before_model_load():
    pytest.importorskip('torch')
    from src.models.reader import candidates
    passage=chunk_book('story','Mira tended the lantern.',words=20,overlap=0)[0]
    with pytest.raises(ValueError,match='Invalid reader window'):candidates('Who?', [passage],stride=300)


def test_production_reader_checks_source_before_inference(tmp_path):
    pytest.importorskip('torch')
    from src.models.grounded_spans import answer
    class Index:
        passages = chunk_book('story', 'Mira tended the lantern.', words=20, overlap=0)
        def verify_sources(self, directory):
            raise ValueError('Source fingerprint mismatch')
    assert answer(Index(), '', 'story', tmp_path, {})['abstained']
    assert answer(Index(), 'Who?', 'missing', tmp_path, {})['abstained']
    with pytest.raises(ValueError, match='fingerprint'):
        answer(Index(), 'Who?', 'story', tmp_path, {})


def test_production_reader_rejects_a_different_model_policy(tmp_path):
    pytest.importorskip('torch')
    from src.models.grounded_spans import answer
    class Index:
        passages = chunk_book('story', 'Mira tended the lantern.', words=20, overlap=0)
        backend = 'lsa'
        def verify_sources(self, directory):
            pass
    with pytest.raises(ValueError, match='frozen selection'):
        answer(Index(), 'Who?', 'story', tmp_path, {'models': {}})
