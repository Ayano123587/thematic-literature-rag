import pytest
from src.reader_cloud import token_f1, choose, evaluate, extract


def test_reader_comparison_scores_real_answers_separately_from_controls():
    predictions = {'q': [{'answer': 'The lantern keeper', 'null_margin': 5, 'retrieval_rank': 1}],
                   'control': [{'answer': 'Invented', 'null_margin': 1, 'retrieval_rank': 1}]}
    gold = [{'id': 'q', 'kind': 'validation', 'answers': ['lantern keeper']},
            {'id': 'control', 'kind': 'control', 'answers': ['']}]
    report = evaluate(predictions, gold, [0, 2, 6], [0])
    assert report['selected']['threshold'] == 2
    assert report['selected']['token_f1'] == 1
    assert report['selected']['exact_match'] == 1
    assert report['selected']['control_false_answer_rate'] == 0
    assert token_f1('irrelevant', 'lantern') == 0
    assert choose([{'null_margin': float('nan')}], 0, 0) is None
    with pytest.raises(ValueError):
        choose([], 0, float('inf'))
    assert choose([{'null_margin':3,'retrieval_rank':0}],0,0) is None
    with pytest.raises(ValueError,match='unknown controls'):
        evaluate({'q':predictions['q']},gold[:1],[0],[0])
    with pytest.raises(ValueError,match='misaligned'):
        evaluate({'missing':[]},gold,[0],[0])


def test_portable_reader_preserves_global_offsets_without_gold(monkeypatch):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    class Encoding(dict):
        def sequence_ids(self, row):
            return [None, 0, None, 1, None]
    class Tokenizer:
        cls_token_id = 0
        def encode(self, *args, **kwargs):
            return [1]
        def __call__(self, *args, **kwargs):
            return Encoding(input_ids=torch.tensor([[0,1,2,3,4]]),
                            overflow_to_sample_mapping=torch.tensor([0]),
                            offset_mapping=torch.tensor([[[0,0],[0,3],[0,0],[0,4],[0,0]]]))
    def model(**kwargs):
        return SimpleNamespace(start_logits=torch.tensor([[0.,0.,0.,5.,0.]]),
                               end_logits=torch.tensor([[0.,0.,0.,5.,0.]]))
    passage = {'id': 'story:100-124', 'book': 'story', 'start': 100,
               'end': 124, 'text': 'Mira tended the lantern.'}
    spans = extract('Who?', [passage], Tokenizer(), model, 'cpu')
    assert spans[0]['answer'] == 'Mira'
    assert spans[0]['answer_start'] == 100 and spans[0]['answer_end'] == 104
    assert spans[0]['citation'] == passage['id']
    assert extract('', [passage], Tokenizer(), model, 'cpu') == []
    with pytest.raises(ValueError,match='positive'):
        extract('Who?', [passage], Tokenizer(), model, 'cpu',batch_size=0)
