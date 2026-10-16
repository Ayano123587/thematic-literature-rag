import copy
import json
import sys
import pytest
from src.features.build_features import chunk_book
from src.models.grounded_spans import expand_passage
from src.models.answer_artifact import make_artifact, verify_artifact, audit_answer, digest_policy


@pytest.fixture
def example():
    text = 'Mirá tended the lantern. Jonas repaired the boat. Mirá returned before dawn.'
    selection = {'models': {'reader': 'fixture', 'reader_revision': 'a' * 40,
                            'embedding_revision': 'b' * 40, 'reranker_revision': 'c' * 40},
                 'selected': {'threshold': 0, 'rank_penalty': .8,
                              'config': {'words': 6, 'overlap': 2, 'expand': 9, 'k': 12, 'sliding': True}}}
    passage = expand_passage(chunk_book('story', text, 6, 2)[0], text, 9)
    result = {'abstained': False, 'answer': 'Mirá', 'book': 'story', 'null_margin': 5,
              'retrieval_rank': 1, 'answer_start': 0, 'answer_end': 4, 'citation': passage.id,
              'evidence': {'citation': passage.id, 'answer_start': 0, 'answer_end': 4}}
    return text, selection, make_artifact(result, text, 'story', selection)


def test_saved_unicode_answer_roundtrip_and_source_drift(tmp_path, example):
    text, selection, artifact = example
    (tmp_path / 'story.txt').write_text(text, encoding='utf-8')
    path = tmp_path / 'answer.json'
    path.write_text(json.dumps(artifact, ensure_ascii=False), encoding='utf-8')
    result = audit_answer(path, tmp_path, selection)
    assert result['source_exact'] and result['frozen_policy_matches']
    assert not result['authenticity_verified'] and not result['semantic_correctness_verified']
    (tmp_path / 'story.txt').write_text(text + ' changed', encoding='utf-8')
    with pytest.raises(ValueError, match='fingerprint'):
        audit_answer(path, tmp_path, selection)


def test_policy_rewrite_and_coordinate_forgery_rejected(example):
    text, selection, original = example
    artifact = copy.deepcopy(original)
    artifact['policy']['threshold'] = 2
    with pytest.raises(ValueError, match='fingerprint'):
        verify_artifact(artifact, text, selection)
    artifact['policy_sha256'] = digest_policy(artifact['policy'])
    consistency = verify_artifact(artifact,text)
    assert consistency['valid'] and not consistency['frozen_policy_matches']
    assert not consistency['authenticity_verified']
    with pytest.raises(ValueError, match='frozen selection'):
        verify_artifact(artifact, text, selection)
    for field, value in [('answer', 'Jonas'), ('answer_start', -1), ('book', '../story'),
                         ('null_margin', float('nan'))]:
        artifact = {**original, field: value}
        with pytest.raises(ValueError):
            verify_artifact(artifact, text, selection)
    artifact = copy.deepcopy(original)
    artifact['citation'] = f'story:0-{len(text)}'
    artifact['evidence']['citation'] = artifact['citation']
    with pytest.raises(ValueError, match='geometry'):
        verify_artifact(artifact, text, selection)


def test_abstention_and_duplicate_key_integrity(tmp_path, example):
    text, selection, original = example
    artifact = make_artifact({'abstained': True, 'answer': None}, text, 'story', selection)
    assert verify_artifact(artifact, text, selection)['abstained']
    artifact['answer'] = 'Mirá'
    with pytest.raises(ValueError, match='Abstention'):
        verify_artifact(artifact, text, selection)
    path = tmp_path / 'duplicate.json'
    path.write_text('{"book":"story","book":"other"}')
    with pytest.raises(ValueError, match='Duplicate'):
        audit_answer(path, tmp_path, selection)


def test_malformed_structures_and_boolean_numeric_values_fail_clearly(tmp_path, example):
    text, selection, original = example
    for bad in [[], {}, {**original,'policy': []},
                {**original,'policy': {**original['policy'],'config': []}},
                {**original,'policy': {**original['policy'],'models': []}},
                {**original,'policy': {**original['policy'],'threshold': True}},
                {**original,'policy': {**original['policy'],'rank_penalty': False}},
                {**original,'policy': {**original['policy'],'threshold': 10**400}},
                {**original,'policy': {**original['policy'],'rank_penalty': 10**400}},
                {**original,'null_margin': 10**400}, {**original,'null_margin': True}]:
        with pytest.raises(ValueError):
            verify_artifact(bad,text,selection)
    path=tmp_path/'oversized.json'
    path.write_bytes(b' ' * 1_000_001)
    with pytest.raises(ValueError,match='large'):
        audit_answer(path,tmp_path,selection)


def test_cli_checks_saved_artifact_without_reader_inference(tmp_path, example, monkeypatch, capsys):
    from src import cli
    text, selection, artifact = example
    (tmp_path / 'story.txt').write_text(text, encoding='utf-8')
    path, policy = tmp_path / 'answer.json', tmp_path / 'selection.json'
    path.write_text(json.dumps(artifact), encoding='utf-8')
    policy.write_text(json.dumps(selection), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['rag', 'audit-answer', str(path), '--sources', str(tmp_path),
                                    '--selection', str(policy)])
    cli.main()
    assert json.loads(capsys.readouterr().out)['source_exact']
    path.write_text('[]',encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code==2 and 'object' in capsys.readouterr().err
    policy.write_text('broken',encoding='utf-8')
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code==2 and 'Traceback' not in capsys.readouterr().err


def test_answer_output_option_writes_utf8_without_shell_encoding(tmp_path, example, monkeypatch):
    from src import cli
    from src.models import grounded_spans
    text, selection, artifact = example
    policy=tmp_path/'policy.json';policy.write_text(json.dumps(selection),encoding='utf-8')
    output=tmp_path/'saved'/'answer.json'
    monkeypatch.setattr(cli.HybridIndex,'load',lambda *args:object())
    monkeypatch.setattr(grounded_spans,'answer',lambda *args:artifact)
    monkeypatch.setattr(sys,'argv',['rag','answer-v2','Fixture question','story',
                                  '--selection',str(policy),'--output',str(output)])
    cli.main()
    assert json.loads(output.read_bytes().decode('utf-8'))==artifact
    assert not output.read_bytes().startswith(b'\xef\xbb\xbf')
