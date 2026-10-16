"""Persist and recheck answer/source consistency; fingerprints are not signatures."""
import hashlib
import json
import math
from pathlib import Path
import re
from src.features.build_features import chunk_book
from src.models.grounded_spans import expand_passage


def finite_number(value):
    if type(value) not in (int,float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def digest_policy(policy):
    raw = json.dumps(policy, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def policy_from_selection(selection):
    if (not isinstance(selection,dict) or not isinstance(selection.get('selected'),dict)
            or not isinstance(selection.get('models'),dict)):
        raise ValueError('Frozen selection must contain models and selected policy objects')
    selected = selection['selected']
    if not {'config','threshold','rank_penalty'} <= set(selected):
        raise ValueError('Frozen selection is missing required policy fields')
    policy={'models': selection['models'], 'config': selected['config'],
            'threshold': selected['threshold'], 'rank_penalty': selected['rank_penalty']}
    validate_policy(policy)
    return policy


def validate_policy(policy):
    if not isinstance(policy,dict) or not {'models','config','threshold','rank_penalty'} <= set(policy):
        raise ValueError('Answer policy must contain required policy fields')
    if (not finite_number(policy['threshold']) or not finite_number(policy['rank_penalty'])
            or policy['rank_penalty'] < 0):
        raise ValueError('Invalid finite selection parameters')
    config = policy['config']
    if not isinstance(config,dict) or not {'words','overlap','k','expand','sliding'} <= set(config):
        raise ValueError('Answer policy has incomplete chunk configuration')
    if (type(config['words']) is not int or type(config['overlap']) is not int
            or not 0 <= config['overlap'] < config['words']
            or type(config['k']) is not int or config['k'] < 1
            or type(config['expand']) is not int or config['expand'] < 0
            or type(config['sliding']) is not bool):
        raise ValueError('Invalid persisted chunk or reader configuration')
    models = policy['models']
    if (not isinstance(models,dict) or not isinstance(models.get('reader'),str) or not models['reader']
            or any(not isinstance(models.get(key),str) or not re.fullmatch(r'[0-9a-f]{40}',models[key])
                   for key in ('embedding_revision','reader_revision','reranker_revision'))):
        raise ValueError('Missing exact persisted model revisions')


def verify_artifact(artifact, text, expected_selection=None):
    if not isinstance(artifact,dict):
        raise ValueError('Answer artifact must be an object')
    if not isinstance(text,str):
        raise ValueError('Source must be Unicode text')
    if not {'policy','policy_sha256','source_sha256','book','abstained'} <= set(artifact):
        raise ValueError('Answer artifact is missing required fields')
    if type(artifact.get('artifact_schema')) is not int or artifact['artifact_schema'] != 1:
        raise ValueError('Unsupported answer artifact schema')
    book = artifact.get('book')
    if not isinstance(book, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', book):
        raise ValueError('Invalid source book identifier')
    if artifact.get('source_sha256') != hashlib.sha256(text.encode('utf-8')).hexdigest():
        raise ValueError('Answer source fingerprint changed')
    policy = artifact['policy']
    validate_policy(policy)
    if artifact.get('policy_sha256') != digest_policy(policy):
        raise ValueError('Answer policy fingerprint changed')
    if expected_selection is not None and policy != policy_from_selection(expected_selection):
        raise ValueError('Artifact differs from the expected frozen selection')
    config = policy['config']
    if type(artifact.get('abstained')) is not bool:
        raise ValueError('Abstention status must be explicit')
    if artifact['abstained']:
        if artifact.get('answer') is not None or any(key in artifact for key in ('citation', 'answer_start', 'answer_end')):
            raise ValueError('Abstention cannot carry an accepted answer')
        return {'valid': True, 'abstained': True, 'source_exact': None,
                'frozen_policy_matches': expected_selection is not None,
                'semantic_correctness_verified': False, 'authenticity_verified': False}
    answer, start, end = artifact.get('answer'), artifact.get('answer_start'), artifact.get('answer_end')
    citation = artifact.get('citation')
    match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9_-]*):(\d+)-(\d+)', citation) if isinstance(citation, str) else None
    if (not match or match[1] != book or type(start) is not int or type(end) is not int
            or not 0 <= int(match[2]) <= start < end <= int(match[3]) <= len(text)
            or not isinstance(answer, str) or text[start:end] != answer):
        raise ValueError('Answer does not match cited source coordinates')
    if (not finite_number(artifact.get('null_margin')) or artifact['null_margin'] <= policy['threshold']
            or type(artifact.get('retrieval_rank')) is not int
            or not 1 <= artifact['retrieval_rank'] <= config['k']):
        raise ValueError('Answer lacks finite eligible reader provenance')
    expected_evidence = {'citation': citation, 'answer_start': start, 'answer_end': end}
    if artifact.get('evidence') != expected_evidence:
        raise ValueError('Evidence metadata disagrees with answer coordinates')
    passages = chunk_book(book, text, config['words'], config['overlap'])
    known = {expand_passage(p, text, config['expand']).id if config['expand'] else p.id for p in passages}
    if citation not in known:
        raise ValueError('Citation does not match declared source chunk geometry')
    return {'valid': True, 'abstained': False, 'source_exact': True,
            'frozen_policy_matches': expected_selection is not None,
            'selection_decision_recomputed': False, 'semantic_correctness_verified': False,
            'authenticity_verified': False}


def make_artifact(result, text, book, selection):
    if not isinstance(result,dict):
        raise ValueError('Answer result must be an object')
    policy = policy_from_selection(selection)
    artifact = {**result, 'artifact_schema': 1, 'book': book,
                'source_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'policy': policy, 'policy_sha256': digest_policy(policy)}
    verify_artifact(artifact, text, selection)
    return artifact


def audit_answer(path, source_directory, expected_selection=None):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate answer artifact key')
            result[key] = value
        return result
    with Path(path).open('rb') as source:
        raw = source.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError('Answer artifact is unexpectedly large')
    artifact = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_pairs)
    if not isinstance(artifact,dict):
        raise ValueError('Answer artifact must be an object')
    book = artifact.get('book')
    if not isinstance(book, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', book):
        raise ValueError('Invalid source book identifier')
    text = (Path(source_directory) / f'{book}.txt').read_text(encoding='utf-8')
    return verify_artifact(artifact, text, expected_selection)
