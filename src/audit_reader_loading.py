"""Inspect pinned QA-head metadata and historical loading logs without inference."""
import hashlib
import json
from pathlib import Path
import urllib.request


def audit(folder=Path('data/interim/v3-kaggle-output')):
    completed = json.loads((folder / 'reader-comparison.json').read_text())
    protocol = completed['protocol']
    records = {}
    for specification in protocol['models']:
        name, revision = specification['name'], specification['revision']
        base = f'https://huggingface.co/{name}/resolve/{revision}'
        request = urllib.request.Request(base + '/model.safetensors',
                                         headers={'Range': 'bytes=0-262143'})
        with urllib.request.urlopen(request, timeout=45) as stream:
            count = int.from_bytes(stream.read(8), 'little')
            if not 0 < count < 262136:
                raise ValueError('Unexpected safetensors header size')
            raw = stream.read(count)
        header = json.loads(raw)
        config_raw = urllib.request.urlopen(base + '/config.json', timeout=30).read()
        config = json.loads(config_raw)
        metadata = json.load(urllib.request.urlopen(
            f'https://huggingface.co/api/models/{name}/revision/{revision}?blobs=true', timeout=30))
        file = next(row for row in metadata['siblings'] if row['rfilename'] == 'model.safetensors')
        official_sha = file['lfs']['sha256']
        if official_sha != completed['weight_sha256'][name]:
            raise ValueError('GPU weight hash differs from official pinned LFS metadata')
        qa = {key: value for key, value in header.items() if key.startswith('qa_outputs.')}
        if set(qa) != {'qa_outputs.weight', 'qa_outputs.bias'}:
            raise ValueError('Checkpoint QA head is missing')
        if qa['qa_outputs.weight']['shape'] != [2, config['hidden_size']] or qa['qa_outputs.bias']['shape'] != [2]:
            raise ValueError('Checkpoint QA head shape does not match architecture')
        records[name] = {'revision': revision, 'config_architectures': config['architectures'],
                         'model_type': config['model_type'],
                         'config_sha256': hashlib.sha256(config_raw).hexdigest(),
                         'qa_head_tensors': qa,
                         'official_lfs_weight_sha256': official_sha,
                         'gpu_weight_sha256_matches_official_lfs': True,
                         'safetensors_header_sha256': hashlib.sha256(raw).hexdigest()}
    log_path = folder / 'portfolio-literary-reader-v3.log'
    events = json.loads(log_path.read_text(encoding='utf-8'))
    text = ''.join(event.get('data', '') for event in events)
    phrases = ['newly initialized', 'not initialized from', 'missing keys',
               'mismatched keys', 'qa_outputs.weight were not']
    flags = {phrase: phrase in text.lower() for phrase in phrases}
    if any(flags.values()):
        raise ValueError('Unexpected uninitialized or missing model weights in GPU log')
    result = {'scope': 'Post-run headers/config/logs; no inference or full candidate-weight download',
              'models': records, 'loading_info_requested_by_version1': False,
              'loading_warning_flags': flags,
              'expected_unused_pooler_warning': 'bert.pooler.dense.bias' in text,
              'expected_unused_pooler_note': 'Unused BERT pooler tensors are expected for the MiniLM QA architecture. No uninitialized QA-head warning appears.',
              'log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest(),
              'runtime_limit': 'Version 1 did not serialize output_loading_info or runtime class strings. Pinned architecture metadata, AutoModel QA mapping, exact weight hashes and logs support pretrained QA-head loading.'}
    Path('docs/v3-loading-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'qa_heads_present': True, 'warning_flags': flags,
                      'architectures': {name: row['config_architectures'] for name, row in records.items()}}))
    return result
