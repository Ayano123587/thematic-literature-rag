"""Replay the v2 public-domain source without expanding the older QA subsets."""
import hashlib
import json
import urllib.request
from pathlib import Path
from src.config import DATA
from src.data.make_dataset import strip_gutenberg, write_processed


def fetch(acknowledge=False, selection=Path('docs/v2-validation-selection.json'), root=DATA):
    if not acknowledge:
        raise ValueError('Check local copyright law and acknowledge local rights')
    expected = json.loads(Path(selection).read_text(encoding='utf-8'))['source']
    raw = urllib.request.urlopen(expected['url'], timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != expected['raw_sha256']:
        raise ValueError('Gutenberg source changed from the evaluated version')
    for folder in ('raw', 'processed'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / 'raw' / 'journey.txt').write_bytes(raw)
    target = root / 'processed' / 'journey.txt'
    write_processed(target, strip_gutenberg(raw.decode('utf-8-sig')))
    actual = hashlib.sha256(target.read_text(encoding='utf-8').encode()).hexdigest()
    if actual != expected['normalized_text_sha256']:
        raise ValueError('Processed source representation differs from evaluated version')
    return {'book': 'journey', 'normalized_text_sha256': actual}
