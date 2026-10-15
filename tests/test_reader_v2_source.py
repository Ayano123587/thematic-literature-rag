import hashlib
import json
import pytest
from src.data import reader_v2
from src.data.make_dataset import strip_gutenberg, write_processed


def test_v2_fetch_requires_rights_and_rejects_source_drift(tmp_path, monkeypatch):
    raw = b'*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\r\nMira returned.\r\n*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***'
    canonical = tmp_path / 'canonical.txt'
    write_processed(canonical, strip_gutenberg(raw.decode()))
    selection = tmp_path / 'selection.json'
    expected = {'url': 'https://example.invalid/source',
                'raw_sha256': hashlib.sha256(raw).hexdigest(),
                'normalized_text_sha256': hashlib.sha256(canonical.read_text().encode()).hexdigest()}
    selection.write_text(json.dumps({'source': expected}))
    class Response:
        def read(self):
            return raw
    monkeypatch.setattr(reader_v2.urllib.request, 'urlopen', lambda *a, **k: Response())
    with pytest.raises(ValueError, match='copyright'):
        reader_v2.fetch(False, selection, tmp_path / 'data')
    assert reader_v2.fetch(True, selection, tmp_path / 'data')['book'] == 'journey'
    expected['raw_sha256'] = '0' * 64
    selection.write_text(json.dumps({'source': expected}))
    with pytest.raises(ValueError, match='changed'):
        reader_v2.fetch(True, selection, tmp_path / 'data')
