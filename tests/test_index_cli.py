import sys
from src import cli
from src.models.train_model import HybridIndex


def test_index_cli_selected_books_and_chunk_policy(tmp_path, monkeypatch, capsys):
    processed = tmp_path / 'data' / 'processed'
    processed.mkdir(parents=True)
    text = ' '.join(f'lantern harbor keeper boat dawn passage{i}' for i in range(25))
    (processed / 'story.txt').write_text(text, encoding='utf-8')
    (processed / 'excluded.txt').write_text('Unrelated source.', encoding='utf-8')
    output = tmp_path / 'selected.lrag'
    monkeypatch.setattr(cli, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(sys, 'argv', ['rag', 'index', '--words', '20', '--overlap', '5',
                                    '--books', 'story', '--output', str(output)])
    cli.main()
    index = HybridIndex.load(output)
    assert {p.book for p in index.passages} == {'story'}
    assert max(len(p.text.split()) for p in index.passages) == 20
    assert index.passages[1].start < index.passages[0].end
    index.verify_sources(processed)
    assert 'passages' in capsys.readouterr().out
