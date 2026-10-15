import pytest
from src.features.build_features import chunk_book
from src.data.make_dataset import strip_gutenberg, fetch, ingest_document
from src.models.train_model import HybridIndex
from src.models.predict_model import thematic_report, character_arc


@pytest.fixture
def index():
    passages = chunk_book("a", "Elizabeth challenged pride and prejudice. Darcy learned humility and affection. Elizabeth rejected vanity and embraced truth.",words=6,overlap=1)
    passages += chunk_book("b", "Victor created a creature. The creature suffered solitude and rejection. Victor feared responsibility and justice.",words=6,overlap=1)
    return HybridIndex(passages)


def test_offsets_cover_original():
    text="  alpha\n beta gamma delta epsilon zeta  "
    ps=chunk_book("book",text,words=3,overlap=1)
    assert all(text[p.start:p.end]==p.text for p in ps)
    assert ps[-1].text.endswith("zeta")
    assert ps[0].id==f"book:{ps[0].start}-{ps[0].end}"


def test_chunk_validation():
    with pytest.raises(ValueError): chunk_book("b","abc",2,2)
    assert chunk_book("b","")==[]


def test_license_stripping_and_failure():
    t="license\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense"
    assert strip_gutenberg(t)=="body"
    with pytest.raises(ValueError): strip_gutenberg("untrusted format")
    with pytest.raises(ValueError): fetch()


def test_retrieval_filter_and_dense(index):
    assert index.dense.shape[0]==len(index.passages)
    hits=index.search("creature solitude",book="b",k=2)
    assert hits and all(p.book=="b" for p,_ in hits)
    assert "creature" in hits[0][0].text.lower()
    assert index.search("qzxwvzznonexistent")==[]
    assert index.search("",k=3)==[]


def test_roundtrip(index,tmp_path):
    path=tmp_path/"index.pkl";index.save(path)
    assert index.search("humility")==HybridIndex.load(path).search("humility")


def test_crossbook_citations(index):
    report=thematic_report(index,"rejection prejudice")
    assert {b['book'] for b in report['books']}=={"a","b"}
    ids={p.id:p.text for p in index.passages}
    assert all(ids[e['citation']]==e['text'] for b in report['books'] for e in b['evidence'])
    assert report['cross_book_connections']
    assert all(c['left'].split(':')[0]!=c['right'].split(':')[0] for c in report['cross_book_connections'])


def test_character_stages(index):
    arc=character_arc(index,"Elizabeth","a")
    assert [s['stage'] for s in arc['stages']]==['early','middle','late']
    assert all("Elizabeth" in e['text'] for s in arc['stages'] for e in s['evidence'])


def test_plaintext_adapter(tmp_path):
    p=tmp_path/"book.txt";p.write_text("narrative",encoding="utf-8")
    assert ingest_document(p)=="narrative"


def test_evaluation_requires_actual_relevant_passages(index):
    from src.evaluation import evaluate
    with pytest.raises(ValueError,match="anchor absent"):
        evaluate(index)
