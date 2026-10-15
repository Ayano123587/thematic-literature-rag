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


def test_pickle_rejected_without_deserialization(tmp_path):
    import pickle
    p=tmp_path/'legacy.pkl';p.write_bytes(pickle.dumps({'not':'an index'}))
    with pytest.raises(ValueError,match='Invalid index'):HybridIndex.load(p)


def test_source_mutation_detected(index,tmp_path):
    import hashlib
    for book in {'a','b'}:
        passages=sorted([p for p in index.passages if p.book==book],key=lambda p:p.start)
        text=[' ']*max(p.end for p in passages)
        for p in passages:text[p.start:p.end]=p.text
        (tmp_path/f'{book}.txt').write_text(''.join(text),encoding='utf-8')
    index.source_hashes={book:hashlib.sha256((tmp_path/f'{book}.txt').read_text(encoding='utf-8').encode()).hexdigest() for book in {'a','b'}}
    index.verify_sources(tmp_path)
    (tmp_path/'a.txt').write_text('changed source',encoding='utf-8')
    with pytest.raises(ValueError,match='Source missing or changed'):index.verify_sources(tmp_path)


def test_invalid_query_and_character(index):
    with pytest.raises(ValueError):index.search(None)
    with pytest.raises(ValueError):index.search('virtue',mode='bogus')
    with pytest.raises(ValueError):character_arc(index,'','a')
    assert character_arc(index,'Hermione','a')['abstained']
    assert index.search('creature',book='unknown')==[]


def test_extractive_synthesis_exact_spans_and_abstention(index):
    from src.models.synthesis import extractive_summary
    report=extractive_summary(index,'rejection prejudice')
    passages={p.id:p for p in index.passages}
    assert report['evidence']
    for e in report['evidence']:
        p=passages[e['citation']]
        assert p.text[e['sentence_start']-p.start:e['sentence_end']-p.start]==e['sentence']
    assert extractive_summary(index,'quantum satellites')['abstained']
    assert extractive_summary(index,'virtue',books=['not-indexed'])['abstained']
    with pytest.raises(ValueError):extractive_summary(index,'virtue',books='a')
    with pytest.raises(ValueError):extractive_summary(index,'virtue',k=0)


def test_real_benchmark_split_declared_disjoint():
    from src.data.narrativeqa import DOCUMENTS
    assert {d['split'] for d in DOCUMENTS.values()}=={'valid','test'}
    assert len({d['book'] for d in DOCUMENTS.values()})==2


def test_generated_answer_requires_valid_citations_and_exact_quotes():
    from src.kaggle_grounded_reader import validate_output
    import json
    row={'id':'fixture','answers':['Mira'],'passages':[{'label':'P1','citation':'fixture:0-24','text':'Mira tended the lantern.'}]}
    valid={'answer':'Mira','abstain':False,'citations':['P1'],'supporting_quotes':[{'citation':'P1','quote':'Mira tended the lantern.'}]}
    scored=validate_output(json.dumps(valid),row)
    assert scored['token_f1']==1 and not scored['abstained']
    assert scored['source_citations']==['fixture:0-24']
    valid['citations']=['P9']
    assert validate_output(json.dumps(valid),row)['abstained']
    valid['citations']=['P1'];valid['supporting_quotes'][0]['quote']='Invented evidence text'
    assert validate_output(json.dumps(valid),row)['abstained']
    valid['citations']={'P1':'not a list'}
    assert not validate_output(json.dumps(valid),row)['schema_valid']
    assert not validate_output('not json',row)['schema_valid']


def test_numeric_archive_rejects_tampered_citation_offsets(index,tmp_path):
    import json,zipfile
    path=tmp_path/'safe.lrag';index.save(path)
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('metadata.json'));arrays=z.read('arrays.npz')
    meta['passages'][0]['start']=-1
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('metadata.json',json.dumps(meta));z.writestr('arrays.npz',arrays)
    with pytest.raises(ValueError,match='Invalid passage offsets'):HybridIndex.load(path)


def test_versioned_processed_writer_and_postwrite_hash(tmp_path):
    import hashlib
    from src.data.make_dataset import write_processed
    p=tmp_path/'processed.txt';write_processed(p,'alpha\r\nbeta\ngamma')
    assert p.read_bytes()==b'alpha\r\r\nbeta\r\ngamma'
    actual=p.read_text(encoding='utf-8')
    assert actual=='alpha\n\nbeta\ngamma'
    assert hashlib.sha256(actual.encode()).hexdigest()!=hashlib.sha256('alpha\r\nbeta\ngamma'.encode()).hexdigest()


def test_neural_revision_is_required_before_model_download(index):
    with pytest.raises(ValueError,match='exact 40-character'):
        HybridIndex(index.passages,backend='sentence-transformer',model_name='custom/model')
    index.backend='sentence-transformer';index.model_revision=None
    with pytest.raises(ValueError,match='exact embedding revision'):index.save(__import__('pathlib').Path('must-not-be-written.lrag'))
    with pytest.raises(ValueError,match='lacks an exact'):index._encoder()


def test_revisionless_numeric_neural_archive_is_forensic_only(index,tmp_path):
    import json,zipfile,io,numpy as np
    path=tmp_path/'legacy-numeric.lrag';index.save(path)
    with zipfile.ZipFile(path) as z:meta=json.loads(z.read('metadata.json'))
    meta['backend']='sentence-transformer';meta.pop('model_revision',None)
    buffer=io.BytesIO();np.savez_compressed(buffer,dense=index.dense)
    with zipfile.ZipFile(path,'w') as z:z.writestr('metadata.json',json.dumps(meta));z.writestr('arrays.npz',buffer.getvalue())
    with pytest.raises(ValueError,match='missing an exact model revision'):HybridIndex.load(path)
    forensic=HybridIndex.load(path,allow_missing_revision=True)
    assert forensic.model_revision is None
    with pytest.raises(ValueError,match='lacks an exact'):forensic._encoder()
