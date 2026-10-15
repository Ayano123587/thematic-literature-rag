"""Real NarrativeQA query evaluation with explicitly weak answer-overlap labels."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.config import DATA
from src.data.narrativeqa import load_subset,COMMIT
from src.features.build_features import chunk_book,tokenize
from src.models.train_model import HybridIndex

RERANKER='cross-encoder/ms-marco-MiniLM-L-6-v2'
POLICIES=[{'name':'bm25','index':'lsa','mode':'bm25'},
 {'name':'lsa_hybrid','index':'lsa','mode':'hybrid'},
 {'name':'minilm_dense','index':'dense','mode':'dense'},
 {'name':'minilm_hybrid','index':'dense','mode':'hybrid'},
 {'name':'minilm_reranked','index':'dense','mode':'hybrid','reranker':RERANKER}]


def build_indexes():
    fixed_books={'pride','frankenstein','dorian','persuasion'}
    texts={p.stem:p.read_text(encoding='utf-8') for p in sorted((DATA/'processed').glob('*.txt')) if p.stem in fixed_books}
    if set(texts)!=fixed_books:raise ValueError('Benchmark requires its four declared source books; run fetch and fetch-qa first')
    passages=[p for book,text in texts.items() for p in chunk_book(book,text)]
    hashes={book:hashlib.sha256(text.encode()).hexdigest() for book,text in texts.items()}
    indexes={}
    for name,backend in [('lsa','lsa'),('dense','sentence-transformer')]:
        index=HybridIndex(passages,backend=backend);index.source_hashes=hashes
        index.save(Path('models')/f'benchmark-{name}.lrag');indexes[name]=index
    return indexes


def content_tokens(text):
    return set(tokenize(text))-ENGLISH_STOP_WORDS


def score_query(index,row,policy,k=5):
    hits=index.search(row['question'],k=k,book=row['book'],mode=policy['mode'],reranker=policy.get('reranker'))
    passages=[p for p,_ in hits]
    def normalized(text):return ' '.join(tokenize(text))
    recalls=[];exact=[]
    for answer in row['answers']:
        terms=content_tokens(answer)
        recalls.append(max((len(terms&content_tokens(p.text))/len(terms) for p in passages),default=0) if terms else 0)
        exact.append(any(normalized(answer) in normalized(p.text) for p in passages) if normalized(answer) else False)
    return {'id':row['id'],'book':row['book'],'gold_content_token_recall_at5':max(recalls),'literal_answer_present_at5':any(exact),'citations':[p.id for p in passages],'source_grounded':all(p.book==row['book'] for p in passages),'retrieved':len(passages)}


def aggregate(rows):
    return {'queries':len(rows),'mean_gold_content_token_recall_at5':float(np.mean([r['gold_content_token_recall_at5'] for r in rows])),'literal_answer_present_rate_at5':float(np.mean([r['literal_answer_present_at5'] for r in rows])),'source_grounding_rate':float(np.mean([r['source_grounded'] for r in rows]))}


def run(output=Path('docs/narrativeqa-locked-test.json')):
    output=Path(output)
    if output.exists():raise ValueError('Locked test exists; preserve it and use a separately named reproduction output')
    rows=load_subset();validation=[r for r in rows if r['split']=='valid'];test=[r for r in rows if r['split']=='test']
    if not validation or not test:raise ValueError('Both official validation and test queries are required')
    if {r['document_id'] for r in validation}&{r['document_id'] for r in test}:raise ValueError('Document split leakage')
    indexes=build_indexes()
    candidates=[]
    for policy in POLICIES:
        result=[score_query(indexes[policy['index']],row,policy) for row in validation]
        candidates.append({'policy':policy,'validation':aggregate(result),'cases':result})
        print(policy['name'],aggregate(result),flush=True)
    selected=max(candidates,key=lambda c:c['validation']['mean_gold_content_token_recall_at5'])['policy']
    protocol={'upstream_commit':COMMIT,'validation_documents':sorted({r['document_id'] for r in validation}),'test_documents':sorted({r['document_id'] for r in test}),'source_hashes':indexes['lsa'].source_hashes,'selection_metric':'Mean per-question maximum gold answer content-token recall in any top-5 passage; weak lexical evidence proxy, not QA accuracy','candidates':candidates,'selected':selected,'limitations':'NarrativeQA questions were written from story summaries, not passage-level evidence. Answers can be paraphrased or absent from the edition. Literal answer presence and token recall do not establish relevance, reasoning, complete answers, or thematic understanding. Two books is a narrow subset.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_name('narrativeqa-validation-selection.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Test questions enter retrieval only after policy selection is serialized.
    result=[score_query(indexes[selected['index']],row,selected) for row in test]
    report={'scope':'Real NarrativeQA official book-disjoint subset: Dorian Gray validation, Persuasion test; no summary is indexed','protocol':protocol,'test':aggregate(result),'test_cases':result}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
