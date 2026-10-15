"""One bounded CPU pass: validation-only span/retrieval policy, then fresh test."""
import csv
import datetime
import hashlib
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
from src.config import DATA
from src.data.make_dataset import strip_gutenberg,write_processed
from src.data.narrativeqa import COMMIT,load_subset
from src.features.build_features import chunk_book
from src.models.train_model import HybridIndex,DENSE_REVISION,RERANKER_REVISION
from src.models.reader import candidates,MODEL,REVISION
from src.models.grounded_spans import expand_passage,select_span,assemble
from src.reader_evaluation import f1,normalize,CONTROLS

RERANKER='cross-encoder/ms-marco-MiniLM-L-6-v2'
CONFIGS=[
 {'name':'baseline180_top5','words':180,'overlap':40,'k':5,'expand':0,'sliding':False},
 {'name':'fine96_top12','words':96,'overlap':48,'k':12,'expand':0,'sliding':True},
 {'name':'fine96_context_top12','words':96,'overlap':48,'k':12,'expand':700,'sliding':True},
]


def score(rows,controls,threshold,penalty):
    predictions=[select_span(r['spans'],threshold,penalty) for r in rows]
    values=[max(f1(s['answer'] if s else '',a) for a in r['answers']) for s,r in zip(predictions,rows)]
    exact=[any(normalize(s['answer'] if s else '')==normalize(a) for a in r['answers']) for s,r in zip(predictions,rows)]
    false_answers=sum(select_span(r['spans'],threshold,penalty) is not None for r in controls)
    return {'queries':len(rows),'token_f1':float(np.mean(values)),'exact_match':float(np.mean(exact)),'abstention_rate':sum(s is None for s in predictions)/len(rows),'control_false_answer_rate':false_answers/len(controls) if controls else None}


def run():
    output=Path('docs/v2-reader-locked-test.json')
    if output.exists():raise ValueError('V2 locked test already exists; preserve the first outcome')
    protocol=json.loads(Path('docs/v2-prespecified-protocol.json').read_text(encoding='utf-8'))
    declared=datetime.datetime.fromisoformat(protocol['declared_at_utc'])
    continuation_path=Path('docs/v2-continuation.json')
    continuation=json.loads(continuation_path.read_text()) if continuation_path.exists() else None
    deadline=declared.timestamp()+(continuation['total_wall_minutes'] if continuation else protocol['cpu_budget_minutes'])*60
    (DATA/'interim').mkdir(parents=True,exist_ok=True)
    def check():
        if time.time()>deadline:raise TimeoutError('Bounded CPU pass expired; preserve checkpoints and do not restart test')
    check()
    url='https://www.gutenberg.org/ebooks/804.txt.utf-8'
    raw=urllib.request.urlopen(url,timeout=45).read()
    body=strip_gutenberg(raw.decode('utf-8-sig'))
    (DATA/'raw/journey.txt').write_bytes(raw);write_processed(DATA/'processed/journey.txt',body)
    texts={book:(DATA/'processed'/f'{book}.txt').read_text(encoding='utf-8') for book in ['dorian','journey']}
    source={'url':url,'raw_sha256':hashlib.sha256(raw).hexdigest(),'normalized_text_sha256':hashlib.sha256(texts['journey'].encode()).hexdigest(),'qa_source_commit':COMMIT,'qa_csv_sha256':hashlib.sha256((DATA/'external/narrativeqa/qaps.csv').read_bytes()).hexdigest(),'source_representation':'CRLF-v1, exact Path.read_text citation coordinates'}
    validation=[r for r in load_subset() if r['split']=='valid']
    controls=[{'id':f'v2-control-{i}','question':q,'book':'dorian','answers':['']} for i,q in enumerate(CONTROLS['valid'])]
    indexes={}
    for words,overlap in [(180,40),(96,48)]:
        check();index_path=Path('models')/f'v2-{words}.lrag'
        if index_path.exists():
            index=HybridIndex.load(index_path);index.verify_sources(DATA/'processed')
        else:
            passages=[p for book,text in texts.items() for p in chunk_book(book,text,words,overlap)]
            index=HybridIndex(passages,backend='sentence-transformer',model_revision=DENSE_REVISION)
            index.chunk_contract={'words':words,'overlap':overlap}
            index.source_hashes={book:hashlib.sha256(text.encode()).hexdigest() for book,text in texts.items()}
            index.save(index_path)
        indexes[words]=index
        print('indexed',words,len(index.passages),flush=True)
    def infer(row,config):
        check();index=indexes[config['words']]
        cache_path=DATA/'interim'/f"v2-{config['name']}-{row['id'].replace(':','-')}.json"
        if cache_path.exists():
            saved=json.loads(cache_path.read_text(encoding='utf-8'))
            if saved['source']!=source or saved['config']!=config:raise ValueError('Inference cache provenance mismatch')
            from src.features.build_features import Passage
            result=saved['result'];result['passages']=[Passage(**p) for p in result['passages']]
            return result
        hits=index.search(row['question'],book=row['book'],k=config['k'],reranker=RERANKER)
        passages=[expand_passage(p,texts[row['book']],config['expand']) if config['expand'] else p for p,_ in hits]
        spans=candidates(row['question'],passages,sliding=config['sliding'])
        result={'id':row['id'],'book':row['book'],'answers':row['answers'],'spans':spans,'passages':passages}
        serialized={**result,'passages':[vars(p) for p in passages]}
        cache_path.write_text(json.dumps({'source':source,'config':config,'result':serialized}),encoding='utf-8')
        print('inferred',config['name'],row['id'],flush=True)
        return result
    all_candidates=[];cache={}
    for config in CONFIGS:
        rows=[infer(r,config) for r in validation];negative=[infer(r,config) for r in controls]
        cache[config['name']]={'rows':rows,'controls':negative}
        for threshold in (0,2,4,6):
            for penalty in (0,.3,.8,1.5,3,6):
                summary=score(rows,negative,threshold,penalty)
                all_candidates.append({'config':config,'threshold':threshold,'rank_penalty':penalty,'validation':summary,'objective':summary['token_f1']-.25*summary['control_false_answer_rate']})
        checkpoint={'completed_configs':list(cache),'candidates':all_candidates,'source':source}
        Path('data/interim/v2-validation-checkpoint.json').write_text(json.dumps(checkpoint,indent=2),encoding='utf-8')
        print(config['name'],max(c['objective'] for c in all_candidates if c['config']['name']==config['name']),flush=True)
    selected=max(all_candidates,key=lambda r:(r['objective'],r['validation']['exact_match'],-r['config']['k'],r['threshold']))
    selection={'protocol':protocol,'continuation':continuation,'source':source,'models':{'embedding_revision':DENSE_REVISION,'reranker_revision':RERANKER_REVISION,'reader':MODEL,'reader_revision':REVISION},'candidates':all_candidates,'selected':selected,'selection_frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retrieval_corpus':'Dorian plus fresh Journey source text, no QA answers or summaries indexed','validation_reused_from_v1':True}
    Path('docs/v2-validation-selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    # Read and infer fresh-test questions only after selection has been serialized.
    test=[]
    for i,row in enumerate(csv.DictReader((DATA/'external/narrativeqa/qaps.csv').open(encoding='utf-8'))):
        if row['document_id']==protocol['fresh_test_document']:
            if row['set']!='test':raise ValueError('Prespecified source is not official test')
            test.append({'id':f'narrativeqa:{i}','question':row['question'],'answers':[row['answer1'],row['answer2']],'book':'journey'})
    rows=[infer(r,selected['config']) for r in test]
    final=score(rows,[],selected['threshold'],selected['rank_penalty'])
    audits=[]
    for row in rows:
        span=select_span(row['spans'],selected['threshold'],selected['rank_penalty'])
        result=assemble(span,row['passages'],row['book'])
        audits.append({'id':row['id'],'accepted':not result['abstained'],'source_exact':span is None or (not result['abstained'] and texts['journey'][span['answer_start']:span['answer_end']]==span['answer']),'citation':span['citation'] if span else None,'token_f1':max(f1(span['answer'] if span else '',a) for a in row['answers'])})
    private={'validation':{key:{'rows':[{k:v for k,v in row.items() if k!='passages'} for row in value['rows']],'controls':[{k:v for k,v in row.items() if k!='passages'} for row in value['controls']]} for key,value in cache.items()},'test':[{k:v for k,v in row.items() if k!='passages'} for row in rows]}
    private_path=DATA/'interim/v2-private-predictions.json';private_path.write_text(json.dumps(private,indent=2),encoding='utf-8')
    report={'scope':'V2 controlled CPU extraction pass on real NarrativeQA; fresh official Journey test after reused Dorian validation','selection':selection,'fresh_test':final,'citation_audit':{'accepted_answers':sum(r['accepted'] for r in audits),'exact_source_checks':sum(r['source_exact'] for r in audits),'total_questions':len(audits),'semantic_entailment_proven':False},'test_cases':audits,'private_predictions_sha256':hashlib.sha256(private_path.read_bytes()).hexdigest(),'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limitations':'Fresh test novel differs from v1 Persuasion and Call of the Wild, so cross-book scores are not direct head-to-head gains. Validation was reused; no broad calibration or literary interpretation claim. All predictions are exact source spans; summary-derived gold answers may require synthesis.'}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(final),flush=True)
    return report
