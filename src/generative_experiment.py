"""Prepare a private bounded generative experiment, without committing book/QA text."""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from src.config import DATA
from src.features.build_features import chunk_book
from src.models.train_model import HybridIndex
from src.data.narrativeqa import load_subset,COMMIT
from src.data.make_dataset import strip_gutenberg,write_processed

FRESH_DOCUMENT='1fa32a0798f79e13ca92ae84eda9ab3eb12c70a4'


def fetch_source(acknowledge=False):
    if not acknowledge:raise ValueError('Check local Gutenberg rights and acknowledge them explicitly')
    protocol=json.loads(Path('docs/generative-prespecified-protocol.json').read_text(encoding='utf-8'))
    target=DATA/'raw/callwild.txt';target.parent.mkdir(parents=True,exist_ok=True)
    raw=target.read_bytes() if target.exists() else urllib.request.urlopen(protocol['source']['url'],timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=protocol['source']['raw_sha256']:raise ValueError('Fresh-source raw snapshot changed; preserve the declared test protocol and investigate')
    if not target.exists():target.write_bytes(raw)
    processed=DATA/'processed/callwild.txt';processed.parent.mkdir(parents=True,exist_ok=True)
    if not processed.exists():write_processed(processed,strip_gutenberg(raw.decode('utf-8-sig')))
    if hashlib.sha256(processed.read_text(encoding='utf-8').encode()).hexdigest()!=protocol['source']['normalized_text_sha256']:raise ValueError('Fresh-source normalized snapshot mismatch')
    return {'book':'callwild','source_verified':True,'normalized_text_sha256':protocol['source']['normalized_text_sha256']}


def prepare(folder=DATA/'interim/kaggle-generative'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'inputs.json').exists():raise ValueError('Private experiment payload already exists; preserve it and choose a separate reproduction folder')
    protocol=json.loads(Path('docs/generative-prespecified-protocol.json').read_text(encoding='utf-8'))
    existing=HybridIndex.load(Path('models/benchmark-dense.lrag'))
    existing.verify_sources(DATA/'processed')
    text=(DATA/'processed/callwild.txt').read_text(encoding='utf-8')
    # Preserve existing corpus embeddings; only the new book requires encoding.
    new=HybridIndex(chunk_book('callwild',text),backend='sentence-transformer')
    passages=existing.passages+new.passages
    import numpy as np
    # Rebuild lexical corpus statistics from all five books, with frozen neural vectors.
    from collections import Counter
    existing.passages=passages;existing.dense=np.concatenate([existing.dense,new.dense])
    existing.tf+=new.tf;existing.lengths=np.concatenate([existing.lengths,new.lengths]);existing.avglen=max(1,existing.lengths.mean());existing.df=Counter(t for c in existing.tf for t in c)
    validation=[r for r in load_subset() if r['split']=='valid']
    test=[]
    for i,row in enumerate(csv.DictReader((DATA/'external/narrativeqa/qaps.csv').open(encoding='utf-8'))):
        if row['document_id']==FRESH_DOCUMENT:
            if row['set']!='test':raise ValueError('Fresh book is not in official test split')
            test.append({'id':f'narrativeqa:{i}','question':row['question'],'answers':[row['answer1'],row['answer2']],'book':'callwild','document_id':FRESH_DOCUMENT})
    if not test:raise ValueError('Fresh benchmark queries missing')
    def pack(row):
        hits=existing.search(row['question'],book=row['book'],k=4,reranker='cross-encoder/ms-marco-MiniLM-L-6-v2')
        return dict(row,passages=[{'label':f'P{i+1}','citation':p.id,'text':p.text} for i,(p,_) in enumerate(hits)])
    payload={'protocol':dict(protocol,upstream_commit=COMMIT,model='Qwen/Qwen2.5-1.5B-Instruct',model_revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',model_license='Apache-2.0',validation_queries=len(validation),fresh_test_queries=len(test),retrieval='Frozen MiniLM BM25 hybrid + MS-MARCO reranking; four passages per question; no gold answer injected into passages'),'validation':[pack(r) for r in validation],'test':[pack(r) for r in test]}
    encoded=json.dumps(payload,ensure_ascii=False)
    (folder/'inputs.json').write_text(encoded,encoding='utf-8')
    payload_hash=hashlib.sha256(encoded.encode()).hexdigest()
    provenance={'input_sha256':payload_hash,'protocol':payload['protocol'],'validation_question_ids':[r['id'] for r in validation],'test_question_ids':[r['id'] for r in test],'payload_contains_private_qa_and_book_excerpts':True}
    (folder/'input-manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    published_manifest=Path('docs/generative-input-manifest.json')
    if not published_manifest.exists():published_manifest.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    template=Path('src/kaggle_grounded_reader.py').read_text(encoding='utf-8')
    (folder/'run.py').write_text('INPUT_JSON = '+repr(encoded)+'\n'+template,encoding='utf-8')
    metadata={'id':'aikyatansinha/portfolio-literary-grounded-qwen','title':'Portfolio Literary Grounded Qwen','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return provenance
