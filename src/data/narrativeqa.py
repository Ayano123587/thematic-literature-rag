"""Pinned, licensed QA metadata; third-party stories remain local."""
import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from src.config import DATA
from src.data.make_dataset import strip_gutenberg,write_processed

COMMIT='904246f6d1fe99a99a08a03501fe3e619af2cee5'
DOCUMENTS={
 '879287f285743b08d86c3bd0831ec1e49fa92e39':{'book':'dorian','id':174,'split':'valid','title':'The Picture of Dorian Gray','author':'Oscar Wilde','author_death':1900},
 '55d68540c6b86bc90f7a9e630c8b13c0920bf834':{'book':'persuasion','id':105,'split':'test','title':'Persuasion','author':'Jane Austen','author_death':1817},
}


def fetch(acknowledge=False):
    if not acknowledge:raise ValueError('Check local rights before downloading Gutenberg books')
    folder=DATA/'external/narrativeqa';folder.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name in ('documents.csv','qaps.csv','LICENSE','README.md'):
        url=f'https://raw.githubusercontent.com/google-deepmind/narrativeqa/{COMMIT}/{name}'
        raw=urllib.request.urlopen(url,timeout=60).read();(folder/name).write_bytes(raw)
        manifest.append({'file':name,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    docs={r['document_id']:r for r in csv.DictReader((folder/'documents.csv').open(encoding='utf-8'))}
    for docid,meta in DOCUMENTS.items():
        row=docs[docid]
        if row['set']!=meta['split'] or str(meta['id']) not in row['story_url']:raise ValueError('Pinned benchmark source mapping changed')
        url=f"https://www.gutenberg.org/ebooks/{meta['id']}.txt.utf-8"
        raw=urllib.request.urlopen(url,timeout=60).read();body=strip_gutenberg(raw.decode('utf-8-sig'))
        (DATA/'raw').mkdir(parents=True,exist_ok=True);(DATA/'processed').mkdir(parents=True,exist_ok=True)
        (DATA/'raw'/f"{meta['book']}.txt").write_bytes(raw)
        write_processed(DATA/'processed'/f"{meta['book']}.txt",body)
        actual=(DATA/'processed'/f"{meta['book']}.txt").read_text(encoding='utf-8')
        manifest.append({'file':meta['book'],'document_id':docid,'split':meta['split'],'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'processed_sha256':hashlib.sha256(actual.encode()).hexdigest(),'processed_hash_definition':'UTF-8 encoding of actual postwrite Path.read_text text (universal newline normalization)','prewrite_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'bytes':len(raw)})
    rows=[]
    for i,row in enumerate(csv.DictReader((folder/'qaps.csv').open(encoding='utf-8'))):
        if row['document_id'] in DOCUMENTS:
            meta=DOCUMENTS[row['document_id']]
            if row['set']!=meta['split']:raise ValueError('QA/document split mismatch')
            rows.append({'id':f'narrativeqa:{i}','document_id':row['document_id'],'book':meta['book'],'split':row['set'],'question':row['question'],'answers':[row['answer1'],row['answer2']]})
    (folder/'subset.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result={'upstream_commit':COMMIT,'license':'Apache-2.0 for NarrativeQA metadata and QA; Gutenberg texts retain their own license','subset_size':len(rows),'documents':DOCUMENTS,'sources':manifest}
    (folder/'subset-manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def load_subset():
    return json.loads((DATA/'external/narrativeqa/subset.json').read_text(encoding='utf-8'))
