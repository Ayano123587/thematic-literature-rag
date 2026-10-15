"""BM25 + dense rank fusion; LSA is learned semantics, not a pretrained model."""
import math
import io
import json
import zipfile
import hashlib
import re
from dataclasses import asdict
from collections import Counter
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from src.features.build_features import tokenize
from src.features.build_features import Passage


DENSE_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
RERANKER_REVISION='233902d25c440f23af6f7d6e94d2946bac0bee0a'


@lru_cache(maxsize=2)
def encoder(model_name,revision=None):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(2)
    return SentenceTransformer(model_name,revision=revision,trust_remote_code=False)


@lru_cache(maxsize=2)
def cross_encoder(model_name):
    from sentence_transformers import CrossEncoder
    import torch
    torch.set_num_threads(2)
    revision=RERANKER_REVISION if model_name=='cross-encoder/ms-marco-MiniLM-L-6-v2' else None
    return CrossEncoder(model_name,revision=revision,trust_remote_code=False)


class HybridIndex:
    def __init__(self, passages, backend="lsa", model_name="sentence-transformers/all-MiniLM-L6-v2",model_revision=None):
        if not passages:
            raise ValueError("Cannot index an empty corpus")
        self.passages, self.backend, self.model_name = passages, backend, model_name
        self.model_revision=model_revision or (DENSE_REVISION if model_name=='sentence-transformers/all-MiniLM-L6-v2' else None)
        if backend=='sentence-transformer' and (not self.model_revision or not re.fullmatch(r'[0-9a-f]{40}',self.model_revision)):raise ValueError('Neural embeddings require an exact 40-character model commit revision')
        self.tf = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = np.array([sum(c.values()) for c in self.tf])
        self.avglen = max(1, self.lengths.mean())
        self.df = Counter(t for c in self.tf for t in c)
        if backend == "lsa":
            self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True)
            matrix = self.vectorizer.fit_transform(p.text for p in passages)
            n = min(128, matrix.shape[0]-1, matrix.shape[1]-1)
            if n < 1:
                raise ValueError("Need two passages and distinct vocabulary terms")
            self.svd = TruncatedSVD(n_components=n, random_state=42)
            self.dense = normalize(self.svd.fit_transform(matrix))
        elif backend == "sentence-transformer":
            self.dense = self._encoder().encode([p.text for p in passages], normalize_embeddings=True)
        else:
            raise ValueError("Unknown embedding backend")

    def _encoder(self):
        revision=getattr(self,'model_revision',None)
        if not revision or not re.fullmatch(r'[0-9a-f]{40}',revision):raise ValueError('Neural index lacks an exact model revision; re-encode it before querying')
        return encoder(self.model_name,revision)

    def search(self, query, k=6, book=None, reranker=None,mode='hybrid',min_dense=None):
        if not isinstance(query,str) or not isinstance(k,int):raise ValueError('Query must be text and k must be an integer')
        if mode not in {'hybrid','bm25','dense'}:raise ValueError('Unknown retrieval mode')
        if not tokenize(query) or k < 1:
            return []
        bm = np.zeros(len(self.passages))
        for term in set(tokenize(query)):
            df = self.df[term]
            idf = math.log(1 + (len(bm)-df+0.5)/(df+0.5))
            f = np.array([c[term] for c in self.tf])
            bm += idf*f*2.5/(f+1.5*(0.25+0.75*self.lengths/self.avglen))
        if self.backend == "lsa":
            q = normalize(self.svd.transform(self.vectorizer.transform([query])))
        else:
            q = self._encoder().encode([query], normalize_embeddings=True)
        dense = (self.dense @ q.T).ravel()
        eligible = [i for i,p in enumerate(self.passages) if (book is None or p.book == book) and (bm[i] > 0 if mode=='bm25' else dense[i]>1e-8 if mode=='dense' else bm[i] > 0 or dense[i] > 1e-8)]
        if min_dense is not None:eligible=[i for i in eligible if dense[i]>=min_dense]
        fused = {i: 0.0 for i in eligible}
        rankings=(bm,dense) if mode=='hybrid' else (bm,) if mode=='bm25' else (dense,)
        for scores in rankings:
            for rank,i in enumerate(sorted(eligible,key=lambda i:(-scores[i],i)),1):
                fused[i] += 1/(60+rank)
        candidates = sorted(eligible,key=lambda i:(-fused[i],i))[:max(k,30) if reranker else k]
        if reranker and candidates:
            scores = cross_encoder(reranker).predict([(query,self.passages[i].text) for i in candidates],batch_size=8,show_progress_bar=False)
            candidates = [i for _,i in sorted(zip(scores,candidates),reverse=True)][:k]
        return [(self.passages[i],fused[i]) for i in candidates[:k]]

    def save(self,path):
        if self.backend=='sentence-transformer' and (not getattr(self,'model_revision',None) or not re.fullmatch(r'[0-9a-f]{40}',self.model_revision)):raise ValueError('Cannot persist neural index without exact embedding revision')
        path.parent.mkdir(parents=True,exist_ok=True)
        metadata={'schema':1,'backend':self.backend,'model_name':self.model_name,'model_revision':getattr(self,'model_revision',None),'chunk_contract':getattr(self,'chunk_contract',None),'passages':[asdict(p) for p in self.passages],'source_hashes':getattr(self,'source_hashes',{})}
        arrays={'dense':self.dense}
        if self.backend=='lsa':
            metadata['vocabulary']=self.vectorizer.vocabulary_
            arrays['idf']=self.vectorizer.idf_
            arrays['components']=self.svd.components_
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('metadata.json',json.dumps(metadata));archive.writestr('arrays.npz',buffer.getvalue())

    @staticmethod
    def load(path,allow_missing_revision=False):
        # No pickle or executable deserialization. Legacy .pkl files require rebuilding.
        try:
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist())!={'metadata.json','arrays.npz'} or sum(i.file_size for i in archive.infolist())>512_000_000:raise ValueError('Invalid or oversized index archive')
                meta=json.loads(archive.read('metadata.json'))
                if meta.get('schema')!=1 or meta.get('backend') not in {'lsa','sentence-transformer'}:raise ValueError('Unsupported index schema')
                npz=archive.read('arrays.npz')
                with zipfile.ZipFile(io.BytesIO(npz)) as inner:
                    if sum(i.file_size for i in inner.infolist())>512_000_000:raise ValueError('Oversized embedding arrays')
                    for entry in inner.infolist():
                        with inner.open(entry) as file:
                            version=np.lib.format.read_magic(file)
                            shape,_,dtype=np.lib.format.read_array_header_1_0(file) if version==(1,0) else np.lib.format.read_array_header_2_0(file)
                            if dtype.hasobject or np.prod(shape,dtype=object)*dtype.itemsize>512_000_000:raise ValueError('Unsafe embedding array header')
                arrays=np.load(io.BytesIO(npz),allow_pickle=False)
                expected={'dense','idf','components'} if meta['backend']=='lsa' else {'dense'}
                if set(arrays.files)!=expected:raise ValueError('Unexpected index arrays')
                if any(arrays[name].dtype.kind not in 'fiu' or not np.isfinite(arrays[name]).all() for name in arrays.files):raise ValueError('Non-numeric or non-finite index arrays')
                index=HybridIndex.__new__(HybridIndex)
                index.passages=[Passage(**p) for p in meta['passages']]
                if not index.passages:raise ValueError('Empty index corpus')
                index.backend=meta['backend'];index.model_name=meta['model_name']
                index.model_revision=meta.get('model_revision')
                if index.backend=='sentence-transformer' and (not index.model_revision or not re.fullmatch(r'[0-9a-f]{40}',index.model_revision)) and not allow_missing_revision:raise ValueError('Neural index is missing an exact model revision; re-encode legacy artifacts')
                index.source_hashes=meta.get('source_hashes',{})
                index.chunk_contract=meta.get('chunk_contract')
                index.dense=arrays['dense'].copy()
                if index.dense.ndim!=2 or index.dense.shape[0]!=len(index.passages) or not np.isfinite(index.dense).all():raise ValueError('Invalid embedding matrix')
                if len({p.id for p in index.passages})!=len(index.passages):raise ValueError('Duplicate passage identifiers')
                if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',p.book) or p.start<0 or p.end<=p.start or len(p.text)!=p.end-p.start or not 0<=p.progress<=1 or p.id!=f'{p.book}:{p.start}-{p.end}' for p in index.passages):raise ValueError('Invalid passage offsets')
                index.tf=[Counter(tokenize(p.text)) for p in index.passages]
                index.lengths=np.array([sum(c.values()) for c in index.tf]);index.avglen=max(1,index.lengths.mean());index.df=Counter(t for c in index.tf for t in c)
                if any(book not in {p.book for p in index.passages} for book in index.source_hashes):raise ValueError('Invalid source manifest book identifier')
                if index.backend=='lsa':
                    vocabulary=meta['vocabulary']
                    if not isinstance(vocabulary,dict) or set(vocabulary.values())!=set(range(len(vocabulary))):raise ValueError('Invalid vocabulary indices')
                    if arrays['idf'].shape!=(len(vocabulary),) or arrays['components'].ndim!=2 or arrays['components'].shape!=(index.dense.shape[1],len(vocabulary)):raise ValueError('Embedding/vocabulary shape mismatch')
                    index.vectorizer=TfidfVectorizer(stop_words='english',min_df=1,sublinear_tf=True,vocabulary=meta['vocabulary'])
                    index.vectorizer.idf_=arrays['idf'].copy()
                    index.svd=TruncatedSVD(n_components=arrays['components'].shape[0],random_state=42)
                    index.svd.components_=arrays['components'].copy();index.svd.n_features_in_=len(meta['vocabulary'])
                    if index.svd.components_.shape[1]!=len(meta['vocabulary']):raise ValueError('Vocabulary/component mismatch')
                return index
        except (zipfile.BadZipFile,KeyError,TypeError) as error:raise ValueError('Invalid index; rebuild legacy pickle artifacts instead of loading them') from error

    def verify_sources(self,folder):
        for book,digest in getattr(self,'source_hashes',{}).items():
            path=folder/f'{book}.txt'
            if not path.exists() or hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()!=digest:raise ValueError(f'Source missing or changed for {book}; rebuild index to restore valid citations')
        for book in {p.book for p in self.passages}:
            path=folder/f'{book}.txt'
            if path.exists():
                text=path.read_text(encoding='utf-8')
                if any(text[p.start:p.end]!=p.text for p in self.passages if p.book==book):raise ValueError(f'Citation/source mismatch for {book}')

    def verify_chunk_contract(self, folder, words, overlap):
        from src.features.build_features import chunk_book
        if getattr(self, 'chunk_contract', None) != {'words': words, 'overlap': overlap}:
            raise ValueError('Index chunk contract differs from the frozen selection')
        expected = [p for book in sorted({p.book for p in self.passages})
                    for p in chunk_book(book, (folder / f'{book}.txt').read_text(encoding='utf-8'), words, overlap)]
        if {p.id: p for p in expected} != {p.id: p for p in self.passages}:
            raise ValueError('Index passages do not satisfy the declared chunk contract')
