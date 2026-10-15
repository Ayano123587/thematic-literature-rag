"""BM25 + dense rank fusion; LSA is learned semantics, not a pretrained model."""
import math
import pickle
from collections import Counter
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from src.features.build_features import tokenize


@lru_cache(maxsize=2)
def encoder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class HybridIndex:
    def __init__(self, passages, backend="lsa", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if not passages:
            raise ValueError("Cannot index an empty corpus")
        self.passages, self.backend, self.model_name = passages, backend, model_name
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
        return encoder(self.model_name)

    def search(self, query, k=6, book=None, reranker=None):
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
        eligible = [i for i,p in enumerate(self.passages) if (book is None or p.book == book) and (bm[i] > 0 or dense[i] > 1e-8)]
        fused = {i: 0.0 for i in eligible}
        for scores in (bm, dense):
            for rank,i in enumerate(sorted(eligible,key=lambda i:(-scores[i],i)),1):
                fused[i] += 1/(60+rank)
        candidates = sorted(eligible,key=lambda i:(-fused[i],i))[:max(k,30) if reranker else k]
        if reranker and candidates:
            from sentence_transformers import CrossEncoder
            scores = CrossEncoder(reranker).predict([(query,self.passages[i].text) for i in candidates])
            candidates = [i for _,i in sorted(zip(scores,candidates),reverse=True)][:k]
        return [(self.passages[i],fused[i]) for i in candidates[:k]]

    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        # Only load indexes you created: pickle can execute code.
        return pickle.loads(path.read_bytes())
