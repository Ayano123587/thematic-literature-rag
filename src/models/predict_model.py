"""Evidence-only synthesis without unsupported claims about literary meaning."""
import re
import numpy as np


def thematic_report(index, query, k=2, reranker=None):
    sections = []
    selected = []
    for book in sorted({p.book for p in index.passages}):
        hits = index.search(query,k=k,book=book,reranker=reranker)
        if hits:
            selected.extend(p for p,_ in hits)
            sections.append({"book":book,"evidence":[{"citation":p.id,"text":p.text,"progress":round(p.progress,4),"rrf_score":score} for p,score in hits]})
    positions = {p.id:i for i,p in enumerate(index.passages)}
    connections = []
    for i,left in enumerate(selected):
        for right in selected[i+1:]:
            if left.book != right.book:
                similarity=float(np.dot(index.dense[positions[left.id]],index.dense[positions[right.id]]))
                connections.append({"left":left.id,"right":right.id,"cosine_similarity":similarity})
    connections.sort(key=lambda c:-c['cosine_similarity'])
    return {"query":query,"method":"Per-book hybrid retrieval; excerpts are evidence, not inferred conclusions.","books":sections,"cross_book_connections":connections,"connection_note":"Embedding similarity proposes passages to compare; it does not prove a shared literary interpretation.","abstained":not sections}


def character_arc(index, character, book, k=2,topic=None,mode='hybrid',reranker=None):
    if not isinstance(character,str) or len(character.strip())<2:raise ValueError('Provide a non-empty character name')
    pattern = re.compile(r"\b"+re.escape(character)+r"\b",re.I)
    if not any(p.book==book and pattern.search(p.text) for p in index.passages):
        return {'character':character,'book':book,'stages':[],'abstained':True,'reason':'Character name is not explicitly present in requested source'}
    hits = index.search(character+' '+(topic or ''),k=len(index.passages),book=book,mode=mode,reranker=reranker)
    stages = []
    for lo,hi,name in [(0,1/3,"early"),(1/3,2/3,"middle"),(2/3,1.01,"late")]:
        selected = [(p,s) for p,s in hits if lo<=p.progress<hi and (topic is not None or pattern.search(p.text))][:k]
        stages.append({"stage":name,"evidence":[{"citation":p.id,"text":p.text,'explicit_name_mention':bool(pattern.search(p.text)),'attribution':'Explicit name occurs in passage; speaker is not resolved' if pattern.search(p.text) else 'Candidate thematic passage; character attribution is unverified'} for p,s in sorted(selected,key=lambda x:x[0].start)]})
    return {"character":character,"book":book,'topic':topic,"stages":stages,'abstained':False,"note":"Chronological candidate evidence, not an inferred psychological arc. Topic mode can retrieve implicit references but marks their attribution unverified."}
