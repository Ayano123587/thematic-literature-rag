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


def character_arc(index, character, book, k=2):
    pattern = re.compile(r"\b"+re.escape(character)+r"\b",re.I)
    hits = index.search(character,k=len(index.passages),book=book)
    stages = []
    for lo,hi,name in [(0,1/3,"early"),(1/3,2/3,"middle"),(2/3,1.01,"late")]:
        selected = [(p,s) for p,s in hits if lo<=p.progress<hi and pattern.search(p.text)][:k]
        stages.append({"stage":name,"evidence":[{"citation":p.id,"text":p.text} for p,s in sorted(selected,key=lambda x:x[0].start)]})
    return {"character":character,"book":book,"stages":stages,"note":"Name mentions by narrative position; not coreference resolution or an inferred psychological arc."}
