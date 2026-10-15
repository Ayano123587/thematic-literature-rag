"""Grounded extractive synthesis: sentences remain exact substrings of cited passages."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.features.build_features import tokenize


def extractive_summary(index,query,books=None,k=3,mode='hybrid',reranker=None):
    if not isinstance(query,str):raise ValueError('Question must be text')
    if not isinstance(k,int) or k<1:raise ValueError('k must be a positive integer')
    if books is not None and (not isinstance(books,(list,tuple,set)) or any(not isinstance(b,str) for b in books)):raise ValueError('books must be a collection of source identifiers')
    terms=set(tokenize(query))-ENGLISH_STOP_WORDS
    known={p.book for p in index.passages}
    requested=sorted(known if books is None else set(books))
    missing=sorted(set(requested)-known)
    if missing:return {'abstained':True,'reason':'Requested source is not indexed','missing_sources':missing,'evidence':[]}
    if not terms:return {'abstained':True,'reason':'No informative query terms','evidence':[]}
    evidence=[]
    for book in requested:
        hits=index.search(query,k=max(k,6),book=book,mode=mode,reranker=reranker)
        choices=[]
        for passage,_ in hits:
            for match in re.finditer(r'[^.!?]+[.!?]?',passage.text):
                sentence=match.group().strip()
                matched=terms&set(tokenize(sentence))
                if not matched:continue
                coverage=len(matched)/len(terms)
                if coverage<.2:continue
                local=passage.text.find(sentence,match.start())
                choices.append({'book':book,'citation':passage.id,'sentence':sentence,'sentence_start':passage.start+local,'sentence_end':passage.start+local+len(sentence),'query_term_coverage':coverage,'matched_terms':sorted(matched)})
        choices.sort(key=lambda e:(-e['query_term_coverage'],e['sentence_start']))
        seen=set()
        for choice in choices:
            if choice['sentence'] not in seen:
                evidence.append(choice);seen.add(choice['sentence'])
                if len(seen)>=k:break
    connections=[]
    for i,a in enumerate(evidence):
        for b in evidence[i+1:]:
            shared=set(a['matched_terms'])&set(b['matched_terms'])
            if a['book']!=b['book'] and shared:
                connections.append({'left':a['citation'],'right':b['citation'],'shared_query_terms':sorted(shared),'claim':'Both cited sentences explicitly contain these query terms; interpretation requires reading the surrounding narrative.'})
    return {'query':query,'abstained':not evidence,'reason':None if evidence else 'Insufficient explicit sentence-level support','method':'Extractive evidence synthesis; quoted sentences, no generated literary conclusions','evidence':evidence,'cross_book_links':connections,'limitations':'Lexical coverage is a transparent support rule, not a calibrated relevance probability. Paraphrases can be missed; lexical matches can be irrelevant.'}
