"""Deterministic source attribution outside learned answer generation."""
from src.features.build_features import Passage


def expand_passage(passage,text,characters=700):
    if text[passage.start:passage.end]!=passage.text:
        raise ValueError('Cannot expand a passage that mismatches its source')
    start=max(0,passage.start-characters)
    end=min(len(text),passage.end+characters)
    # Avoid cutting words while retaining exact source coordinates.
    while start>0 and not text[start-1].isspace():start-=1
    while end<len(text) and not text[end].isspace():end+=1
    return Passage(f'{passage.book}:{start}-{end}',passage.book,start,end,text[start:end],passage.progress)


def select_span(spans,threshold=0,rank_penalty=0):
    eligible=[s for s in spans if s['null_margin']>threshold]
    if not eligible:return None
    return max(eligible,key=lambda s:s['null_margin']-rank_penalty*(s.get('retrieval_rank',1)-1))


def assemble(span,passages,book):
    if span is None:return {'abstained':True,'answer':None,'reason':'No span exceeded the frozen null threshold'}
    known={p.id:p for p in passages}
    passage=known.get(span['citation'])
    if passage is None or passage.book!=book:
        return {'abstained':True,'answer':None,'reason':'Span source is absent or belongs to another book'}
    start=span['answer_start']-passage.start;end=span['answer_end']-passage.start
    if not 0<=start<end<=len(passage.text) or passage.text[start:end]!=span['answer']:
        return {'abstained':True,'answer':None,'reason':'Answer does not exactly match cited source coordinates'}
    return {**span,'abstained':False,'evidence':{'citation':passage.id,'answer_start':span['answer_start'],'answer_end':span['answer_end']},'citation_method':'Deterministically assembled from exact source offsets; no model-generated quote JSON','semantic_correctness_verified':False}


def answer(index, question, book, source_directory, selection):
    """Apply a frozen validation policy to a verified persisted index."""
    from pathlib import Path
    from src.models.reader import candidates
    if not isinstance(question, str) or not question.strip():
        return {'abstained': True, 'answer': None, 'reason': 'Empty question'}
    if len(question) > 1000:
        return {'abstained': True, 'answer': None, 'reason': 'Question exceeds the supported 1000-character limit'}
    if book not in {p.book for p in index.passages}:
        return {'abstained': True, 'answer': None, 'reason': 'Requested source is not indexed'}
    index.verify_sources(Path(source_directory))
    policy = selection['selected']
    config = policy['config']
    hits = index.search(question, book=book, k=config['k'],
                        reranker='cross-encoder/ms-marco-MiniLM-L-6-v2')
    text = (Path(source_directory) / f'{book}.txt').read_text(encoding='utf-8')
    passages = [expand_passage(p, text, config['expand']) if config['expand'] else p
                for p, _ in hits]
    spans = candidates(question, passages, sliding=config['sliding'])
    return assemble(select_span(spans, policy['threshold'], policy['rank_penalty']),
                    passages, book)
