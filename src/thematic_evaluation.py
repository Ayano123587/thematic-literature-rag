"""Small original evidence-annotation diagnostic, separate from real benchmark scores."""
import json
from pathlib import Path
from src.models.predict_model import character_arc
from src.models.synthesis import extractive_summary

THEMES=[
 {'id':'theme-judgment','split':'development','query':'How are pride vanity and virtue distinguished from moral corruption?','evidence':{'pride':['Vanity and pride are different'],'frankenstein':['I was benevolent and good']}},
 {'id':'theme-injured-affection','split':'test','query':'How does injured affection turn into resentment or revenge instead of love?','evidence':{'pride':['resentment, once created'],'frankenstein':['if I cannot inspire love']}},
 {'id':'theme-self-recognition','split':'test','query':'How do vanity ignorance and horror expose errors in former self judgment?','evidence':{'pride':['Till this moment, I never knew myself'],'frankenstein':['I had desired it with an ardour']}},
]
ARCS=[{'character':'Elizabeth','book':'pride','topic':'vanity ignorance prejudice self knowledge','anchor':'Till this moment, I never knew myself'},
 {'character':'Victor','book':'frankenstein','topic':'ambition creation ardour horror dream','anchor':'I had desired it with an ardour'}]


def relevant_ids(index,book,anchors):
    found={p.id for p in index.passages if p.book==book and any(a.lower() in ' '.join(p.text.lower().split()) for a in anchors)}
    if not found:raise ValueError(f'Authored evidence anchor absent from {book}')
    return found


def run(index,output=Path('docs/thematic-diagnostic.json')):
    rows=[]
    for case in THEMES:
        books=[]
        for book,anchors in case['evidence'].items():
            gold=relevant_ids(index,book,anchors)
            hits=index.search(case['query'],book=book,k=10)
            retrieved={p.id for p,_ in hits}
            books.append({'book':book,'gold_evidence_ids':sorted(gold),'retrieved_ids':sorted(retrieved),'gold_passage_recall_at10':len(gold&retrieved)/len(gold)})
        summary=extractive_summary(index,case['query'],books=list(case['evidence']))
        positions={p.id:p for p in index.passages}
        grounded=all(e['sentence'] in positions[e['citation']].text for e in summary['evidence'])
        rows.append({'id':case['id'],'split':case['split'],'books':books,'both_books_have_annotated_evidence':all(b['gold_passage_recall_at10']>0 for b in books),'extractive_sentences':len(summary['evidence']),'all_sentence_spans_grounded':grounded,'cross_book_lexical_links':len(summary['cross_book_links'])})
    arcs=[]
    for case in ARCS:
        gold=relevant_ids(index,case['book'],[case['anchor']])
        arc=character_arc(index,case['character'],case['book'],topic=case['topic'],k=3)
        retrieved={e['citation'] for stage in arc['stages'] for e in stage['evidence']}
        arcs.append({'character':case['character'],'book':case['book'],'gold_ids':sorted(gold),'retrieved_ids':sorted(retrieved),'gold_passage_recall':len(gold&retrieved)/len(gold),'note':'Gold anchor denotes a relevant narrative turning point; no independent speaker/coreference annotation is available'})
    report={'scope':'Three original authored cross-book evidence queries and two character-turning-point diagnostics; small, non-independent annotation study, not a public benchmark or interpretation accuracy score','annotation':'One anchor per book/question chosen by the implementing agent after reading source excerpts, without independent adjudication. Development/test labels prevent query reuse but do not establish external generalization. Policies are not tuned on these test queries. Gold passages can overlap.','themes':rows,'character_arcs':arcs}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
