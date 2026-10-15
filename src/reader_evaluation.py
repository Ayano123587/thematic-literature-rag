"""Actual extractive QA scores with validation-only null-threshold selection."""
import json
import re
import string
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from src.data.narrativeqa import load_subset
from src.models.train_model import HybridIndex
from src.models.reader import candidates,MODEL,REVISION

CONTROLS={
 'valid':["What is Dorian Gray's email address?","Which Nobel Prize did Lord Henry win?","What is Basil's smartphone password?","What degree did Dorian obtain online?"],
 'test':["What is Anne Elliot's email address?","How many satellites did Captain Wentworth launch?","What is Sir Walter's Wi-Fi password?","Which Nobel Prize did Anne Elliot win?"]}


def normalize(text):
    text=re.sub(r'\b(a|an|the)\b',' ',text.lower().translate(str.maketrans('','',string.punctuation)))
    return text.split()


def f1(prediction,answer):
    pred=Counter(normalize(prediction));gold=Counter(normalize(answer))
    overlap=sum((pred&gold).values())
    if not overlap:return 0.0
    p=overlap/sum(pred.values());r=overlap/sum(gold.values())
    return 2*p*r/(p+r)


def predict(index,row,policy):
    hits=index.search(row['question'],k=5,book=row['book'],mode=policy['mode'],reranker=policy.get('reranker'))
    spans=candidates(row['question'],[p for p,_ in hits])
    return {'id':row['id'],'answers':row['answers'],'best':spans[0] if spans else None}


def summarize(rows,controls,threshold):
    scores=[];exact=[];abstained=0
    for row in rows:
        best=row['best'];prediction=best['answer'] if best and best['null_margin']>threshold else ''
        abstained+=not bool(prediction)
        scores.append(max(f1(prediction,a) for a in row['answers']))
        exact.append(any(normalize(prediction)==normalize(a) for a in row['answers']))
    false_answers=sum(bool(r['best'] and r['best']['null_margin']>threshold) for r in controls)
    return {'questions':len(rows),'mean_token_f1':float(np.mean(scores)),'exact_match':float(np.mean(exact)),'abstention_rate':abstained/len(rows),'authored_no_answer_controls':len(controls),'control_false_answer_rate':false_answers/len(controls)}


def run(output=Path('docs/reader-locked-test.json')):
    output=Path(output)
    if output.exists():raise ValueError('Reader locked test exists; preserve it')
    protocol=json.loads(Path('docs/narrativeqa-validation-selection.json').read_text(encoding='utf-8'))
    policy=protocol['selected'];index=HybridIndex.load(Path('models')/f"benchmark-{policy['index']}.lrag")
    index.verify_sources(Path('data/processed'))
    rows=load_subset()
    def split_rows(split):return [predict(index,row,policy) for row in rows if row['split']==split]
    def controls(split):
        book='dorian' if split=='valid' else 'persuasion'
        return [predict(index,{'id':f'authored-{split}-{i}','question':q,'book':book,'answers':['']},policy) for i,q in enumerate(CONTROLS[split])]
    validation=split_rows('valid');validation_controls=controls('valid')
    options=[]
    for threshold in (0,2,4,6,8):
        summary=summarize(validation,validation_controls,threshold)
        options.append({'threshold':threshold,'validation':summary,'objective':summary['mean_token_f1']-.25*summary['control_false_answer_rate']})
    selected=max(options,key=lambda o:(o['objective'],o['threshold']))['threshold']
    selection={'model':MODEL,'revision':REVISION,'model_license':'CC BY 4.0; attribution deepset','retrieval_policy':policy,'threshold_options':options,'selected_threshold':selected,'objective':'Published QA mean token F1 minus .25*false-answer rate on four authored impossible-modern-technology controls; controls are not NarrativeQA labels'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_name('reader-validation-selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    test=split_rows('test');test_controls=controls('test')
    result={'scope':'Pretrained SQuAD2 MiniLM exact-span QA on 29 official NarrativeQA Persuasion questions after Dorian validation; answers remain exact cited substrings','selection':selection,'test':summarize(test,test_controls,selected),'test_cases':[{'id':r['id'],'best':r['best'],'token_f1_at_selected_threshold':max(f1(r['best']['answer'] if r['best'] and r['best']['null_margin']>selected else '',a) for a in r['answers'])} for r in test],'no_answer_controls':[{'id':r['id'],'answered':bool(r['best'] and r['best']['null_margin']>selected)} for r in test_controls],'limitations':'Token F1 is a conventional automatic answer score, not semantic correctness or thematic reasoning. Published answers were written from summaries; a novel passage may not contain a contiguous answer. Small book subset and authored no-answer controls do not establish out-of-domain calibration.'}
    private=Path('data/interim/reader-predictions.json');private.parent.mkdir(parents=True,exist_ok=True)
    private.write_text(json.dumps(result,indent=2),encoding='utf-8')
    result['private_predictions_sha256']=hashlib.sha256(private.read_bytes()).hexdigest()
    for case in result['test_cases']:
        if case['best']:case['best'].pop('answer',None)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
