"""Public experiment template; prepared private payload is supplied separately."""
import hashlib
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

START=time.monotonic()
LIMIT_SECONDS=3300


def normalized(text):
    text=re.sub(r'\b(a|an|the)\b',' ',text.lower().translate(str.maketrans('','',string.punctuation)))
    return text.split()


def token_f1(prediction,answer):
    p=Counter(normalized(prediction));a=Counter(normalized(answer));overlap=sum((p&a).values())
    return 2*overlap/(sum(p.values())+sum(a.values())) if overlap else 0.0


PROMPTS={
 'direct':'Use only the supplied passages. Answer the question concisely. If the passages do not support an answer, abstain. Return JSON only with keys answer (string), abstain (boolean), citations (list of passage labels such as P1), supporting_quotes (list of objects with citation and quote). Every nonempty answer must cite at least one provided passage and include an exact supporting quote from that passage. Do not use outside knowledge.',
 'strict':'You are an evidence-bound literary reader. First check whether the retrieved passages explicitly support the requested fact. Similar themes, familiar character names and your prior knowledge are insufficient. If evidence is incomplete, set abstain=true and answer="". Otherwise answer in at most 15 words. Output only valid JSON: {"answer":"...","abstain":false,"citations":["P1"],"supporting_quotes":[{"citation":"P1","quote":"an exact substring of P1"}]}. Use only labels present below. Quotes must support the answer. Never guess.'}


def validate_output(raw,row):
    try:
        left=raw.index('{');right=raw.rindex('}')+1
        obj=json.loads(raw[left:right])
        if not isinstance(obj.get('answer'),str) or not isinstance(obj.get('abstain'),bool):raise ValueError('Malformed answer schema')
        citations=obj.get('citations',[]);quotes=obj.get('supporting_quotes',[])
        if not isinstance(citations,list) or not isinstance(quotes,list):raise ValueError('Citations and supporting_quotes must be lists')
        allowed={p['label']:p['text'] for p in row['passages']}
        abstain=obj['abstain'] or not obj['answer'].strip()
        citation_valid=bool(citations) and all(isinstance(c,str) and c in allowed for c in citations)
        quote_valid=bool(quotes) and all(isinstance(q,dict) and q.get('citation') in allowed and q.get('citation') in citations and isinstance(q.get('quote'),str) and len(q['quote'].strip())>=8 and q['quote'] in allowed[q['citation']] for q in quotes)
        quote_valid=quote_valid and all(c in {q['citation'] for q in quotes} for c in citations)
        # Invalid support is a fail-closed abstention; retain raw protocol-failure flags.
        accepted=not abstain and citation_valid and quote_valid
        prediction=obj['answer'] if accepted else ''
        source_citations=[p['citation'] for p in row['passages'] if p['label'] in citations]
        gold=row.get('answers') or []
        return {'id':row['id'],'answer':prediction,'abstained':not accepted,'raw_answered':not abstain,'citation_valid':citation_valid,'quote_exact_valid':quote_valid,'schema_valid':True,'citations':citations,'source_citations':source_citations,'raw':raw,'token_f1':max((token_f1(prediction,a) for a in gold),default=0.0),'exact_match':any(normalized(prediction)==normalized(a) for a in gold)}
    except (ValueError,TypeError,KeyError):return {'id':row['id'],'answer':'','abstained':True,'raw_answered':False,'citation_valid':False,'quote_exact_valid':False,'schema_valid':False,'citations':[],'raw':raw,'token_f1':0.0,'exact_match':False}


def summary(rows):
    n=len(rows)
    answered=[r for r in rows if r['raw_answered']]
    return {'queries':n,'token_f1':sum(r['token_f1'] for r in rows)/n,'exact_match':sum(r['exact_match'] for r in rows)/n,'abstention_rate':sum(r['abstained'] for r in rows)/n,'schema_valid_rate':sum(r['schema_valid'] for r in rows)/n,'raw_answer_citation_valid_rate':sum(r['citation_valid'] for r in answered)/len(answered) if answered else None,'raw_answer_quote_exact_rate':sum(r['quote_exact_valid'] for r in answered)/len(answered) if answered else None,'raw_answered':len(answered)}


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():raise RuntimeError('This bounded experiment requires the authorized free GPU runtime')
    payload=json.loads(INPUT_JSON if 'INPUT_JSON' in globals() else Path('inputs.json').read_text(encoding='utf-8'))
    protocol=payload['protocol'];model_id=protocol['model'];revision=protocol['model_revision']
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,trust_remote_code=False,torch_dtype=torch.float16,use_safetensors=True).to('cuda').eval()
    root=Path(hf_hub_download(model_id,'config.json',revision=revision)).parent
    weight_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.safetensors')}
    environment={'python':os.sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'weight_hashes':weight_hashes,'model_revision':revision,'input_sha256':hashlib.sha256((INPUT_JSON if 'INPUT_JSON' in globals() else Path('inputs.json').read_text()).encode()).hexdigest(),'limit_seconds':LIMIT_SECONDS,'generation':{'do_sample':False,'max_new_tokens':192,'max_context_tokens':3072,'max_time_per_generation_seconds':75}}
    def evaluate(rows,prompt):
        results=[]
        for row in rows:
            if time.monotonic()-START>LIMIT_SECONDS:raise TimeoutError('Bounded experiment budget reached; no restart')
            context='\n\n'.join(f"[{p['label']}] {p['text']}" for p in row['passages'])
            messages=[{'role':'system','content':PROMPTS[prompt]},{'role':'user','content':f"Question: {row['question']}\n\nPassages:\n{context}"}]
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(text,return_tensors='pt',truncation=True,max_length=3072).to('cuda')
            with torch.inference_mode():generated=model.generate(**encoded,do_sample=False,max_new_tokens=192,max_time=75,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
            result=validate_output(raw,row);results.append(result)
            print(prompt,row['id'],'f1',round(result['token_f1'],3),'abstained',result['abstained'],flush=True)
        return results
    validation={name:evaluate(payload['validation'],name) for name in PROMPTS}
    summaries={name:summary(rows) for name,rows in validation.items()}
    selected=max(summaries,key=lambda name:(summaries[name]['token_f1'],summaries[name]['schema_valid_rate']))
    selection={'environment':environment,'protocol':protocol,'validation':summaries,'selected_prompt':selected,'prompts':PROMPTS,'selected_before_test_generation':True}
    Path('validation-selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    # Fresh test generation is strictly after validation selection has been written.
    test=evaluate(payload['test'],selected)
    # Complete raw model outputs remain private; public results contain IDs and measurements only.
    Path('private-generations.json').write_text(json.dumps({'validation':validation,'test':test},indent=2),encoding='utf-8')
    report={'selection':selection,'fresh_test':summary(test),'test_cases':[{k:v for k,v in r.items() if k not in {'answer','raw'}} for r in test],'limitations':'Valid citation labels and exact quotes prove source linkage only, not answer entailment or thematic reasoning. Fail-closed formatting/support abstentions affect EM/F1. No examples were discarded and no parameters were selected from fresh test outcomes. Model generation is pretrained, not newly finetuned.'}
    Path('generative-locked-test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['fresh_test'],indent=2),flush=True)


if __name__=='__main__':main()
