"""Local optional grounded generation using the same tested Kaggle experiment contract."""
import json
from src.kaggle_grounded_reader import PROMPTS,validate_output

MODEL='Qwen/Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def generate(index,question,books,prompt='strict',device='auto',reranker=None):
    if not question.strip() or not books:return {'abstained':True,'reason':'Provide a question and explicit source books'}
    known={p.book for p in index.passages}
    if set(books)-known:return {'abstained':True,'reason':'Requested source is not indexed','missing_sources':sorted(set(books)-known)}
    if len(books)>4:raise ValueError('Bounded generator accepts at most four source books')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    target='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,torch_dtype=torch.float16 if target=='cuda' else torch.float32,use_safetensors=True).to(target).eval()
    evidence=[]
    for book in books:
        for passage,_ in index.search(question,book=book,k=max(1,4//len(books)),reranker=reranker):
            evidence.append({'label':f'P{len(evidence)+1}','citation':passage.id,'text':passage.text})
    if not evidence:return {'abstained':True,'reason':'No retrieved evidence'}
    row={'id':'user-query','passages':evidence}
    context='\n\n'.join(f"[{p['label']}] {p['text']}" for p in evidence)
    messages=[{'role':'system','content':PROMPTS[prompt]},{'role':'user','content':f'Question: {question}\n\nPassages:\n{context}'}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    encoded=tokenizer(text,return_tensors='pt',max_length=3072,truncation=True).to(target)
    with torch.inference_mode():out=model.generate(**encoded,do_sample=False,max_new_tokens=192,max_time=75,pad_token_id=tokenizer.eos_token_id)
    raw=tokenizer.decode(out[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
    result=validate_output(raw,row)
    for key in ('raw','token_f1','exact_match'):result.pop(key,None)
    result.update(model=MODEL,revision=REVISION,device=target,retrieval_backend=index.backend,reranker=reranker,warning='Citation validity and exact supporting quotes do not prove entailment. CPU inference needs several GB of RAM; benchmark GPU experiment is evaluated separately.')
    return result
