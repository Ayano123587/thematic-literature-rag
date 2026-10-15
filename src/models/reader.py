"""Pretrained extractive QA; every emitted answer is an exact cited source span."""
from functools import lru_cache
import torch
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
from src.models.loading import validate_loading_info

MODEL='deepset/minilm-uncased-squad2'
REVISION='934656cdda79824eabf503ed56e15c01ddbdbe3f'


@lru_cache(maxsize=1)
def reader():
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    model,loading_info=AutoModelForQuestionAnswering.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,use_safetensors=True,output_loading_info=True)
    validate_loading_info(loading_info)
    model.eval()
    return tokenizer,model


def candidates(question,passages,sliding=False,max_length=384,stride=96,batch_size=8):
    if not passages:return []
    if not isinstance(question,str) or not question.strip():return []
    if not 128<=max_length<=512 or not 0<=stride<max_length//2 or batch_size<1:
        raise ValueError('Invalid reader window or batch configuration')
    tokenizer,model=reader()
    if len(tokenizer.encode(question,add_special_tokens=False))>max_length-32:
        return []
    options={'return_overflowing_tokens':True,'stride':stride} if sliding else {}
    encoded=tokenizer([question]*len(passages),[p.text for p in passages],padding=True,truncation='only_second',max_length=max_length,return_offsets_mapping=True,return_tensors='pt',**options)
    mapping=encoded.pop('overflow_to_sample_mapping').tolist() if sliding else list(range(len(passages)))
    offsets=encoded.pop('offset_mapping').tolist()
    starts_all=[];ends_all=[]
    for lo in range(0,len(mapping),batch_size):
        batch={key:value[lo:lo+batch_size] for key,value in encoded.items()}
        with torch.inference_mode():output=model(**batch)
        starts_all.extend(output.start_logits);ends_all.extend(output.end_logits)
    result=[]
    for row,passage_id in enumerate(mapping):
        p=passages[passage_id]
        context=[i for i,seq in enumerate(encoded.sequence_ids(row)) if seq==1 and offsets[row][i][1]>offsets[row][i][0]]
        if not context:continue
        starts=starts_all[row];ends=ends_all[row]
        if not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
            continue
        top_starts=sorted(context,key=lambda i:-float(starts[i]))[:20]
        top_ends=sorted(context,key=lambda i:-float(ends[i]))[:20]
        valid=[(float(starts[a]+ends[b]),a,b) for a in top_starts for b in top_ends if a<=b<a+30]
        if not valid:continue
        score,a,b=max(valid)
        cls=(encoded['input_ids'][row]==tokenizer.cls_token_id).nonzero()[0].item()
        margin=score-float(starts[cls]+ends[cls])
        start,end=offsets[row][a][0],offsets[row][b][1]
        result.append({'answer':p.text[start:end],'citation':p.id,'book':p.book,'answer_start':p.start+start,'answer_end':p.start+end,'null_margin':margin,'reader_window':row,'retrieval_rank':passage_id+1})
    return sorted(result,key=lambda r:-r['null_margin'])


def answer(index,question,book,policy=None,threshold=0,k=5):
    if not isinstance(question,str) or not question.strip():return {'abstained':True,'reason':'Empty question','answer':None}
    if book not in {p.book for p in index.passages}:return {'abstained':True,'reason':'Requested source is not indexed','answer':None}
    policy=policy or {'mode':'hybrid'}
    hits=index.search(question,k=k,book=book,mode=policy.get('mode','hybrid'),reranker=policy.get('reranker'))
    spans=candidates(question,[p for p,_ in hits])
    if not spans or spans[0]['null_margin']<=threshold:return {'abstained':True,'reason':'Reader prefers null or lacks sufficient span margin','answer':None,'threshold':threshold}
    return dict(spans[0],abstained=False,method='SQuAD2 pretrained extractive reader; exact source span, not generated prose',score_note='Null-logit margin is not a calibrated probability; novel-domain correctness is evaluated separately')
