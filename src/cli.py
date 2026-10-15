import argparse
import json
import hashlib
from pathlib import Path
from src.config import DATA, MODEL
from src.data.make_dataset import fetch, ingest_document
from src.features.build_features import chunk_book
from src.models.train_model import HybridIndex
from src.models.predict_model import thematic_report, character_arc


def main():
    parser=argparse.ArgumentParser(description="Traceable literary retrieval")
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fetch"); f.add_argument("--acknowledge-local-rights",action="store_true")
    t=sub.add_parser("index"); t.add_argument("--backend",choices=["lsa","sentence-transformer"],default="lsa")
    q=sub.add_parser("query"); q.add_argument("text"); q.add_argument("--reranker")
    a=sub.add_parser("arc"); a.add_argument("character"); a.add_argument("book");a.add_argument('--topic')
    n=sub.add_parser('fetch-qa');n.add_argument('--acknowledge-local-rights',action='store_true')
    b=sub.add_parser('benchmark');b.add_argument('--output',default='docs/narrativeqa-locked-test.json')
    er=sub.add_parser('evaluate-reader');er.add_argument('--output',default='docs/reader-locked-test.json')
    sub.add_parser('evaluate-themes')
    s=sub.add_parser('synthesize');s.add_argument('text');s.add_argument('--books',nargs='+')
    answer=sub.add_parser('answer');answer.add_argument('text');answer.add_argument('book');answer.add_argument('--threshold',type=float,default=None)
    improved=sub.add_parser('answer-v2');improved.add_argument('text');improved.add_argument('book');improved.add_argument('--selection',default='docs/v2-validation-selection.json');improved.add_argument('--index')
    g=sub.add_parser('generate');g.add_argument('text');g.add_argument('--books',nargs='+',required=True);g.add_argument('--device',default='auto',choices=['auto','cpu','cuda']);g.add_argument('--prompt',choices=['direct','strict']);g.add_argument('--reranker')
    pg=sub.add_parser('prepare-generation');pg.add_argument('--acknowledge-local-rights',action='store_true');pg.add_argument('--folder',default='data/interim/kaggle-generative')
    d=sub.add_parser("ingest"); d.add_argument("path"); d.add_argument("book")
    sub.add_parser("evaluate")
    args=parser.parse_args()
    if args.command=="fetch":
        result=fetch(args.acknowledge_local_rights)
    elif args.command=='fetch-qa':
        from src.data.narrativeqa import fetch as fetch_qa
        result=fetch_qa(args.acknowledge_local_rights)
    elif args.command=='benchmark':
        from src.benchmark import run
        result=run(Path(args.output))
    elif args.command=='evaluate-reader':
        from src.reader_evaluation import run
        result=run(Path(args.output))
    elif args.command=='prepare-generation':
        from src.generative_experiment import fetch_source,prepare
        fetch_source(args.acknowledge_local_rights);result=prepare(Path(args.folder))
    elif args.command=="ingest":
        if not args.book.replace("-", "").isalnum():
            parser.error("book must be an alphanumeric identifier")
        (DATA/"processed").mkdir(parents=True,exist_ok=True)
        target=DATA/"processed"/f"{args.book}.txt"
        target.write_text(ingest_document(args.path),encoding="utf-8")
        result={"processed":str(target),"note":"Citations reference normalized converted text, not PDF page coordinates"}
    elif args.command=='answer-v2':
        from src.models.grounded_spans import answer as grounded_answer
        selection=json.loads(Path(args.selection).read_text(encoding='utf-8'))
        path=Path(args.index) if args.index else Path('models')/f"v2-{selection['selected']['config']['words']}.lrag"
        index=HybridIndex.load(path)
        result=grounded_answer(index,args.text,args.book,DATA/'processed',selection)
    elif args.command=="index":
        empty=[p.name for p in (DATA/'processed').glob('*.txt') if not p.read_text(encoding='utf-8').strip()]
        if empty:parser.error('Empty source documents cannot be indexed: '+', '.join(empty))
        passages=[p for path in sorted((DATA/"processed").glob("*.txt")) for p in chunk_book(path.stem,path.read_text(encoding="utf-8"))]
        index=HybridIndex(passages,backend=args.backend)
        index.source_hashes={path.stem:hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() for path in sorted((DATA/'processed').glob('*.txt'))}
        index.save(MODEL)
        result={"passages":len(passages),"backend":args.backend}
    else:
        index=HybridIndex.load(MODEL);index.verify_sources(DATA/'processed')
        if args.command=='generate':
            from src.models.generative import generate
            chosen=args.prompt
            completed=Path('docs/generative-locked-test.json')
            if chosen is None:chosen=json.loads(completed.read_text())['selection']['selected_prompt'] if completed.exists() else 'strict'
            result=generate(index,args.text,args.books,device=args.device,prompt=chosen,reranker=args.reranker)
        elif args.command=='synthesize':
            from src.models.synthesis import extractive_summary
            result=extractive_summary(index,args.text,books=args.books)
        elif args.command=='answer':
            from src.models.reader import answer
            selection_path=Path('docs/reader-validation-selection.json')
            threshold=args.threshold
            policy=None
            if selection_path.exists():
                selection=json.loads(selection_path.read_text());policy=selection['retrieval_policy']
                index=HybridIndex.load(Path('models')/f"benchmark-{policy['index']}.lrag");index.verify_sources(DATA/'processed')
                if threshold is None:threshold=selection['selected_threshold']
            elif threshold is None:threshold=0
            result=answer(index,args.text,args.book,threshold=threshold,policy=policy)
        elif args.command=='evaluate-themes':
            from src.thematic_evaluation import run
            result=run(index)
        elif args.command=="evaluate":
            from src.evaluation import evaluate
            result=evaluate(index)
        elif args.command=="query":
            result=thematic_report(index,args.text,reranker=args.reranker)
        else:
            result=character_arc(index,args.character,args.book,topic=args.topic)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
