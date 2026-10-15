import argparse
import json
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
    a=sub.add_parser("arc"); a.add_argument("character"); a.add_argument("book")
    d=sub.add_parser("ingest"); d.add_argument("path"); d.add_argument("book")
    sub.add_parser("evaluate")
    args=parser.parse_args()
    if args.command=="fetch":
        result=fetch(args.acknowledge_local_rights)
    elif args.command=="ingest":
        if not args.book.replace("-", "").isalnum():
            parser.error("book must be an alphanumeric identifier")
        (DATA/"processed").mkdir(parents=True,exist_ok=True)
        target=DATA/"processed"/f"{args.book}.txt"
        target.write_text(ingest_document(args.path),encoding="utf-8")
        result={"processed":str(target),"note":"Citations reference normalized converted text, not PDF page coordinates"}
    elif args.command=="index":
        passages=[p for path in sorted((DATA/"processed").glob("*.txt")) for p in chunk_book(path.stem,path.read_text(encoding="utf-8"))]
        index=HybridIndex(passages,backend=args.backend); index.save(MODEL)
        result={"passages":len(passages),"backend":args.backend}
    elif args.command=="evaluate":
        from src.evaluation import evaluate
        result=evaluate(HybridIndex.load(MODEL))
    elif args.command=="query":
        result=thematic_report(HybridIndex.load(MODEL),args.text,reranker=args.reranker)
    else:
        result=character_arc(HybridIndex.load(MODEL),args.character,args.book)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
