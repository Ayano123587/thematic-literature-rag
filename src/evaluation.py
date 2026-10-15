"""Known-passage diagnostic, not an independently annotated thematic benchmark."""
CASES = [
    ("pride", "single man possession good fortune want wife", "a single man in possession of a good fortune"),
    ("pride", "Elizabeth vanity pride prejudices Darcy", "vanity"),
    ("frankenstein", "creature Adam fallen angel", "thy Adam"),
    ("frankenstein", "life death ideal bounds break through", "ideal bounds"),
]


def evaluate(index,k=5):
    rows=[]
    for book,query,anchor in CASES:
        relevant={p.id for p in index.passages if p.book==book and anchor.lower() in " ".join(p.text.lower().split())}
        if not relevant:
            raise ValueError(f"Evaluation anchor absent in {book}: {anchor}")
        hits=index.search(query,k=k,book=book)
        ranks=[rank for rank,(p,_) in enumerate(hits,1) if p.id in relevant]
        rows.append({"book":book,"query":query,"relevant_passages":len(relevant),"hit":bool(ranks),"reciprocal_rank":1/min(ranks) if ranks else 0})
    return {"scope":"Four manually specified known-passage smoke queries; not thematic reasoning quality", "backend":index.backend,"k":k,"hit_rate":sum(r['hit'] for r in rows)/len(rows),"mrr":sum(r['reciprocal_rank'] for r in rows)/len(rows),"cases":rows}
