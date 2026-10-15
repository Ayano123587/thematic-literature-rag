from collections import Counter


def corpus_summary(passages):
    """Book-level chunk counts for exploratory notebooks."""
    return dict(Counter(p.book for p in passages))
