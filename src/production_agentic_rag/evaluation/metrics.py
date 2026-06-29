"""Retrieval and generation evaluation metrics."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

from ..text import tokenize


@dataclass
class RetrievalMetrics:
    hit_rate: float
    mrr: float
    recall_at_k: float
    ndcg_at_k: float


def evaluate_retrieval(rankings: Sequence[Sequence[str]],
                       relevant: Sequence[set[str]], k: int = 5) -> RetrievalMetrics:
    hits = mrr = recall = ndcg = 0.0
    n = len(rankings) or 1
    for ranked, gold in zip(rankings, relevant):
        topk = list(ranked[:k])
        if any(d in gold for d in topk):
            hits += 1
        for rank, d in enumerate(topk):
            if d in gold:
                mrr += 1.0 / (rank + 1)
                break
        if gold:
            recall += len(set(topk) & gold) / len(gold)
        dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(topk) if d in gold)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gold), k)))
        ndcg += (dcg / idcg) if idcg else 0.0
    return RetrievalMetrics(round(hits / n, 3), round(mrr / n, 3),
                            round(recall / n, 3), round(ndcg / n, 3))


def faithfulness(answer: str, contexts: list[str]) -> float:
    a = [t for t in tokenize(answer)]
    if not a:
        return 0.0
    ctx = set().union(*(set(tokenize(c)) for c in contexts)) if contexts else set()
    return round(sum(1 for t in a if t in ctx) / len(a), 3)


def answer_correctness(answer: str, reference: str) -> float:
    """Token-F1 between the generated answer and the reference answer."""
    a, r = set(tokenize(answer)), set(tokenize(reference))
    if not a or not r:
        return 0.0
    tp = len(a & r)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(a), tp / len(r)
    return round(2 * prec * rec / (prec + rec), 3)
