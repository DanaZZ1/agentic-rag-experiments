"""Okapi BM25 lexical retrieval (from scratch, stemming-aware)."""
from __future__ import annotations
import math
from collections import Counter

from ..text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.ids: list[str] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def index(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.docs.append(toks)
        self.ids.append(doc_id)
        for term in set(toks):
            self.df[term] += 1
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores: list[tuple[str, float]] = []
        for doc_id, toks in zip(self.ids, self.docs):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self._idf(term) * num / den
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
