"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then reranked."""
from __future__ import annotations
from dataclasses import dataclass

from ..ingestion.chunking import Chunk
from ..providers.embeddings import Embedder
from ..text import tokenize
from .bm25 import BM25
from .vectorstore import VectorStore


@dataclass
class Scored:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, embedder: Embedder | None = None, rrf_k: int = 60) -> None:
        self.bm25 = BM25()
        self.vectors = VectorStore(embedder)
        self.rrf_k = rrf_k
        self._chunks: dict[str, Chunk] = {}

    def index(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.bm25.index(c.id, c.text)
            self.vectors.add(c)
            self._chunks[c.id] = c

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _rrf(self, rankings: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return fused

    def retrieve(self, query: str, k: int = 4, *, candidate_k: int = 12,
                 rerank: bool = True) -> list[Scored]:
        lexical = [cid for cid, _ in self.bm25.search(query, candidate_k)]
        dense = [cid for cid, _ in self.vectors.search(query, candidate_k)]
        fused = self._rrf([lexical, dense])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = [Scored(self._chunks[cid], score) for cid, score in ranked]
        if rerank:
            results = self._rerank(query, results)
        return results[:k]

    def _rerank(self, query: str, results: list[Scored]) -> list[Scored]:
        q = set(tokenize(query))
        for r in results:
            overlap = len(q & set(tokenize(r.chunk.text)))
            phrase = 0.5 if query.lower() in r.chunk.text.lower() else 0.0
            r.score = r.score + 0.2 * overlap + phrase
        results.sort(key=lambda s: s.score, reverse=True)
        return results
