"""In-memory dense vector store with cosine search + embedding cache."""
from __future__ import annotations
import hashlib
from collections import OrderedDict

from ..ingestion.chunking import Chunk
from ..providers.embeddings import Embedder, HashingEmbedder
from ..text import cosine


class VectorStore:
    def __init__(self, embedder: Embedder | None = None, cache_size: int = 4096) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._vecs: dict[str, dict[int, float]] = {}
        self._chunks: dict[str, Chunk] = {}
        self._cache: "OrderedDict[str, dict[int, float]]" = OrderedDict()
        self._cap = cache_size
        self.hits = self.misses = 0

    def _embed_cached(self, text: str) -> dict[int, float]:
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        vec = self.embedder.embed(text)
        self._cache[key] = vec
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return vec

    def add(self, chunk: Chunk) -> None:
        self._vecs[chunk.id] = self._embed_cached(chunk.text)
        self._chunks[chunk.id] = chunk

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        qv = self.embedder.embed(query)
        scored = [(cid, cosine(qv, v)) for cid, v in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(cid, s) for cid, s in scored[:k] if s > 0]

    def chunk(self, cid: str) -> Chunk:
        return self._chunks[cid]
