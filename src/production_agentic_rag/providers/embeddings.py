"""Embedding providers: deterministic offline hashing + real OpenAI/Azure adapters."""
from __future__ import annotations
import hashlib
import math
import os
from typing import Protocol, runtime_checkable

from ..text import tokenize


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> dict[int, float]: ...


class HashingEmbedder:
    """Feature-hashing embedder → sparse L2-normalised vectors. Deterministic and
    dependency-free; a solid local stand-in for dense embeddings in tests/CI."""

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode()).digest()
        return int.from_bytes(h[:4], "big") % self.dim, (1.0 if h[4] & 1 else -1.0)

    def embed(self, text: str) -> dict[int, float]:
        vec: dict[int, float] = {}
        for tok in tokenize(text):
            idx, sign = self._bucket(tok)
            vec[idx] = vec.get(idx, 0.0) + sign
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


class OpenAIEmbedder:
    """Real dense embeddings (OpenAI or Azure OpenAI via the same SDK)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.dim = 1536

    def embed(self, text: str) -> dict[int, float]:
        from openai import AzureOpenAI, OpenAI  # lazy

        client = AzureOpenAI() if os.getenv("AZURE_OPENAI_ENDPOINT") else OpenAI()
        vec = client.embeddings.create(model=self.model, input=text).data[0].embedding
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return {i: x / norm for i, x in enumerate(vec)}


def build_embedder(provider: str, model: str) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model)
    return HashingEmbedder()
