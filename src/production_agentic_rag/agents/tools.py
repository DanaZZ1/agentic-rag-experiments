"""Tools the agent can call. Retrieval is exposed as a first-class tool."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from ..retrieval.hybrid import HybridRetriever, Scored


@dataclass
class RetrievalTool:
    retriever: HybridRetriever
    k: int = 4
    candidate_k: int = 12
    rerank: bool = True

    name: str = "search_knowledge_base"
    description: str = "Search the indexed corpus and return the most relevant chunks."

    def __call__(self, query: str) -> list[Scored]:
        return self.retriever.retrieve(query, k=self.k,
                                       candidate_k=self.candidate_k, rerank=self.rerank)
