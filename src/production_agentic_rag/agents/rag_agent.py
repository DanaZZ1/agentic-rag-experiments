"""Agentic RAG loop: plan -> multi-hop retrieve -> synthesize -> self-correct."""
from __future__ import annotations
from dataclasses import dataclass, field

from ..guardrails import grounding_score
from ..observability import Tracer
from ..providers.llm import LLMProvider
from ..retrieval.hybrid import Scored
from .planner import plan
from .tools import RetrievalTool

_SYSTEM = (
    "You are a precise assistant. Answer ONLY from the provided context. "
    "If the context is insufficient, say so. Be concise and cite nothing you "
    "cannot support from the context."
)


@dataclass
class Hop:
    sub_question: str
    contexts: list[Scored]


@dataclass
class AgentAnswer:
    question: str
    answer: str
    hops: list[Hop]
    faithfulness: float
    iterations: int
    citations: list[str] = field(default_factory=list)


def _build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n---\n".join(contexts)
    return f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"


@dataclass
class AgenticRAG:
    llm: LLMProvider
    tool: RetrievalTool
    model: str = "mock"
    max_iterations: int = 2
    faithfulness_threshold: float = 0.6

    def run(self, question: str, tracer: Tracer | None = None) -> AgentAnswer:
        tracer = tracer or Tracer()
        best: AgentAnswer | None = None
        query = question
        with tracer.span("agent.run", question=question):
            for iteration in range(1, self.max_iterations + 1):
                hops: list[Hop] = []
                with tracer.span("plan"):
                    sub_questions = plan(query)
                seen: dict[str, Scored] = {}
                for sub in sub_questions:
                    with tracer.span("retrieve", sub_question=sub):
                        results = self.tool(sub)
                    hops.append(Hop(sub, results))
                    for r in results:
                        seen.setdefault(r.chunk.id, r)
                contexts = [r.chunk.text for r in seen.values()]
                with tracer.span("synthesize"):
                    result = self.llm.complete(_SYSTEM, _build_prompt(question, contexts))
                    tracer.record_usage(self.model, result.usage)
                faith = grounding_score(result.text, contexts)
                candidate = AgentAnswer(question, result.text, hops, faith, iteration,
                                        citations=list(seen.keys()))
                if best is None or faith > best.faithfulness:
                    best = candidate
                if faith >= self.faithfulness_threshold:
                    break
                query = self._reformulate(question, contexts)  # self-correction
        assert best is not None
        return best

    @staticmethod
    def _reformulate(question: str, contexts: list[str]) -> str:
        expansion = " ".join(contexts)[:240]
        return f"{question} {expansion}"
