"""Top-level RAG pipeline: ingest → guardrails → agentic retrieval+generation → report."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .agents.rag_agent import AgenticRAG
from .agents.tools import RetrievalTool
from .config import Settings
from .guardrails import GuardrailReport, InputBlocked, check_input, grounding_score, redact_pii
from .ingestion.chunking import CHUNKERS
from .ingestion.loaders import Document
from .observability import Tracer
from .providers.embeddings import build_embedder
from .providers.llm import build_llm
from .retrieval.hybrid import HybridRetriever, Scored


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: list[str]
    contexts: list[Scored]
    faithfulness: float
    iterations: int
    blocked: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.text for c in self.contexts]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.retriever = HybridRetriever(
            embedder=build_embedder(self.settings.embed_provider, self.settings.embed_model))
        self.llm = build_llm(self.settings.llm_provider, self.settings.llm_model)
        self.tool = RetrievalTool(self.retriever, k=self.settings.top_k,
                                  candidate_k=self.settings.candidate_k,
                                  rerank=self.settings.rerank)
        self.agent = AgenticRAG(
            llm=self.llm, tool=self.tool,
            model=("mock" if self.settings.llm_provider == "mock" else self.settings.llm_model),
            max_iterations=self.settings.max_iterations,
            faithfulness_threshold=self.settings.faithfulness_threshold)
        self._engine = self._select_engine()

    def _select_engine(self):
        """Use the LangGraph engine if requested and installed, else the built-in agent."""
        if self.settings.engine == "langgraph":
            try:
                from .graph import LangGraphRAG
                return LangGraphRAG(
                    llm=self.llm, tool=self.tool,
                    model=("mock" if self.settings.llm_provider == "mock" else self.settings.llm_model),
                    max_iterations=self.settings.max_iterations,
                    faithfulness_threshold=self.settings.faithfulness_threshold)
            except Exception:
                pass  # fall back gracefully
        return self.agent

    # ingestion ---------------------------------------------------------------
    def add_document(self, text: str, doc_id: str = "doc", strategy: str = "section") -> int:
        chunker = CHUNKERS[strategy]
        chunks = chunker(text, doc_id=doc_id)
        self.retriever.index(chunks)
        return len(chunks)

    def add_documents(self, docs: list[Document], strategy: str = "section") -> int:
        return sum(self.add_document(d.text, d.doc_id, strategy) for d in docs)

    # query -------------------------------------------------------------------
    def query(self, question: str) -> QueryResult:
        tracer = Tracer()
        try:
            question = check_input(question)
        except InputBlocked as exc:
            return QueryResult(question, f"Request blocked: {exc}", [], [], 0.0, 0,
                               blocked=True, trace=tracer.summary())
        with tracer.span("pipeline.query"):
            agent_answer = self._engine.run(question, tracer=tracer)
            chunks_by_id = {r.chunk.id: r for h in agent_answer.hops for r in h.contexts}
            contexts = [chunks_by_id[c] for c in agent_answer.citations if c in chunks_by_id]
            answer, _ = redact_pii(agent_answer.answer)
            faith = grounding_score(answer, [c.chunk.text for c in contexts])
        return QueryResult(question, answer, agent_answer.citations, contexts,
                           faith, agent_answer.iterations, trace=tracer.summary())
