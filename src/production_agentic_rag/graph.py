"""LangGraph engine for the agentic RAG flow (optional).

Implements the same plan → retrieve → generate → self-correct loop as the built-in
engine, but as an explicit LangGraph ``StateGraph`` with conditional edges. LangGraph
is an optional dependency: ``pip install '.[langgraph]'``. When it is not installed,
the pipeline transparently falls back to the built-in engine.

Graph topology:

    START → plan → retrieve → generate ─┬─(grounded or max iters)→ END
                     ▲                  │
                     └── reformulate ◀──┘  (faithfulness < threshold)
"""
from __future__ import annotations
from typing import Any, TypedDict

from .agents.planner import plan
from .agents.rag_agent import AgentAnswer, Hop, _SYSTEM, _build_prompt
from .guardrails import grounding_score
from .observability import Tracer

try:                                    # availability flag (real import is lazy)
    import langgraph  # noqa: F401
    HAS_LANGGRAPH = True
except Exception:                       # pragma: no cover
    HAS_LANGGRAPH = False


class RAGState(TypedDict, total=False):
    question: str
    query: str
    sub_questions: list[str]
    contexts: list[Any]      # list[Scored]
    citations: list[str]
    answer: str
    faithfulness: float
    iterations: int


def build_rag_graph(*, llm, tool, model: str = "mock",
                    max_iterations: int = 2, faithfulness_threshold: float = 0.6,
                    tracer: Tracer | None = None):
    """Compile and return a LangGraph app implementing the agentic RAG loop."""
    from langgraph.graph import END, START, StateGraph   # lazy import

    tr = tracer or Tracer()

    def plan_node(state: RAGState) -> dict:
        with tr.span("plan"):
            subs = plan(state.get("query", state["question"]))
        return {"sub_questions": subs, "iterations": state.get("iterations", 0) + 1}

    def retrieve_node(state: RAGState) -> dict:
        seen: dict[str, Any] = {}
        with tr.span("retrieve"):
            for sub in state["sub_questions"]:
                for r in tool(sub):
                    seen.setdefault(r.chunk.id, r)
        return {"contexts": list(seen.values()), "citations": list(seen.keys())}

    def generate_node(state: RAGState) -> dict:
        ctx = [r.chunk.text for r in state["contexts"]]
        with tr.span("synthesize"):
            res = llm.complete(_SYSTEM, _build_prompt(state["question"], ctx))
            tr.record_usage(model, res.usage)
        return {"answer": res.text, "faithfulness": grounding_score(res.text, ctx)}

    def reformulate_node(state: RAGState) -> dict:
        expansion = " ".join(r.chunk.text for r in state["contexts"])[:240]
        return {"query": f"{state['question']} {expansion}"}

    def should_continue(state: RAGState) -> str:
        if state["faithfulness"] >= faithfulness_threshold or state["iterations"] >= max_iterations:
            return "end"
        return "retry"

    graph = StateGraph(RAGState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", should_continue,
                                {"end": END, "retry": "reformulate"})
    graph.add_edge("reformulate", "plan")
    app = graph.compile()
    app._prag_tracer = tr   # expose tracer for the wrapper
    return app


class LangGraphRAG:
    """Wrapper exposing the same ``.run(question)`` contract as the built-in agent,
    backed by a compiled LangGraph app."""

    def __init__(self, *, llm, tool, model: str = "mock",
                 max_iterations: int = 2, faithfulness_threshold: float = 0.6) -> None:
        if not HAS_LANGGRAPH:
            raise RuntimeError("LangGraph not installed; pip install '.[langgraph]'")
        self.llm, self.tool, self.model = llm, tool, model
        self.max_iterations = max_iterations
        self.faithfulness_threshold = faithfulness_threshold

    def run(self, question: str, tracer: Tracer | None = None) -> AgentAnswer:
        tracer = tracer or Tracer()
        app = build_rag_graph(llm=self.llm, tool=self.tool, model=self.model,
                              max_iterations=self.max_iterations,
                              faithfulness_threshold=self.faithfulness_threshold,
                              tracer=tracer)
        final: RAGState = app.invoke({"question": question, "query": question, "iterations": 0})
        # carry the retrieved Scored contexts so the pipeline can rebuild + score them
        hops = [Hop("aggregated", final.get("contexts", []))]
        return AgentAnswer(
            question=question,
            answer=final.get("answer", ""),
            hops=hops,
            faithfulness=final.get("faithfulness", 0.0),
            iterations=final.get("iterations", 1),
            citations=final.get("citations", []),
        )
