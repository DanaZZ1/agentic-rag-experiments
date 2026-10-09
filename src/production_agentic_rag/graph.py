"""Optional LangGraph adapter using the built-in agent's exact V2 semantics."""
from __future__ import annotations

from typing import TypedDict

from .agents.rag_agent import AgentAnswer, AgenticRAG
from .observability import Tracer

try:
    import langgraph  # noqa: F401
    HAS_LANGGRAPH = True
except Exception:  # pragma: no cover
    HAS_LANGGRAPH = False


class RAGState(TypedDict, total=False):
    question: str
    result: AgentAnswer


def build_rag_graph(*, llm, tool, model: str = "mock", max_iterations: int = 2,
                    faithfulness_threshold: float = 0.6, tracer: Tracer | None = None):
    """Compile a graph wrapper around the shared V2 control loop.

    The graph intentionally delegates sufficiency, verification, and retries to
    AgenticRAG so the optional engine cannot silently use different semantics.
    """
    from langgraph.graph import END, START, StateGraph

    agent = AgenticRAG(llm, tool, model, max_iterations, faithfulness_threshold)
    tr = tracer or Tracer()
    graph = StateGraph(RAGState)
    graph.add_node("agent_v2", lambda state: {"result": agent.run(state["question"], tracer=tr)})
    graph.add_edge(START, "agent_v2")
    graph.add_edge("agent_v2", END)
    app = graph.compile()
    app._prag_tracer = tr
    return app


class LangGraphRAG:
    """LangGraph-backed wrapper with the same run contract and V2 verifier."""

    def __init__(self, *, llm, tool, model: str = "mock", max_iterations: int = 2,
                 faithfulness_threshold: float = 0.6) -> None:
        if not HAS_LANGGRAPH:
            raise RuntimeError("LangGraph not installed; pip install '.[langgraph]'")
        self.llm, self.tool, self.model = llm, tool, model
        self.max_iterations = max_iterations
        self.faithfulness_threshold = faithfulness_threshold

    def run(self, question: str, tracer: Tracer | None = None) -> AgentAnswer:
        tracer = tracer or Tracer()
        app = build_rag_graph(llm=self.llm, tool=self.tool, model=self.model,
                              max_iterations=self.max_iterations,
                              faithfulness_threshold=self.faithfulness_threshold, tracer=tracer)
        result = app.invoke({"question": question}).get("result")
        if not isinstance(result, AgentAnswer):
            raise RuntimeError("LangGraph did not return an AgentAnswer")
        return result
