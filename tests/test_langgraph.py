"""LangGraph engine test — runs only when LangGraph is installed; otherwise it
passes trivially (the engine is an optional dependency)."""
from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.graph import HAS_LANGGRAPH


def test_langgraph_engine_when_available():
    if not HAS_LANGGRAPH:
        return  # optional dependency not present
    settings = Settings()
    settings.engine = "langgraph"
    pipe = RAGPipeline(settings)
    pipe.add_document("# Returns\nUnused items can be returned within 30 days for a full refund.", doc_id="kb")
    res = pipe.query("what is the return window?")
    assert "kb:0" in res.citations
    assert "30 days" in res.answer


def test_pipeline_defaults_to_builtin_engine():
    # default engine is the built-in agent; must work with no extra deps
    pipe = RAGPipeline()
    pipe.add_document("# A\nalpha beta gamma delta", doc_id="d")
    assert pipe.query("alpha").answer
