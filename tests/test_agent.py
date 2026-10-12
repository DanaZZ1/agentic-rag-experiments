from production_agentic_rag.agents import AgenticRAG, RetrievalTool, plan
from production_agentic_rag.ingestion.chunking import markdown_section_chunks
from production_agentic_rag.providers.llm import MockLLM
from production_agentic_rag.retrieval import HybridRetriever

MD = ("# Returns\nReturn unused items within 30 days for a full refund.\n\n"
      "# Refunds\nApproved refunds are issued within 5 business days.\n\n"
      "# Warranty\nElectronics have a 1-year limited warranty.")


def _agent():
    r = HybridRetriever()
    r.index(markdown_section_chunks(MD, doc_id="kb"))
    return AgenticRAG(llm=MockLLM(), tool=RetrievalTool(r, k=2))


def test_planner_multipart():
    assert len(plan("What is the return window and how long do refunds take?")) >= 2


def test_multi_hop_citations():
    a = _agent().run("What is the return window and how long do refunds take?")
    assert len(a.citations) >= 2 and a.faithfulness > 0.4


def test_single_hop_grounded():
    a = _agent().run("what does the warranty cover?")
    # This fixture states that electronics have a limited warranty, but says
    # nothing about what it covers. The evidence-aware agent must abstain.
    assert not a.citations
    assert a.task_answers[0].status == "not_in_corpus"
    assert a.task_answers[0].final_state == "not_in_corpus"
    assert "couldn't complete verification" not in a.answer.lower()
