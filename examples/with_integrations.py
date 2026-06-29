"""Demo of the optional security + observability integrations (graceful fallback)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_agentic_rag import RAGPipeline
from production_agentic_rag.integrations import (HAS_GATEWAY, HAS_OBSERVABILITY,
                                                 SecuredRetrievalTool, export_observability)

KB = "# Returns\nUnused items can be returned within 30 days for a full refund."


def main() -> None:
    pipe = RAGPipeline()
    pipe.add_document(KB, doc_id="kb")
    print("mcp-tool-gateway installed :", HAS_GATEWAY)
    print("llm-observability installed:", HAS_OBSERVABILITY)

    if HAS_GATEWAY:
        # route the agent's retrieval through the secure gateway
        pipe.agent.tool = SecuredRetrievalTool(pipe.retriever, k=pipe.settings.top_k)
        print("retrieval is now secured by the gateway (auth + injection inspection + audit)")

    res = pipe.query("what is the return window?")
    print("answer     :", res.answer)
    print("observability report:", export_observability  # function ref
          and "(see below)")
    # export the trace via llm-observability if present (else built-in summary)
    from production_agentic_rag.observability import Tracer
    print("trace summary:", res.trace)


if __name__ == "__main__":
    main()
