"""Minimal end-to-end usage of the production-agentic-rag pipeline (offline)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from production_agentic_rag import RAGPipeline

KB = """
# Returns Policy
Unused items can be returned within 30 days of delivery for a full refund.

# Refunds
Approved refunds are issued to the original payment method within 5 business days.
"""


def main() -> None:
    rag = RAGPipeline()
    rag.add_document(KB, doc_id="kb")
    res = rag.query("What is the return window and how long do refunds take?")
    print("answer      :", res.answer)
    print("citations   :", res.citations)
    print("faithfulness:", res.faithfulness)
    print("trace       :", res.trace)


if __name__ == "__main__":
    main()
