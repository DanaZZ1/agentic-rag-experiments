from production_agentic_rag.ingestion.chunking import fixed_size_chunks, markdown_section_chunks
from production_agentic_rag.retrieval import BM25, HybridRetriever


def test_fixed_chunk_overlap():
    chunks = fixed_size_chunks(" ".join(map(str, range(300))), size=100, overlap=20)
    assert len(chunks) >= 3 and all(len(c.text.split()) <= 100 for c in chunks)


def test_bm25_ranking():
    bm = BM25()
    bm.index("a", "the cat sat on the mat")
    bm.index("b", "quarterly revenue and financial report")
    assert bm.search("financial revenue", 1)[0][0] == "b"


def test_hybrid_section_retrieval():
    md = "# Shipping\nExpress shipping arrives in 1-2 business days.\n\n# Returns\nReturn within 30 days."
    r = HybridRetriever()
    r.index(markdown_section_chunks(md, doc_id="kb"))
    top = r.retrieve("how fast is express shipping", k=1)
    assert top and "Shipping" in top[0].chunk.metadata["section"]
