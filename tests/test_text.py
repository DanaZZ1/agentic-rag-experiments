from production_agentic_rag.text import cosine, sentences, stem, tokenize


def test_stem_and_stopwords():
    toks = tokenize("Returns are processed and refunded")
    assert "return" in toks and "are" not in toks  # stemmed + stopword removed


def test_cosine_self():
    from production_agentic_rag.providers.embeddings import HashingEmbedder
    v = HashingEmbedder().embed("regulatory trade reporting")
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_sentences_strips_headers():
    s = sentences("# Title\nFirst sentence. Second sentence.")
    assert s[0] == "First sentence."
