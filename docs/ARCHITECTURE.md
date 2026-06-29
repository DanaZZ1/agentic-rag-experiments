# Architecture

## Design goals
1. **Production concerns first** — planning, grounding, guardrails, evaluation, observability are first-class, not afterthoughts.
2. **Provider-agnostic** — a one-line switch moves between an offline deterministic mock and OpenAI/Azure OpenAI.
3. **Offline-deterministic** — the entire control flow (retrieval, fusion, agent loop, eval) is unit-testable with no network or keys.
4. **Swappable components** — every stage is a small module behind a clear interface.

## Components

### Providers (`providers/`)
- `LLMProvider` protocol → `MockLLM` (extractive, deterministic), `OpenAILLM` (OpenAI/Azure).
- `Embedder` protocol → `HashingEmbedder` (feature hashing, L2-normalised), `OpenAIEmbedder`.

### Ingestion (`ingestion/`)
- Loaders for files/directories; chunkers: fixed-size sliding window (with overlap) and markdown-section.

### Retrieval (`retrieval/`)
- `BM25` lexical scorer (IDF + length normalisation), stemming-aware.
- `VectorStore` dense cosine search with an LRU embedding cache.
- `HybridRetriever` fuses lexical + dense rankings via **Reciprocal Rank Fusion**
  (`1/(k+rank)`), then reranks by stopword-filtered lexical overlap + exact-phrase bonus.

### Agent (`agents/`)
- `plan()` decomposes a question into sub-questions (LLM-swappable contract).
- `RetrievalTool` exposes retrieval as a callable tool.
- `AgenticRAG` runs: plan → per-hop retrieve (dedup) → synthesize (LLM) → score grounding →
  if below `faithfulness_threshold`, reformulate (pseudo-relevance feedback) and retry up to `max_iterations`.

### Guardrails (`guardrails.py`)
- Input: length cap + prompt-injection patterns (fail-closed, raises `InputBlocked`).
- Output: PII redaction (SSN/card/email) + grounding score.

### Observability (`observability.py`)
- Hierarchical `Tracer` spans with durations; token + USD cost accounting via a price book.

### Evaluation (`evaluation/`)
- `evaluate_retrieval`: hit@k, MRR, recall@k, nDCG.
- Generation: `faithfulness` (answer ⊆ context), `answer_correctness` (token-F1 vs reference).
- `run_eval` executes the pipeline over a JSONL dataset and returns an aggregate report.

### Pipeline (`pipeline.py`)
Wires config → providers → retriever → agent, applies guardrails, and returns a
`QueryResult` (answer, citations, contexts, faithfulness, trace summary).

## Extending
- **New provider**: implement `complete()` / `embed()` and register in `build_llm` / `build_embedder`.
- **New chunker**: add to `CHUNKERS`.
- **LLM-based planning/reranking**: replace the heuristics in `agents/planner.py` / `retrieval/hybrid.py` — interfaces are unchanged.
