"""FastAPI service exposing the RAG pipeline (optional; needs fastapi+uvicorn)."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from pydantic import BaseModel

from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.ingestion.loaders import load_directory

app = FastAPI(title="Production Agentic RAG", version="0.1.0")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(Settings.from_env())
        corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
        if corpus.exists():
            _pipeline.add_documents(load_directory(corpus))
    return _pipeline


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    faithfulness: float
    iterations: int
    blocked: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    res = get_pipeline().query(req.question)
    return AskResponse(answer=res.answer, citations=res.citations,
                       faithfulness=res.faithfulness, iterations=res.iterations,
                       blocked=res.blocked)
