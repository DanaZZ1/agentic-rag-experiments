"""Command-line interface: ingest a corpus and ask, or run evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .config import Settings
from .evaluation.harness import load_dataset, run_eval
from .ingestion.loaders import load_directory
from .pipeline import RAGPipeline

_DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
_DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "eval" / "qa.jsonl"


def _build(corpus: Path) -> RAGPipeline:
    pipe = RAGPipeline(Settings.from_env())
    pipe.add_documents(load_directory(corpus))
    return pipe


def cmd_ask(args: argparse.Namespace) -> None:
    import os
    if getattr(args, 'engine', None):
        os.environ['RAG_ENGINE'] = args.engine
    pipe = _build(Path(args.corpus))
    res = pipe.query(args.question)
    print(json.dumps({"answer": res.answer, "citations": res.citations,
                      "faithfulness": res.faithfulness, "iterations": res.iterations,
                      "trace": res.trace}, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    pipe = _build(Path(args.corpus))
    report = run_eval(pipe, load_dataset(args.dataset), k=args.k)
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prag", description="Production agentic RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("ask", help="ask a question over the corpus")
    a.add_argument("question")
    a.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    a.add_argument("--engine", choices=["agent", "langgraph"], default=None)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    e.add_argument("--dataset", default=str(_DEFAULT_EVAL))
    e.add_argument("--k", type=int, default=5)
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
