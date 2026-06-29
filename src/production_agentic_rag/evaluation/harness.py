"""End-to-end evaluation runner over a labelled QA dataset."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import answer_correctness, evaluate_retrieval, faithfulness


@dataclass
class EvalCase:
    question: str
    answer: str
    relevant_ids: set[str]


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(EvalCase(obj["question"], obj.get("answer", ""),
                              set(obj.get("relevant_ids", []))))
    return cases


def run_eval(pipeline: Any, cases: list[EvalCase], k: int = 5) -> dict[str, Any]:
    rankings, gold, faiths, corrects = [], [], [], []
    for case in cases:
        result = pipeline.query(case.question)
        rankings.append(result.citations)
        gold.append(case.relevant_ids)
        faiths.append(faithfulness(result.answer, result.context_texts))
        if case.answer:
            corrects.append(answer_correctness(result.answer, case.answer))
    rm = evaluate_retrieval(rankings, gold, k=k)
    return {
        "n": len(cases),
        "retrieval": rm.__dict__,
        "faithfulness_mean": round(sum(faiths) / (len(faiths) or 1), 3),
        "answer_correctness_mean": round(sum(corrects) / (len(corrects) or 1), 3) if corrects else None,
    }
