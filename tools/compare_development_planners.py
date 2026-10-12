#!/usr/bin/env python3
"""Offline structural planner comparison on the isolated development-v5 set only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_planner(tree: Path):
    sys.path.insert(0, str(tree / "src"))
    try:
        return importlib.import_module("production_agentic_rag.agents.planner").plan
    finally:
        for name in list(sys.modules):
            if name == "production_agentic_rag" or name.startswith("production_agentic_rag."):
                del sys.modules[name]
        sys.path.pop(0)


def analyze(tree: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    planner = load_planner(tree)
    output = []
    for record in records:
        tasks = planner(record["question"])
        serialized = [asdict(task) for task in tasks]
        facts = [fact for task in tasks for fact in task.required_facts]
        comparisons = [task for task in tasks if task.comparison]
        output.append({
            "question_id": record["question_id"],
            "source_category": record.get("category", "unknown"),
            "planned_task_count": len(tasks),
            "planned_fact_count": len(facts),
            "comparison_operand_count": sum(len(task.required_facts) for task in comparisons),
            "comparison_conclusion_count": sum(bool(task.comparison and
                task.comparison.get("conclusion_required")) for task in tasks),
            "dependency_edge_count": sum(len(dep.get("depends_on", []))
                for task in tasks for dep in getattr(task, "dependencies", [])),
            "generic_unverifiable_fact_ids": [fact.id for fact in facts if not fact.verifiable],
            "planning_uncertainty": {task.id: list(getattr(task, "planning_uncertainty", [])) for task in tasks},
            "tasks": serialized,
        })
    return {
        "worktree": str(tree),
        "records": output,
        "summary": {
            "record_count": len(records),
            "task_count": sum(item["planned_task_count"] for item in output),
            "fact_count": sum(item["planned_fact_count"] for item in output),
            "comparison_operand_count": sum(item["comparison_operand_count"] for item in output),
            "comparison_conclusion_count": sum(item["comparison_conclusion_count"] for item in output),
            "dependency_edge_count": sum(item["dependency_edge_count"] for item in output),
            "generic_unverifiable_fact_count": sum(len(item["generic_unverifiable_fact_ids"]) for item in output),
            "questions_with_planning_uncertainty": sum(bool(any(item["planning_uncertainty"].values())) for item in output),
            "model_calls": 0,
            "retrieval_calls": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.dataset)
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    baseline, candidate = analyze(args.baseline, records), analyze(args.candidate, records)
    base_by_id = {row["question_id"]: row for row in baseline["records"]}
    candidate_by_id = {row["question_id"]: row for row in candidate["records"]}
    regressions = []
    for question_id, before in base_by_id.items():
        after = candidate_by_id[question_id]
        if after["planned_fact_count"] < before["planned_fact_count"]:
            regressions.append({"question_id": question_id, "kind": "fewer_planned_facts",
                                "before": before["planned_fact_count"], "after": after["planned_fact_count"]})
        if after["comparison_operand_count"] < before["comparison_operand_count"]:
            regressions.append({"question_id": question_id, "kind": "fewer_comparison_operands",
                                "before": before["comparison_operand_count"], "after": after["comparison_operand_count"]})
    result = {
        "dataset": str(args.dataset),
        "dataset_sha256": digest,
        "evaluation_scope": "structural diagnostics only; no semantic accuracy claim",
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {key: candidate["summary"][key] - baseline["summary"][key]
                   for key in baseline["summary"] if key not in ("record_count", "model_calls", "retrieval_calls")},
        "structural_regressions": regressions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dataset_sha256": digest, "baseline": baseline["summary"],
                      "candidate": candidate["summary"], "structural_regressions": len(regressions),
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
