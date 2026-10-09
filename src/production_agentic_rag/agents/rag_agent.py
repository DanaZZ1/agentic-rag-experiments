"""Single-agent plan, retrieve, generate, verify, and targeted-retry loop."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..guardrails import grounding_score, typed_value_request, verify_task_answer
from ..observability import Tracer
from ..providers.llm import LLMProvider
from ..retrieval.hybrid import Scored
from .planner import PlannedTask, plan
from .tools import RetrievalTool

_SYSTEM = ("Answer each task only from its supplied evidence. Return JSON with atomic claims. "
          "Every factual claim must include entity, attribute, fact_ids, and exact evidence_refs. "
          "The answer must be exactly the ordered claim texts; do not add uncited prose.")


@dataclass
class EvidenceRef:
    chunk_id: str
    quote: str
    fact_id: str = ""


@dataclass
class Claim:
    claim_id: str
    entity: str
    attribute: str
    text: str
    fact_ids: list[str]
    evidence_refs: list[EvidenceRef] = field(default_factory=list)


@dataclass
class TaskAnswer:
    task_id: str
    status: str
    answer: str
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    reason: str = ""
    retries: int = 0
    missing_fact_ids: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)


@dataclass
class Hop:
    sub_question: str
    contexts: list[Scored]
    task_id: str = ""


@dataclass
class AgentAnswer:
    question: str
    answer: str
    hops: list[Hop]
    faithfulness: float
    iterations: int
    citations: list[str] = field(default_factory=list)
    task_answers: list[TaskAnswer] = field(default_factory=list)


def _task_dict(task: PlannedTask) -> dict:
    return asdict(task)


def _build_prompt(question: str, tasks: list[dict]) -> str:
    return "RAG_TASKS_V3\n" + json.dumps({"question": question, "tasks": tasks}, ensure_ascii=False)


@dataclass
class AgenticRAG:
    llm: LLMProvider
    tool: RetrievalTool
    model: str = "mock"
    max_iterations: int = 2
    faithfulness_threshold: float = 0.6  # retained for API compatibility; diagnostic only

    def run(self, question: str, tracer: Tracer | None = None) -> AgentAnswer:
        tracer = tracer or Tracer()
        tasks = plan(question)
        hops: list[Hop] = []
        retrieved: dict[str, dict[str, dict[str, Scored]]] = {
            t.id: {fact.id: {} for fact in t.required_facts} for t in tasks}
        attempts: dict[str, int] = {t.id: 0 for t in tasks}
        verified: dict[str, TaskAnswer] = {}
        pending = {t.id for t in tasks}
        task_by_id = {t.id: t for t in tasks}
        iterations = 0

        with tracer.span("agent.run", question=question):
            for round_no in range(1, max(1, self.max_iterations) + 1):
                if not pending:
                    break
                iterations = round_no
                payload_tasks = []
                with tracer.span("plan", round=round_no):
                    for tid in list(pending):
                        task = task_by_id[tid]
                        failed = verified.get(tid)
                        fact_ids = set(failed.missing_fact_ids) if failed else set()
                        # Retry only the failed task, using its specific missing-fact guidance.
                        target_facts = [f for f in task.required_facts if not fact_ids or f.id in fact_ids]
                        for fact in target_facts:
                            retry_query = fact.retrieval_query
                            if failed and failed.reason:
                                retry_query = f"{fact.retrieval_query} {fact.description} {failed.reason}"
                            # Inspect all indexed chunks for typed absence checks; retrieval stack is unchanged.
                            full = typed_value_request(task.question) is not None
                            k = max(self.tool.k, self.tool.retriever.size) if full else self.tool.k
                            results = self.tool(retry_query, k=k,
                                                candidate_k=max(self.tool.candidate_k, self.tool.retriever.size) if full else None)
                            hops.append(Hop(retry_query, results, tid))
                            bucket = retrieved[tid][fact.id]
                            for result in results:
                                bucket.setdefault(result.chunk.id, result)
                        evidence_groups = []
                        for fact in task.required_facts:
                            chunk_list = list(retrieved[tid][fact.id].values())
                            evidence_groups.append({"fact_id": fact.id,
                                "query": fact.retrieval_query,
                                "chunks": [{"chunk_id": r.chunk.id, "text": r.chunk.text} for r in chunk_list]})
                        td = _task_dict(task)
                        td["evidence"] = evidence_groups
                        payload_tasks.append(td)

                with tracer.span("generate", round=round_no):
                    result = self.llm.complete(_SYSTEM, _build_prompt(question, payload_tasks))
                    tracer.record_usage(self.model, result.usage)
                try:
                    generated = json.loads(result.text).get("tasks", [])
                except (json.JSONDecodeError, AttributeError):
                    generated = []
                generated_by_id = {str(x.get("task_id")): x for x in generated if isinstance(x, dict)}

                for tid in list(pending):
                    task = task_by_id[tid]
                    raw = generated_by_id.get(tid, {})
                    per_task_chunks = {cid: item.chunk.text for fact_chunks in retrieved[tid].values()
                                       for cid, item in fact_chunks.items()}
                    verification = verify_task_answer(_task_dict(task), raw, per_task_chunks,
                                                      require_claims=True)
                    if verification.status in ("supported", "not_in_corpus"):
                        valid_claims = []
                        if verification.status == "supported":
                            for claim in raw.get("claims", []):
                                valid_claims.append(Claim(
                                    claim_id=str(claim.get("claim_id", "")),
                                    entity=str(claim.get("entity", "unknown")),
                                    attribute=str(claim.get("attribute", "general")),
                                    text=str(claim.get("text", "")),
                                    fact_ids=[str(fid) for fid in claim.get("fact_ids", [])],
                                    evidence_refs=[EvidenceRef(**ref) for ref in claim.get("evidence_refs", [])],
                                ))
                        verified[tid] = TaskAnswer(tid, verification.status, verification.answer,
                            [EvidenceRef(**ref) for ref in verification.evidence_refs], verification.reason,
                            attempts[tid], claims=valid_claims)
                        pending.remove(tid)
                    else:
                        attempts[tid] += 1
                        verified[tid] = TaskAnswer(tid, "needs_retry", raw.get("answer", ""),
                            [EvidenceRef(**ref) for ref in verification.evidence_refs],
                            verification.reason, attempts[tid], verification.missing_fact_ids or [])

        # Never expose an unverified candidate when the bounded retry budget is exhausted.
        for tid in pending:
            task = task_by_id[tid]
            verified[tid] = TaskAnswer(tid, "not_in_corpus",
                f"The requested information is not present in the corpus: {task.question}", [],
                verified[tid].reason if tid in verified else "Verification did not pass", attempts[tid])
        ordered = [verified[t.id] for t in tasks]
        answer_parts = [ta.answer for ta in ordered]
        final_answer = "\n".join(answer_parts)
        citations = list(dict.fromkeys(ref.chunk_id for ta in ordered if ta.status == "supported" for ref in ta.evidence_refs))
        all_contexts = [r.chunk.text for h in hops for r in h.contexts]
        faith = grounding_score(final_answer, all_contexts)
        return AgentAnswer(question, final_answer, hops, faith, iterations, citations, ordered)
