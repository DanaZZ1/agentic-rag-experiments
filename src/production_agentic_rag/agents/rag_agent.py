"""Single-agent plan, retrieve, generate, verify, and targeted-retry loop."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..guardrails import (EvidenceVerification, bind_evidence_references,
                          classify_verification_failure, grounding_score,
                          typed_value_request, verify_task_answer)
from ..observability import Tracer
from ..providers.llm import LLMProvider
from ..retrieval.hybrid import Scored
from .planner import PlannedTask, plan
from .tools import RetrievalTool

_SYSTEM = '''Answer each task only from its supplied evidence. Return exactly one JSON object matching this contract:

{
  "tasks": [
    {
      "task_id": "task_1",
      "status": "supported",
      "answer": "Returns are accepted within 30 days.",
      "claims": [
        {
          "claim_id": "claim_fact_1",
          "entity": "return_eligibility",
          "attribute": "duration",
          "text": "Returns are accepted within 30 days.",
          "fact_ids": ["fact_1"],
          "evidence_refs": [
            {
              "fact_id": "fact_1",
              "chunk_id": "handbook:0",
              "quote": "Returns are accepted within 30 days."
            }
          ]
        }
      ],
      "evidence_refs": [
        {
          "fact_id": "fact_1",
          "chunk_id": "handbook:0",
          "quote": "Returns are accepted within 30 days."
        }
      ]
    }
  ]
}

Return one task object for every input task. Copy each input task's id exactly into task_id.
The only allowed status values are "supported" and "not_in_corpus". Use "supported" only
when the supplied evidence supports the answer. For "not_in_corpus", give an explicit
abstention in answer and use empty claims and evidence_refs arrays.
For supported tasks, answer must equal the ordered claim text joined with newlines. Every
claim must have claim_id, entity, attribute, text, fact_ids, and evidence_refs. Copy entity
and attribute exactly from the associated planned required_fact. Entity means the fact's
subject; attribute means the requested relation/property. Never infer labels from your answer
or change them to fit a passage. A fact marked verifiable=false must not be answered as
supported. Each evidence reference must have fact_id, chunk_id, and a candidate source quote.
Keep each quote to one contiguous source span; use separate references for separate source
sentences. The runtime resolves quotes only against chunks retrieved for that fact and
replaces both chunk_id and quote with exact source values. Do not concatenate distant text,
paraphrase quotes, or guess IDs. For comparisons, include one evidence-linked claim per
operand plus a comparison claim with entity="comparison", the planned comparison
claim_attribute, both operand fact IDs, and evidence refs for both. Do not give a yes/no
conclusion when either operand or its comparison value is absent. For not_in_corpus, use a
clear plain-language refusal (including "There is no evidence to determine...") and empty
claims and evidence_refs arrays. Do not add prose outside the JSON object, use Markdown fences,
or copy example values. If a task contains retry_guidance, address that specific failure.
Reuse supplied evidence for contract, citation, entity/attribute, and comparison repairs;
do not claim evidence is absent when it is present. Request/retrieve more evidence only when
the guidance says evidence is insufficient.'''


class LLMOutputContractError(ValueError):
    """Raised when a V3 completion cannot be mapped to the planned task contract."""


def _strip_json_fence(text: str) -> str:
    """Remove a single whole-response Markdown JSON fence, if present."""
    import re

    match = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", text, re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def _parse_generated_tasks(text: str, expected_task_ids: set[str]) -> dict[str, dict]:
    """Parse and validate the task envelope before invoking evidence verification."""
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise LLMOutputContractError(f"V3 completion is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise LLMOutputContractError('V3 completion must be an object with a "tasks" array')

    generated_by_id: dict[str, dict] = {}
    for index, task in enumerate(payload["tasks"]):
        if not isinstance(task, dict):
            raise LLMOutputContractError(f"V3 tasks[{index}] must be an object")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise LLMOutputContractError(f"V3 tasks[{index}] is missing a non-empty task_id")
        if task_id in generated_by_id:
            raise LLMOutputContractError(f"V3 completion contains duplicate task_id {task_id!r}")
        if task.get("status") not in ("supported", "not_in_corpus"):
            raise LLMOutputContractError(
                f"V3 task {task_id!r} has invalid status {task.get('status')!r}; "
                'expected "supported" or "not_in_corpus"')
        generated_by_id[task_id] = task

    missing = sorted(expected_task_ids - generated_by_id.keys())
    unexpected = sorted(generated_by_id.keys() - expected_task_ids)
    if missing or unexpected:
        raise LLMOutputContractError(
            f"V3 task_id mismatch: missing={missing}, unexpected={unexpected}")
    return generated_by_id


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
    failure_type: str = ""
    # Public outcome is distinct from the verifier's internal status.
    final_state: str = ""
    original_answer: str = ""
    error_reason: str = ""


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
    diagnostics: list[dict] = field(default_factory=list)
    final_state: str = ""


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
        diagnostics: list[dict] = []
        pending = {t.id for t in tasks}
        task_by_id = {t.id: t for t in tasks}
        iterations = 0

        with tracer.span("agent.run", question=question):
            for round_no in range(1, max(1, self.max_iterations) + 1):
                if not pending:
                    break
                iterations = round_no
                payload_tasks = []
                retrieval_performed_tasks: set[str] = set()
                with tracer.span("plan", round=round_no):
                    for tid in list(pending):
                        task = task_by_id[tid]
                        failed = verified.get(tid)
                        # Retrieve on the initial attempt and only when verification
                        # says the available evidence is insufficient. Contract,
                        # citation, and comparison repairs reuse the current corpus view.
                        retrieve_again = failed is None or failed.failure_type == "insufficient_evidence"
                        fact_ids = set(failed.missing_fact_ids) if failed else set()
                        target_facts = [f for f in task.required_facts if not fact_ids or f.id in fact_ids]
                        for fact in target_facts if retrieve_again else []:
                            retrieval_performed_tasks.add(tid)
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
                                "entity": fact.entity, "entity_aliases": fact.entity_aliases,
                                "attribute": fact.attribute, "verifiable": fact.verifiable,
                                "query": fact.retrieval_query,
                                "chunks": [{"chunk_id": r.chunk.id, "text": r.chunk.text} for r in chunk_list]})
                        td = _task_dict(task)
                        td["evidence"] = evidence_groups
                        if failed:
                            td["retry_guidance"] = {
                                "failure_type": failed.failure_type,
                                "reason": failed.reason,
                                "action": _retry_action(failed.failure_type),
                                "retrieval_performed": retrieve_again,
                            }
                        payload_tasks.append(td)

                with tracer.span("generate", round=round_no):
                    try:
                        result = self.llm.complete(_SYSTEM, _build_prompt(question, payload_tasks))
                        tracer.record_usage(self.model, result.usage)
                    except Exception as exc:
                        reason = f"Model generation failed: {type(exc).__name__}: {exc}"
                        for tid in list(pending):
                            verified[tid] = TaskAnswer(
                                tid, "execution_failed", "I couldn't complete verification for this request.",
                                reason=reason, retries=attempts[tid], failure_type="internal_execution_failure",
                                final_state="execution_failed", error_reason=reason)
                            diagnostics.append({"round": round_no, "task_id": tid, "phase": "generation",
                                "status": "failed", "category": "internal_execution_failure",
                                "failure_type": "internal_execution_failure", "reason": reason,
                                "retry_action": "none", "retrieval_performed": tid in retrieval_performed_tasks,
                                "verifier_invoked": False})
                            pending.remove(tid)
                        break
                expected_task_ids = set(pending)
                try:
                    generated_by_id = _parse_generated_tasks(result.text, expected_task_ids)
                except LLMOutputContractError as exc:
                    for tid in list(pending):
                        attempts[tid] += 1
                        verified[tid] = TaskAnswer(tid, "needs_retry", "", [], str(exc),
                                                   attempts[tid], failure_type="output_contract")
                        diagnostics.append({"round": round_no, "task_id": tid, "phase": "output_contract",
                                            "status": "failed", "failure_type": "output_contract",
                                            "reason": str(exc), "retry_action": "regenerate_from_existing_evidence",
                                            "retrieval_performed": tid in retrieval_performed_tasks,
                                            "verifier_invoked": False})
                    continue

                for tid in list(pending):
                    task = task_by_id[tid]
                    raw = generated_by_id.get(tid, {})
                    per_task_chunks = {cid: item.chunk.text for fact_chunks in retrieved[tid].values()
                                       for cid, item in fact_chunks.items()}
                    retrieved_by_fact = {
                        fid: {cid: item.chunk.text for cid, item in fact_chunks.items()}
                        for fid, fact_chunks in retrieved[tid].items()
                    }
                    candidate = raw
                    generated_answer = str(raw.get("answer", ""))
                    binding_error = ""
                    binding_repairs = []
                    if raw.get("status") == "supported":
                        binding = bind_evidence_references(_task_dict(task), raw, retrieved_by_fact)
                        candidate, binding_error = binding.candidate, binding.reason
                        binding_repairs = binding.repairs or []
                    if binding_error:
                        verification = EvidenceVerification("needs_retry", str(raw.get("answer", "")),
                                                            [], binding_error,
                                                            failure_type=binding.failure_type or "invalid_citation")
                        phase = "binding"
                    else:
                        # Structured claims are the only candidate source for answer text.
                        # This repairs serialization drift without another generation call;
                        # all claims are still independently checked by the verifier below.
                        candidate = dict(candidate)
                        claims = candidate.get("claims")
                        if candidate.get("status") == "supported" and isinstance(claims, list) and all(
                                isinstance(c, dict) and isinstance(c.get("text"), str) for c in claims):
                            ordered_claims = _ordered_claims(task, claims)
                            assembled = "\n".join(str(c["text"]).strip() for c in ordered_claims)
                            candidate["claims"] = ordered_claims
                            candidate["answer"] = assembled
                        try:
                            verification = verify_task_answer(_task_dict(task), candidate, per_task_chunks,
                                                              require_claims=True,
                                                              retrieved_by_fact=retrieved_by_fact)
                        except Exception as exc:
                            verification = EvidenceVerification("needs_retry", generated_answer, [],
                                f"Verifier execution failed: {type(exc).__name__}: {exc}",
                                failure_type="internal_execution_failure")
                        phase = "verifier"
                        if verification.status == "needs_retry" and not verification.failure_type:
                            verification.failure_type = classify_verification_failure(verification.reason)
                    category = _failure_category(phase, verification.status, verification.failure_type)
                    diagnostics.append({"round": round_no, "task_id": tid, "phase": phase,
                                        "status": verification.status,
                                        "category": category,
                                        "failure_type": verification.failure_type,
                                        "reason": verification.reason,
                                        "generated_answer": generated_answer,
                                        "assembled_answer": str(candidate.get("answer", "")),
                                        "serialization_repaired": generated_answer != str(candidate.get("answer", "")),
                                        "binding_repairs": binding_repairs,
                                        "retry_action": _retry_action(verification.failure_type),
                                        "retrieval_performed": tid in retrieval_performed_tasks,
                                        "verifier_invoked": phase == "verifier"})
                    if verification.failure_type == "internal_execution_failure":
                        verified[tid] = TaskAnswer(tid, "execution_failed",
                            "I couldn't complete verification for this request.", [], verification.reason,
                            attempts[tid], failure_type="internal_execution_failure",
                            final_state="execution_failed", original_answer=generated_answer,
                            error_reason=verification.reason)
                        pending.remove(tid)
                        continue
                    if verification.status in ("supported", "not_in_corpus"):
                        valid_claims = []
                        if verification.status == "supported":
                            for claim in candidate.get("claims", []):
                                valid_claims.append(Claim(
                                    claim_id=str(claim.get("claim_id", "")),
                                    entity=str(claim.get("entity", "unknown")),
                                    attribute=str(claim.get("attribute", "general")),
                                    text=str(claim.get("text", "")),
                                    fact_ids=[str(fid) for fid in claim.get("fact_ids", [])],
                                    evidence_refs=[EvidenceRef(**ref) for ref in claim.get("evidence_refs", [])],
                                ))
                        outcome = "answered" if verification.status == "supported" else "not_in_corpus"
                        verified[tid] = TaskAnswer(tid, verification.status, verification.answer,
                            [EvidenceRef(**ref) for ref in verification.evidence_refs], verification.reason,
                            attempts[tid], claims=valid_claims, final_state=outcome,
                            original_answer=generated_answer)
                        pending.remove(tid)
                    else:
                        attempts[tid] += 1
                        verified[tid] = TaskAnswer(tid, "needs_retry", raw.get("answer", ""),
                            [EvidenceRef(**ref) for ref in verification.evidence_refs],
                            verification.reason, attempts[tid], verification.missing_fact_ids or [],
                            failure_type=verification.failure_type or classify_verification_failure(verification.reason))

        # Never expose an unverified candidate when the bounded retry budget is exhausted.
        for tid in pending:
            task = task_by_id[tid]
            previous = verified.get(tid)
            reason = previous.reason if previous else "Verification did not pass"
            failure_type = previous.failure_type if previous else "retry_exhausted"
            evidence_absent = failure_type == "insufficient_evidence"
            state = "not_in_corpus" if evidence_absent else "execution_failed"
            answer = (f"The available evidence does not answer: {task.question}"
                      if evidence_absent else "I couldn't verify an answer from the available evidence.")
            verified[tid] = TaskAnswer(tid, state, answer, [], reason, attempts[tid],
                failure_type=failure_type, final_state=state, error_reason="" if evidence_absent else reason,
                original_answer=previous.answer if previous else "")
            diagnostics.append({"task_id": tid, "phase": "recovery", "status": state,
                "category": "insufficient_evidence" if evidence_absent else "retry_exhausted",
                "failure_type": "insufficient_evidence" if evidence_absent else "retry_exhausted", "reason": reason,
                "underlying_failure_type": failure_type, "retry_action": "none",
                "retrieval_performed": False, "verifier_invoked": False})
        ordered = [verified[t.id] for t in tasks]
        answer_parts = [ta.answer for ta in ordered]
        final_answer = "\n".join(answer_parts)
        citations = list(dict.fromkeys(ref.chunk_id for ta in ordered if ta.status == "supported" for ref in ta.evidence_refs))
        all_contexts = [r.chunk.text for h in hops for r in h.contexts]
        faith = grounding_score(final_answer, all_contexts)
        final_state = ("execution_failed" if any(t.final_state == "execution_failed" for t in ordered)
                       else "answered" if any(t.final_state == "answered" for t in ordered)
                       else "not_in_corpus")
        return AgentAnswer(question, final_answer, hops, faith, iterations, citations, ordered, diagnostics,
                           final_state)


def _ordered_claims(task: PlannedTask, claims: list[dict]) -> list[dict]:
    """Stable task-fact ordering; a comparison conclusion follows its operands."""
    fact_order = {fact.id: index for index, fact in enumerate(task.required_facts)}
    indexed = list(enumerate(claims))
    return [claim for _, claim in sorted(indexed, key=lambda pair: (
        pair[1].get("entity") == "comparison",
        min((fact_order.get(str(fid), len(fact_order)) for fid in pair[1].get("fact_ids", [])),
            default=len(fact_order)), pair[0]))]


def _failure_category(phase: str, status: str, failure_type: str) -> str:
    if status in ("supported", "not_in_corpus"):
        return "verified_conclusion"
    if failure_type in ("output_contract", "output_contract_failure"):
        return "output_contract_failure"
    if phase == "binding" or failure_type in ("missing_evidence_reference", "invalid_citation",
            "ambiguous_citation", "incorrect_fact_association"):
        return "evidence_binding_failure"
    if failure_type == "insufficient_evidence":
        return "insufficient_evidence"
    if failure_type == "internal_execution_failure":
        return "internal_execution_failure"
    return "claim_verification_failure"


def _retry_action(failure_type: str) -> str:
    if not failure_type:
        return "none"
    if failure_type in ("output_contract", "missing_evidence_reference", "invalid_citation",
                        "ambiguous_citation", "incorrect_fact_association", "comparison_structure",
                        "entity_attribute_mismatch"):
        return "regenerate_from_existing_evidence"
    if failure_type == "insufficient_evidence":
        return "reformulate_retrieval_query"
    if failure_type in ("unsupported_claim", "contradictory_evidence"):
        return "recheck_claim_against_existing_evidence"
    return "retry_with_existing_evidence"
