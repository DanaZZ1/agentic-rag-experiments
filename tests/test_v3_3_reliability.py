"""Deterministic regressions for the V3.3 reliability repair."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, replace

from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.agents.planner import plan
from production_agentic_rag.guardrails import bind_evidence_references, verify_task_answer
from production_agentic_rag.providers.llm import LLMResult


def _fact_task(question):
    planned = plan(question)[0]
    return planned, {**asdict(planned), "required_facts": [asdict(f) for f in planned.required_facts]}


def _candidate(fact, quote, *, claim_text=None, answer=None, refs=True):
    claim_text = claim_text or quote
    evidence_refs = ([{"fact_id": fact["id"], "chunk_id": "c:0", "quote": quote}]
                     if refs else [])
    claim = {"claim_id": "claim_1", "entity": fact["entity"],
             "attribute": fact["attribute"], "text": claim_text,
             "fact_ids": [fact["id"]], "evidence_refs": evidence_refs}
    return {"status": "supported", "answer": answer if answer is not None else claim_text,
            "claims": [claim], "evidence_refs": evidence_refs}


def _verify(question, quote, candidate):
    _, task = _fact_task(question)
    fact = task["required_facts"][0]
    ref = {"fact_id": fact["id"], "chunk_id": "c:0", "quote": quote}
    candidate["evidence_refs"] = [ref] if candidate["claims"][0]["evidence_refs"] else []
    candidate["claims"][0]["evidence_refs"] = [ref] if candidate["claims"][0]["evidence_refs"] else []
    by_fact = {fact["id"]: {"c:0": quote}}
    binding = bind_evidence_references(task, candidate, by_fact)
    if binding.reason:
        return task, binding, None
    return task, binding, verify_task_answer(task, binding.candidate, {"c:0": quote},
                                              require_claims=True, retrieved_by_fact=by_fact)


def test_valid_location_claim_accepts_alternative_linguistic_formulation():
    question = "Where will the annual meeting take place?"
    quote = "The annual meeting will be held at Civic Hall in Riverton."
    task_plan, task = _fact_task(question)
    fact = task["required_facts"][0]
    # The planner's surface entity is question-specific, so this test holds the
    # entity constant and exercises the location expression itself.
    fact["entity"] = "annual_meeting"
    fact["entity_aliases"] = ["annual meeting"]
    task["required_facts"] = [fact]
    ref = {"fact_id": fact["id"], "chunk_id": "c:0", "quote": quote}
    claim = {"claim_id": "c", "entity": "annual_meeting", "attribute": "location",
             "text": quote, "fact_ids": [fact["id"]], "evidence_refs": [ref]}
    candidate = {"status": "supported", "answer": quote, "claims": [claim], "evidence_refs": [ref]}
    by_fact = {fact["id"]: {"c:0": quote}}
    bound = bind_evidence_references(task, candidate, by_fact)
    result = verify_task_answer(task, bound.candidate, {"c:0": quote}, require_claims=True,
                                retrieved_by_fact=by_fact)
    assert result.status == "supported"


def test_director_alternative_form_is_supported_but_wrong_entity_and_attribute_fail():
    quote = 'Alex Rivera is set to direct "River Road".'
    task = {"id": "t", "question": "Who directs River Road?", "comparison": None,
            "required_facts": [{"id": "f", "entity": "river_road", "entity_aliases": ["River Road"],
                                "attribute": "director", "verifiable": True, "evidence_terms": []}]}
    ref = {"fact_id": "f", "chunk_id": "c:0", "quote": quote}
    good = {"status": "supported", "answer": quote, "evidence_refs": [ref],
            "claims": [{"claim_id": "c", "entity": "river_road", "attribute": "director",
                        "text": quote, "fact_ids": ["f"], "evidence_refs": [ref]}]}
    assert verify_task_answer(task, good, {"c:0": quote}, require_claims=True).status == "supported"
    for entity, attribute in (("another_film", "director"), ("river_road", "location")):
        wrong = {**good, "claims": [{**good["claims"][0], "entity": entity, "attribute": attribute}]}
        assert verify_task_answer(task, wrong, {"c:0": quote}, require_claims=True).status == "needs_retry"


def test_wrong_location_claim_and_contradictory_source_values_fail_closed():
    question = "Where is Northport Arena?"
    quote = "Northport Arena will host the final in Tampa, Florida."
    planned, task = _fact_task(question)
    fact = task["required_facts"][0]
    # Explicitly pin the intended entity/attribute; this test targets value
    # consistency, not the planner's entity extraction.
    fact.update(entity="northport_arena", entity_aliases=["Northport Arena"], attribute="location", verifiable=True)
    task["required_facts"] = [fact]
    ref = {"fact_id": fact["id"], "chunk_id": "c:0", "quote": quote}
    claim_text = "Northport Arena will host the final in Glendale, Arizona."
    claim = {"claim_id": "c", "entity": fact["entity"], "attribute": "location", "text": claim_text,
             "fact_ids": [fact["id"]], "evidence_refs": [ref]}
    candidate = {"status": "supported", "answer": claim_text, "claims": [claim], "evidence_refs": [ref]}
    result = verify_task_answer(task, candidate, {"c:0": quote}, require_claims=True)
    assert result.status == "needs_retry"
    assert result.failure_type == "contradictory_evidence"


def test_binding_reports_missing_reference_wrong_chunk_and_wrong_fact():
    _, task = _fact_task("What is the return window?")
    fact = task["required_facts"][0]
    quote = "Returns are accepted within 30 days."
    by_fact = {fact["id"]: {"c:0": quote}}
    no_ref = _candidate(fact, quote, refs=False)
    rebound = bind_evidence_references(task, no_ref, by_fact)
    assert rebound.failure_type == ""
    assert rebound.repairs[0]["kind"] == "missing_reference_recovered_from_claim_span"
    wrong_chunk = _candidate(fact, quote)
    wrong_chunk["claims"][0]["evidence_refs"][0]["chunk_id"] = "invented:0"
    wrong_chunk["evidence_refs"][0]["chunk_id"] = "invented:0"
    bound = bind_evidence_references(task, wrong_chunk, by_fact)
    assert bound.failure_type == ""
    assert bound.repairs[0]["kind"] == "invalid_chunk_id_rebound_by_unique_source_span"
    wrong_span = _candidate(fact, quote)
    wrong_span["claims"][0]["evidence_refs"][0]["quote"] = "Text absent from this chunk."
    bound = bind_evidence_references(task, wrong_span, by_fact)
    assert bound.failure_type == "invalid_citation"
    wrong_fact = _candidate(fact, quote)
    wrong_fact["claims"][0]["evidence_refs"][0]["fact_id"] = "unplanned"
    bound = bind_evidence_references(task, wrong_fact, by_fact)
    assert bound.failure_type == "incorrect_fact_association"


def _comparison_candidate(task_plan, qa, qb, relation, *, comparison_refs=True, answer_include=True):
    a, b = task_plan.required_facts
    ra = {"fact_id": a.id, "chunk_id": "a:0", "quote": qa}
    rb = {"fact_id": b.id, "chunk_id": "b:0", "quote": qb}
    comparison_text = f"{task_plan.comparison['left_label']} and {task_plan.comparison['right_label']} have {relation}."
    claims = [
        {"claim_id": "a", "entity": a.entity, "attribute": a.attribute, "text": qa,
         "fact_ids": [a.id], "evidence_refs": [ra]},
        {"claim_id": "b", "entity": b.entity, "attribute": b.attribute, "text": qb,
         "fact_ids": [b.id], "evidence_refs": [rb]},
        {"claim_id": "cmp", "entity": "comparison", "attribute": task_plan.comparison["claim_attribute"],
         "text": comparison_text, "fact_ids": [a.id, b.id],
         "evidence_refs": [ra, rb] if comparison_refs else []},
    ]
    answer = "\n".join(c["text"] for c in claims) if answer_include else "\n".join(c["text"] for c in claims[:2])
    refs = [ra, rb] + ([ra, rb] if comparison_refs else [])
    return {"status": "supported", "answer": answer, "claims": claims, "evidence_refs": refs}


def test_comparison_requires_conclusion_and_references_and_checks_same_different_values():
    same_plan = plan("Were Ada North and Ben Lake of the same nationality?")[0]
    qa, qb = "Ada North is an American pilot.", "Ben Lake was an American engineer."
    by_fact = {"operand_a": {"a:0": qa}, "operand_b": {"b:0": qb}}
    task = asdict(same_plan)
    missing_conclusion = _comparison_candidate(same_plan, qa, qb, "the same nationality", answer_include=False)
    result = verify_task_answer(task, missing_conclusion, {"a:0": qa, "b:0": qb}, require_claims=True,
                                retrieved_by_fact=by_fact)
    assert result.failure_type == "output_contract"
    missing_refs = _comparison_candidate(same_plan, qa, qb, "the same nationality", comparison_refs=False)
    binding = bind_evidence_references(task, missing_refs, by_fact)
    assert binding.failure_type == ""
    assert any(r["kind"] == "claim_reference_reused_from_task_reference" for r in binding.repairs)
    assert len(binding.candidate["claims"][2]["evidence_refs"]) == 2
    good = _comparison_candidate(same_plan, qa, qb, "the same nationality")
    binding = bind_evidence_references(task, good, by_fact)
    result = verify_task_answer(task, binding.candidate, {"a:0": qa, "b:0": qb}, require_claims=True,
                                retrieved_by_fact=by_fact)
    assert result.status == "supported"


def test_conflicting_operand_values_are_not_reduced_to_the_last_match():
    planned = plan("Were Ada North and Ben Lake of the same nationality?")[0]
    a, b = planned.required_facts
    qa = "Ada North was American and Canadian during her career."
    qb = "Ben Lake was American."
    candidate = _comparison_candidate(planned, qa, qb, "the same nationality")
    by_fact = {"operand_a": {"a:0": qa}, "operand_b": {"b:0": qb}}
    bound = bind_evidence_references(asdict(planned), candidate, by_fact)
    result = verify_task_answer(asdict(planned), bound.candidate,
                                {"a:0": qa, "b:0": qb}, require_claims=True, retrieved_by_fact=by_fact)
    assert result.status == "needs_retry"
    assert result.failure_type == "contradictory_evidence"

    different_plan = plan("Were Ada North and Ben Lake of different nationalities?")[0]
    qa, qb = "Ada North is an American pilot.", "Ben Lake was a Canadian engineer."
    by_fact = {"operand_a": {"a:0": qa}, "operand_b": {"b:0": qb}}
    candidate = _comparison_candidate(different_plan, qa, qb, "different nationalities")
    binding = bind_evidence_references(asdict(different_plan), candidate, by_fact)
    result = verify_task_answer(asdict(different_plan), binding.candidate,
                                {"a:0": qa, "b:0": qb}, require_claims=True, retrieved_by_fact=by_fact)
    assert result.status == "supported"


class _RepairLLM:
    def __init__(self, failure):
        self.failure = failure
        self.calls = 0
        self.retry_guidance = []

    def complete(self, system, user):
        payload = json.loads(user.split("\n", 1)[1])
        task = payload["tasks"][0]
        self.retry_guidance.append(task.get("retry_guidance"))
        fact = task["required_facts"][0]
        chunks = task["evidence"][0]["chunks"]
        if self.calls == 0 and self.failure == "insufficient":
            quote = chunks[0]["text"]
        else:
            quote = next(text.strip() for chunk in chunks for text in re.split(r"(?<=[.!?])\s+", chunk["text"])
                         if "30 days" in text)
        ref = {"fact_id": fact["id"], "chunk_id": next(c["chunk_id"] for c in chunks if quote in c["text"]),
               "quote": quote}
        claim_refs = [] if self.calls == 0 and self.failure == "missing_ref" else [ref]
        claim_text = ("Returns are accepted within 30 days." if self.calls == 0 and self.failure == "insufficient"
                      else quote)
        answer = claim_text + (" Extra unsupported sentence." if self.calls == 0 and self.failure == "contract" else "")
        claim = {"claim_id": "c", "entity": fact["entity"], "attribute": fact["attribute"],
                 "text": claim_text, "fact_ids": [fact["id"]], "evidence_refs": claim_refs}
        candidate = {"task_id": task["id"], "status": "supported", "answer": answer,
                     "claims": [claim], "evidence_refs": claim_refs}
        self.calls += 1
        return LLMResult(json.dumps({"tasks": [candidate]}))


class _CountingTool:
    def __init__(self, tool, *, weak_first=False):
        self.tool = tool
        self.retriever = tool.retriever
        self.k, self.candidate_k = tool.k, tool.candidate_k
        self.calls = 0
        self.queries = []
        self.weak_first = weak_first

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.queries.append(args[0])
        results = self.tool(*args, **kwargs)
        if self.weak_first and self.calls == 1 and results:
            return [replace(results[0], chunk=replace(results[0].chunk, id="weak:0",
                         text="Returns are handled according to the policy."))]
        return results


def _repair_pipeline(failure, *, weak_first=False):
    pipe = RAGPipeline(Settings(embed_provider="hashing", top_k=4, candidate_k=12,
                                rerank=True, max_iterations=2))
    from pathlib import Path
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus" / "handbook.md"
    pipe.add_document(corpus.read_text(encoding="utf-8"), doc_id="handbook")
    llm = _RepairLLM(failure)
    tool = _CountingTool(pipe.agent.tool, weak_first=weak_first)
    pipe.agent.llm = llm
    pipe.agent.tool = tool
    return pipe, llm, tool


def test_answer_contract_is_repaired_without_regeneration_or_retrieval():
    pipe, llm, tool = _repair_pipeline("contract")
    result = pipe.query("What is the return window?")
    assert result.task_answers[0].status == "supported"
    assert result.final_state == "answered"
    assert result.task_answers[0].original_answer.endswith("Extra unsupported sentence.")
    assert "Extra unsupported sentence" not in result.answer
    assert llm.calls == 1 and tool.calls == 1
    assert result.diagnostics[0]["serialization_repaired"] is True
    assert result.diagnostics[0]["verifier_invoked"] is True


def test_missing_reference_repairs_from_unique_claim_span_without_regeneration():
    pipe, llm, tool = _repair_pipeline("missing_ref")
    result = pipe.query("What is the return window?")
    assert result.task_answers[0].status == "supported"
    assert llm.calls == 1 and tool.calls == 1
    assert result.diagnostics[0]["binding_repairs"][0]["kind"] == "missing_reference_recovered_from_claim_span"
    assert result.diagnostics[0]["verifier_invoked"] is True


def test_insufficient_evidence_triggers_a_query_retry_and_retrieval():
    pipe, llm, tool = _repair_pipeline("insufficient", weak_first=True)
    result = pipe.query("What is the return window?")
    assert result.task_answers[0].status == "supported"
    assert llm.calls == 2 and tool.calls == 2
    assert tool.queries[0] != tool.queries[1]
    assert result.diagnostics[0]["failure_type"] == "insufficient_evidence"
    assert result.diagnostics[1]["retrieval_performed"] is True
    assert llm.retry_guidance[1]["action"] == "reformulate_retrieval_query"


class _NeverValidLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        payload = json.loads(user.split("\n", 1)[1])
        task = payload["tasks"][0]
        fact = task["required_facts"][0]
        chunks = task["evidence"][0]["chunks"]
        quote = next(text.strip() for chunk in chunks for text in re.split(r"(?<=[.!?])\s+", chunk["text"])
                     if "30 days" in text)
        ref = {"fact_id": fact["id"], "chunk_id": next(c["chunk_id"] for c in chunks if quote in c["text"]),
               "quote": quote}
        claim = {"claim_id": "bad", "entity": "unrelated_entity", "attribute": fact["attribute"],
                 "text": quote, "fact_ids": [fact["id"]], "evidence_refs": [ref]}
        self.calls += 1
        return LLMResult(json.dumps({"tasks": [{"task_id": task["id"], "status": "supported",
            "answer": quote, "claims": [claim], "evidence_refs": [ref]}]}))


def test_retry_exhaustion_with_available_evidence_is_execution_failure_not_abstention():
    pipe, _, tool = _repair_pipeline("contract")
    llm = _NeverValidLLM()
    pipe.agent.llm = llm
    pipe.agent.tool = tool
    result = pipe.query("What is the return window?")
    assert llm.calls == 2 and tool.calls == 1
    assert result.final_state == "execution_failed"
    assert result.task_answers[0].final_state == "execution_failed"
    assert "not present in the corpus" not in result.answer.lower()
    assert result.diagnostics[-1]["category"] == "retry_exhausted"


def test_held_out_synthetic_entity_attribute_and_paraphrase_are_verified():
    question = "Where will the Lumen Forum be staged?"
    quote = "Organizers will host the Lumen Forum at Cedar Pavilion."
    _, task = _fact_task(question)
    fact = task["required_facts"][0]
    fact.update(entity="lumen_forum", entity_aliases=["Lumen Forum"], attribute="location", verifiable=True)
    task["required_facts"] = [fact]
    ref = {"fact_id": fact["id"], "chunk_id": "heldout:0", "quote": quote}
    claim = {"claim_id": "h", "entity": "lumen_forum", "attribute": "location", "text": quote,
             "fact_ids": [fact["id"]], "evidence_refs": [ref]}
    candidate = {"status": "supported", "answer": quote, "claims": [claim],
                 "evidence_refs": [ref]}
    bound = bind_evidence_references(task, candidate, {fact["id"]: {"heldout:0": quote}})
    result = verify_task_answer(task, bound.candidate, {"heldout:0": quote}, require_claims=True,
                                retrieved_by_fact={fact["id"]: {"heldout:0": quote}})
    assert result.status == "supported"
    assert result.answer == quote


def test_planner_splits_independent_coordinated_predicates_without_domain_question():
    tasks = plan("What does the warranty cover and exclude?")
    facts = [task.required_facts[0] for task in tasks]
    assert [(f.entity, f.attribute) for f in facts] == [
        ("warranty", "coverage"), ("warranty", "exclusions")]
