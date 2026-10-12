"""Held-out deterministic reliability checks for V3.2's planning and evidence contract."""
from dataclasses import asdict

import pytest

from production_agentic_rag.agents.planner import plan
from production_agentic_rag.guardrails import bind_evidence_references, verify_task_answer
from production_agentic_rag import RAGPipeline, Settings


def _task(question, facts, comparison=None):
    planned = next(t for t in plan(question) if t.comparison == comparison) if comparison else plan(question)[0]
    return {"id": planned.id, "question": planned.question,
            "required_facts": [asdict(f) for f in facts], "comparison": comparison}


def _candidate(task, claim_specs, answer=None, status="supported"):
    claims = []
    refs = []
    for entity, attribute, text, fact_ids, evidence_refs in claim_specs:
        claims.append({"claim_id": f"claim_{len(claims)}", "entity": entity,
                       "attribute": attribute, "text": text, "fact_ids": fact_ids,
                       "evidence_refs": evidence_refs})
        refs.extend(evidence_refs)
    return {"task_id": task["id"], "status": status,
            "answer": answer if answer is not None else "\n".join(c["text"] for c in claims),
            "claims": claims, "evidence_refs": refs}


def _resolved(task, candidate, by_fact):
    binding = bind_evidence_references(task, candidate, by_fact)
    assert not binding.reason, binding.reason
    flat = {cid: text for chunks in by_fact.values() for cid, text in chunks.items()}
    result = verify_task_answer(task, binding.candidate, flat, require_claims=True,
                                retrieved_by_fact=by_fact)
    return binding.candidate, result


def _handbook_pipeline():
    pipeline = RAGPipeline(Settings(embed_provider="hashing", top_k=4, candidate_k=12,
                                    rerank=True, max_iterations=2))
    from pathlib import Path
    corpus = Path(__file__).resolve().parents[1] / "data" / "corpus" / "handbook.md"
    pipeline.add_document(corpus.read_text(encoding="utf-8"), doc_id="handbook")
    return pipeline


def test_planner_emits_meaningful_entity_attribute_and_query_across_domains():
    cases = [
        ("Where do I find signal information?", "signal_information", "path"),
        ("Who is the director of 'Just Call Out My Name'?", "just_call_out_my_name", "director"),
        ("What is the return window?", "return_eligibility", "duration"),
        ("Does support ask for passwords over the phone?", "support_password_policy", "policy"),
    ]
    for question, entity, attribute in cases:
        fact = plan(question)[0].required_facts[0]
        assert (fact.entity, fact.attribute) == (entity, attribute)
        assert fact.verifiable is True
        assert entity.replace("_", " ").split()[0] in fact.retrieval_query.lower()


def test_unresolved_attribute_is_explicit_and_cannot_certify_supported_answer():
    task_plan = plan("What color is the fictional Zorblax device?")[0]
    fact = task_plan.required_facts[0]
    assert fact.entity != "unknown"
    assert fact.attribute == "unresolved"
    assert not fact.verifiable
    task = {"id": task_plan.id, "question": task_plan.question,
            "required_facts": [asdict(fact)], "comparison": None}
    quote = "The fictional Zorblax device is blue."
    ref = {"fact_id": fact.id, "chunk_id": "c:0", "quote": quote}
    candidate = _candidate(task, [(fact.entity, fact.attribute, quote, [fact.id], [ref])])
    _, result = _resolved(task, candidate, {fact.id: {"c:0": quote}})
    assert result.status == "needs_retry"
    assert "could not normalize" in result.reason.lower()


def test_coordinated_setup_and_transfer_actions_become_independent_facts():
    tasks = plan("How do I choose a wallpaper for Ambient Mode and import photos from my phone?")
    facts = [task.required_facts[0] for task in tasks]
    assert [(fact.attribute, fact.entity) for fact in facts] == [
        ("wallpaper_selection", "ambient_mode"),
        ("photo_import", "mobile_photo_import"),
    ]
    assert facts[0].id != facts[1].id
    assert "wallpaper" in facts[0].retrieval_query.lower()
    assert "import" in facts[1].retrieval_query.lower()


@pytest.mark.parametrize("entity,attribute,text", [
    ("express_shipping", "duration", "Standard shipping takes 3-5 business days."),
    ("standard_shipping", "coverage", "Standard shipping takes 3-5 business days."),
])
def test_strict_claim_entity_and_attribute_mismatches_remain_rejected(entity, attribute, text):
    fact = plan("Does standard shipping take longer than express shipping?")[0].required_facts[0]
    task = {"id": "t", "question": "What is standard shipping duration?",
            "required_facts": [asdict(fact)], "comparison": None}
    ref = {"fact_id": fact.id, "chunk_id": "c:0", "quote": text}
    candidate = _candidate(task, [(entity, attribute, text, [fact.id], [ref])])
    # The direct verifier remains strict; the task's question string does not
    # override the actual planned entity/attribute contract.
    result = verify_task_answer(task, candidate, {"c:0": text}, require_claims=True)
    assert result.status == "needs_retry"
    assert "entity or attribute" in result.reason.lower()


@pytest.mark.parametrize("question,expected", [
    ("Does standard shipping take longer than express shipping?", ("standard_shipping", "express_shipping", "duration", "longer")),
    ("Does express delivery take longer than standard delivery?", ("express_shipping", "standard_shipping", "duration", "longer")),
    ("Are Northbridge and Southport in the same neighborhood?", ("northbridge", "southport", "neighborhood", "same")),
    ("Were Ava Stone and Ben Lake of different nationalities?", ("ava_stone", "ben_lake", "nationality", "different")),
])
def test_comparison_plans_two_ordered_operands_and_explicit_operation(question, expected):
    planned = plan(question)
    assert len(planned) == 1
    task = planned[0]
    assert task.comparison is not None
    left, right = task.required_facts
    assert (left.entity, right.entity, task.comparison["attribute"], task.comparison["operator"]) == expected
    assert task.comparison["left_fact_id"] == left.id == "operand_a"
    assert task.comparison["right_fact_id"] == right.id == "operand_b"
    assert left.retrieval_query != right.retrieval_query


def test_unsupported_comparison_attribute_uses_non_certifying_fallback():
    task = plan("Are Ava Stone and Ben Lake the same age?")[0]
    assert task.comparison is None
    assert any(not fact.verifiable for fact in task.required_facts)


def test_comparison_verifier_accepts_supported_same_relation_only_with_both_operands():
    planned = plan("Were Scott Derrickson and Ed Wood of the same nationality?")[0]
    task = asdict(planned)
    a, b = planned.required_facts
    qa = "Scott Derrickson is an American director."
    qb = "Ed Wood was an American filmmaker."
    ra = {"fact_id": a.id, "chunk_id": "a:0", "quote": qa}
    rb = {"fact_id": b.id, "chunk_id": "b:0", "quote": qb}
    specs = [
        (a.entity, a.attribute, qa, [a.id], [ra]),
        (b.entity, b.attribute, qb, [b.id], [rb]),
        ("comparison", "same_different_relation", "Scott Derrickson and Ed Wood have the same nationality.",
         [a.id, b.id], [ra, rb]),
    ]
    candidate = _candidate(task, specs)
    _, result = _resolved(task, candidate, {a.id: {"a:0": qa}, b.id: {"b:0": qb}})
    assert result.status == "supported"


def test_comparison_verifier_rejects_missing_operand_and_wrong_relation():
    planned = plan("Were Ava Stone and Ben Lake of the same nationality?")[0]
    task = asdict(planned)
    a, b = planned.required_facts
    qa = "Ava Stone is an American engineer."
    qb = "Ben Lake was a Canadian pilot."
    ra = {"fact_id": a.id, "chunk_id": "a:0", "quote": qa}
    rb = {"fact_id": b.id, "chunk_id": "b:0", "quote": qb}
    only_a = _candidate(task, [(a.entity, a.attribute, qa, [a.id], [ra]),
                              ("comparison", "same_different_relation", "They have the same nationality.",
                               [a.id, b.id], [ra])])
    result = verify_task_answer(task, only_a, {"a:0": qa}, require_claims=True,
                                retrieved_by_fact={a.id: {"a:0": qa}, b.id: {}})
    assert result.status == "needs_retry"
    assert "operand" in result.reason.lower() or "evidence" in result.reason.lower()

    wrong = _candidate(task, [
        (a.entity, a.attribute, qa, [a.id], [ra]),
        (b.entity, b.attribute, qb, [b.id], [rb]),
        ("comparison", "same_different_relation", "Ava Stone and Ben Lake have the same nationality.",
         [a.id, b.id], [ra, rb]),
    ])
    _, result = _resolved(task, wrong, {a.id: {"a:0": qa}, b.id: {"b:0": qb}})
    assert result.status == "needs_retry"
    assert "comparison conclusion" in result.reason.lower()


@pytest.mark.parametrize("wording", [
    "There is no evidence to determine the return window.",
    "The return window is not specified in the available text.",
    "I cannot determine the return window from these passages.",
])
def test_answerable_fact_with_paraphrased_abstention_is_rejected_when_evidence_is_sufficient(wording):
    fact = plan("What is the return window?")[0].required_facts[0]
    task = {"id": "task_1", "question": "What is the return window?",
            "required_facts": [asdict(fact)], "comparison": None}
    evidence = "Return any unused item within 30 days for a full refund."
    result = verify_task_answer(task, {"status": "not_in_corpus", "answer": wording,
                                       "claims": [], "evidence_refs": []},
                                {"c:0": evidence}, require_claims=True,
                                retrieved_by_fact={fact.id: {"c:0": evidence}})
    assert result.status == "needs_retry"
    assert "abstention is not justified" in result.reason.lower()


@pytest.mark.parametrize("wording", [
    "There is no evidence to determine the requested fact.",
    "That information is not specified in the supplied passages.",
    "I cannot determine it from the available corpus.",
    "The evidence does not provide information on this fact.",
    "No evidence found for this requested detail.",
])
def test_genuinely_unsupported_fact_accepts_varied_clear_refusals(wording):
    fact = plan("What is the return window?")[0].required_facts[0]
    task = {"id": "task_1", "question": "What is the return window?",
            "required_facts": [asdict(fact)], "comparison": None}
    irrelevant = "Electronics have a one-year limited warranty."
    result = verify_task_answer(task, {"status": "not_in_corpus", "answer": wording,
                                       "claims": [], "evidence_refs": []},
                                {"w:0": irrelevant}, require_claims=True,
                                retrieved_by_fact={fact.id: {"w:0": irrelevant}})
    assert result.status == "not_in_corpus"


def test_exact_quote_binding_preserves_original_chunk_and_span():
    fact = plan("What is the return window?")[0].required_facts[0]
    task = {"id": "task_1", "question": "What is the return window?",
            "required_facts": [asdict(fact)], "comparison": None}
    source = "Returns are accepted within 30 days."
    ref = {"fact_id": fact.id, "chunk_id": "c:0", "quote": source}
    candidate = _candidate(task, [(fact.entity, fact.attribute, source, [fact.id], [ref])])
    bound, result = _resolved(task, candidate, {fact.id: {"c:0": source}})
    assert result.status == "supported"
    assert bound["evidence_refs"][0] == ref


def test_unique_punctuation_and_whitespace_difference_resolves_to_exact_source_substring():
    fact = plan("How do I enter Ambient Mode?")[0].required_facts[0]
    task = {"id": "task_1", "question": "How do I enter Ambient Mode?",
            "required_facts": [asdict(fact)], "comparison": None}
    source = "To enter Ambient Mode , press the button."
    ref = {"fact_id": fact.id, "chunk_id": "actual:0", "quote": "To enter Ambient Mode, press the button"}
    candidate = _candidate(task, [(fact.entity, fact.attribute, source, [fact.id], [ref])])
    bound = bind_evidence_references(task, candidate, {fact.id: {"actual:0": source}})
    assert not bound.reason
    actual = bound.candidate["evidence_refs"][0]
    assert actual["chunk_id"] == "actual:0"
    assert actual["quote"] in source
    assert actual["quote"] == "To enter Ambient Mode , press the button"


def test_noncontiguous_combined_quote_is_rejected_but_separate_refs_bind():
    fact = plan("How do I enter Ambient Mode?")[0].required_facts[0]
    task = {"id": "task_1", "question": "How do I enter Ambient Mode?",
            "required_facts": [asdict(fact)], "comparison": None}
    source = "First source statement. Intervening unrelated text. Second source statement."
    combined = {"fact_id": fact.id, "chunk_id": "c:0",
                "quote": "First source statement. Second source statement."}
    candidate = _candidate(task, [(fact.entity, fact.attribute, "First source statement. Second source statement.",
                                   [fact.id], [combined])])
    binding = bind_evidence_references(task, candidate, {fact.id: {"c:0": source}})
    assert "non-contiguous" in binding.reason.lower()

    separate = [dict(combined, quote="First source statement."),
                dict(combined, quote="Second source statement.")]
    candidate = _candidate(task, [(fact.entity, fact.attribute, "First source statement. Second source statement.",
                                   [fact.id], separate)])
    binding = bind_evidence_references(task, candidate, {fact.id: {"c:0": source}})
    assert not binding.reason
    refs = binding.candidate["evidence_refs"]
    assert len(refs) == 2
    assert all(ref["quote"] in source for ref in refs)
    assert all("Intervening" not in ref["quote"] for ref in refs)


def test_ambiguous_quote_fails_closed():
    fact = plan("What is the return window?")[0].required_facts[0]
    task = {"id": "task_1", "question": "What is the return window?",
            "required_facts": [asdict(fact)], "comparison": None}
    quote = "Returns accepted within 30 days"
    ref = {"fact_id": fact.id, "chunk_id": "a:0", "quote": quote}
    candidate = _candidate(task, [(fact.entity, fact.attribute, quote, [fact.id], [ref])])
    binding = bind_evidence_references(task, candidate,
        {fact.id: {"a:0": "Returns, accepted within 30 days. Returns accepted, within 30 days.",
                   "b:0": quote}})
    assert "ambiguous" in binding.reason.lower()


def test_invalid_chunk_id_rebinds_only_when_source_span_is_unique_for_fact():
    fact = plan("What is the return window?")[0].required_facts[0]
    task = {"id": "task_1", "question": "What is the return window?",
            "required_facts": [asdict(fact)], "comparison": None}
    source = "Returns are accepted within 30 days."
    candidate = _candidate(task, [(fact.entity, fact.attribute, source, [fact.id], [
        {"fact_id": fact.id, "chunk_id": "returns_other:0", "quote": source}])])
    binding = bind_evidence_references(task, candidate, {fact.id: {"returns_0:0": source}})
    assert binding.failure_type == ""
    assert binding.repairs == [{"kind": "invalid_chunk_id_rebound_by_unique_source_span",
                                "fact_id": fact.id, "chunk_id": "returns_0:0"}]

    bad = _candidate(task, [(fact.entity, fact.attribute, source, [fact.id], [
        {"fact_id": fact.id, "chunk_id": "missing:0", "quote": "That text is absent."}])])
    binding = bind_evidence_references(task, bad, {fact.id: {"returns_0:0": source}})
    assert "chunk id" in binding.reason.lower()


def test_evidence_cannot_be_reassigned_to_a_different_planned_fact():
    planned = plan("What is the return window and how long do refunds take?")
    task = {"id": "task_1", "question": "multi fact",
            "required_facts": [asdict(planned[0].required_facts[0]), asdict(planned[1].required_facts[0])],
            "comparison": None}
    a, b = task["required_facts"]
    quote_a = "Return any unused item within 30 days."
    ref = {"fact_id": b["id"], "chunk_id": "return:0", "quote": quote_a}
    candidate = _candidate(task, [(b["entity"], b["attribute"], quote_a, [b["id"]], [ref])])
    binding = bind_evidence_references(task, candidate, {
        a["id"]: {"return:0": quote_a}, b["id"]: {"refund:0": "Approved refunds are issued within 5 business days."}})
    assert "chunk id" in binding.reason.lower()


def test_heldout_single_fact_pipeline_preserves_original_grounded_behavior():
    result = _handbook_pipeline().query("What does the warranty cover?")
    assert result.task_answers[0].status == "supported"
    assert "manufacturing defects" in result.answer.lower()
    assert result.citations == ["handbook:2"]


def test_heldout_multi_fact_pipeline_keeps_return_and_refund_facts_separate():
    result = _handbook_pipeline().query("What is the return window, and when does refund processing start?")
    assert [task.status for task in result.task_answers] == ["supported", "supported"]
    assert "30 days" in result.task_answers[0].answer
    assert "5 business days" not in result.task_answers[0].answer
    assert "5 business days" in result.task_answers[1].answer
    assert "received and inspected" in result.task_answers[1].answer.lower()
    assert set(result.citations) == {"handbook:0", "handbook:3"}


def test_heldout_reversed_duration_comparison_preserves_operands_and_evidence():
    result = _handbook_pipeline().query("Does express shipping take longer than standard shipping?")
    assert result.task_answers[0].status == "supported"
    assert "does not take longer" in result.answer.lower()
    assert "1-2 business days" in result.answer
    assert "3-5 business days" in result.answer
    assert set(result.citations) == {"handbook:1"}


def test_heldout_answerable_question_with_shorter_duration_wording_still_answers():
    result = _handbook_pipeline().query("How many days are returns accepted?")
    assert result.task_answers[0].status == "supported"
    assert "30 days" in result.answer
    assert result.citations == ["handbook:0"]


def test_heldout_typed_unsupported_contact_request_still_abstains():
    result = _handbook_pipeline().query("What is the support phone number?")
    assert result.task_answers[0].status == "not_in_corpus"
    assert "does not provide a phone number" in result.answer.lower()
    assert not result.citations


@pytest.mark.parametrize("question", [
    "How long must an account password be?",
    "What is the minimum password length in characters?",
    "How many characters are required for passwords?",
])
def test_password_length_is_a_password_requirement_not_a_time_duration(question):
    fact = plan(question)[0].required_facts[0]
    assert fact.entity == "support_password_policy"
    assert fact.attribute == "password_requirement"


def test_heldout_password_length_pipeline_answers_without_temporal_confusion():
    result = _handbook_pipeline().query("How long must an account password be?")
    assert result.task_answers[0].status == "supported"
    assert "12 characters" in result.answer
    assert "days" not in result.answer.lower()
