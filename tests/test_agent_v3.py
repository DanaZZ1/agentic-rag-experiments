"""Deterministic adversarial coverage for claim-aware V3 behavior."""
from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.agents.planner import plan
from production_agentic_rag.guardrails import typed_value_request, verify_task_answer


def _pipeline():
    pipe = RAGPipeline(Settings(embed_provider="hashing", top_k=4, candidate_k=12,
                                rerank=True, max_iterations=2))
    with open("data/corpus/handbook.md", encoding="utf-8") as f:
        pipe.add_document(f.read(), doc_id="handbook")
    return pipe


def test_standard_vs_express_preserves_comparison_operands():
    pipe = _pipeline()
    result = pipe.query("Does standard shipping take longer than express shipping?")
    task = result.task_answers[0]
    assert "standard shipping takes longer than express shipping" in task.answer.lower()
    assert {f.entity for f in plan(result.question)[0].required_facts} == {
        "standard_shipping", "express_shipping"}
    assert {ref.chunk_id for ref in task.evidence_refs} == {"handbook:1"}
    assert any("Standard shipping" in ref.quote for ref in task.evidence_refs)
    assert any("Express shipping" in ref.quote for ref in task.evidence_refs)
    assert task.status == "supported"


def test_express_vs_standard_reversed_operands():
    result = _pipeline().query("Does express shipping take longer than standard shipping?")
    answer = result.answer.lower()
    assert "express shipping does not take longer than standard shipping" in answer
    assert "1-2 business days" in answer and "3-5 business days" in answer


def test_refund_vs_standard_shipping_keeps_attributes_and_conditions():
    result = _pipeline().query("Does standard shipping take longer than refunds?")
    answer = result.answer.lower()
    assert "3-5 business days" in answer
    assert "5 business days" in answer
    assert "received" in answer and "inspected" in answer
    assert "overlap" in answer
    refs = result.task_answers[0].evidence_refs
    assert any("Standard shipping" in ref.quote for ref in refs)
    assert any("Approved refunds" in ref.quote for ref in refs)


def test_warranty_coordinated_coverage_claims_keep_polarity():
    question = "The warranty covers manufacturing defects and cosmetic damage, right?"
    planned = plan(question)[0]
    assert len(planned.required_facts) == 2
    result = _pipeline().query(question)
    answer = result.answer.lower()
    assert "manufacturing defects" in answer
    assert "does not cover" in answer and "cosmetic damage" in answer
    claims = result.task_answers[0].claims
    assert len(claims) == 2
    assert all(c.entity == "warranty" and c.attribute == "coverage" and c.evidence_refs for c in claims)


def test_negated_return_premise_is_corrected():
    result = _pipeline().query("Does the return policy reject items sent back within 30 days?")
    assert "return any unused item within 30 days" in result.answer.lower()
    assert "not accepted" not in result.answer.lower()
    assert result.task_answers[0].evidence_refs[0].chunk_id == "handbook:0"


def test_password_policy_over_phone_is_not_a_phone_number_request():
    question = "Does support ask for passwords over the phone?"
    assert typed_value_request(question) is None
    result = _pipeline().query(question)
    assert "never ask" in result.answer.lower()
    assert "password" in result.answer.lower() and "phone" in result.answer.lower()
    assert result.task_answers[0].status == "supported"


def test_missing_support_phone_abstains_as_typed_value_request():
    question = "What is the support phone number?"
    assert typed_value_request(question) == "phone_number"
    result = _pipeline().query(question)
    assert "does not provide a phone number" in result.answer.lower()
    assert result.task_answers[0].status == "not_in_corpus"
    assert not result.citations


def test_password_policy_and_missing_refund_email_are_separate_tasks():
    question = "Does support ask for passwords by email, and which email address should I use for refunds?"
    planned = plan(question)
    assert len(planned) == 2
    assert typed_value_request(planned[0].question) is None
    assert typed_value_request(planned[1].question) == "email_address"
    result = _pipeline().query(question)
    assert "never ask" in result.task_answers[0].answer.lower()
    assert result.task_answers[0].status == "supported"
    assert "does not provide an email address" in result.task_answers[1].answer.lower()
    assert result.task_answers[1].status == "not_in_corpus"


def test_refund_duration_and_start_condition_are_explicit_facts():
    question = "When are approved refunds issued, and what starts that period?"
    tasks = plan(question)
    assert len(tasks) == 2
    facts = [task.required_facts[0] for task in tasks]
    assert [(fact.entity, fact.attribute) for fact in facts] == [
        ("refund_processing", "duration"),
        ("refund_processing", "start_condition"),
    ]
    assert "received" in facts[1].retrieval_query.lower()
    assert "inspected" in facts[1].retrieval_query.lower()
    result = _pipeline().query(question)
    assert "5 business days" in result.task_answers[0].answer
    assert result.task_answers[1].status == "supported"
    assert "received" in result.task_answers[1].answer.lower()
    assert "inspected" in result.task_answers[1].answer.lower()


def test_return_window_and_refund_processing_do_not_share_duration_fact():
    result = _pipeline().query("What is the return window, and how long does refund processing take?")
    planned = plan(result.question)
    assert [(t.required_facts[0].entity, t.required_facts[0].attribute) for t in planned] == [
        ("return_eligibility", "duration"), ("refund_processing", "duration")]
    first, second = result.task_answers
    assert "30 days" in first.answer and "5 business days" not in first.answer
    assert "5 business days" in second.answer and "30 days" not in second.answer


def test_verifier_rejects_claim_subject_that_conflicts_with_quote():
    planned = plan("Does standard shipping take longer than express shipping?")[0]
    fact = planned.required_facts[0]
    quote = "Standard shipping takes 3-5 business days."
    ref = {"fact_id": fact.id, "chunk_id": "handbook:1", "quote": quote}
    candidate = {"status": "supported", "answer": "Express shipping takes 3-5 business days.",
                 "evidence_refs": [ref], "claims": [{"claim_id": "bad", "entity": "express_shipping",
                 "attribute": "duration", "text": "Express shipping takes 3-5 business days.",
                 "fact_ids": [fact.id], "evidence_refs": [ref]}]}
    result = verify_task_answer({"question": planned.question,
                                 "required_facts": [fact.__dict__], "comparison": None},
                                candidate, {"handbook:1": quote}, require_claims=True)
    assert result.status == "needs_retry"
    assert "entity or attribute" in result.reason or "subject" in result.reason


def test_verifier_rejects_claim_text_subject_that_disagrees_with_required_fact():
    fact = plan("Does standard shipping take longer than express shipping?")[0].required_facts[0]
    quote = "Standard shipping takes 3-5 business days."
    ref = {"fact_id": fact.id, "chunk_id": "handbook:1", "quote": quote}
    claim_text = "Express shipping takes 3-5 business days."
    candidate = {"status": "supported", "answer": claim_text, "evidence_refs": [ref],
                 "claims": [{"entity": fact.entity, "attribute": fact.attribute, "text": claim_text,
                            "fact_ids": [fact.id], "evidence_refs": [ref]}]}
    result = verify_task_answer({"question": "Does standard shipping take longer than express shipping?",
                                 "required_facts": [fact.__dict__], "comparison": None},
                                candidate, {"handbook:1": quote}, require_claims=True)
    assert result.status == "needs_retry"
    assert "subject" in result.reason.lower()


def test_verifier_rejects_comparison_claim_without_both_operands():
    planned = plan("Does standard shipping take longer than express shipping?")[0]
    left, right = planned.required_facts
    quote = "Standard shipping takes 3-5 business days."
    ref = {"fact_id": left.id, "chunk_id": "handbook:1", "quote": quote}
    candidate = {"status": "supported", "answer": "Standard shipping takes longer than express shipping.",
                 "evidence_refs": [ref], "claims": [{"claim_id": "cmp", "entity": "comparison",
                 "attribute": "duration_relation", "text": "Standard shipping takes longer than express shipping.",
                 "fact_ids": [left.id, right.id], "evidence_refs": [ref]}]}
    result = verify_task_answer({"question": planned.question,
                                 "required_facts": [left.__dict__, right.__dict__],
                                 "comparison": planned.comparison},
                                candidate, {"handbook:1": quote}, require_claims=True)
    assert result.status == "needs_retry"
