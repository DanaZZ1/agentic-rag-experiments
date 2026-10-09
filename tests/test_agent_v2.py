import pytest

from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.guardrails import verify_task_answer


@pytest.fixture(scope="module")
def pipeline():
    pipe = RAGPipeline(Settings(embed_provider="hashing"))
    with open("data/corpus/handbook.md", encoding="utf-8") as f:
        pipe.add_document(f.read(), doc_id="handbook")
    return pipe


def test_acceptance_phone_abstains(pipeline):
    result = pipeline.query("What is the support phone number?")
    assert "does not provide a phone number" in result.answer.lower()
    assert result.coverage["not_in_corpus"] == 1


def test_acceptance_return_and_email(pipeline):
    result = pipeline.query("What is the return window, and which email address should I use to get a refund?")
    assert "30 days" in result.answer
    assert "does not provide an email address" in result.answer.lower()
    assert result.coverage["supported"] == 1
    assert result.coverage["not_in_corpus"] == 1


def test_acceptance_comparison_preserves_condition(pipeline):
    result = pipeline.query("Do express deliveries take longer than refunds?")
    assert "1-2 business days" in result.answer
    assert "5 business days" in result.answer
    assert "received and inspected" in result.answer.lower()
    assert "does not take longer" in result.answer.lower()
    assert result.task_answers[0].status == "supported"


def test_acceptance_corrects_false_warranty_premise(pipeline):
    result = pipeline.query("The warranty covers cosmetic damage, right?")
    assert "does not cover" in result.answer.lower()
    assert "cosmetic damage" in result.answer.lower()


def test_acceptance_send_back_maps_to_return_window(pipeline):
    result = pipeline.query("How many days do I have to send back a product in its original package?")
    assert "30 days" in result.answer
    assert "5 business days" not in result.answer


def _task():
    return {"id": "t", "question": "How long is the policy?", "required_facts": [
        {"id": "f", "description": "duration", "retrieval_query": "duration"}], "comparison": None}


def _candidate(answer="30 days", refs=None, status="supported"):
    return {"task_id": "t", "status": status, "answer": answer,
            "evidence_refs": refs if refs is not None else [
                {"fact_id": "f", "chunk_id": "c1", "quote": "Returns are accepted within 30 days."}]}


def test_verifier_rejects_invalid_chunk_id():
    result = verify_task_answer(_task(), _candidate(refs=[{"fact_id": "f", "chunk_id": "missing", "quote": "30 days"}]), {"c1": "Returns are accepted within 30 days."})
    assert result.status == "needs_retry"
    assert "not retrieved" in result.reason


def test_verifier_rejects_quote_not_found():
    result = verify_task_answer(_task(), _candidate(refs=[{"fact_id": "f", "chunk_id": "c1", "quote": "30 days accepted"}]), {"c1": "Returns are accepted within 30 days."})
    assert result.status == "needs_retry"
    assert "verbatim" in result.reason


@pytest.mark.parametrize("question,claim", [
    ("What is the phone number?", "Call 555-123-4567."),
    ("Which email address?", "Email help@example.com."),
])
def test_verifier_rejects_unsupported_typed_value(question, claim):
    task = {**_task(), "question": question}
    result = verify_task_answer(task, _candidate(answer=claim), {"c1": "Support is available online."})
    assert result.status == "not_in_corpus"
    assert "does not provide" in result.answer


def test_verifier_rejects_number_mismatch():
    result = verify_task_answer(_task(), _candidate(answer="60 days"), {"c1": "Returns are accepted within 30 days."})
    assert result.status == "needs_retry"
    assert "number" in result.reason


def test_verifier_rejects_negation_mismatch():
    task = {**_task(), "question": "Does warranty cover cosmetic damage?"}
    text = "The warranty does not cover cosmetic damage."
    candidate = _candidate(answer="The warranty covers cosmetic damage.", refs=[{"fact_id": "f", "chunk_id": "c1", "quote": text}])
    result = verify_task_answer(task, candidate, {"c1": text})
    assert result.status == "needs_retry"
    assert "polarity" in result.reason


def test_verifier_rejects_incomplete_comparison():
    task = {"id": "cmp", "question": "Does A take longer than B?", "required_facts": [
        {"id": "a", "description": "A", "retrieval_query": "A"},
        {"id": "b", "description": "B", "retrieval_query": "B"}],
        "comparison": {"operator": "longer", "left_fact_id": "a", "right_fact_id": "b", "required_conditions": []}}
    cand = {"task_id": "cmp", "status": "supported", "answer": "A takes longer.",
            "evidence_refs": [{"fact_id": "a", "chunk_id": "c1", "quote": "A takes 8 business days."}]}
    result = verify_task_answer(task, cand, {"c1": "A takes 8 business days."})
    assert result.status == "needs_retry"
    assert "evidence" in result.reason
