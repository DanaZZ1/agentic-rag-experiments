from dataclasses import asdict

from production_agentic_rag.agents.planner import plan
from production_agentic_rag.agents.rag_agent import _retry_action
from production_agentic_rag.guardrails import (
    _verify_claims,
    classify_verification_failure,
    resolve_entity_alias,
    verify_task_answer,
)


def test_development_style_comparisons_have_two_operands_and_conclusions():
    questions = [
        "Who was born earlier, Mira Vale or Tomas Reed?",
        "Which orchestra, the North Ensemble or Harbor Players, had more members?",
        "Are both Luma Palm and Vero Succulent classified as a genus?",
        "Do the drinks Silver Comet and Bitter Orchard both contain juniper?",
    ]
    for question in questions:
        task = plan(question)[0]
        assert task.comparison is not None
        assert len(task.required_facts) == 2
        assert task.comparison["left_fact_id"] == task.required_facts[0].id
        assert task.comparison["right_fact_id"] == task.required_facts[1].id
        assert task.comparison["conclusion_required"] is True


def test_two_independent_entities_are_not_collapsed_into_one_fact():
    task = plan("Where are Ember Ridge Park and Bluewater Reserve located?")[0]
    assert task.comparison is None
    assert [fact.entity for fact in task.required_facts] == [
        "ember_ridge_park", "bluewater_reserve"]
    assert len(task.dependencies) == 2
    assert all(dep["relation"] == "independent" for dep in task.dependencies)


def test_bridge_planning_records_dependency_and_uncertainty():
    tasks = plan("Where is the headquarters of the studio that Mira Vale worked for as an engineer?")
    assert len(tasks) == 1
    task = tasks[0]
    assert len(task.required_facts) == 2
    assert task.dependencies[1]["depends_on"] == [task.required_facts[0].id]
    assert task.planning_uncertainty


def test_unrecognized_comparison_property_is_explicitly_uncertain():
    task = plan("Which artist was known for an unusual performance style, Mira Vale or Tomas Reed?")[0]
    assert task.comparison is not None
    assert task.planning_uncertainty
    assert all(not fact.verifiable for fact in task.required_facts)


def test_entity_resolution_normalizes_format_but_rejects_name_overlap():
    assert resolve_entity_alias("northport_arena", "Northport-Arena")["matched"]
    assert resolve_entity_alias("Harbor Light", "Harbor Light Theater",
                                ["Harbor Light"])["matched"] is False
    assert resolve_entity_alias("Harbor Light", "The Beacon",
                                ["Harbor Light"],
                                ["Harbor Light, also known as The Beacon, opened in 1998."])["method"] == \
        "explicit_identity_relation_in_evidence"


def test_normalized_entity_claim_is_accepted_but_wrong_entity_is_rejected():
    quote = "Northport Arena is located in Cedar Bay."
    fact = {"id": "f", "entity": "northport_arena", "entity_aliases": ["Northport Arena"],
            "attribute": "location", "verifiable": True, "evidence_terms": []}
    ref = {"fact_id": "f", "chunk_id": "c", "quote": quote}
    task = {"question": "Where is Northport Arena?", "required_facts": [fact], "comparison": None}
    claim = {"claim_id": "c", "entity": "Northport-Arena", "attribute": "location",
             "text": quote, "fact_ids": ["f"], "evidence_refs": [ref]}
    assert _verify_claims(task, quote, [claim], [ref], {"c": quote}) is None
    wrong = {**claim, "entity": "Northport Arena Annex"}
    assert _verify_claims(task, quote, [wrong], [ref], {"c": quote}) is not None


def test_evidence_explicitly_establishing_an_alias_is_accepted():
    quote = "Harbor Light, also known as The Beacon, is located in Cedar Bay."
    fact = {"id": "f", "entity": "harbor_light", "entity_aliases": ["Harbor Light"],
            "attribute": "location", "verifiable": True, "evidence_terms": []}
    ref = {"fact_id": "f", "chunk_id": "c", "quote": quote}
    task = {"question": "Where is Harbor Light?", "required_facts": [fact], "comparison": None}
    claim_text = "The Beacon is located in Cedar Bay."
    claim = {"claim_id": "c", "entity": "The Beacon", "attribute": "location",
             "text": claim_text, "fact_ids": ["f"], "evidence_refs": [ref]}
    assert _verify_claims(task, claim_text, [claim], [ref], {"c": quote}) is None


def test_failure_taxonomy_selects_distinct_recovery_classes():
    assert classify_verification_failure("Evidence chunk 'x' was not retrieved") == "invalid_citation"
    assert classify_verification_failure("Evidence does not contain the required attribute") == "insufficient_evidence"
    assert classify_verification_failure("Retrieved evidence meets checks; abstention is not justified") == "unjustified_abstention"
    assert classify_verification_failure("Comparison claim is missing") == "incomplete_comparison"
    assert classify_verification_failure("Claim subject does not match evidence subject") == "entity_resolution_failure"
    assert classify_verification_failure("At least one required fact has no evidence-linked claim") == "missing_required_fact"
    assert _retry_action("insufficient_evidence") == "reformulate_retrieval_query"
    assert _retry_action("invalid_citation") == "regenerate_from_existing_evidence"
    assert _retry_action("unjustified_abstention") == "recheck_claim_against_existing_evidence"


def _comparison_candidate(task, facts, quotes, conclusion):
    refs = [{"fact_id": fact["id"], "chunk_id": f"c{index}", "quote": quote}
            for index, (fact, quote) in enumerate(zip(facts, quotes))]
    claims = [{"claim_id": f"operand{index}", "entity": fact["entity"],
               "attribute": fact["attribute"], "text": quote,
               "fact_ids": [fact["id"]], "evidence_refs": [refs[index]]}
              for index, (fact, quote) in enumerate(zip(facts, quotes))]
    claims.append({"claim_id": "conclusion", "entity": "comparison",
                   "attribute": task["comparison"]["claim_attribute"], "text": conclusion,
                   "fact_ids": [fact["id"] for fact in facts], "evidence_refs": refs})
    answer = "\n".join(claim["text"] for claim in claims)
    return {"status": "supported", "answer": answer, "claims": claims, "evidence_refs": refs}


def test_numeric_comparison_conclusion_is_checked_against_both_operands():
    planned = plan("Which group, North Ensemble or Harbor Players, had more members?")[0]
    task = asdict(planned)
    facts = task["required_facts"]
    quotes = ["North Ensemble has 5 members.", "Harbor Players has 3 members."]
    chunks = {"c0": quotes[0], "c1": quotes[1]}
    good = _comparison_candidate(task, facts, quotes,
        "North Ensemble had more members than Harbor Players.")
    assert verify_task_answer(task, good, chunks, require_claims=True).status == "supported"
    wrong = _comparison_candidate(task, facts, quotes,
        "Harbor Players had more members than North Ensemble.")
    assert verify_task_answer(task, wrong, chunks, require_claims=True).status == "needs_retry"


def test_shared_property_requires_both_operands_and_the_requested_value():
    planned = plan("Do both Silver Comet and Bitter Orchard contain gin?")[0]
    task = asdict(planned)
    facts = task["required_facts"]
    quotes = ["Silver Comet contains gin.", "Bitter Orchard contains gin."]
    chunks = {"c0": quotes[0], "c1": quotes[1]}
    good = _comparison_candidate(task, facts, quotes,
        "Both Silver Comet and Bitter Orchard contain gin.")
    assert verify_task_answer(task, good, chunks, require_claims=True).status == "supported"
    wrong = _comparison_candidate(task, facts, quotes,
        "Both Silver Comet and Bitter Orchard contain vodka.")
    assert verify_task_answer(task, wrong, chunks, require_claims=True).status == "needs_retry"
