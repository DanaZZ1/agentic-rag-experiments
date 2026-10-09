"""Input and output guardrails for the RAG pipeline."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from .text import tokenize

_INJECTION = [
    re.compile(r"ignore\s+(all|any|previous|prior)[\s\w]{0,24}instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous|above)", re.I),
    re.compile(r"(reveal|print|show)\s+[\w\s]{0,20}(system prompt|api[\s_-]?key|secret|password)", re.I),
]
_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b\d{16}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
]


class InputBlocked(Exception):
    pass


@dataclass
class GuardrailReport:
    grounded: bool
    faithfulness: float
    redactions: int


def check_input(query: str, *, max_len: int = 2000) -> str:
    if len(query) > max_len:
        raise InputBlocked("query too long")
    for pat in _INJECTION:
        if pat.search(query):
            raise InputBlocked("possible prompt-injection detected in query")
    return query.strip()


def redact_pii(text: str) -> tuple[str, int]:
    count = 0
    for pat, repl in _PII:
        text, n = pat.subn(repl, text)
        count += n
    return text, count


def grounding_score(answer: str, contexts: list[str]) -> float:
    a = [t for t in tokenize(answer)]
    if not a:
        return 0.0
    ctx = set().union(*(set(tokenize(c)) for c in contexts)) if contexts else set()
    return round(sum(1 for t in a if t in ctx) / len(a), 3)


@dataclass
class EvidenceVerification:
    status: str
    answer: str
    evidence_refs: list[dict[str, str]]
    reason: str = ""
    missing_fact_ids: list[str] | None = None


def typed_value_request(question: str) -> str | None:
    """Classify direct requests for contact values, not policy mentions."""
    q = question.lower()
    ask = r"\b(?:what|which|give|provide|tell me|is there|can i get)\b"
    if re.search(ask + r".{0,55}\b(?:phone|telephone)\s*(?:number|#)\b", q):
        return "phone_number"
    if re.search(ask + r".{0,65}\b(?:e-?mail)(?:\s+address)?\b", q):
        return "email_address"
    return None


def verify_task_answer(task: dict[str, Any], candidate: dict[str, Any],
                       retrieved: dict[str, str], *, require_claims: bool = False) -> EvidenceVerification:
    """Validate provenance and simple factual consistency; this is not entailment."""
    question = task["question"]
    status = candidate.get("status", "")
    answer = str(candidate.get("answer", ""))
    refs = candidate.get("evidence_refs", [])
    claims = candidate.get("claims")
    if not refs and isinstance(claims, list):
        refs = [ref for claim in claims if isinstance(claim, dict)
                for ref in claim.get("evidence_refs", []) if isinstance(ref, dict)]
    facts = task.get("required_facts", [])
    comparison = task.get("comparison")
    unsupported = lambda msg: EvidenceVerification("not_in_corpus", msg, [])

    value_type = typed_value_request(question)
    typed_value_present = False
    if value_type:
        is_email = value_type == "email_address"
        pattern = r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b" if is_email else r"\b(?:\+?\d[\d() .-]{5,}\d)\b"
        typed_value_present = any(re.search(pattern, text, re.I) for text in retrieved.values())
        if not typed_value_present:
            label = "an email address" if is_email else "a phone number"
            return unsupported(f"The corpus does not provide {label}.")

    valid_refs: list[dict[str, str]] = []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, dict):
            return EvidenceVerification("needs_retry", answer, [], "Malformed evidence reference")
        cid, quote, fid = str(ref.get("chunk_id", "")), str(ref.get("quote", "")), str(ref.get("fact_id", ""))
        if cid not in retrieved:
            return EvidenceVerification("needs_retry", answer, [], f"Evidence chunk {cid!r} was not retrieved")
        if not quote or quote not in retrieved[cid]:
            return EvidenceVerification("needs_retry", answer, [], f"Evidence quote not found verbatim in {cid}")
        valid_refs.append({"chunk_id": cid, "quote": quote, "fact_id": fid})

    if status == "not_in_corpus":
        if typed_value_present:
            return EvidenceVerification("needs_retry", answer, [], "A matching typed value exists in retrieved evidence")
        if re.search(r"\b(not|doesn't|does not|isn't|is not|unable|no information|not provided|not present)\b", answer, re.I):
            return EvidenceVerification("not_in_corpus", answer, valid_refs)
        return EvidenceVerification("needs_retry", answer, [], "Not-in-corpus status lacks explicit abstention")
    if status != "supported":
        return EvidenceVerification("needs_retry", answer, [], "Unknown task status")

    quotes = [r["quote"] for r in valid_refs]
    present_facts = {r["fact_id"] for r in valid_refs}
    missing = [f["id"] for f in facts if f["id"] not in present_facts]
    for fact in facts:
        if fact.get("evidence_terms") and fact["id"] in present_facts:
            quote = " ".join(r["quote"] for r in valid_refs if r["fact_id"] == fact["id"]).lower()
            if any(term.lower() not in quote for term in fact["evidence_terms"]):
                missing.append(fact["id"])
    if missing:
        return EvidenceVerification("needs_retry", answer, valid_refs, "Missing evidence for required fact", sorted(set(missing)))
    if not quotes:
        return EvidenceVerification("needs_retry", answer, [], "Supported answer has no evidence")

    # Typed values and numerical claims must be present in cited text.
    cited = " ".join(quotes)
    for pattern in (r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", r"\b(?:\+?\d[\d() .-]{5,}\d)\b"):
        for value in re.findall(pattern, answer, re.I):
            if value not in cited:
                return EvidenceVerification("needs_retry", answer, valid_refs, "Typed value is unsupported")
    def numbers(s: str) -> set[str]:
        found = set(re.findall(r"\b\d+(?:\.\d+)?\b", s))
        words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "seven": "7", "ten": "10", "thirty": "30"}
        found.update(words[w] for w in re.findall(r"\b(?:one|two|three|four|five|seven|ten|thirty)\b", s.lower()))
        return found
    if numbers(answer) - numbers(cited):
        return EvidenceVerification("needs_retry", answer, valid_refs, "Answer contains a number absent from evidence")

    # Legacy outputs lack claim-local evidence. V3 checks polarity against each
    # claim's own quote below.
    for quote in quotes if not isinstance(claims, list) else []:
        terms = [t for t in tokenize(quote) if len(t) > 4]
        for term in terms:
            qpos, apos = quote.lower().find(term), answer.lower().find(term)
            if apos < 0 or qpos < 0:
                continue
            qneg = bool(re.search(r"\b(no|not|never|exclude|excluded|without|cannot|doesn't|isn't)\b", quote[max(0, qpos-45):qpos+len(term)+30], re.I))
            aneg = bool(re.search(r"\b(no|not|never|exclude|excluded|without|cannot|doesn't|isn't)\b", answer[max(0, apos-45):apos+len(term)+30], re.I))
            if qneg != aneg and len(set(terms) & set(tokenize(answer))) >= 2:
                # Only flag when the quote's distinctive phrase is actually reused.
                phrase = re.search(r"\b(?:cosmetic damage|accidental damage|liquid damage)\b", quote, re.I)
                if phrase and phrase.group(0).lower() in answer.lower():
                    return EvidenceVerification("needs_retry", answer, valid_refs, "Answer polarity conflicts with evidence")
                break

    if comparison:
        required_ids = {comparison["left_fact_id"], comparison["right_fact_id"]}
        if not required_ids.issubset(present_facts):
            return EvidenceVerification("needs_retry", answer, valid_refs, "Comparison is missing evidence for an operand", list(required_ids - present_facts))
        # Ensure conditions in operand quotes are retained (e.g. after receipt and inspection).
        all_text = " ".join(quotes).lower()
        for condition in comparison.get("required_conditions", []):
            if condition.lower() not in all_text or condition.lower() not in answer.lower():
                return EvidenceVerification("needs_retry", answer, valid_refs, f"Comparison omits condition: {condition}")
        by_fact = {r["fact_id"]: r["quote"] for r in valid_refs}
        def duration(text: str):
            m = re.search(r"\b(\d+)(?:\s*-\s*(\d+))?\s+business days?\b", text, re.I)
            return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None
        left = duration(by_fact[comparison["left_fact_id"]])
        right = duration(by_fact[comparison["right_fact_id"]])
        relation = comparison.get("operator")
        statement = answer.lower()
        if not left or not right:
            return EvidenceVerification("needs_retry", answer, valid_refs, "Could not compare durations")
        if relation == "longer":
            phrase_ok = any(x in statement for x in ("longer", "shorter", "take more", "take less"))
            consistent = ("does not take longer" in statement or "not longer" in statement or "shorter" in statement) if left[1] < right[0] else ("takes longer" in statement or "longer" in statement and "not longer" not in statement)
        else:
            phrase_ok = any(x in statement for x in ("shorter", "longer", "take more", "take less"))
            consistent = ("shorter" in statement or "does not take longer" in statement) if left[1] < right[0] else "takes longer" in statement
        if not phrase_ok or not consistent:
            return EvidenceVerification("needs_retry", answer, valid_refs, "Comparison conclusion is missing or conflicts with durations")
    if require_claims and not isinstance(claims, list):
        return EvidenceVerification("needs_retry", answer, valid_refs, "Structured claim list is required")
    if isinstance(claims, list):
        claim_verification = _verify_claims(task, answer, claims, valid_refs, retrieved)
        if claim_verification:
            return claim_verification
    return EvidenceVerification("supported", answer, valid_refs)


def _entity_aliases(entity: str) -> tuple[str, ...]:
    aliases = {
        "standard_shipping": ("standard shipping", "standard delivery"),
        "express_shipping": ("express shipping", "express delivery", "express deliveries"),
        "refund_processing": ("refund", "refunds"),
        "return_eligibility": ("return", "returns", "send back"),
        "warranty": ("warranty",),
        "support_password_policy": ("support", "password", "passwords"),
        "account_security": ("account", "accounts", "two-factor", "password"),
        "support_contact": ("support",),
        "unknown": (),
    }
    return aliases.get(entity, (entity.replace("_", " "),))


def _mentions_entity(entity: str, text: str) -> bool:
    if entity in ("unknown", "comparison"):
        return True
    lower = text.lower()
    return any(re.search(r"\b" + re.escape(alias) + r"\b", lower)
               for alias in _entity_aliases(entity))


def _attribute_supported(attribute: str, quote: str) -> bool:
    q = quote.lower()
    if attribute == "duration":
        return bool(re.search(r"\b\d+(?:\s*-\s*\d+)?\s*-?\s*(?:business\s+)?(?:days?|weeks?|months?|years?)\b", q))
    if attribute == "packaging":
        return bool(re.search(r"\boriginal\s+packag(?:e|ing)\b", q))
    if attribute == "start_condition":
        return "received" in q and "inspected" in q
    if attribute == "coverage":
        return "cover" in q
    if attribute == "exclusions":
        return bool(re.search(r"\b(not|never|exclude|excluded|doesn't)\b", q)) and "damage" in q
    if attribute == "fee":
        return "fee" in q or "charge" in q
    if attribute == "password_requirement":
        return "password" in q and "character" in q
    if attribute == "phone_number":
        return bool(re.search(r"\b(?:\+?\d[\d() .-]{5,}\d)\b", q))
    if attribute == "email_address":
        return bool(re.search(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", q))
    return True


def _polarity_conflict(claim: str, quote: str) -> bool:
    negation = re.compile(r"\b(?:not|no|never|excluded|exclude|without|cannot|doesn't|isn't)\b", re.I)
    claim_terms = set(tokenize(claim))
    shared = [t for t in tokenize(quote) if len(t) > 4 and t in claim_terms]
    for term in shared:
        qpos, cpos = quote.lower().find(term), claim.lower().find(term)
        if qpos < 0 or cpos < 0:
            continue
        qneg = bool(negation.search(quote[max(0, qpos - 40):qpos + len(term) + 20]))
        cneg = bool(negation.search(claim[max(0, cpos - 40):cpos + len(term) + 20]))
        if qneg != cneg and len(shared) >= 2:
            return True
    return False


def _verify_claims(task: dict[str, Any], answer: str, claims: list[Any],
                   valid_refs: list[dict[str, str]], retrieved: dict[str, str]) -> EvidenceVerification | None:
    facts = {str(f.get("id")): f for f in task.get("required_facts", [])}
    if not claims:
        return EvidenceVerification("needs_retry", answer, valid_refs, "Supported answer has no claims")
    top_refs = {(r["fact_id"], r["chunk_id"], r["quote"]) for r in valid_refs}
    seen_fact_ids: set[str] = set()
    claim_texts: list[str] = []

    for claim in claims:
        if not isinstance(claim, dict):
            return EvidenceVerification("needs_retry", answer, valid_refs, "Malformed claim")
        text = str(claim.get("text", "")).strip()
        entity = str(claim.get("entity", "unknown"))
        attribute = str(claim.get("attribute", "general"))
        fact_ids = [str(fid) for fid in claim.get("fact_ids", [])]
        if not fact_ids and claim.get("fact_id") is not None:
            fact_ids = [str(claim["fact_id"])]
        refs = claim.get("evidence_refs", [])
        if not text or not fact_ids or not isinstance(refs, list) or not refs:
            return EvidenceVerification("needs_retry", answer, valid_refs,
                                        "Claim is missing text, fact association, or evidence")
        claim_texts.append(text)

        comparison_claim = entity == "comparison"
        comparison = task.get("comparison")
        required_comparison_ids = ({str(comparison["left_fact_id"]), str(comparison["right_fact_id"])}
                                   if comparison else set())
        if comparison_claim:
            if not comparison or attribute != "duration_relation" or not required_comparison_ids.issubset(fact_ids):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Comparison claim does not identify both planned operands")
        elif len({(str(facts.get(fid, {}).get("entity", "unknown")),
                   str(facts.get(fid, {}).get("attribute", "general"))) for fid in fact_ids}) != 1:
            return EvidenceVerification("needs_retry", answer, valid_refs,
                                        "A claim combines facts with different entities or attributes")

        seen_refs: set[str] = set()
        evidence_text_by_fact: dict[str, list[str]] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Malformed claim evidence reference")
            fid, cid, quote = (str(ref.get(k, "")) for k in ("fact_id", "chunk_id", "quote"))
            if (fid, cid, quote) not in top_refs:
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim evidence is not in verified task evidence")
            if fid not in fact_ids or fid not in facts:
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim evidence is linked to the wrong fact")
            fact = facts[fid]
            expected_entity = str(fact.get("entity", "unknown"))
            expected_attribute = str(fact.get("attribute", "general"))
            if not comparison_claim and (entity != expected_entity or attribute != expected_attribute):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim entity or attribute does not match required fact")
            if not _mentions_entity(expected_entity, quote):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim subject does not match evidence subject")
            if not comparison_claim and not _mentions_entity(expected_entity, text):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim text subject does not match required entity")
            if not comparison_claim and not _attribute_supported(expected_attribute, quote):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Evidence does not contain the required attribute")
            if not comparison_claim and not _attribute_supported(expected_attribute, text):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim text does not express the required attribute")
            if not comparison_claim and _polarity_conflict(text, quote):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim polarity conflicts with its evidence")
            evidence_text_by_fact.setdefault(fid, []).append(quote)
            seen_refs.add(fid)
        if not set(fact_ids).issubset(seen_refs):
            return EvidenceVerification("needs_retry", answer, valid_refs,
                                        "Claim lacks evidence for every associated fact")
        seen_fact_ids.update(seen_refs)

        if comparison_claim:
            for fid in required_comparison_ids:
                if not _mentions_entity(str(facts[fid].get("entity", "unknown")), text):
                    return EvidenceVerification("needs_retry", answer, valid_refs,
                                                "Comparison claim mislabels an operand")

    missing = set(facts) - seen_fact_ids
    if missing:
        return EvidenceVerification("needs_retry", answer, valid_refs,
                                    "At least one required fact has no evidence-linked claim", sorted(missing))
    expected_answer = " ".join(claim_texts)
    if re.sub(r"\s+", " ", answer).strip() != re.sub(r"\s+", " ", expected_answer).strip():
        return EvidenceVerification("needs_retry", answer, valid_refs,
                                    "Answer contains text outside its evidence-linked claims")
    return None
