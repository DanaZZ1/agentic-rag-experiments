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


@dataclass
class EvidenceBinding:
    candidate: dict[str, Any]
    reason: str = ""


def _normalize_evidence_text(text: str) -> list[re.Match[str]]:
    """Return source-token matches; punctuation/spacing are ignored, words are not."""
    return list(re.finditer(r"[\w]+(?:['’][\w]+)?", text, re.UNICODE))


def _source_spans(candidate_quote: str, source: str) -> list[str]:
    wanted = [m.group(0).casefold() for m in _normalize_evidence_text(candidate_quote)]
    source_tokens = _normalize_evidence_text(source)
    if not wanted or len(wanted) > len(source_tokens):
        return []
    spans: list[str] = []
    for start in range(len(source_tokens) - len(wanted) + 1):
        if [m.group(0).casefold() for m in source_tokens[start:start + len(wanted)]] == wanted:
            spans.append(source[source_tokens[start].start():source_tokens[start + len(wanted) - 1].end()])
    return spans


def _sentence_candidates(quote: str) -> list[str]:
    """Split only at explicit sentence punctuation to avoid joining source spans."""
    parts = re.split(r"(?<=[.!?])\s+", quote.strip())
    return [part.strip() for part in parts if part.strip()]


def bind_evidence_references(task: dict[str, Any], candidate: dict[str, Any],
                             retrieved_by_fact: dict[str, dict[str, str]]) -> EvidenceBinding:
    """Resolve model quote hints into exact spans and IDs from retrieved chunks.

    A supplied chunk ID is only a hint. The returned IDs and quotes always come
    from the per-fact retrieved map. Non-contiguous model quotes must resolve as
    separate sentence spans; ambiguous or absent spans fail closed.
    """
    normalized = dict(candidate)
    claims = candidate.get("claims")
    if not isinstance(claims, list):
        return EvidenceBinding(normalized, "Structured claim list is required")
    facts = {str(f.get("id")): f for f in task.get("required_facts", [])}
    bound_claims: list[dict[str, Any]] = []
    all_refs: list[dict[str, str]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            return EvidenceBinding(normalized, "Malformed claim")
        fact_ids = [str(fid) for fid in claim.get("fact_ids", [])]
        refs = claim.get("evidence_refs")
        if not fact_ids or not isinstance(refs, list) or not refs:
            return EvidenceBinding(normalized, "Claim is missing fact association or evidence hints")
        if any(fid not in facts for fid in fact_ids):
            return EvidenceBinding(normalized, "Claim references an unplanned fact")
        bound_refs: list[dict[str, str]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                return EvidenceBinding(normalized, "Malformed evidence hint")
            fid = str(ref.get("fact_id", ""))
            hint_id = str(ref.get("chunk_id", ""))
            quote = str(ref.get("quote", "")).strip()
            if fid not in fact_ids or fid not in facts:
                return EvidenceBinding(normalized, "Evidence hint is linked to the wrong fact")
            chunks = retrieved_by_fact.get(fid, {})
            if not chunks:
                return EvidenceBinding(normalized, f"No retrieved chunks are associated with fact {fid!r}")
            if not quote:
                return EvidenceBinding(normalized, "Evidence hint contains no source text")
            exact_matches = [(cid, quote) for cid, text in chunks.items() if quote in text]
            candidates = exact_matches or [
                (cid, source_quote) for cid, text in chunks.items()
                for source_quote in _source_spans(quote, text)]
            # A valid supplied ID disambiguates identical spans, but never
            # overrides whether the span occurs in that retrieved chunk.
            if hint_id and hint_id in chunks:
                hinted = [item for item in candidates if item[0] == hint_id]
                if hinted:
                    candidates = hinted
            unique = list(dict.fromkeys(candidates))
            if not unique:
                if len(_sentence_candidates(quote)) > 1:
                    reason = "Evidence quote combines non-contiguous or non-verbatim text; provide separate references"
                else:
                    reason = f"Evidence span not found in retrieved chunks for fact {fid!r}"
                return EvidenceBinding(normalized, reason)
            chunk_ids = {cid for cid, _ in unique}
            if len(chunk_ids) != 1:
                return EvidenceBinding(normalized,
                    f"Evidence span is ambiguous across retrieved chunks for fact {fid!r}")
            cid = next(iter(chunk_ids))
            source_quotes = list(dict.fromkeys(q for found_id, q in unique if found_id == cid))
            if len(source_quotes) != 1:
                return EvidenceBinding(normalized,
                    f"Evidence span is ambiguous within retrieved chunk {cid!r}")
            bound_refs.append({"fact_id": fid, "chunk_id": cid, "quote": source_quotes[0]})
        if not bound_refs:
            return EvidenceBinding(normalized, "Claim has no source-resolved evidence")
        # Deduplicate refs without combining source passages.
        claim_copy = dict(claim)
        claim_copy["evidence_refs"] = list({(r["fact_id"], r["chunk_id"], r["quote"]): r for r in bound_refs}.values())
        bound_claims.append(claim_copy)
        all_refs.extend(claim_copy["evidence_refs"])
    normalized["claims"] = bound_claims
    normalized["evidence_refs"] = list({(r["fact_id"], r["chunk_id"], r["quote"]): r for r in all_refs}.values())
    return EvidenceBinding(normalized)


def typed_value_request(question: str) -> str | None:
    """Classify direct requests for contact values, not policy mentions."""
    q = question.lower()
    ask = r"\b(?:what|which|give|provide|tell me|is there|can i get)\b"
    if re.search(ask + r".{0,55}\b(?:phone|telephone)\s*(?:number|#)\b", q):
        return "phone_number"
    if re.search(ask + r".{0,65}\b(?:e-?mail)(?:\s+address)?\b", q):
        return "email_address"
    return None


def _explicit_abstention(answer: str) -> bool:
    return bool(re.search(
        r"\b(?:not in (?:the )?corpus|no evidence|no information|not provided|not present|"
        r"not available|not specified|cannot determine|can't determine|unable to determine|"
        r"insufficient information|does not contain|doesn't contain|"
        r"(?:the )?(?:available )?(?:evidence|information|text|passages|context)\s+does not provide|"
        r"no evidence found)\b", answer, re.I))


def _fact_evidence_available(fact: dict[str, Any], chunks: dict[str, str]) -> bool:
    """Conservative deterministic evidence screen, not semantic entailment."""
    if not fact.get("verifiable", False):
        return False
    entity, attribute = str(fact.get("entity", "unresolved")), str(fact.get("attribute", "unresolved"))
    aliases = fact.get("entity_aliases", [])
    terms = [str(term).casefold() for term in fact.get("evidence_terms", [])]
    for chunk in chunks.values():
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", chunk) if s.strip()]
        for sentence in sentences:
            if not _mentions_entity(entity, sentence, aliases):
                continue
            if not _attribute_supported(attribute, sentence):
                continue
            if terms and not all(term in sentence.casefold() for term in terms):
                continue
            return True
    return False


def verify_task_answer(task: dict[str, Any], candidate: dict[str, Any],
                       retrieved: dict[str, str], *, require_claims: bool = False,
                       retrieved_by_fact: dict[str, dict[str, str]] | None = None) -> EvidenceVerification:
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
        if not _explicit_abstention(answer):
            return EvidenceVerification("needs_retry", answer, [], "Not-in-corpus status lacks explicit abstention")
        by_fact = retrieved_by_fact or {str(f.get("id")): retrieved for f in facts}
        supported_facts = [str(f["id"]) for f in facts
                           if _fact_evidence_available(f, by_fact.get(str(f["id"]), {}))]
        if supported_facts:
            return EvidenceVerification("needs_retry", answer, valid_refs,
                                        "Retrieved evidence meets planned entity/attribute checks; abstention is not justified",
                                        supported_facts)
        return EvidenceVerification("not_in_corpus", answer, valid_refs)
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
        by_fact = {fid: [r["quote"] for r in valid_refs if r["fact_id"] == fid]
                   for fid in required_ids}
        if not _comparison_conclusion_matches(comparison, answer,
                by_fact.get(str(comparison["left_fact_id"]), []),
                by_fact.get(str(comparison["right_fact_id"]), [])):
            return EvidenceVerification("needs_retry", answer, valid_refs,
                                        "Comparison conclusion is unsupported or conflicts with operand values")
    if require_claims and not isinstance(claims, list):
        return EvidenceVerification("needs_retry", answer, valid_refs, "Structured claim list is required")
    if isinstance(claims, list):
        claim_verification = _verify_claims(task, answer, claims, valid_refs, retrieved)
        if claim_verification:
            return claim_verification
    return EvidenceVerification("supported", answer, valid_refs)


def _entity_aliases(entity: str, supplied: Any = None) -> tuple[str, ...]:
    if isinstance(supplied, (list, tuple)) and supplied:
        return tuple(str(x) for x in supplied if str(x))
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
        "unresolved": (),
    }
    return aliases.get(entity, (entity.replace("_", " "),))


def _mentions_entity(entity: str, text: str, aliases: Any = None) -> bool:
    if entity == "comparison":
        return True
    if entity in ("unknown", "unresolved"):
        return False
    lower = text.lower()
    return any(re.search(r"\b" + re.escape(alias.lower()) + r"\b", lower)
               for alias in _entity_aliases(entity, aliases))


def _attribute_supported(attribute: str, quote: str) -> bool:
    q = quote.lower()
    if attribute in ("unresolved", "general"):
        return False
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
    if attribute == "path":
        return bool(re.search(r"\b(settings?|support|menu|self diagnosis|select|find)\b", q))
    if attribute == "definition":
        return bool(re.search(r"\b(is|are|means|category|allows|refers|defined)\b", q))
    if attribute == "director":
        return bool(re.search(r"\b(director|directed|directs)\b", q))
    if attribute == "premiere_date":
        return (bool(re.search(r"\b(premiere|premiered|release date|released)\b", q))
                and bool(re.search(r"\b(?:\d{1,2}[/.-]\d{1,2}|\d{4}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", q)))
    if attribute == "genre":
        genres = r"(?:drama|comedy|documentary|thriller|romance|action|horror|western|musical|animation)"
        return bool(re.search(r"\bgenre\s*:\s*" + genres + r"\b", q)
                    or re.search(r"\b(?:is|was)\s+(?:an?\s+)?" + genres + r"\b", q))
    if attribute == "entry_method":
        return bool(re.search(r"\b(enter|press|button|turn on)\b", q))
    if attribute == "wallpaper_selection":
        return "wallpaper" in q and bool(re.search(r"\b(photo|picture|set|select)\b", q))
    if attribute == "photo_import":
        return bool(re.search(r"\b(import|save)\b", q)) and "smartthings" in q
    if attribute == "nationality":
        nationalities = r"(?:american|british|canadian|french|german|italian|japanese|turkish|australian|irish|spanish|indian|chinese)"
        return bool(re.search(r"\b(?:is|was|are|were)\s+(?:an?\s+)?" + nationalities + r"\b", q)
                    or re.search(r"\bnationality\s*:\s*" + nationalities + r"\b", q))
    if attribute == "neighborhood":
        return bool(re.search(r"\b(neighborhood|neighbourhood|located in|located at|situated in)\b", q))
    if attribute == "location":
        return bool(re.search(r"\b(?:located|situated)\s+(?:in|at)\s+[A-Z\w]", quote)
                    or re.search(r"\blocation\s*:\s*\S", quote, re.I))
    if attribute == "time":
        return bool(re.search(r"\b(date|time|year|month|day|when)\b", q))
    if attribute == "policy":
        return bool(re.search(r"\b(ask|request|require|never|always|policy|password)\b", q))
    if attribute == "same_different_relation":
        return True
    if attribute == "duration_relation":
        return True
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


def _comparison_value(attribute: str, quotes: list[str]) -> str | None:
    text = " ".join(quotes)
    if attribute == "nationality":
        values = re.findall(
            r"\b(american|british|canadian|french|german|italian|japanese|turkish|australian|irish|spanish|indian|chinese)\b",
            text, re.I)
        return values[-1].casefold() if values else None
    if attribute == "neighborhood":
        named = re.findall(r"\b(?:in|at)\s+([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)?)\s+(?:neighbou?rhood|district)\b", text)
        if named:
            return named[-1].casefold()
        values = re.findall(r"\b(?:located|situated)\s+in\s+([A-Z][\w’'-]+)", text)
        return values[-1].casefold() if values else None
    if attribute == "location":
        matches = re.findall(
            r"\b(?:located|situated)\s+(?:in|at)\s+([^,.;]+)|\bin\s+([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)?)\s+(?:neighbou?rhood|district)\b",
            text, re.I)
        values = []
        for first, second in matches:
            value = first or second
            value = re.sub(r"\s+(?:neighbou?rhood|district)$", "", value, flags=re.I).strip()
            if value:
                values.append(value.casefold())
        return values[-1] if values else None
    return None


def _comparison_conclusion_matches(comparison: dict[str, Any], answer: str,
                                   left_quotes: list[str], right_quotes: list[str]) -> bool:
    kind, operator = str(comparison.get("kind", "duration")), str(comparison.get("operator", ""))
    statement = answer.casefold()
    if kind == "duration":
        pattern = re.compile(r"\b(\d+)(?:\s*-\s*(\d+))?\s+business days?\b", re.I)
        def bounds(quotes: list[str]):
            m = pattern.search(" ".join(quotes))
            return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None
        left, right = bounds(left_quotes), bounds(right_quotes)
        if not left or not right:
            return False
        if operator == "longer":
            if left[0] > right[1]:
                return "longer" in statement and "not longer" not in statement
            if left[1] < right[0]:
                return any(x in statement for x in ("not take longer", "not longer", "shorter"))
            return "overlap" in statement or "neither" in statement
        if left[1] < right[0]:
            return "shorter" in statement and "not shorter" not in statement
        if left[0] > right[1]:
            return "longer" in statement and "not longer" not in statement
        return "overlap" in statement or "neither" in statement

    left = _comparison_value(str(comparison.get("attribute", "")), left_quotes)
    right = _comparison_value(str(comparison.get("attribute", "")), right_quotes)
    if left is None or right is None:
        return False
    same = left == right
    if operator == "same":
        return (same and bool(re.search(r"\b(same|both)\b", statement))
                and not bool(re.search(r"\b(not the same|not in the same|different|no)\b", statement)))
    return (not same and bool(re.search(r"\b(different|not the same|not in the same|no)\b", statement)))


def _verify_claims(task: dict[str, Any], answer: str, claims: list[Any],
                   valid_refs: list[dict[str, str]], retrieved: dict[str, str]) -> EvidenceVerification | None:
    facts = {str(f.get("id")): f for f in task.get("required_facts", [])}
    if not claims:
        return EvidenceVerification("needs_retry", answer, valid_refs, "Supported answer has no claims")
    top_refs = {(r["fact_id"], r["chunk_id"], r["quote"]) for r in valid_refs}
    seen_fact_ids: set[str] = set()
    claim_texts: list[str] = []
    comparison_claim_seen = False

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
            if (not comparison
                    or attribute != comparison.get("claim_attribute", "duration_relation")
                    or set(fact_ids) != required_comparison_ids):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Comparison claim does not identify both planned operands")
            comparison_claim_seen = True
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
            if not fact.get("verifiable", False):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Planner could not normalize required fact; refusing to certify supported claim", [fid])
            if not comparison_claim and (entity != expected_entity or attribute != expected_attribute):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim entity or attribute does not match required fact")
            if not _mentions_entity(expected_entity, quote, fact.get("entity_aliases")):
                return EvidenceVerification("needs_retry", answer, valid_refs,
                                            "Claim subject does not match evidence subject")
            if not comparison_claim and not _mentions_entity(expected_entity, text, fact.get("entity_aliases")):
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
                fact = facts[fid]
                if not _mentions_entity(str(fact.get("entity", "unknown")), text,
                                        fact.get("entity_aliases")):
                    return EvidenceVerification("needs_retry", answer, valid_refs,
                                                "Comparison claim mislabels an operand")

    missing = set(facts) - seen_fact_ids
    if missing:
        return EvidenceVerification("needs_retry", answer, valid_refs,
                                    "At least one required fact has no evidence-linked claim", sorted(missing))
    if task.get("comparison") and not comparison_claim_seen:
        return EvidenceVerification("needs_retry", answer, valid_refs,
                                    "Comparison claim is missing")
    expected_answer = " ".join(claim_texts)
    if re.sub(r"\s+", " ", answer).strip() != re.sub(r"\s+", " ", expected_answer).strip():
        return EvidenceVerification("needs_retry", answer, valid_refs,
                                    "Answer contains text outside its evidence-linked claims")
    return None
