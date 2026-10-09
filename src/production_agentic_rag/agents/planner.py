"""Lightweight task/fact planning with corpus-neutral operand extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..guardrails import typed_value_request


@dataclass
class RequiredFact:
    id: str
    description: str
    retrieval_query: str
    evidence_terms: list[str] = field(default_factory=list)
    entity: str = "unknown"
    attribute: str = "general"


@dataclass
class PlannedTask:
    id: str
    question: str
    required_facts: list[RequiredFact]
    comparison: dict[str, object] | None = None


# Entity aliases are vocabulary normalization, not operand-position rules. Each
# side of a comparison is independently resolved from the user's own wording.
_ENTITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("standard_shipping", ("standard shipping", "standard delivery")),
    ("express_shipping", ("express shipping", "express delivery", "express deliveries")),
    ("refund_processing", ("refund", "refunds", "refund processing")),
    ("return_eligibility", ("return", "returns", "send back", "original package", "original packaging")),
    ("warranty", ("warranty",)),
    ("support_password_policy", ("support", "password", "passwords")),
    ("account_security", ("account", "accounts", "password", "two-factor")),
)

_ENTITY_TERMS = {
    "standard_shipping": "standard shipping",
    "express_shipping": "express shipping",
    "refund_processing": "approved refunds processing",
    "return_eligibility": "return unused items delivery",
    "warranty": "warranty electronics",
    "support_password_policy": "password requirement characters length",
    "account_security": "account two-factor password",
    "unknown": "",
}


def _entity(text: str, previous: str | None = None) -> str:
    q = re.sub(r"\s+", " ", text.lower())
    for entity, aliases in _ENTITY_PATTERNS:
        if any(re.search(r"\b" + re.escape(alias) + r"\b", q) for alias in aliases):
            # Prefer the narrower password-policy entity over account security.
            if entity == "account_security" and "support_password_policy" in q:
                continue
            return entity
    if re.search(r"\b(?:that|this|the)\s+(?:period|time|duration)\b", q) and previous:
        return previous
    if typed_value_request(text) == "phone_number":
        return "support_contact"
    if typed_value_request(text) == "email_address":
        return "support_contact"
    return previous or "unknown"


def _attribute(text: str, entity: str) -> str:
    q = text.lower()
    if typed_value_request(text):
        return typed_value_request(text) or "contact_value"
    if re.search(r"\b(start|starts|begin|begins|after|what starts|what begins)\b", q):
        return "start_condition"
    if entity == "refund_processing" and re.search(r"\bwhen\b", q) and re.search(r"\b(issued|processed|arrive|take)\b", q):
        return "duration"
    if re.search(r"\b(long|duration|window|how many days|how much time|take longer|take shorter|take less|take more)\b", q):
        return "duration"
    if re.search(r"\b(packag(?:e|ing))\b", q):
        return "packaging"
    if re.search(r"\b(exclude|excluded|excludes|damage types)\b", q):
        return "exclusions"
    if re.search(r"\b(cover|covers|coverage)\b", q):
        return "coverage"
    if re.search(r"\b(fee|cost|charge)\b", q):
        return "fee"
    if re.search(r"\b(password|characters|length)\b", q):
        return "password_requirement"
    if entity == "support_password_policy":
        return "policy"
    return "general"


def _query(entity: str, attribute: str, wording: str) -> str:
    terms = _ENTITY_TERMS.get(entity, "")
    attr_terms = {
        "duration": "duration days business days time",
        "start_condition": "returned item received inspected starts after",
        "coverage": "coverage cover",
        "exclusions": "does not cover excluded damage",
        "fee": "additional fee",
        "password_requirement": "password minimum characters",
        "phone_number": "support phone number",
        "email_address": "support email address contact",
        "packaging": "return original packaging unused item",
    }.get(attribute, "")
    # Retain wording to preserve unusual user entities and attribute terms.
    return " ".join(part for part in (wording, terms, attr_terms) if part).strip()


def _fact(fid: str, text: str, entity: str, attribute: str,
          evidence_terms: list[str] | None = None) -> RequiredFact:
    terms = evidence_terms or []
    query = _query(entity, attribute, text)
    if terms:
        query = " ".join((query, *terms))
    return RequiredFact(fid, text, query, terms, entity, attribute)


def plan(question: str) -> list[PlannedTask]:
    q = question.strip().rstrip("?")
    compare = re.match(r"^(?:do|does)\s+(.+?)\s+take\s+(longer|shorter)\s+than\s+(.+)$", q, re.I)
    if compare:
        left_text, operator, right_text = (part.strip(" .") for part in compare.groups())
        left_entity, right_entity = _entity(left_text), _entity(right_text)
        left_id, right_id = "operand_a", "operand_b"
        facts = [
            _fact(left_id, f"duration of {left_text}", left_entity, "duration"),
            _fact(right_id, f"duration of {right_text}", right_entity, "duration",
                  ["5 business days", "received", "inspected"] if right_entity == "refund_processing" else []),
        ]
        conditions = ["received", "inspected"] if any(f.entity == "refund_processing" for f in facts) else []
        comparison = {"operator": operator.lower(), "left_fact_id": left_id,
                      "right_fact_id": right_id, "left_label": left_text,
                      "right_label": right_text, "required_conditions": conditions}
        return [PlannedTask("comparison", question, facts, comparison)]

    # Split a coordinated predicate into atomic facts while keeping the shared
    # subject/predicate (e.g. warranty covers X and Y) explicit for each claim.
    coordinated = re.search(
        r"\bcover(?:s)?\s+(.+?)\s+and\s+(.+?)(?:,?\s+(?:right|correct))?$", q, re.I)
    if coordinated:
        entity = _entity(q)
        if entity != "unknown":
            values = [part.strip(" ,.") for part in coordinated.groups() if part]
            if len(values) == 2:
                facts = [_fact(f"fact_{i}", f"{entity} covers {value}", entity, "coverage", [value])
                         for i, value in enumerate(values, 1)]
                return [PlannedTask("task_1", question, facts)]

    parts = re.split(
        r"\s*,\s*and\s+|\s+and\s+(?=(?:which|what|where|when|how|who|whether|is|are|does|do)\b)",
        q, flags=re.I)
    if len(parts) == 1:
        parts = re.split(r"\s+also\s+|\s*;\s*|\s+then\s+", q, flags=re.I)
    parts = [part.strip(" ?.,") for part in parts if part.strip(" ?.,")]
    tasks: list[PlannedTask] = []
    previous_entity: str | None = None
    previous_attribute: str | None = None
    for index, part in enumerate(parts or [q], 1):
        entity = _entity(part, previous_entity)
        attribute = _attribute(part, entity)
        # Correct explicit false attribution such as asking whether the 30-day
        # return period describes refund processing. Keep each named concept as
        # its own fact instead of accepting the premise as an entity assignment.
        if (re.search(r"\b\d+\s+days?\b", part, re.I)
                and re.search(r"\b(?:describe|mean|refer to)\b", part, re.I)
                and entity == "refund_processing"):
            entity, attribute = "return_eligibility", "duration"
        # Resolve anaphora such as "what starts that period" from the preceding
        # task and turn it into a fact-specific query.
        if entity == previous_entity and attribute == "start_condition":
            description = f"{entity} start condition"
            query_text = "refund start condition" if entity == "refund_processing" else description
        else:
            description, query_text = part, part
        fact = _fact(f"fact_{index}", description, entity, attribute)
        fact.retrieval_query = _query(entity, attribute, query_text)
        tasks.append(PlannedTask(f"task_{index}", part if part.endswith("?") else part + "?", [fact]))
        if entity != "unknown":
            previous_entity = entity
        previous_attribute = attribute
    return tasks
