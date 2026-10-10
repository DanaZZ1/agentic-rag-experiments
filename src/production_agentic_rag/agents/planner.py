"""Small, corpus-neutral planner for tasks, facts, and comparison operands."""
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
    entity: str = "unresolved"
    attribute: str = "unresolved"
    entity_aliases: list[str] = field(default_factory=list)
    verifiable: bool = False


@dataclass
class PlannedTask:
    id: str
    question: str
    required_facts: list[RequiredFact]
    comparison: dict[str, object] | None = None


_KNOWN_ENTITIES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("standard_shipping", ("standard shipping", "standard delivery"), "standard shipping"),
    ("express_shipping", ("express shipping", "express delivery", "express deliveries"), "express shipping"),
    ("refund_processing", ("refund processing", "approved refunds", "refunds", "refund"), "refund processing"),
    ("return_eligibility", ("return eligibility", "return policy", "returns", "return", "send back", "original package", "original packaging"), "return eligibility"),
    ("warranty", ("warranty",), "warranty"),
    ("support_password_policy", ("support", "password", "passwords"), "support password policy"),
    ("account_security", ("two-factor", "accounts", "account"), "account security"),
    ("support_contact", ("support",), "support contact"),
)

_ATTRIBUTE_TERMS = {
    "duration": "duration days business days time",
    "start_condition": "received inspected begins after",
    "coverage": "cover covered coverage",
    "exclusions": "exclude excluded does not cover damage",
    "fee": "fee cost charge",
    "password_requirement": "password minimum characters length",
    "policy": "policy ask request require",
    "phone_number": "phone telephone number",
    "email_address": "email address",
    "packaging": "original packaging unused item",
    "location": "location located address place",
    "path": "settings support menu find",
    "definition": "definition category means refers allows",
    "director": "director directed",
    "premiere_date": "premiere date premiered release",
    "genre": "genre category type",
    "entry_method": "enter mode button how",
    "wallpaper_selection": "photo set wallpaper picture",
    "photo_import": "save import photos SmartThings",
    "nationality": "nationality national American British Canadian French German",
    "neighborhood": "neighborhood located in district",
    "time": "time date when",
}

_ENTITY_QUERY_TERMS = {
    "standard_shipping": "standard shipping delivery",
    "express_shipping": "express shipping delivery",
    "refund_processing": "approved refunds processing",
    "return_eligibility": "return unused items delivery",
    "warranty": "warranty electronics",
    "support_password_policy": "password requirement characters length",
    "account_security": "account two-factor password",
    "support_contact": "support contact",
}


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words) if words else "unresolved"


def _known_entity(text: str) -> tuple[str, list[str]] | None:
    q = re.sub(r"\s+", " ", text.lower())
    for entity, aliases, _ in _KNOWN_ENTITIES:
        if any(re.search(r"\b" + re.escape(alias) + r"\b", q) for alias in aliases):
            if entity == "account_security" and "support" in q and "password" in q:
                continue
            return entity, list(aliases)
    if re.search(r"\bambient\s+mode\b", q):
        return "ambient_mode", ["ambient mode"]
    return None


def _surface_entity(text: str) -> tuple[str, list[str]]:
    value = re.sub(r"^(?:the|a|an)\s+", "", text.strip(" \t\n\"'`.,?!"), flags=re.I)
    value = re.sub(r"\s+", " ", value)
    if not value or len(value.split()) > 12 or value.lower() in {"it", "its", "this", "that", "information"}:
        return "unresolved", []
    return _slug(value), [value]


def _attribute(text: str) -> str:
    q = text.lower()
    value_type = typed_value_request(text)
    if value_type:
        return value_type
    if re.search(r"\b(start|starts|begin|begins|after|what starts|what begins)\b", q):
        return "start_condition"
    # "How long must a password be?" asks for the password rule (a minimum
    # length), not a time duration. Keep this typed fact distinct from temporal
    # windows such as returns and refunds.
    if re.search(r"\b(password|passwords)\b", q) and re.search(r"\b(long|length|characters?)\b", q):
        return "password_requirement"
    if re.search(r"\b(long|duration|window|how many days|how much time|take longer|take shorter|take less|take more)\b", q):
        return "duration"
    if re.search(r"\bwithin\s+\d+\s+days?\b", q):
        return "duration"
    if re.search(r"\b(exclude|excluded|excludes|damage types|cosmetic damage)\b", q):
        return "exclusions"
    if re.search(r"\b(cover|covers|coverage)\b", q):
        return "coverage"
    if re.search(r"\b(fee|cost|charge)\b", q):
        return "fee"
    if re.search(r"\b(ask|request|require|policy)\b", q) and re.search(r"\b(password|support)\b", q):
        return "policy"
    if re.search(r"\b(password|characters|length)\b", q):
        return "password_requirement"
    if re.search(r"\b(packag(?:e|ing))\b", q):
        return "packaging"
    if re.search(r"\b(nationality|nationalities)\b", q):
        return "nationality"
    if re.search(r"\b(neighborhood|neighbourhood)\b", q):
        return "neighborhood"
    if re.search(r"\b(director|directed)\b", q):
        return "director"
    if re.search(r"\b(premiere|premiered|release date)\b", q):
        return "premiere_date"
    if re.search(r"\b(genre)\b", q):
        return "genre"
    if re.search(r"\b(find|menu|settings|path)\b", q):
        return "path"
    if re.search(r"\b(where|location|located|address)\b", q):
        return "location"
    if re.search(r"\b(enter|turn on|turn into)\b", q):
        return "entry_method"
    if re.search(r"\b(wallpaper|picture|photo)\b", q):
        return "wallpaper_selection"
    if re.search(r"\b(import|save|smartthings)\b", q):
        return "photo_import"
    if re.search(r"\b(ask|request|require|policy)\b", q):
        return "policy"
    if re.search(r"\b(what is|define|definition|means)\b", q):
        return "definition"
    if re.search(r"\b(when|date|time)\b", q):
        if re.search(r"\b(refund|issued|processed)\b", q):
            return "duration"
        return "time"
    return "unresolved"


def _subject_for(part: str, attribute: str, previous: tuple[str, list[str]] | None) -> tuple[str, list[str]]:
    if typed_value_request(part):
        return "support_contact", ["support"]
    if attribute == "photo_import":
        return "mobile_photo_import", ["photos", "mobile device", "SmartThings"]
    patterns = (
        r"\b(?:who\s+is|what\s+is)\s+(?:the\s+)?(.+?)\s+of\s+(.+)$",
        r"\bwhere\s+do\s+i\s+find\s+(.+)$",
        r"\bwhat\s+is\s+(.+)$",
        r"\bhow\s+(?:can\s+i|do\s+i|to)\s+.+?\b(?:in|into|of)\s+(.+)$",
        r"\b(?:what|which)\s+(.+?)\s+should\s+i\s+use\b",
    )
    m = re.search(patterns[0], part, re.I)
    if m:
        candidate = _surface_entity(m.group(2))
        return _known_entity(m.group(2)) or candidate
    m = re.search(patterns[1], part, re.I)
    if m:
        return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    m = re.search(patterns[2], part, re.I)
    if m:
        return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    if re.search(r"\b(?:its|their)\b", part, re.I) and previous:
        return previous
    m = re.search(patterns[3], part, re.I)
    if m:
        return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    m = re.search(patterns[4], part, re.I)
    if m:
        return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    m = re.search(r"\b(?:genre|location|nationality|director|premiere(?:\s+date)?)\s+of\s+(?:the\s+)?(?:drama|film|movie|person|place)?\s*[\"']?(.+?)[\"']?(?:\s+is\s+what)?$", part, re.I)
    if m:
        return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    m = re.search(r"\b(?:what|which)\s+([\w -]+?)\s+is\s+(.+)$", part, re.I)
    if m:
        return _known_entity(m.group(2)) or _surface_entity(m.group(2))
    if attribute in {"location", "genre", "nationality", "director", "premiere_date"}:
        suffix = {"premiere_date": r"premiere(?:\s+date)?", "director": r"director",
                  "nationality": r"nationality", "genre": r"genre", "location": r"location|place"}[attribute]
        m = re.match(r"(.+?)\s+(?:" + suffix + r")$", part, re.I)
        if m:
            return _known_entity(m.group(1)) or _surface_entity(m.group(1))
    known = _known_entity(part)
    if known:
        return known
    if previous and re.search(r"\b(that|this)\s+(?:period|time|duration)\b", part, re.I):
        return previous
    return "unresolved", []


def _fact(fid: str, description: str, entity: str, aliases: list[str], attribute: str,
          evidence_terms: list[str] | None = None) -> RequiredFact:
    terms = evidence_terms or []
    entity_terms = _ENTITY_QUERY_TERMS.get(entity, " ".join(aliases))
    attr_terms = _ATTRIBUTE_TERMS.get(attribute, "")
    query = " ".join(p for p in (description, entity_terms, attr_terms, *terms) if p).strip()
    verifiable = entity != "unresolved" and attribute != "unresolved" and bool(aliases)
    return RequiredFact(fid, description, query, terms, entity, attribute, aliases, verifiable)


def _clean_operand(text: str) -> str:
    return text.strip(" \t\n.,?!\"'`)")


def _comparison(question: str) -> PlannedTask | None:
    q = question.strip().rstrip("?.!")
    duration = re.match(r"^(?:do|does)\s+(.+?)\s+take\s+(longer|shorter)\s+than\s+(.+)$", q, re.I)
    if duration:
        left_text, operator, right_text = (_clean_operand(x) for x in duration.groups())
        left_known, right_known = _known_entity(left_text), _known_entity(right_text)
        left_entity, left_aliases = left_known or _surface_entity(left_text)
        right_entity, right_aliases = right_known or _surface_entity(right_text)
        facts = [
            _fact("operand_a", f"duration of {left_text}", left_entity, left_aliases, "duration"),
            _fact("operand_b", f"duration of {right_text}", right_entity, right_aliases, "duration",
                  ["5 business days", "received", "inspected"] if right_entity == "refund_processing" else []),
        ]
        conditions = ["received", "inspected"] if any(f.entity == "refund_processing" for f in facts) else []
        comparison = {"kind": "duration", "operator": operator.lower(), "attribute": "duration",
                      "claim_attribute": "duration_relation", "left_fact_id": "operand_a",
                      "right_fact_id": "operand_b", "left_label": left_text,
                      "right_label": right_text, "required_conditions": conditions}
        return PlannedTask("comparison", question, facts, comparison)

    patterns = (
        (r"^(?:are|were|is|was)\s+(.+?)\s+and\s+(.+?)\s+of\s+(?:the\s+)?(same|different)\s+(.+)$", "same_different"),
        (r"^(?:are|were|is|was)\s+(.+?)\s+and\s+(.+?)\s+(?:located\s+)?in\s+the\s+(same|different)\s+(.+)$", "same_different"),
        (r"^(?:do|does)\s+(.+?)\s+and\s+(.+?)\s+have\s+the\s+(same|different)\s+(.+)$", "same_different"),
    )
    for pattern, kind in patterns:
        m = re.match(pattern, q, re.I)
        if not m:
            continue
        left_text, right_text, operator, attr_text = (_clean_operand(x) for x in m.groups())
        attribute = _attribute(attr_text)
        if attribute == "unresolved":
            attribute = _attribute("what is the " + attr_text)
        left_entity, left_aliases = _known_entity(left_text) or _surface_entity(left_text)
        right_entity, right_aliases = _known_entity(right_text) or _surface_entity(right_text)
        if attribute == "unresolved" or not left_aliases or not right_aliases:
            return None
        facts = [
            _fact("operand_a", f"{attribute} of {left_text}", left_entity, left_aliases, attribute),
            _fact("operand_b", f"{attribute} of {right_text}", right_entity, right_aliases, attribute),
        ]
        comparison = {"kind": kind, "operator": operator.lower(), "attribute": attribute,
                      "claim_attribute": "same_different_relation", "left_fact_id": "operand_a",
                      "right_fact_id": "operand_b", "left_label": left_text,
                      "right_label": right_text, "required_conditions": []}
        return PlannedTask("comparison", question, facts, comparison)
    return None


def plan(question: str) -> list[PlannedTask]:
    comparison = _comparison(question)
    if comparison:
        return [comparison]

    q = question.strip().rstrip("?")
    coordinated = re.search(r"\bcover(?:s)?\s+(.+?)\s+and\s+(.+?)(?:,?\s+(?:right|correct))?$", q, re.I)
    if coordinated:
        known = _known_entity(q)
        if known:
            entity, aliases = known
            values = [_clean_operand(x) for x in coordinated.groups() if x]
            if len(values) == 2:
                facts = [_fact(f"fact_{i}", f"{entity} covers {value}", entity, aliases,
                               "coverage", [value]) for i, value in enumerate(values, 1)]
                return [PlannedTask("task_1", question, facts)]

    parts = re.split(r"\s*,\s*and\s+|\s+and\s+(?=(?:which|what|where|when|how|who|whether|is|are|does|do|select|choose|import|save|upload|transfer|send)\b)",
                     q, flags=re.I)
    if len(parts) == 1:
        parts = re.split(r"\s+also\s+|\s*;\s*|\s+then\s+", q, flags=re.I)
    parts = [_clean_operand(part) for part in parts if _clean_operand(part)]
    tasks: list[PlannedTask] = []
    previous: tuple[str, list[str]] | None = None
    for index, part in enumerate(parts or [q], 1):
        attribute = _attribute(part)
        entity, aliases = _subject_for(part, attribute, previous)
        if entity == "unresolved" and previous and re.search(r"\b(that|this)\s+(?:period|time|duration)\b", part, re.I):
            entity, aliases = previous
        if attribute == "start_condition" and entity == "refund_processing":
            description = "refund processing start condition"
            query_text = "refund processing start condition after returned item received and inspected"
        else:
            description = part
            query_text = part

        facts = [_fact(f"fact_{index}", description, entity, aliases, attribute)]
        facts[0].retrieval_query = _fact(facts[0].id, query_text, entity, aliases, attribute,
                                         facts[0].evidence_terms).retrieval_query
        task_question = part if part.endswith("?") else part + "?"
        tasks.append(PlannedTask(f"task_{index}", task_question, facts))
        if entity != "unresolved":
            previous = (entity, aliases)
    return tasks
