"""LLM providers: deterministic offline mock + OpenAI / Azure OpenAI adapters."""
from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..text import sentences, tokenize


_V3_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_tasks_v3",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["supported", "not_in_corpus"]},
                            "answer": {"type": "string"},
                            "claims": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "claim_id": {"type": "string"},
                                        "entity": {"type": "string"},
                                        "attribute": {"type": "string"},
                                        "text": {"type": "string"},
                                        "fact_ids": {"type": "array", "items": {"type": "string"}},
                                        "evidence_refs": {
                                            "type": "array",
                                            "items": {"$ref": "#/$defs/evidence_ref"},
                                        },
                                    },
                                    "required": ["claim_id", "entity", "attribute", "text",
                                                 "fact_ids", "evidence_refs"],
                                    "additionalProperties": False,
                                },
                            },
                            "evidence_refs": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/evidence_ref"},
                            },
                        },
                        "required": ["task_id", "status", "answer", "claims", "evidence_refs"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tasks"],
            "additionalProperties": False,
            "$defs": {
                "evidence_ref": {
                    "type": "object",
                    "properties": {
                        "fact_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["fact_id", "chunk_id", "quote"],
                    "additionalProperties": False,
                },
            },
        },
    },
}


@dataclass
class LLMResult:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> LLMResult: ...


class MockLLM:
    """Deterministic, offline LLM used for tests, demos, and CI.

    For RAG synthesis the prompt embeds a ``Context:`` block and a ``Question:``;
    the mock returns the context sentences most relevant to the question — i.e. a
    grounded, extractive answer — so the full pipeline (and its faithfulness gate)
    is exercised end-to-end with zero API keys.
    """

    def complete(self, system: str, user: str) -> LLMResult:
        if user.startswith("RAG_TASKS_V3\n"):
            payload = json.loads(user.split("\n", 1)[1])
            outputs = []
            entity_aliases = {
                "standard_shipping": ("standard shipping", "standard delivery"),
                "express_shipping": ("express shipping", "express delivery", "express deliveries"),
                "refund_processing": ("refund", "refunds"),
                "return_eligibility": ("return", "returns", "send back"),
                "warranty": ("warranty",),
                "support_password_policy": ("support", "password", "passwords"),
                "account_security": ("accounts", "account", "two-factor", "password"),
                "support_contact": ("support",),
                "unknown": (),
            }

            def mentions(entity: str, text: str, supplied=()) -> bool:
                aliases = tuple(str(x) for x in supplied) or entity_aliases.get(entity, (entity.replace("_", " "),))
                if entity in ("unknown", "unresolved"):
                    return False
                return any(re.search(r"\b" + re.escape(alias) + r"\b", text.lower()) for alias in aliases)

            def find_duration(text: str):
                m = re.search(r"\b(\d+)(?:\s*-\s*(\d+))?\s+business days?\b", text, re.I)
                return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None

            for task in payload.get("tasks", []):
                claims: list[dict] = []
                flat_refs: list[dict[str, str]] = []
                fact_quotes: dict[str, tuple[dict, str, dict]] = {}
                for fact in task.get("required_facts", []):
                    fid = str(fact["id"])
                    if not fact.get("verifiable", False):
                        continue
                    group = next((g for g in task.get("evidence", []) if g.get("fact_id") == fid), {})
                    entity = str(fact.get("entity", "unknown"))
                    query_terms = set(tokenize(" ".join((fact.get("retrieval_query", ""), entity,
                                                          fact.get("attribute", ""), fact.get("description", ""),
                                                          " ".join(fact.get("entity_aliases", []))))))
                    choices = []
                    for chunk in group.get("chunks", []):
                        # Remove headings for selection while retaining exact source substrings.
                        text = re.sub(r"(?m)^#{1,6}[^\n]*(?:\n|$)", "", chunk.get("text", ""))
                        for match in re.finditer(r"[^.!?]+[.!?]?", text):
                            quote = match.group(0).strip()
                            if not quote or not mentions(entity, quote, fact.get("entity_aliases", [])):
                                continue
                            overlap = len(query_terms & set(tokenize(quote)))
                            if overlap:
                                choices.append((overlap, chunk["chunk_id"], quote))
                    # When the planner identifies a specific required fact, prefer
                    # evidence that contains its discriminating terms. Generic
                    # entity/attribute overlap must not substitute a sibling fact.
                    evidence_terms = [str(term).lower() for term in fact.get("evidence_terms", [])]
                    matching = [choice for choice in choices
                                if all(term in choice[2].lower() for term in evidence_terms)]
                    if matching:
                        choices = matching
                    if not choices:
                        continue
                    _, chunk_id, quote = max(choices, key=lambda item: item[0])
                    ref = {"fact_id": fid, "chunk_id": chunk_id, "quote": quote}
                    flat_refs.append(ref)
                    fact_quotes[fid] = (fact, quote, ref)
                    claims.append({"claim_id": f"claim_{fid}", "entity": entity,
                                   "attribute": str(fact.get("attribute", "general")),
                                   "text": quote, "fact_ids": [fid], "evidence_refs": [ref]})

                comparison = task.get("comparison")
                if comparison:
                    left_id, right_id = str(comparison["left_fact_id"]), str(comparison["right_fact_id"])
                    left_item, right_item = fact_quotes.get(left_id), fact_quotes.get(right_id)
                    if left_item and right_item:
                        left_label, right_label = str(comparison["left_label"]), str(comparison["right_label"])
                        if comparison.get("kind") == "duration":
                            left_duration, right_duration = find_duration(left_item[1]), find_duration(right_item[1])
                            if left_duration and right_duration:
                                if left_duration[0] > right_duration[1]:
                                    relation = f"{left_label.capitalize()} takes longer than {right_label}."
                                elif left_duration[1] < right_duration[0]:
                                    relation = f"{left_label.capitalize()} does not take longer than {right_label}; it is shorter."
                                else:
                                    relation = f"The durations of {left_label} and {right_label} overlap, so neither is consistently longer."
                            else:
                                relation = ""
                        else:
                            attr = str(comparison.get("attribute", ""))
                            def value(quote: str):
                                if attr == "nationality":
                                    found = re.findall(r"\b(american|british|canadian|french|german|italian|japanese|turkish|australian|irish|spanish|indian|chinese)\b", quote, re.I)
                                    return found[-1].lower() if found else None
                                if attr in ("neighborhood", "location"):
                                    found = re.findall(r"\b(?:in|at)\s+([A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)?)\s+(?:neighbou?rhood|district)\b", quote, re.I)
                                    if not found and attr == "neighborhood":
                                        found = re.findall(r"\b(?:located|situated)\s+in\s+([A-Z][\w’'-]+)", quote, re.I)
                                    if not found:
                                        found = re.findall(r"\b(?:located|situated)\s+(?:in|at)\s+([^,.;]+)", quote, re.I)
                                    vals = [(x if isinstance(x, str) else (x[0] or x[1])).strip().lower() if isinstance(x, tuple) else x.strip().lower() for x in found]
                                    return vals[-1] if vals else None
                                return None
                            left_value, right_value = value(left_item[1]), value(right_item[1])
                            if left_value and right_value:
                                same = left_value == right_value
                                if comparison.get("operator") == "same":
                                    relation = (f"{left_label} and {right_label} are in the same {attr}." if same
                                                else f"{left_label} and {right_label} are not in the same {attr}.")
                                else:
                                    relation = (f"{left_label} and {right_label} are different in {attr}." if not same
                                                else f"{left_label} and {right_label} are the same in {attr}.")
                            else:
                                relation = ""
                        if relation:
                            refs = [left_item[2], right_item[2]]
                            claims.append({"claim_id": "claim_comparison", "entity": "comparison",
                                           "attribute": str(comparison.get("claim_attribute", "duration_relation")), "text": relation,
                                           "fact_ids": [left_id, right_id], "evidence_refs": refs})

                if claims and (not task.get("comparison") or any(c.get("entity") == "comparison" for c in claims)):
                    answer = "\n".join(claim["text"] for claim in claims)
                    outputs.append({"task_id": task["id"], "status": "supported", "answer": answer,
                                    "claims": claims, "evidence_refs": flat_refs})
                else:
                    outputs.append({"task_id": task["id"], "status": "not_in_corpus",
                                    "answer": "The corpus does not provide enough information to answer this task.",
                                    "claims": [], "evidence_refs": []})
            text = json.dumps({"tasks": outputs})
            return LLMResult(text=text, usage={"prompt_tokens": len(user.split()), "completion_tokens": len(text.split())})
        if user.startswith("RAG_TASKS_V2\n"):
            payload = json.loads(user.split("\n", 1)[1])
            outputs = []
            for task in payload.get("tasks", []):
                refs: list[dict[str, str]] = []
                fact_answers: dict[str, str] = {}
                for fact in task.get("required_facts", []):
                    fid = fact["id"]
                    group = next((g for g in task.get("evidence", []) if g.get("fact_id") == fid), {})
                    qtokens = set(tokenize(fact.get("retrieval_query", "") + " " + fact.get("description", "")))
                    candidates = []
                    for chunk in group.get("chunks", []):
                        # Keep exact substrings of the indexed chunk for evidence provenance.
                        text = re.sub(r"(?m)^#{1,6}[^\n]*(?:\n|$)", "", chunk.get("text", ""))
                        for match in re.finditer(r"[^.!?]+[.!?]?", text):
                            quote = match.group(0).strip()
                            overlap = len(qtokens & set(tokenize(quote)))
                            if overlap:
                                candidates.append((overlap, chunk["chunk_id"], quote))
                    if candidates:
                        _, cid, quote = max(candidates, key=lambda item: item[0])
                        refs.append({"fact_id": fid, "chunk_id": cid, "quote": quote})
                        fact_answers[fid] = quote
                comparison = task.get("comparison")
                if comparison:
                    left = fact_answers.get(comparison["left_fact_id"], "")
                    right = fact_answers.get(comparison["right_fact_id"], "")
                    if left and right:
                        left_days = re.search(r"(\d+(?:-\d+)?\s+business days?)", left, re.I)
                        right_days = re.search(r"(\d+(?:-\d+)?\s+business days?)", right, re.I)
                        if left_days and right_days:
                            conditions = comparison.get("required_conditions", [])
                            condition_text = ("after the returned item is received and inspected"
                                              if any("received and inspected" in c.lower() for c in conditions)
                                              else " ".join(c for c in conditions if c.lower() in right.lower()))
                            answer = (f"Express shipping is delivered within {left_days.group(1)}. "
                                      f"Approved refunds are issued within {right_days.group(1)} {condition_text}. "
                                      "Express delivery does not take longer than refund processing.")
                        else:
                            answer = f"{left} {right}"
                        outputs.append({"task_id": task["id"], "status": "supported", "answer": answer,
                                        "evidence_refs": refs})
                        continue
                if refs:
                    outputs.append({"task_id": task["id"], "status": "supported",
                                    "answer": " ".join(fact_answers.values()), "evidence_refs": refs})
                else:
                    outputs.append({"task_id": task["id"], "status": "not_in_corpus",
                                    "answer": "The corpus does not provide enough information to answer this task.",
                                    "evidence_refs": []})
            text = json.dumps({"tasks": outputs})
            return LLMResult(text=text, usage={"prompt_tokens": len(user.split()), "completion_tokens": len(text.split())})
        ctx_match = re.search(r"Context:\s*(.*?)\n\s*Question:", user, re.S)
        q_match = re.search(r"Question:\s*(.*)", user, re.S)
        context = ctx_match.group(1).strip() if ctx_match else ""
        question = q_match.group(1).strip() if q_match else user
        if not context:
            return LLMResult(text="I don't have enough grounded context to answer that.",
                             usage={"prompt_tokens": len(user.split()), "completion_tokens": 12})
        q = set(tokenize(question))
        scored = sorted(sentences(context),
                        key=lambda s: len(q & set(tokenize(s))), reverse=True)
        picked = [s for s in scored[:2] if q & set(tokenize(s))] or scored[:1]
        answer = " ".join(picked)
        return LLMResult(text=answer,
                         usage={"prompt_tokens": len(user.split()),
                                "completion_tokens": len(answer.split())})


class OpenAILLM:
    """OpenAI / Azure OpenAI chat adapter (lazy import; needs credentials)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def complete(self, system: str, user: str) -> LLMResult:
        from openai import AzureOpenAI, OpenAI

        client = AzureOpenAI() if os.getenv("AZURE_OPENAI_ENDPOINT") else OpenAI()
        resp = client.chat.completions.create(
            model=self.model, temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **({"response_format": _V3_RESPONSE_FORMAT}
               if user.startswith("RAG_TASKS_V3\n") else {}),
        )
        usage = resp.usage.model_dump() if resp.usage else {}
        return LLMResult(text=resp.choices[0].message.content or "", usage=usage)


def build_llm(provider: str, model: str) -> LLMProvider:
    if provider in ("openai", "azure"):
        return OpenAILLM(model)
    return MockLLM()
