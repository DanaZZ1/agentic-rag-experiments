"""Input and output guardrails for the RAG pipeline."""
from __future__ import annotations
import re
from dataclasses import dataclass

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
