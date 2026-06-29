"""Query planning: decompose a complex question into ordered sub-questions."""
from __future__ import annotations
import re


def plan(question: str) -> list[str]:
    """Heuristic decomposition (LLM-swappable: same return contract)."""
    parts = re.split(r"\?\s*|\band also\b|\band\b|;|\bthen\b", question, flags=re.I)
    subs = [p.strip(" ?.") for p in parts if len(p.strip(" ?.")) > 3]
    return subs or [question.strip()]
