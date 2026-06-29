"""Tracing, token-cost accounting, and run metrics."""
from __future__ import annotations
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

PRICE_BOOK = {  # USD / 1K tokens (illustrative)
    "gpt-4o": (0.0025, 0.010), "gpt-4o-mini": (0.00015, 0.0006),
    "azure-gpt-4o": (0.0025, 0.010), "mock": (0.0, 0.0),
}


def estimate_cost(model: str, inp: int, out: int) -> float:
    pin, pout = PRICE_BOOK.get(model, (0.0, 0.0))
    return round(inp / 1000 * pin + out / 1000 * pout, 6)


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end is None else round((self.end - self.start) * 1000, 2)


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(name, uuid.uuid4().hex[:8], self._stack[-1] if self._stack else None,
                 time.time(), attributes=dict(attrs))
        self.spans.append(s)
        self._stack.append(s.span_id)
        try:
            yield s
        finally:
            s.end = time.time()
            self._stack.pop()

    def record_usage(self, model: str, usage: dict[str, int]) -> None:
        inp = int(usage.get("prompt_tokens", 0))
        out = int(usage.get("completion_tokens", 0))
        self.input_tokens += inp
        self.output_tokens += out
        self.cost_usd = round(self.cost_usd + estimate_cost(model, inp, out), 6)

    def tree(self) -> str:
        by_parent: dict[str | None, list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(pid: str | None, depth: int) -> None:
            for s in by_parent.get(pid, []):
                lines.append("  " * depth + f"{s.name} ({s.duration_ms} ms)")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {"spans": len(self.spans), "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost_usd": self.cost_usd}
