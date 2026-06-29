"""LLM providers: deterministic offline mock + OpenAI / Azure OpenAI adapters."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..text import sentences, tokenize


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
        )
        usage = resp.usage.model_dump() if resp.usage else {}
        return LLMResult(text=resp.choices[0].message.content or "", usage=usage)


def build_llm(provider: str, model: str) -> LLMProvider:
    if provider in ("openai", "azure"):
        return OpenAILLM(model)
    return MockLLM()
