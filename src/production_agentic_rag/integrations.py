"""Optional integrations with sibling portfolio libraries (graceful fallback).

If `mcp-tool-gateway` is installed, every retrieval tool call can be routed through
a secure plane (JWT auth + rate limiting + prompt-injection inspection + audit).
If `llm-observability` is installed, the pipeline trace can be exported into its
Monitor report (token/cost/latency). Both are optional — the core works without them.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .retrieval.hybrid import HybridRetriever, Scored

try:                                  # optional: mcp-tool-gateway
    from mcp_gateway import MCPServer, MCPTool, SecureToolGateway, encode
    HAS_GATEWAY = True
except Exception:                     # pragma: no cover - depends on install
    HAS_GATEWAY = False

try:                                  # optional: llm-observability
    from llm_observability import Monitor
    HAS_OBSERVABILITY = True
except Exception:                     # pragma: no cover
    HAS_OBSERVABILITY = False


@dataclass
class SecuredRetrievalTool:
    """Retrieval tool whose every call first passes through the mcp-tool-gateway
    security plane (auth, rate limit, prompt-injection inspection, audit) before
    the underlying hybrid retriever executes. Defense-in-depth even for in-process
    tool calls. Requires `pip install '.[gateway]'`."""

    retriever: HybridRetriever
    secret: str = "rag-gateway-secret"
    k: int = 4
    name: str = "search_knowledge_base"
    description: str = "Secure retrieval over the indexed corpus."

    def __post_init__(self) -> None:
        if not HAS_GATEWAY:
            raise RuntimeError("mcp-tool-gateway not installed; pip install '.[gateway]'")
        server = MCPServer()
        server.register(MCPTool(
            self.name, self.description,
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=lambda query: "authorized",
            required_scope="kb:read",
        ))
        self._gateway = SecureToolGateway(server=server, secret=self.secret)
        self._token = encode({"sub": "rag-agent", "scopes": ["kb:read"]}, self.secret)

    def __call__(self, query: str) -> list[Scored]:
        resp = self._gateway.handle(
            {"id": 1, "method": "tools/call",
             "params": {"name": self.name, "arguments": {"query": query}}},
            self._token,
        )
        if "error" in resp:
            raise PermissionError(resp["error"]["message"])
        return self.retriever.retrieve(query, k=self.k)

    @property
    def audit(self) -> Any:
        return self._gateway.audit


def export_observability(tracer: Any, model: str = "mock") -> dict[str, Any]:
    """Export a pipeline Tracer into an llm-observability Monitor report if available,
    else fall back to the built-in trace summary."""
    if not HAS_OBSERVABILITY:
        return tracer.summary()
    mon = Monitor()
    mon.record(model=model, name="rag.run",
               input_tokens=getattr(tracer, "input_tokens", 0),
               output_tokens=getattr(tracer, "output_tokens", 0),
               latency_ms=0.0)
    return mon.report()
