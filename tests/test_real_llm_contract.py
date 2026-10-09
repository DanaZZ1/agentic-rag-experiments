"""Tests for the V3 real-provider JSON contract boundary."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from production_agentic_rag.agents.rag_agent import (
    LLMOutputContractError,
    _SYSTEM,
    _parse_generated_tasks,
)
from production_agentic_rag.providers import llm as llm_module


def _task(task_id="task_1"):
    ref = {"fact_id": "fact_1", "chunk_id": "handbook:0",
           "quote": "Returns are accepted within 30 days."}
    claim = {"claim_id": "claim_fact_1", "entity": "return_eligibility",
             "attribute": "duration", "text": ref["quote"],
             "fact_ids": ["fact_1"], "evidence_refs": [ref]}
    return {"task_id": task_id, "status": "supported", "answer": ref["quote"],
            "claims": [claim], "evidence_refs": [ref]}


def test_prompt_spells_out_the_existing_task_and_evidence_contract():
    for required in ('"tasks"', '"task_id"', '"status"', '"supported"',
                     '"not_in_corpus"', '"answer"', '"claims"', '"claim_id"',
                     '"entity"', '"attribute"', '"text"', '"fact_ids"',
                     '"evidence_refs"', '"chunk_id"', '"quote"'):
        assert required in _SYSTEM


def test_parser_accepts_contract_json_and_whole_response_markdown_fence():
    raw = json.dumps({"tasks": [_task()]})
    assert _parse_generated_tasks(raw, {"task_1"})["task_1"]["status"] == "supported"
    assert _parse_generated_tasks(f"```json\n{raw}\n```", {"task_1"})["task_1"]["task_id"] == "task_1"


@pytest.mark.parametrize("raw, message", [
    ('{"claims": []}', 'object with a "tasks" array'),
    ('{"tasks": [{"status": "supported"}]}', "missing a non-empty task_id"),
    ('{"tasks": [{"task_id": "task_1", "status": "unknown"}]}', "invalid status"),
    ('{"tasks": []}', "task_id mismatch"),
    ('```json\nnot json\n```', "not valid JSON"),
])
def test_parser_fails_clearly_for_invalid_contract(raw, message):
    with pytest.raises(LLMOutputContractError, match=message):
        _parse_generated_tasks(raw, {"task_1"})


def test_openai_adapter_uses_json_schema_only_for_v3_calls(monkeypatch):
    calls = []

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="{}")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda: fake_client
    fake_openai.AzureOpenAI = lambda: fake_client
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    adapter = llm_module.OpenAILLM("gpt-4o-mini")
    adapter.complete("system", "RAG_TASKS_V3\n{}")
    adapter.complete("system", "RAG_TASKS_V1\n{}")

    schema_request = calls[0]["response_format"]
    assert schema_request["type"] == "json_schema"
    assert schema_request["json_schema"]["strict"] is True
    assert "response_format" not in calls[1]
    assert calls[0]["temperature"] == calls[1]["temperature"] == 0.0
