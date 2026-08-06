"""Tests for the OpenAI adapter's Responses API routing (reasoning + function tools)."""

from __future__ import annotations

from anymodel.providers._openai import OpenAIAdapter


def _request(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "Research Texas LLC fees."}],
    }
    base.update(overrides)
    return base


def test_uses_responses_shape_and_translates_function_tools_when_reasoning_is_configured() -> None:
    adapter = OpenAIAdapter("test-key")
    request = _request(
        reasoning={"effort": "low"},
        tools=[{"type": "function", "function": {"name": "list_pages", "description": "List pages", "parameters": {"type": "object"}}}],
        tool_choice={"type": "function", "function": {"name": "list_pages"}},
    )

    assert adapter._uses_responses(request) is True
    body = adapter._build_responses_body(request)

    assert body["input"] == [{"role": "user", "content": "Research Texas LLC fees."}]
    assert body["reasoning"] == {"effort": "low"}
    assert body["tools"] == [{"type": "function", "name": "list_pages", "description": "List pages", "parameters": {"type": "object"}}]
    assert body["tool_choice"] == {"type": "function", "name": "list_pages"}


def test_translates_responses_function_calls_into_common_tool_calls() -> None:
    adapter = OpenAIAdapter("test-key")
    completion = adapter._translate_responses_response({
        "id": "resp_456",
        "model": "gpt-5.6-luna",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": [{"type": "function_call", "call_id": "call_1", "name": "list_pages", "arguments": '{"customer":"c"}'}],
    })

    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["choices"][0]["message"]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "list_pages", "arguments": '{"customer":"c"}'}},
    ]


def test_chat_completions_path_unaffected_when_reasoning_not_set() -> None:
    adapter = OpenAIAdapter("test-key")
    request = _request(max_tokens=100)

    assert adapter._uses_responses(request) is False
    body = adapter._build_request_body(request)
    assert "reasoning" not in body
    # gpt-5.6-luna matches the max_completion_tokens translation rule
    assert body["max_completion_tokens"] == 100


def test_prompt_cache_key_and_ttl_applied_to_chat_and_responses_bodies() -> None:
    adapter = OpenAIAdapter("test-key")
    request = _request(cache={"key": "workflow-v3", "ttl": "24h"})

    chat_body = adapter._build_request_body(request)
    assert chat_body["prompt_cache_key"] == "workflow-v3"
    assert chat_body["prompt_cache_retention"] == "24h"

    responses_body = adapter._build_responses_body(_request(cache={"key": "workflow-v3", "ttl": "5m"}, reasoning={"effort": "low"}))
    assert responses_body["prompt_cache_key"] == "workflow-v3"
    assert responses_body["prompt_cache_retention"] == "in_memory"
