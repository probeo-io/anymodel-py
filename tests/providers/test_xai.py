"""Tests for the xAI provider adapter."""

from __future__ import annotations

from anymodel.providers._xai import XAIAdapter


def test_chat_response_preserves_citations() -> None:
    adapter = XAIAdapter("test-key")
    completion = adapter._translate_chat_response({
        "id": "abc123",
        "model": "grok-4.5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "citations": ["https://example.com"],
    })
    assert completion["model"] == "xai/grok-4.5"
    assert completion["citations"] == ["https://example.com"]
    assert completion["id"] == "gen-abc123"


def test_responses_response_extracts_text_and_citations() -> None:
    adapter = XAIAdapter("test-key")
    completion = adapter._translate_responses_response({
        "id": "resp_1",
        "model": "grok-4.5",
        "output_text": "The answer is 42.",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "citations": ["https://example.com"],
    })
    assert completion["choices"][0]["message"]["content"] == "The answer is 42."
    assert completion["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert completion["citations"] == ["https://example.com"]


def test_build_chat_body_includes_prompt_cache_key() -> None:
    adapter = XAIAdapter("test-key")
    body = adapter._build_chat_body({
        "model": "grok-4.5",
        "messages": [{"role": "user", "content": "hi"}],
        "cache": {"key": "conv-42"},
    })
    assert body["prompt_cache_key"] == "conv-42"


def test_supports_parameter_includes_cache() -> None:
    adapter = XAIAdapter("test-key")
    assert adapter.supports_parameter("cache") is True
    assert adapter.supports_parameter("nonexistent") is False


def test_list_models_returns_grok_models_with_context_limits() -> None:
    import asyncio

    adapter = XAIAdapter("test-key")
    models = asyncio.run(adapter.list_models())
    ids = [m["id"] for m in models]
    assert "xai/grok-4.5" in ids
    assert all(m["context_length"] == 256000 for m in models)
