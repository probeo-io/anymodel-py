"""Tests for Router — parameter stripping, alias resolution, missing provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from anymodel._types import AnyModelError
from anymodel._router import Router
from anymodel.providers._registry import ProviderRegistry
from anymodel.utils._id import generate_id


def _make_mock_adapter(
    name: str,
    supported_params: set[str] | None = None,
) -> Any:
    """Create a mock provider adapter with configurable supported params."""
    params = supported_params or set()
    last_request: dict[str, Any] = {}

    mock_response: dict[str, Any] = {
        "id": generate_id(),
        "object": "chat.completion",
        "created": 1000,
        "model": f"{name}/test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    class _Adapter:
        @property
        def name(self) -> str:
            return name

        async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
            last_request.update(request)
            return dict(mock_response)

        async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
            async def _empty() -> AsyncIterator[dict[str, Any]]:
                return
                yield  # type: ignore[misc]
            return _empty()

        async def list_models(self) -> list[dict[str, Any]]:
            return []

        def supports_parameter(self, param: str) -> bool:
            return param in params

        def supports_batch(self) -> bool:
            return False

        def translate_error(self, error: Exception) -> dict[str, Any]:
            return {"code": 500, "message": "error", "metadata": {}}

        def get_last_request(self) -> dict[str, Any]:
            return last_request

    return _Adapter()


async def test_strips_unsupported_parameters() -> None:
    registry = ProviderRegistry()
    adapter = _make_mock_adapter(
        "test",
        {"temperature", "max_tokens", "top_p", "stop", "stream", "tools", "tool_choice"},
    )
    registry.register("test", adapter)

    router = Router(registry)
    request = {
        "model": "test/some-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_k": 40,
        "seed": 42,
        "frequency_penalty": 0.5,
    }

    await router.complete(request)
    sent = adapter.get_last_request()

    assert sent["temperature"] == 0.7
    assert "top_k" not in sent
    assert "seed" not in sent
    assert "frequency_penalty" not in sent


async def test_resolves_aliases() -> None:
    registry = ProviderRegistry()
    adapter = _make_mock_adapter("anthropic", {"temperature"})
    registry.register("anthropic", adapter)

    router = Router(registry, aliases={"smart": "anthropic/claude-sonnet-4-6"})

    await router.complete({
        "model": "smart",
        "messages": [{"role": "user", "content": "Hi"}],
    })

    sent = adapter.get_last_request()
    assert sent["model"] == "claude-sonnet-4-6"


async def test_throws_on_missing_provider() -> None:
    registry = ProviderRegistry()
    router = Router(registry)

    with pytest.raises(AnyModelError):
        await router.complete({
            "model": "unknown/model",
            "messages": [{"role": "user", "content": "Hi"}],
        })
