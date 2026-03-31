"""Tests for BatchManager native batch routing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from anymodel.batch._manager import BatchManager
from anymodel.batch._store import BatchStore
from anymodel.providers._adapter import NativeBatchStatus


def _mock_router() -> Any:
    """Create a mock router."""

    class _Router:
        complete = AsyncMock()
        stream = AsyncMock()

    return _Router()


def _mock_batch_adapter(**overrides: Any) -> Any:
    """Create a mock batch adapter."""

    class _Adapter:
        create_batch = overrides.get("create_batch", AsyncMock(return_value={
            "providerBatchId": "provider-batch-123",
            "metadata": {"some": "data"},
        }))
        poll_batch = overrides.get("poll_batch", AsyncMock(return_value=NativeBatchStatus(
            status="completed", total=2, completed=2, failed=0,
        )))
        get_batch_results = overrides.get("get_batch_results", AsyncMock(return_value=[
            {
                "custom_id": "req-1",
                "status": "success",
                "response": {
                    "id": "gen-1",
                    "object": "chat.completion",
                    "created": 1000,
                    "model": "openai/gpt-4o",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello 1"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                "error": None,
            },
            {
                "custom_id": "req-2",
                "status": "success",
                "response": {
                    "id": "gen-2",
                    "object": "chat.completion",
                    "created": 1000,
                    "model": "openai/gpt-4o",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello 2"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                "error": None,
            },
        ]))
        cancel_batch = overrides.get("cancel_batch", AsyncMock())

    return _Adapter()


async def test_uses_native_adapter_when_registered(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter()
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    batch = await manager.create({
        "model": "openai/gpt-4o",
        "requests": [
            {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
            {"custom_id": "req-2", "messages": [{"role": "user", "content": "Hello"}]},
        ],
    })

    assert batch["batch_mode"] == "native"
    assert batch["provider_name"] == "openai"

    # Give background tasks a moment
    await asyncio.sleep(0.1)

    adapter.create_batch.assert_called_once()
    call_args = adapter.create_batch.call_args
    assert call_args[0][0] == "gpt-4o"


async def test_falls_back_to_concurrent_when_no_native_adapter(tmp_path: Path) -> None:
    router = _mock_router()
    router.complete = AsyncMock(return_value={
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1000,
        "model": "google/gemini-2.0-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    })

    manager = BatchManager(router, dir=str(tmp_path))

    batch = await manager.create({
        "model": "google/gemini-2.0-flash",
        "requests": [
            {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
        ],
    })

    assert batch["batch_mode"] == "concurrent"


async def test_persists_provider_state_for_native_batches(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter()
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    batch = await manager.create({
        "model": "openai/gpt-4o",
        "requests": [
            {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
        ],
    })

    await asyncio.sleep(0.1)

    store = BatchStore(str(tmp_path))
    state = await store.load_provider_state(batch["id"])
    assert state is not None
    assert state["providerBatchId"] == "provider-batch-123"


async def test_polls_and_downloads_results(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter()
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    results = await manager.create_and_poll(
        {
            "model": "openai/gpt-4o",
            "requests": [
                {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
                {"custom_id": "req-2", "messages": [{"role": "user", "content": "Hello"}]},
            ],
        },
        interval=0.05,
    )

    assert results["status"] == "completed"
    assert len(results["results"]) == 2
    assert results["results"][0]["custom_id"] == "req-1"
    assert results["results"][1]["custom_id"] == "req-2"
    assert results["usage_summary"]["total_prompt_tokens"] == 20
    assert results["usage_summary"]["total_completion_tokens"] == 10


async def test_cancels_native_batch_at_provider(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter(
        poll_batch=AsyncMock(return_value=NativeBatchStatus(
            status="processing", total=2, completed=0, failed=0,
        )),
    )
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("anthropic", adapter)

    batch = await manager.create({
        "model": "anthropic/claude-sonnet-4-6",
        "requests": [
            {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
        ],
    })

    await asyncio.sleep(0.1)

    cancelled = await manager.cancel(batch["id"])
    assert cancelled["status"] == "cancelled"
    adapter.cancel_batch.assert_called_with("provider-batch-123")


async def test_handles_native_batch_creation_failure(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter(
        create_batch=AsyncMock(side_effect=RuntimeError("Upload failed")),
    )
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    with pytest.raises(RuntimeError, match="Upload failed"):
        await manager.create({
            "model": "openai/gpt-4o",
            "requests": [
                {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
            ],
        })

    # Batch should be marked as failed — retrieve via list
    batches = await manager.list()
    assert len(batches) == 1
    assert batches[0]["status"] == "failed"


async def test_handles_per_item_errors_in_results(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter(
        get_batch_results=AsyncMock(return_value=[
            {
                "custom_id": "req-1",
                "status": "success",
                "response": {
                    "id": "gen-1",
                    "object": "chat.completion",
                    "created": 1000,
                    "model": "openai/gpt-4o",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
                "error": None,
            },
            {
                "custom_id": "req-2",
                "status": "error",
                "response": None,
                "error": {"code": 400, "message": "Invalid request"},
            },
        ]),
        poll_batch=AsyncMock(return_value=NativeBatchStatus(
            status="completed", total=2, completed=1, failed=1,
        )),
    )
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    results = await manager.create_and_poll(
        {
            "model": "openai/gpt-4o",
            "requests": [
                {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
                {"custom_id": "req-2", "messages": [{"role": "user", "content": "Bad request"}]},
            ],
        },
        interval=0.05,
    )

    assert results["status"] == "completed"
    assert len(results["results"]) == 2
    assert results["results"][0]["status"] == "success"
    assert results["results"][1]["status"] == "error"
    assert results["results"][1]["error"]["message"] == "Invalid request"


async def test_reports_progress_via_callback(tmp_path: Path) -> None:
    router = _mock_router()
    adapter = _mock_batch_adapter()
    manager = BatchManager(router, dir=str(tmp_path))
    manager.register_batch_adapter("openai", adapter)

    progress_updates: list[dict[str, Any]] = []

    await manager.create_and_poll(
        {
            "model": "openai/gpt-4o",
            "requests": [
                {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}]},
            ],
        },
        interval=0.05,
        on_progress=lambda batch: progress_updates.append(dict(batch)),
    )

    assert len(progress_updates) > 0
