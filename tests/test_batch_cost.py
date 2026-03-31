"""Tests for batch cost calculation with service_tier discounts."""

import shutil
from pathlib import Path
from typing import Any

import pytest

from anymodel.batch._manager import BatchManager

try:
    from anymodel.generated.pricing import calculate_cost
except ImportError:
    def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0


def _make_completion(model: str) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


class _MockRouter:
    """Minimal mock router for batch tests."""

    def __init__(self, model: str) -> None:
        self._completion = _make_completion(model)

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._completion

    async def complete_with_meta(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"completion": self._completion, "meta": {"headers": {}}}


MODEL = "openai/gpt-4o"
REQUESTS = [
    {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hello"}]},
    {"custom_id": "req-2", "messages": [{"role": "user", "content": "World"}]},
]


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "batch_cost_test"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_concurrent_without_flex_full_price(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    results = await manager.create_and_poll(
        {"model": MODEL, "requests": REQUESTS, "batch_mode": "concurrent"},
        interval=0.05,
    )
    full_cost = calculate_cost(MODEL, 100, 50)
    assert results["usage_summary"]["estimated_cost"] == pytest.approx(full_cost * 2, abs=1e-10)


@pytest.mark.asyncio
async def test_concurrent_with_flex_50_discount(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    results = await manager.create_and_poll(
        {"model": MODEL, "requests": REQUESTS, "batch_mode": "concurrent", "options": {"service_tier": "flex"}},
        interval=0.05,
    )
    full_cost = calculate_cost(MODEL, 100, 50)
    assert results["usage_summary"]["estimated_cost"] == pytest.approx(full_cost * 2 * 0.5, abs=1e-10)


@pytest.mark.asyncio
async def test_concurrent_with_auto_full_price(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    results = await manager.create_and_poll(
        {"model": MODEL, "requests": REQUESTS, "batch_mode": "concurrent", "options": {"service_tier": "auto"}},
        interval=0.05,
    )
    full_cost = calculate_cost(MODEL, 100, 50)
    assert results["usage_summary"]["estimated_cost"] == pytest.approx(full_cost * 2, abs=1e-10)


@pytest.mark.asyncio
async def test_service_tier_persisted(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    batch = await manager.create(
        {"model": MODEL, "requests": REQUESTS, "batch_mode": "concurrent", "options": {"service_tier": "flex"}},
    )
    assert batch.get("service_tier") == "flex"

    retrieved = await manager.get(batch["id"])
    assert retrieved is not None
    assert retrieved.get("service_tier") == "flex"


@pytest.mark.asyncio
async def test_service_tier_defaults_to_none(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    batch = await manager.create(
        {"model": MODEL, "requests": REQUESTS, "batch_mode": "concurrent"},
    )
    assert batch.get("service_tier") is None


@pytest.mark.asyncio
async def test_service_tier_falls_back_to_first_request(batch_dir: Path):
    manager = BatchManager(_MockRouter(MODEL), dir=str(batch_dir))
    batch = await manager.create({
        "model": MODEL,
        "requests": [
            {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hi"}], "service_tier": "flex"},
            {"custom_id": "req-2", "messages": [{"role": "user", "content": "Hi"}]},
        ],
        "batch_mode": "concurrent",
    })
    assert batch.get("service_tier") == "flex"
