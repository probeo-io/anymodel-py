"""Tests for batch cost calculation with service_tier discounts."""

import shutil
from pathlib import Path
from typing import Any

import pytest

from anymodel.batch._manager import BatchManager

try:
    from anymodel.generated.pricing import calculate_cost, get_model_pricing
except ImportError:
    def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        return 0.0

    def get_model_pricing(model_id: str):
        return None


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


def test_calculate_cost_charges_cache_reads_and_writes_at_generated_rates():
    # Find any model in the live pricing table with both cache_read and cache_write rates.
    from anymodel.generated.pricing import MODEL_PRICING

    model_id, pricing = next(
        (mid, p) for mid, p in MODEL_PRICING.items() if "cache_read" in p and "cache_write" in p
    )

    prompt_tokens, completion_tokens = 1_000_000, 100_000
    cache_read_tokens, cache_write_tokens = 400_000, 100_000
    uncached_prompt_tokens = prompt_tokens - cache_read_tokens - cache_write_tokens

    cost = calculate_cost(model_id, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens)
    expected = (
        uncached_prompt_tokens * pricing["prompt"]
        + cache_read_tokens * pricing["cache_read"]
        + cache_write_tokens * pricing["cache_write"]
        + completion_tokens * pricing["completion"]
    )
    assert cost == pytest.approx(expected, rel=1e-9)


@pytest.mark.asyncio
async def test_flex_discount_is_openai_specific_policy(batch_dir: Path):
    """A non-OpenAI provider's flex service_tier must not get OpenAI's 50% discount."""
    xai_model = "xai/grok-4.5"
    manager = BatchManager(_MockRouter(xai_model), dir=str(batch_dir))
    results = await manager.create_and_poll(
        {"model": xai_model, "requests": REQUESTS, "batch_mode": "concurrent", "options": {"service_tier": "flex"}},
        interval=0.05,
    )
    full_cost = calculate_cost(xai_model, 100, 50)
    # No provider policy entry for xai + flex, so the multiplier stays 1.0 — full price, no discount.
    assert results["usage_summary"]["estimated_cost"] == pytest.approx(full_cost * 2, abs=1e-10)


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
