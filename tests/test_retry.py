"""Tests for retry with exponential backoff."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from anymodel._types import AnyModelError
from anymodel.utils._retry import with_retry


async def test_returns_on_first_success() -> None:
    fn = AsyncMock(return_value="ok")
    result = await with_retry(fn, max_retries=3, base_delay=0.001, max_delay=0.01)
    assert result == "ok"
    assert fn.call_count == 1


async def test_retries_on_429_and_succeeds() -> None:
    fn = AsyncMock(side_effect=[AnyModelError(429, "Rate limited"), "ok"])
    result = await with_retry(fn, max_retries=2, base_delay=0.001, max_delay=0.01)
    assert result == "ok"
    assert fn.call_count == 2


async def test_retries_on_502_and_succeeds() -> None:
    fn = AsyncMock(side_effect=[AnyModelError(502, "Bad gateway"), "ok"])
    result = await with_retry(fn, max_retries=2, base_delay=0.001, max_delay=0.01)
    assert result == "ok"
    assert fn.call_count == 2


async def test_does_not_retry_on_400() -> None:
    fn = AsyncMock(side_effect=AnyModelError(400, "Bad request"))
    with pytest.raises(AnyModelError, match="Bad request"):
        await with_retry(fn, max_retries=3, base_delay=0.001, max_delay=0.01)
    assert fn.call_count == 1


async def test_throws_after_max_retries_exhausted() -> None:
    fn = AsyncMock(side_effect=AnyModelError(429, "Rate limited"))
    with pytest.raises(AnyModelError, match="Rate limited"):
        await with_retry(fn, max_retries=2, base_delay=0.001, max_delay=0.01)
    assert fn.call_count == 3  # initial + 2 retries


async def test_does_not_retry_non_anymodel_error() -> None:
    fn = AsyncMock(side_effect=RuntimeError("Network error"))
    with pytest.raises(RuntimeError, match="Network error"):
        await with_retry(fn, max_retries=3, base_delay=0.001, max_delay=0.01)
    assert fn.call_count == 1
