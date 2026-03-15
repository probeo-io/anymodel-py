"""Retry with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from anymodel._types import AnyModelError

T = TypeVar("T")

RETRYABLE_CODES = frozenset({429, 502, 503, 529})


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> T:
    """Call fn with retries on transient errors (429, 502, 503, 529)."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except AnyModelError as e:
            last_error = e
            if e.code not in RETRYABLE_CODES or attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
            await asyncio.sleep(delay)
        except Exception:
            raise

    # Should never reach here, but satisfy type checker
    assert last_error is not None
    raise last_error
