"""Adaptive concurrency controller with TCP-style slow-start + AIMD."""

from __future__ import annotations

import math
import time
from typing import Any


class AdaptiveConcurrencyController:
    """Adaptive concurrency controller with TCP-style slow-start + AIMD.

    Phase 1 -- Slow-start: doubles concurrency each window (exponential ramp)
    until a 429 or header-driven backoff occurs.

    Phase 2 -- Congestion avoidance (AIMD): after the first throttle sets
    a threshold, switches to additive increase (+1 per window).

    On 429: multiplicative decrease (halve), set threshold to pre-throttle / 2.
    """

    def __init__(
        self,
        *,
        initial: int = 5,
        min: int = 1,
        max: int = 500,
        decrease_factor: float = 0.5,
    ) -> None:
        self._current: float = initial
        self._min = min
        self._max = max
        self._decrease_factor = decrease_factor
        self._pause_until: float = 0
        self._success_count = 0
        self._ssthresh: float = math.inf

    @property
    def max_concurrency(self) -> int:
        """Current allowed concurrency level."""
        return math.floor(self._current)

    def record_success(self, meta: dict[str, Any] | None = None) -> None:
        """Record a successful response.

        Optionally pass a ``ResponseMeta``-like dict with a ``headers``
        key to allow header-driven proactive adjustment.
        """
        self._success_count += 1

        if self._success_count >= self._current:
            if self._current < self._ssthresh:
                # Slow-start phase: double each window
                self._current = min(self._current * 2, self._max)
            else:
                # Congestion avoidance: additive increase (+1 per window)
                self._current = min(self._current + 1, self._max)
            self._success_count = 0

        # Proactive backoff from rate-limit headers
        if meta and meta.get('headers'):
            headers = meta['headers']
            remaining_str = (
                headers.get('x-ratelimit-remaining-requests')
                or headers.get('anthropic-ratelimit-requests-remaining')
            )
            if remaining_str is not None:
                try:
                    remaining_num = int(remaining_str)
                except (ValueError, TypeError):
                    remaining_num = None
                if remaining_num is not None and remaining_num < self._current:
                    self._ssthresh = max(self._min, remaining_num)
                    self._current = max(self._min, remaining_num)
                    self._success_count = 0

    def record_throttle(self, retry_after_ms: int | None = None) -> None:
        """Record a rate-limit (429) response.

        Halves concurrency, sets slow-start threshold, and optionally
        pauses for *retry_after_ms* milliseconds.
        """
        self._ssthresh = max(self._min, math.floor(self._current * self._decrease_factor))
        self._current = max(self._min, math.floor(self._current * self._decrease_factor))
        self._success_count = 0

        if retry_after_ms and retry_after_ms > 0:
            self._pause_until = time.monotonic() + retry_after_ms / 1000.0

    def get_delay(self) -> float:
        """Return milliseconds to wait before sending the next request (0 if none)."""
        return max(0, (self._pause_until - time.monotonic()) * 1000.0)
