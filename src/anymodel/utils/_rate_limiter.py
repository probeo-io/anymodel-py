"""Per-provider rate limit tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _RateLimitState:
    remaining: int | None = None
    reset_at: float | None = None
    retry_after: float | None = None
    last_updated: float = field(default_factory=time.monotonic)


class RateLimitTracker:
    """Tracks rate limit state per provider."""

    def __init__(self) -> None:
        self._states: dict[str, _RateLimitState] = {}

    def record(
        self,
        provider: str,
        *,
        remaining: int | None = None,
        reset_at: float | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Record rate limit headers from a provider response."""
        state = self._states.setdefault(provider, _RateLimitState())
        if remaining is not None:
            state.remaining = remaining
        if reset_at is not None:
            state.reset_at = reset_at
        if retry_after is not None:
            state.retry_after = time.monotonic() + retry_after
        state.last_updated = time.monotonic()

    def is_rate_limited(self, provider: str) -> bool:
        """Check if a provider is currently rate limited."""
        state = self._states.get(provider)
        if state is None:
            return False

        now = time.monotonic()

        if state.retry_after and now < state.retry_after:
            return True

        if state.remaining is not None and state.remaining <= 0:
            if state.reset_at and now < state.reset_at:
                return True

        return False

    def get_wait_time(self, provider: str) -> float:
        """Get seconds to wait before retrying a rate-limited provider."""
        state = self._states.get(provider)
        if state is None:
            return 0

        now = time.monotonic()

        if state.retry_after and now < state.retry_after:
            return state.retry_after - now

        if state.remaining is not None and state.remaining <= 0 and state.reset_at:
            if now < state.reset_at:
                return state.reset_at - now

        return 0

    def clear(self, provider: str) -> None:
        """Clear rate limit state for a provider."""
        self._states.pop(provider, None)
