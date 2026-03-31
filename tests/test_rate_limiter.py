"""Tests for RateLimitTracker."""

import time

from anymodel.utils._rate_limiter import RateLimitTracker


def test_not_rate_limited_by_default() -> None:
    tracker = RateLimitTracker()
    assert tracker.is_rate_limited("openai") is False


def test_tracks_rate_limit_from_record() -> None:
    tracker = RateLimitTracker()
    tracker.record("openai", retry_after=5.0)
    assert tracker.is_rate_limited("openai") is True
    assert tracker.get_wait_time("openai") > 0


def test_tracks_remaining_zero_with_future_reset() -> None:
    tracker = RateLimitTracker()
    future_reset = time.monotonic() + 60
    tracker.record("anthropic", remaining=0, reset_at=future_reset)

    assert tracker.is_rate_limited("anthropic") is True


def test_not_rate_limited_when_remaining_positive() -> None:
    tracker = RateLimitTracker()
    tracker.record("openai", remaining=50)
    assert tracker.is_rate_limited("openai") is False
