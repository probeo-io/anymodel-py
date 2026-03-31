"""Tests for AdaptiveConcurrencyController."""

import asyncio

import pytest

from anymodel.utils._adaptive_concurrency import AdaptiveConcurrencyController


def test_starts_at_configured_initial():
    controller = AdaptiveConcurrencyController(initial=10)
    assert controller.max_concurrency == 10


def test_defaults_to_initial_5():
    controller = AdaptiveConcurrencyController()
    assert controller.max_concurrency == 5


# --- Slow-start phase (exponential) ---


def test_slow_start_doubles_after_first_window():
    controller = AdaptiveConcurrencyController(initial=5)
    for _ in range(5):
        controller.record_success()
    assert controller.max_concurrency == 10


def test_slow_start_keeps_doubling():
    controller = AdaptiveConcurrencyController(initial=5)
    # Window 1: 5 successes -> 10
    for _ in range(5):
        controller.record_success()
    assert controller.max_concurrency == 10
    # Window 2: 10 successes -> 20
    for _ in range(10):
        controller.record_success()
    assert controller.max_concurrency == 20
    # Window 3: 20 successes -> 40
    for _ in range(20):
        controller.record_success()
    assert controller.max_concurrency == 40


def test_slow_start_reaches_high_concurrency_quickly():
    controller = AdaptiveConcurrencyController(initial=5)
    # 5 -> 10 -> 20 -> 40 -> 80 -> 160
    total = 0
    window = 5
    while window <= 80:
        for _ in range(window):
            controller.record_success()
        total += window
        window *= 2
    assert controller.max_concurrency == 160
    assert total == 5 + 10 + 20 + 40 + 80  # 155


def test_does_not_increase_before_full_window():
    controller = AdaptiveConcurrencyController(initial=5)
    for _ in range(4):
        controller.record_success()
    assert controller.max_concurrency == 5


# --- Congestion avoidance (AIMD) ---


def test_switches_to_additive_after_throttle():
    controller = AdaptiveConcurrencyController(initial=10)
    controller.record_throttle()
    assert controller.max_concurrency == 5

    # Congestion avoidance: +1 per window
    for _ in range(5):
        controller.record_success()
    assert controller.max_concurrency == 6

    for _ in range(6):
        controller.record_success()
    assert controller.max_concurrency == 7


def test_multiplicative_decrease_halves():
    controller = AdaptiveConcurrencyController(initial=10)
    controller.record_throttle()
    assert controller.max_concurrency == 5


def test_respects_min_floor():
    controller = AdaptiveConcurrencyController(initial=1)
    controller.record_throttle()
    assert controller.max_concurrency == 1


def test_respects_max_ceiling():
    controller = AdaptiveConcurrencyController(initial=8, max=10)
    # 8 successes -> would double to 16, clamped to 10
    for _ in range(8):
        controller.record_success()
    assert controller.max_concurrency == 10
    # Another window: stays at 10
    for _ in range(10):
        controller.record_success()
    assert controller.max_concurrency == 10


# --- Header-driven proactive backoff ---


def test_proactive_backoff_clamps():
    controller = AdaptiveConcurrencyController(initial=20)
    controller.record_success({"headers": {"x-ratelimit-remaining-requests": "3"}})
    assert controller.max_concurrency == 3


def test_proactive_backoff_switches_to_congestion_avoidance():
    controller = AdaptiveConcurrencyController(initial=20)
    controller.record_success({"headers": {"x-ratelimit-remaining-requests": "10"}})
    assert controller.max_concurrency == 10

    # Should be additive now
    for _ in range(10):
        controller.record_success()
    assert controller.max_concurrency == 11


def test_does_not_reduce_below_min_from_headers():
    controller = AdaptiveConcurrencyController(initial=5, min=2)
    controller.record_success({"headers": {"x-ratelimit-remaining-requests": "0"}})
    assert controller.max_concurrency == 2


def test_ignores_remaining_when_exceeds_current():
    controller = AdaptiveConcurrencyController(initial=5)
    controller.record_success({"headers": {"x-ratelimit-remaining-requests": "1000"}})
    assert controller.max_concurrency == 5


def test_normalizes_anthropic_headers():
    controller = AdaptiveConcurrencyController(initial=20)
    controller.record_success({
        "headers": {
            "anthropic-ratelimit-requests-remaining": "5",
            "x-ratelimit-remaining-requests": "5",
        },
    })
    assert controller.max_concurrency == 5


# --- Delay / retry-after ---


def test_sets_retry_after_delay():
    controller = AdaptiveConcurrencyController(initial=5)
    controller.record_throttle(5000)
    delay = controller.get_delay()
    assert delay > 4900
    assert delay <= 5000


def test_returns_0_delay_when_no_throttle():
    controller = AdaptiveConcurrencyController()
    assert controller.get_delay() == 0


@pytest.mark.asyncio
async def test_delay_expires_after_waiting():
    controller = AdaptiveConcurrencyController(initial=5)
    controller.record_throttle(50)
    assert controller.get_delay() > 0
    await asyncio.sleep(0.06)
    assert controller.get_delay() == 0


# --- Recovery scenarios ---


def test_resets_success_counter_on_throttle():
    controller = AdaptiveConcurrencyController(initial=5)
    for _ in range(3):
        controller.record_success()
    controller.record_throttle()
    assert controller.max_concurrency == 2  # floor(5 * 0.5) = 2
    # Congestion avoidance: 2 successes -> 3
    controller.record_success()
    assert controller.max_concurrency == 2
    controller.record_success()
    assert controller.max_concurrency == 3


def test_custom_decrease_factor():
    controller = AdaptiveConcurrencyController(initial=10, decrease_factor=0.75)
    controller.record_throttle()
    assert controller.max_concurrency == 7


def test_slow_start_resumes_up_to_ssthresh():
    controller = AdaptiveConcurrencyController(initial=20)
    # Ramp to 40
    for _ in range(20):
        controller.record_success()
    assert controller.max_concurrency == 40

    # Throttle: current = 20, ssthresh = 20
    controller.record_throttle()
    assert controller.max_concurrency == 20

    # Next window: current (20) >= ssthresh (20), additive
    for _ in range(20):
        controller.record_success()
    assert controller.max_concurrency == 21  # +1, not *2
