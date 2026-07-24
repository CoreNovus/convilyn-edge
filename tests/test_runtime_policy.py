"""Unit tests for the deterministic runtime policies (retry/backoff, cycle-guard)."""

from __future__ import annotations

from random import Random

import pytest

from convilyn_edge.runtime.policy import (
    CycleGuard,
    RetryPolicy,
    full_jitter,
    retry_async,
)

# ── RetryPolicy: logic ───────────────────────────────────────────────────────


def test_delay_grows_exponentially():
    policy = RetryPolicy(base_delay_ms=50, factor=2.0, max_delay_ms=10_000)

    assert [policy.delay_ms(n) for n in (1, 2, 3)] == [50.0, 100.0, 200.0]


def test_delay_is_capped_at_max():
    policy = RetryPolicy(base_delay_ms=50, factor=10.0, max_delay_ms=100)

    assert policy.delay_ms(3) == 100.0


def test_should_retry_true_within_budget():
    policy = RetryPolicy(max_attempts=3)

    assert policy.should_retry(1) is True


def test_should_retry_false_at_budget():
    policy = RetryPolicy(max_attempts=3)

    assert policy.should_retry(3) is False


def test_no_jitter_by_default_is_deterministic():
    policy = RetryPolicy(base_delay_ms=80)

    assert policy.delay_ms(1) == policy.delay_ms(1)


# ── RetryPolicy: boundary ────────────────────────────────────────────────────


def test_max_attempts_one_means_no_retry():
    policy = RetryPolicy(max_attempts=1)

    assert policy.should_retry(1) is False


def test_jitter_never_returns_negative():
    # A jitter function that would push below zero is clamped to 0.
    policy = RetryPolicy(base_delay_ms=50, jitter=lambda _d: -999.0)

    assert policy.delay_ms(1) == 0.0


def test_full_jitter_stays_within_bound():
    policy = RetryPolicy(base_delay_ms=100, factor=1.0, jitter=full_jitter(Random(7)))

    assert 0.0 <= policy.delay_ms(1) <= 100.0


# ── RetryPolicy: error (invalid construction) ────────────────────────────────


def test_max_attempts_below_one_rejected():
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_factor_below_one_rejected():
    with pytest.raises(ValueError, match="factor"):
        RetryPolicy(factor=0.5)


def test_max_delay_below_base_rejected():
    with pytest.raises(ValueError, match="max_delay_ms"):
        RetryPolicy(base_delay_ms=100, max_delay_ms=10)


# ── CycleGuard: object-state ─────────────────────────────────────────────────


def test_guard_not_stuck_before_threshold():
    guard = CycleGuard(max_consecutive=3)

    assert guard.record("x") is False


def test_guard_stuck_at_threshold():
    guard = CycleGuard(max_consecutive=3)
    guard.record("x")
    guard.record("x")

    assert guard.record("x") is True


def test_different_signature_resets_streak():
    guard = CycleGuard(max_consecutive=2)
    guard.record("x")

    # A new signature resets the count, so this is the FIRST "y", not stuck.
    assert guard.record("y") is False


def test_reset_clears_streak():
    guard = CycleGuard(max_consecutive=2)
    guard.record("x")
    guard.reset()

    assert guard.record("x") is False


def test_guard_max_consecutive_below_one_rejected():
    with pytest.raises(ValueError, match="max_consecutive"):
        CycleGuard(max_consecutive=0)


# ── retry_async: logic + error ───────────────────────────────────────────────


async def _nosleep(_seconds: float) -> None:
    return None


async def test_retry_async_returns_on_first_success():
    calls = {"n": 0}

    async def op() -> str:
        calls["n"] += 1
        return "ok"

    result = await retry_async(op, RetryPolicy(max_attempts=3), sleep=_nosleep)

    assert result == "ok" and calls["n"] == 1


async def test_retry_async_recovers_after_transient():
    attempts = {"n": 0}

    async def op() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("down")
        return "recovered"

    result = await retry_async(op, RetryPolicy(max_attempts=3), sleep=_nosleep)

    assert result == "recovered"


async def test_retry_async_exhausts_and_raises_transient():
    async def op() -> str:
        raise TimeoutError("still down")

    with pytest.raises(TimeoutError):
        await retry_async(op, RetryPolicy(max_attempts=2), sleep=_nosleep)


async def test_retry_async_does_not_retry_unclassified_error():
    calls = {"n": 0}

    async def op() -> str:
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await retry_async(op, RetryPolicy(max_attempts=5), sleep=_nosleep)
    assert calls["n"] == 1  # not retried — a ValueError is not transient
