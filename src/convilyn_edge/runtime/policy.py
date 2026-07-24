"""Deterministic runtime policies — retry/backoff, cycle-guard, stuck-abort.

These are the edge runtime's **deterministic** control-flow gates. The project's
error-routing boundary makes retry eligibility and backoff timing NEVER
LLM-driven: a runaway retry is a platform-level failure, and the only safe
rate-limit shape is a fixed exponential backoff. This module is the edge SDK's
own zero-dependency expression of that contract — it imports nothing from the
backend (the shape is shared, the code is not).

**Deterministic by default (replay-safe).** ``RetryPolicy`` applies NO jitter
unless a jitter function is injected, and ``retry_async`` takes an injectable
``sleep`` — so a driven run is fully reproducible (the SDK's replay guarantee)
and a test never actually waits. A production caller opts into jitter with
:func:`full_jitter` to avoid a thundering herd. Stdlib only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from random import Random
from typing import TypeVar

T = TypeVar("T")

#: The default transient-error classifier: which exceptions are retry-eligible.
#: DETERMINISTIC — a fixed tuple of exception types, never an LLM deciding
#: "should we retry?". Mirrors the backend's ``error_class`` transient set
#: (throttling / timeout / network) by shape, restricted to what a stdlib-only
#: edge transport can raise. Callers widen it explicitly for their transport.
DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


def _no_jitter(delay_ms: float) -> float:
    """The default: return the delay unchanged (deterministic, replay-safe)."""
    return delay_ms


def full_jitter(rng: Random) -> Callable[[float], float]:
    """Build a *full-jitter* function ``d -> uniform(0, d)`` from an injected RNG.

    Pass a seeded ``Random`` for a reproducible-but-jittered schedule, or a fresh
    ``Random()`` in production to spread retries across a fleet (anti-thundering
    herd). Kept opt-in so the default stays deterministic.
    """
    return lambda delay_ms: rng.uniform(0.0, delay_ms)


@dataclass(frozen=True)
class RetryPolicy:
    """Deterministic exponential-backoff retry budget.

    ``max_attempts`` bounds TOTAL attempts (the first try + retries), so a value
    of 1 means "no retry". ``delay_ms`` grows ``base_delay_ms * factor**(n-1)``,
    capped at ``max_delay_ms``, then optionally jittered. All timing is data, not
    control flow an LLM can lobby to extend.
    """

    max_attempts: int = 3
    base_delay_ms: float = 50.0
    factor: float = 2.0
    max_delay_ms: float = 5_000.0
    jitter: Callable[[float], float] = _no_jitter

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_ms < 0:
            raise ValueError("base_delay_ms must be >= 0")
        if self.factor < 1:
            raise ValueError("factor must be >= 1 (backoff must not shrink)")
        if self.max_delay_ms < self.base_delay_ms:
            raise ValueError("max_delay_ms must be >= base_delay_ms")

    def should_retry(self, attempt: int) -> bool:
        """True if another attempt is allowed after ``attempt`` (1-based) failed."""
        return attempt < self.max_attempts

    def delay_ms(self, attempt: int) -> float:
        """Backoff before the retry that follows ``attempt`` (1-based, just failed).

        Never negative even under a jitter function that could undershoot.
        """
        raw = self.base_delay_ms * (self.factor ** (attempt - 1))
        capped = min(raw, self.max_delay_ms)
        return max(0.0, self.jitter(capped))


@dataclass
class CycleGuard:
    """Stuck-state abort — the edge analogue of the backend's consecutive-same-call
    guard. When the SAME no-progress ``signature`` repeats ``max_consecutive``
    times in a row, the run is stuck and must abort; an LLM cannot lobby for "just
    one more cycle". A *different* signature (real progress) resets the counter.
    """

    max_consecutive: int = 3
    _last: Hashable | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")

    def record(self, signature: Hashable) -> bool:
        """Record one step's outcome ``signature``; return True if now STUCK.

        Stuck = this signature has repeated ``max_consecutive`` times with no
        intervening change. Any new signature resets the streak to 1.
        """
        if signature == self._last:
            self._count += 1
        else:
            self._last = signature
            self._count = 1
        return self._count >= self.max_consecutive

    def reset(self) -> None:
        """Forget the streak (e.g. after a successful side effect)."""
        self._last = None
        self._count = 0


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``operation`` with deterministic backoff retries on ``retry_on`` errors.

    Retries ONLY the classified transient exceptions; anything else propagates
    immediately (a rejected event is not a rate-limit). After the budget is
    exhausted the last transient exception propagates so the caller can take its
    fallback/offline path. ``sleep`` is injectable so tests don't wait and replay
    stays fast; delays are in seconds (``delay_ms / 1000``).
    """
    attempt = 1
    while True:
        try:
            return await operation()
        except retry_on:
            if not policy.should_retry(attempt):
                raise
            await sleep(policy.delay_ms(attempt) / 1000.0)
            attempt += 1


__all__ = [
    "DEFAULT_RETRYABLE",
    "full_jitter",
    "RetryPolicy",
    "CycleGuard",
    "retry_async",
]
