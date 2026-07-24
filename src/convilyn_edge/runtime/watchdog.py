"""Timer / watchdog runtime primitives — the deterministic clock the loop lacks.

The event-driven :class:`~convilyn_edge.runtime.driver.WorkflowDriver` and the declarative
:class:`~convilyn_edge.runtime.pipeline.Pipeline` are purely *reactive*: nothing happens
until an event arrives. Two runtime concerns need a **clock** instead, and this module is
that clock — both deterministic, neither LLM-driven:

* **Review expiry → ``default_action``.** A :class:`~convilyn_edge.spi.review.HumanReview`
  may block on a human who never answers. :func:`resolve_review` honours
  :attr:`ReviewRequest.expires_at`: when that absolute UTC deadline elapses with no answer,
  it resolves to the request's ``default_action`` (``stop`` / ``escalate`` / ``continue``) —
  the fail-safe the *workflow* declared, never a hardcoded one and never an LLM deciding to
  "wait a little longer". This is what makes "escalate to the sister after 10 min" a
  declaration the runtime enforces.

* **Source-health polling.** :func:`poll_health` drives an
  :meth:`~convilyn_edge.spi.source.EventSource.health` on a fixed interval so a device that
  goes *silent* between events (a camera that simply stops producing frames) is still
  observed. Each reading is handed to a deterministic ``on_health`` callback: the watchdog
  *observes*; the removable Solution Pack *decides* what an offline transition means — this
  module encodes no scenario reaction.

Both take injectable clocks (``now`` / ``sleep``) so a driven run stays reproducible (the
SDK's replay guarantee) and a test never actually waits. Generic capability, scenario-free.
Stdlib only — no ``import app.*``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from convilyn_edge.spi.review import HumanReview, ReviewOutcome, ReviewRequest
from convilyn_edge.spi.source import DeviceHealth, EventSource, HealthStatus

_log = logging.getLogger(__name__)

# ── Review expiry → default_action ───────────────────────────────────────────


def _utc_now() -> datetime:
    """The single wall-clock read for expiry math (injectable for replay/tests)."""
    return datetime.now(timezone.utc)


def _parse_iso_utc(value: str) -> datetime:
    """Parse a UTC ISO-8601 timestamp to an aware ``datetime`` (raises ``ValueError``).

    Accepts the ``+00:00`` form minted by the envelope clock and a trailing ``Z`` (which
    ``datetime.fromisoformat`` only learned to parse in 3.11 — the SDK floor is 3.10). A
    naive timestamp is read as UTC rather than rejected.
    """
    text = value.strip()
    if text[-1:] in {"Z", "z"}:
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def seconds_until_expiry(expires_at: str | None, *, now: datetime) -> float | None:
    """Seconds from ``now`` until ``expires_at`` — ``None`` when there is no deadline.

    Returns ``None`` for a missing OR unparseable ``expires_at``: an un-readable deadline
    must degrade to "no deadline" (await the human as today), never silently fire the
    default action nor crash the review. A negative/zero result means the deadline has
    already passed. ``now`` is treated as UTC when naive.
    """
    if expires_at is None:
        return None
    try:
        deadline = _parse_iso_utc(expires_at)
    except (ValueError, TypeError, AttributeError, OverflowError):
        # Unparseable OR the wrong type entirely (a pack that built expires_at from event
        # JSON may hand an int / dict / bytes): degrade to "no deadline", never crash the
        # review. TypeError/AttributeError cover the non-str inputs .strip()/fromisoformat
        # reject; ValueError covers a malformed string; OverflowError covers a boundary date
        # (year 0001/9999) with an offset that overflows the UTC conversion.
        return None
    reference = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return (deadline - reference.astimezone(timezone.utc)).total_seconds()


def _expired_outcome(request: ReviewRequest) -> ReviewOutcome:
    """The synthesized outcome when a review expires: the workflow's ``default_action``.

    ``ReviewRequest.default_action`` is a subset of :data:`ReviewDecision`, so it maps
    straight onto ``ReviewOutcome.decision`` — the pack's downstream binding routes on it
    exactly as it would a live human decision (``dispatch.review_disposition`` is total).
    ``decision_source="expiry_default"`` marks it as machine-synthesized (not a human
    answer) so an approval gate can refuse to treat a timeout as a human grant.
    """
    return ReviewOutcome(
        decision=request.default_action,
        note="expired: default_action",
        decision_source="expiry_default",
    )


async def _drain(task: asyncio.Future) -> None:
    """Cancel an abandoned child task and discard its result / exception.

    Suppresses the child's own ``CancelledError`` (from the ``cancel()``) and any exception
    it completed with — this task's result is being thrown away. It deliberately does NOT
    suppress ``KeyboardInterrupt`` / ``SystemExit``, which must still propagate. (Fully
    distinguishing the child's cancellation from an *outer* cancellation delivered during
    the drain needs ``Task.cancelling()`` — 3.11+, above the SDK's 3.10 floor — so that
    micro-window is an accepted limitation; the primary park point handles outer cancel.)
    """
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def resolve_review(
    review: HumanReview,
    request: ReviewRequest,
    *,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ReviewOutcome:
    """Await a human decision, falling back to ``request.default_action`` on expiry.

    * No (or unparseable) ``expires_at`` → await ``review.request`` exactly as before, so
      an expiry-less review is byte-for-byte the prior behaviour (additive primitive).
    * Deadline already passed at call time → resolve to ``default_action`` immediately,
      without a pointless ask (deterministic; no zero-timeout race).
    * Otherwise race the human answer against the deadline: a human who answers by the
      deadline wins (ties resolve to the answer — a real decision is honoured); the timer
      firing first abandons the ask and resolves to ``default_action``.

    Deterministic: the deadline is a fixed clock (``now`` + ``sleep``, both injectable —
    ``None`` uses the real UTC clock / ``asyncio.sleep``), so a replay that pins them never
    waits and never varies. An exception raised by ``review.request`` (when it wins the
    race) propagates unchanged.
    """
    now = now if now is not None else _utc_now
    sleep = sleep if sleep is not None else asyncio.sleep
    remaining = seconds_until_expiry(request.expires_at, now=now())
    if remaining is None:
        return await review.request(request)
    if remaining <= 0.0:
        return _expired_outcome(request)

    answer = asyncio.ensure_future(review.request(request))
    timer = asyncio.ensure_future(sleep(remaining))
    try:
        done, _pending = await asyncio.wait({answer, timer}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # The awaiting task itself was cancelled — tear both children down, then re-raise.
        await _drain(answer)
        await _drain(timer)
        raise

    if answer in done:
        await _drain(timer)
        return answer.result()  # re-raises the human-request's exception, if any
    await _drain(answer)
    return _expired_outcome(request)


# ── Source-health polling (the watchdog clock) ───────────────────────────────

#: A deterministic reaction to one health reading. May be sync or async; its return value
#: is ignored. It MUST NOT drive scenario control flow into the substrate — it is the
#: Solution Pack's hook, injected by the pack, never a branch inside this module.
HealthObserver = Callable[[DeviceHealth], "Awaitable[None] | None"]


@dataclass(frozen=True)
class WatchdogPolicy:
    """A deterministic health-poll schedule.

    Every ``interval_s`` seconds :func:`poll_health` calls ``source.health()`` and hands the
    reading to ``on_health``. ``transitions_only`` (the default) fires the observer only when
    the coarse :data:`~convilyn_edge.spi.source.HealthStatus` *changes*, so a steady-state
    device stays quiet and the pack sees edges (connected→disconnected), not a heartbeat.
    """

    interval_s: float
    on_health: HealthObserver
    transitions_only: bool = True

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("interval_s must be > 0")


async def poll_health(
    source: EventSource,
    policy: WatchdogPolicy,
    *,
    stop: asyncio.Event,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll ``source.health()`` on ``policy``'s interval until ``stop`` is set.

    Sleeps first (no poll at t=0 — the source has only just started), then reads health and,
    per ``policy``, notifies ``on_health``. **Offline-first:** a failing ``health()`` or a
    raising ``on_health`` is swallowed (logged) and polling continues — a poll hiccup is
    itself a symptom, never a reason to crash the driver it runs beside. Deterministic: a
    fixed interval, never an LLM deciding when to check. Returns once ``stop`` is set.

    ``last`` advances only after a transition is *successfully delivered* to ``on_health``:
    if the observer raises on an edge (most likely the connected→disconnected edge, where the
    observer's cloud/alert call is itself failing), the edge is NOT recorded, so the next
    identical reading retries delivery instead of being de-duplicated and lost. Steady-state
    dedup resumes once delivery succeeds.
    """
    last: HealthStatus | None = None
    while not stop.is_set():
        await sleep(policy.interval_s)
        if stop.is_set():
            break
        try:
            health = await source.health()
        except Exception:  # noqa: BLE001 — a failed poll is a symptom, not a crash
            _log.warning("watchdog health poll failed", exc_info=True)
            continue
        if policy.transitions_only and health.status == last:
            continue
        try:
            result = policy.on_health(health)
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — an observer error must not kill the watchdog
            _log.warning("watchdog on_health observer failed", exc_info=True)
            continue  # leave ``last`` unchanged → retry this transition on the next reading
        last = health.status  # record the edge only once the pack has actually seen it


__all__ = [
    "seconds_until_expiry",
    "resolve_review",
    "HealthObserver",
    "WatchdogPolicy",
    "poll_health",
]
