"""Unit tests for the timer/watchdog primitives — review expiry + health polling (#2792 PR-a)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.runtime.driver import WorkflowDriver
from convilyn_edge.runtime.watchdog import (
    WatchdogPolicy,
    poll_health,
    resolve_review,
    seconds_until_expiry,
)
from convilyn_edge.spi.review import ReviewOutcome, ReviewRequest
from convilyn_edge.spi.source import DeviceHealth, HealthStatus, SourceContext


async def _yield(_seconds: float) -> None:
    """A sleep that returns 'instantly' but yields to the loop (no tight-spin starvation)."""
    await asyncio.sleep(0)


def _at(iso: str):
    return lambda: datetime.fromisoformat(iso)


# ── resolve_review: no / unparseable deadline → await the human (status quo) ──


class _Answers:
    """A HumanReview that resolves immediately to a fixed outcome, counting calls."""

    def __init__(self, outcome: ReviewOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        self.calls += 1
        return self.outcome


class _NeverAnswers:
    """A HumanReview that blocks forever — the human who never responds."""

    def __init__(self) -> None:
        self.cancelled = False
        self.calls = 0

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        self.calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")  # pragma: no cover


async def test_no_deadline_awaits_the_human():
    review = _Answers(ReviewOutcome(decision="selected", choice_id="keep"))

    outcome = await resolve_review(review, ReviewRequest("t", "m"))  # expires_at=None

    assert outcome.decision == "selected" and review.calls == 1


async def test_unparseable_deadline_degrades_to_awaiting_the_human():
    review = _Answers(ReviewOutcome(decision="continue"))

    outcome = await resolve_review(review, ReviewRequest("t", "m", expires_at="not-a-timestamp"))

    assert outcome.decision == "continue" and review.calls == 1


# ── resolve_review: expiry → default_action ──────────────────────────────────


async def test_already_expired_fires_default_without_asking():
    review = _Answers(ReviewOutcome(decision="selected"))
    request = ReviewRequest(
        "t", "m", default_action="escalate", expires_at="2020-01-01T00:00:00+00:00"
    )

    outcome = await resolve_review(review, request, now=_at("2026-01-01T00:00:00+00:00"))

    # deadline already passed → default_action, and the human is never even asked
    assert outcome.decision == "escalate"
    assert outcome.note == "expired: default_action"
    assert outcome.decision_source == "expiry_default"  # marked machine-synthesized, not human
    assert review.calls == 0


async def test_deadline_elapses_before_answer_fires_default():
    review = _NeverAnswers()
    request = ReviewRequest("t", "m", default_action="stop", expires_at="2026-01-01T00:10:00+00:00")

    # now is 10 min before the deadline → a positive remaining; the injected instant sleep
    # makes the timer win the race against the never-answering human.
    outcome = await resolve_review(
        review, request, now=_at("2026-01-01T00:00:00+00:00"), sleep=_yield
    )

    assert outcome.decision == "stop"
    assert outcome.decision_source == "expiry_default"
    assert review.cancelled is True  # the abandoned human ask was cancelled, not leaked


async def test_default_action_variants_all_surface():
    for action in ("stop", "escalate", "continue"):
        request = ReviewRequest(
            "t", "m", default_action=action, expires_at="2020-01-01T00:00:00+00:00"
        )
        outcome = await resolve_review(
            _Answers(ReviewOutcome(decision="selected")),
            request,
            now=_at("2026-01-01T00:00:00+00:00"),
        )
        assert outcome.decision == action


async def test_human_answer_before_deadline_wins():
    # A far-future deadline with a human who answers immediately: the answer wins the race
    # (real asyncio.sleep for the timer is cancelled the moment the human resolves — no wait).
    review = _Answers(ReviewOutcome(decision="selected", choice_id="keep-both"))
    request = ReviewRequest("t", "m", default_action="stop", expires_at="2099-01-01T00:00:00+00:00")

    outcome = await resolve_review(review, request, now=_at("2026-01-01T00:00:00+00:00"))

    assert outcome.decision == "selected" and outcome.choice_id == "keep-both"
    assert outcome.decision_source == "human"  # a real human answer, not a synthesized default


async def test_human_exception_propagates_when_it_wins_the_race():
    class _Boom:
        async def request(self, req: ReviewRequest) -> ReviewOutcome:
            raise RuntimeError("review backend down")

    request = ReviewRequest("t", "m", expires_at="2099-01-01T00:00:00+00:00")

    try:
        await resolve_review(_Boom(), request, now=_at("2026-01-01T00:00:00+00:00"))
    except RuntimeError as exc:
        assert str(exc) == "review backend down"
    else:  # pragma: no cover
        raise AssertionError("expected the human-request exception to propagate")


# ── seconds_until_expiry ─────────────────────────────────────────────────────


def test_seconds_until_expiry_none_for_missing_deadline():
    assert seconds_until_expiry(None, now=datetime(2026, 1, 1, tzinfo=timezone.utc)) is None


def test_seconds_until_expiry_none_for_unparseable():
    assert seconds_until_expiry("garbage", now=datetime(2026, 1, 1, tzinfo=timezone.utc)) is None


def test_seconds_until_expiry_none_for_overflowing_boundary_deadline():
    # A boundary date (year 9999/0001) with an offset parses but overflows the UTC conversion —
    # it must degrade to "no deadline", never crash the review with an uncaught OverflowError.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for boundary in ("9999-12-31T23:59:59-14:00", "0001-01-01T00:00:00+14:00"):
        assert seconds_until_expiry(boundary, now=now) is None


def test_seconds_until_expiry_none_for_non_string_types():
    # A pack that builds expires_at from event JSON may hand a non-string; the "unparseable
    # → no deadline" contract must hold (never crash the review), not just for bad strings.
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for bad in (123, 12.5, {"a": 1}, ["x"], b"2026-01-01T00:00:00+00:00", True):
        assert seconds_until_expiry(bad, now=now) is None  # type: ignore[arg-type]


def test_seconds_until_expiry_positive_before_deadline():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    remaining = seconds_until_expiry("2026-01-01T00:10:00+00:00", now=now)

    assert remaining == 600.0


def test_seconds_until_expiry_negative_after_deadline():
    now = datetime(2026, 1, 1, 0, 20, 0, tzinfo=timezone.utc)

    assert seconds_until_expiry("2026-01-01T00:10:00+00:00", now=now) == -600.0


def test_seconds_until_expiry_accepts_z_suffix():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert seconds_until_expiry("2026-01-01T00:05:00Z", now=now) == 300.0


def test_seconds_until_expiry_treats_naive_now_as_utc():
    now = datetime(2026, 1, 1, 0, 0, 0)  # naive

    assert seconds_until_expiry("2026-01-01T00:01:00+00:00", now=now) == 60.0


def test_seconds_until_expiry_treats_naive_deadline_as_utc():
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert seconds_until_expiry("2026-01-01T00:02:00", now=now) == 120.0  # naive deadline → UTC


async def test_resolve_review_propagates_outer_cancellation():
    # The awaiting resolve_review task is cancelled while parked in the race — it must tear
    # down the in-flight human ask (not leak it) and re-raise the cancellation.
    review = _NeverAnswers()
    request = ReviewRequest("t", "m", default_action="stop", expires_at="2099-01-01T00:00:00+00:00")
    task = asyncio.ensure_future(
        resolve_review(review, request, now=_at("2026-01-01T00:00:00+00:00"))
    )
    await asyncio.sleep(0)  # let it reach the asyncio.wait race point
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the outer cancellation to propagate")

    assert review.cancelled is True


# ── poll_health: transitions, dedup, error-swallow ───────────────────────────


class _ScriptedHealthSource:
    """A minimal EventSource whose health() replays a scripted status sequence."""

    def __init__(self, statuses: Sequence[HealthStatus]) -> None:
        self._statuses = list(statuses)
        self._i = 0

    async def health(self) -> DeviceHealth:
        status = self._statuses[min(self._i, len(self._statuses) - 1)]
        self._i += 1
        return DeviceHealth(status=status)

    def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:  # pragma: no cover
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover
        return None


async def test_poll_fires_only_on_transitions():
    seen: list[HealthStatus] = []
    stop = asyncio.Event()
    source = _ScriptedHealthSource(["connected", "connected", "disconnected", "disconnected"])

    def on_health(health: DeviceHealth) -> None:
        seen.append(health.status)
        if len(seen) >= 2:
            stop.set()

    await poll_health(source, WatchdogPolicy(1.0, on_health), stop=stop, sleep=_yield)

    assert seen == ["connected", "disconnected"]  # steady-state repeats de-duplicated


async def test_poll_fires_every_reading_when_transitions_only_false():
    seen: list[HealthStatus] = []
    stop = asyncio.Event()
    source = _ScriptedHealthSource(["connected", "connected"])

    def on_health(health: DeviceHealth) -> None:
        seen.append(health.status)
        if len(seen) >= 2:
            stop.set()

    await poll_health(
        source, WatchdogPolicy(1.0, on_health, transitions_only=False), stop=stop, sleep=_yield
    )

    assert seen == ["connected", "connected"]


async def test_poll_swallows_a_failing_health_read():
    class _FlakyHealth(_ScriptedHealthSource):
        def __init__(self) -> None:
            super().__init__(["degraded"])
            self.calls = 0

        async def health(self) -> DeviceHealth:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("blip")
            return DeviceHealth(status="degraded")

    seen: list[HealthStatus] = []
    stop = asyncio.Event()
    source = _FlakyHealth()

    def on_health(health: DeviceHealth) -> None:
        seen.append(health.status)
        stop.set()

    await poll_health(source, WatchdogPolicy(1.0, on_health), stop=stop, sleep=_yield)

    assert seen == ["degraded"]  # recovered after the failed poll, no crash


async def test_poll_swallows_a_raising_observer():
    seen: list[HealthStatus] = []
    stop = asyncio.Event()
    calls = {"n": 0}
    source = _ScriptedHealthSource(["degraded", "disconnected"])

    def on_health(health: DeviceHealth) -> None:
        calls["n"] += 1
        seen.append(health.status)
        if calls["n"] == 1:
            raise ValueError("observer boom")
        stop.set()

    await poll_health(source, WatchdogPolicy(1.0, on_health), stop=stop, sleep=_yield)

    assert seen == ["degraded", "disconnected"]  # observer error did not kill the watchdog


async def test_poll_retries_the_transition_after_an_observer_failure():
    # Regression: if on_health raises on the connected→disconnected EDGE and the device
    # STAYS disconnected, the edge must NOT be swallowed forever — ``last`` only advances on
    # a successful delivery, so the next identical reading retries. (Before the fix this
    # de-duplicated the lost edge and the offline alert never fired — the test would hang.)
    delivered: list[HealthStatus] = []
    stop = asyncio.Event()
    calls = {"n": 0}
    source = _ScriptedHealthSource(["disconnected", "disconnected"])  # stays offline

    def on_health(health: DeviceHealth) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("cloud unreachable — the offline edge itself failing")
        delivered.append(health.status)
        stop.set()

    await poll_health(source, WatchdogPolicy(1.0, on_health), stop=stop, sleep=_yield)

    assert delivered == ["disconnected"]  # retried and delivered on the second poll


async def test_poll_awaits_an_async_observer():
    seen: list[HealthStatus] = []
    stop = asyncio.Event()
    source = _ScriptedHealthSource(["disconnected"])

    async def on_health(health: DeviceHealth) -> None:
        await asyncio.sleep(0)
        seen.append(health.status)
        stop.set()

    await poll_health(source, WatchdogPolicy(1.0, on_health), stop=stop, sleep=_yield)

    assert seen == ["disconnected"]


async def test_poll_breaks_when_stopped_during_the_sleep():
    # stop is set *while the interval sleep is in flight* → the watchdog breaks right after
    # the sleep without doing a (now-pointless) health read.
    stop = asyncio.Event()
    source = _ScriptedHealthSource(["connected"])
    polled = {"n": 0}

    async def _sleep_then_stop(_seconds: float) -> None:
        stop.set()
        await asyncio.sleep(0)

    async def _counting_health() -> DeviceHealth:  # pragma: no cover - must never run
        polled["n"] += 1
        return DeviceHealth(status="connected")

    source.health = _counting_health  # type: ignore[method-assign]

    await poll_health(
        source, WatchdogPolicy(1.0, lambda h: None), stop=stop, sleep=_sleep_then_stop
    )

    assert polled["n"] == 0  # broke after the sleep, before any health read


def test_watchdog_policy_rejects_non_positive_interval():
    try:
        WatchdogPolicy(0.0, lambda h: None)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected interval_s > 0 to be enforced")


# ── driver integration: watchdog runs beside the event loop ──────────────────


def _events(n: int) -> list[EventEnvelope]:
    return [
        new_envelope(
            event_type="t",
            event_schema="s",
            source=EventSourceRef("dev", "sim", "0.1.0"),
            data={"i": i},
        )
        for i in range(n)
    ]


async def _ok(_e: EventEnvelope) -> str:
    return "ok"


async def test_driver_runs_watchdog_beside_the_loop():
    fired = asyncio.Event()

    class _EventAndHealthSource:
        def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
            async def _gen() -> AsyncIterator[EventEnvelope]:
                for envelope in _events(1):
                    yield envelope
                await fired.wait()  # hold the stream open until the watchdog has observed

            return _gen()

        async def health(self) -> DeviceHealth:
            return DeviceHealth(status="degraded")

        async def stop(self) -> None:
            return None

    def on_health(_health: DeviceHealth) -> None:
        fired.set()

    report = await WorkflowDriver(sleep=_yield).run(
        _EventAndHealthSource(),
        SourceContext(device_id="dev"),
        _ok,
        watchdog=WatchdogPolicy(interval_s=1.0, on_health=on_health),
    )

    assert fired.is_set()  # the watchdog polled health() while the loop was event-idle
    assert report.processed == 1


async def test_driver_does_not_leak_watchdog_when_start_raises():
    # Regression: the watchdog is scheduled only AFTER source.start() succeeds, so a source
    # whose start() raises (port/camera not ready) does not orphan a poll task that would
    # keep polling a source that never started.
    class _StartBoomSource:
        def __init__(self) -> None:
            self.health_calls = 0

        def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
            raise ConnectionError("camera not ready")

        async def health(self) -> DeviceHealth:
            self.health_calls += 1
            return DeviceHealth(status="disconnected")

        async def stop(self) -> None:
            return None

    source = _StartBoomSource()
    before = len(asyncio.all_tasks())
    try:
        await WorkflowDriver(sleep=_yield).run(
            source,
            SourceContext(device_id="dev"),
            _ok,
            watchdog=WatchdogPolicy(interval_s=1.0, on_health=lambda h: None),
        )
    except ConnectionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected start() failure to propagate")

    await asyncio.sleep(0)  # give any (erroneously) scheduled task a chance to run
    assert len(asyncio.all_tasks()) <= before  # no orphaned watchdog task lingering
    assert source.health_calls == 0  # the watchdog never polled a source that never started


async def test_driver_without_watchdog_is_unchanged():
    class _ListSource:
        def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
            async def _gen() -> AsyncIterator[EventEnvelope]:
                for envelope in _events(2):
                    yield envelope

            return _gen()

        async def health(self) -> DeviceHealth:
            return DeviceHealth(status="connected")

        async def stop(self) -> None:
            return None

    report = await WorkflowDriver(sleep=_yield).run(
        _ListSource(), SourceContext(device_id="dev"), _ok
    )

    assert report.processed == 2 and report.aborted is False
