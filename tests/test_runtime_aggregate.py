"""Unit tests for the stateful threshold aggregator (#2792 PR-b)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from convilyn_edge.envelope import EventEnvelope, EventSourceRef
from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.probe import DeviceCapabilityManifest
from convilyn_edge.runtime.aggregate import Occurrence, ThresholdAggregator, ThresholdState
from convilyn_edge.runtime.pipeline import Pipeline
from convilyn_edge.spi.state import EventContext
from convilyn_edge.storage.selection import select_storage_provider

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class _Clock:
    """A controllable trusted clock — the window reference (never event content)."""

    def __init__(self, start: datetime) -> None:
        self._t = start

    def __call__(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += timedelta(seconds=seconds)


def _event(event_id: str, *, kind: str = "anomaly", time: str = "2000-01-01T00:00:00+00:00"):
    """An envelope whose ``time`` is deliberately unrelated to the window — the aggregator anchors
    on the trusted clock, so ``envelope.time`` (attacker-settable) must never move the window."""
    return EventEnvelope(
        event_id=event_id,
        event_type="sensor",
        event_schema="s",
        source=EventSourceRef("dev", "sim", "0.1.0"),
        time=time,
        data={"kind": kind},
    )


def _ctx(envelope: EventEnvelope) -> EventContext:
    return EventContext(envelope=envelope)


def _store(tmp_path: Path, name: str = "q.jsonl") -> DurableQueue:
    return DurableQueue(tmp_path / name)


def _is_anomaly(envelope: EventEnvelope) -> str | None:
    kind = envelope.data.get("kind")
    return str(kind) if kind is not None else None


def _plant(store: DurableQueue, record: dict) -> None:
    """Write a raw record straight to the store file (simulating a pre-planted / foreign record)."""
    store.path.parent.mkdir(parents=True, exist_ok=True)
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ── accumulation + threshold (window driven by the trusted clock) ────────────


async def test_first_occurrence_is_not_breached(tmp_path: Path):
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=_Clock(_T0)
    )

    state = await agg.snapshot(_ctx(_event("e1")))

    assert (state.key, state.count, state.breached) == ("anomaly", 1, False)


async def test_second_occurrence_in_window_breaches(tmp_path: Path):
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("e2")))

    assert state.count == 2 and state.breached is True


async def test_occurrence_outside_window_does_not_count(tmp_path: Path):
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(660)  # 11 min later — e1 has aged out of the 10-min window
    state = await agg.snapshot(_ctx(_event("e2")))

    assert state.count == 1 and state.breached is False


async def test_aged_out_occurrences_are_pruned_from_the_store(tmp_path: Path):
    clock = _Clock(_T0)
    store = _store(tmp_path)
    agg = ThresholdAggregator(store, threshold=5, window_s=600, key_of=_is_anomaly, now=clock)

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(660)
    await agg.snapshot(_ctx(_event("e2")))  # e1 now out of window → pruned

    assert [r["idempotency_key"] for r in store.pending()] == ["e2"]


# ── non-qualifying events ────────────────────────────────────────────────────


async def test_non_qualifying_event_is_not_recorded(tmp_path: Path):
    store = _store(tmp_path)
    agg = ThresholdAggregator(
        store, threshold=2, window_s=600, key_of=lambda e: None, now=_Clock(_T0)
    )

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.key is None and state.count == 0 and state.breached is False
    assert store.pending() == []


# ── idempotency within the retained window (replay-stable) ───────────────────


async def test_replayed_event_in_window_is_not_double_counted(tmp_path: Path):
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1")))
    state = await agg.snapshot(_ctx(_event("e1")))  # same event_id replayed within window

    assert state.count == 1 and state.breached is False  # apply-once by event_id


async def test_event_id_reuse_collapses_distinct_occurrences(tmp_path: Path):
    # Documented precondition: event_id is a trusted unique identity (the envelope contract). Two
    # DISTINCT occurrences reusing one id collapse to a single count — a source must not reuse ids.
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("dup")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("dup")))  # a different real occurrence, same id

    assert state.count == 1 and state.breached is False  # collapsed — the precondition's effect


# ── per-bucket separation ────────────────────────────────────────────────────


async def test_distinct_buckets_accumulate_separately(tmp_path: Path):
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1", kind="feeder_jam")))
    clock.advance(30)
    state_water = await agg.snapshot(_ctx(_event("e2", kind="water_low")))
    clock.advance(30)
    state_feeder = await agg.snapshot(_ctx(_event("e3", kind="feeder_jam")))

    assert state_water.count == 1 and state_water.breached is False
    assert state_feeder.count == 2 and state_feeder.breached is True


async def test_occurrences_carry_event_ids_and_receive_times(tmp_path: Path):
    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("e2")))

    # timestamps are the TRUSTED receive times (clock), not the events' own (year-2000) time.
    assert state.occurrences == (
        Occurrence(event_id="e1", at=_T0.isoformat()),
        Occurrence(event_id="e2", at=(_T0 + timedelta(seconds=60)).isoformat()),
    )


# ── integrity: event content cannot move the window (CRITICAL/HIGH root) ──────


async def test_spoofed_far_future_event_time_does_not_affect_the_count(tmp_path: Path):
    # A far-future envelope.time must NOT become the window reference (else it would prune every
    # real record / never age out). The trusted clock governs; the spoofed time is inert.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    await agg.snapshot(_ctx(_event("e1", time="3000-01-01T00:00:00+00:00")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("e2", time="3000-01-01T00:00:00+00:00")))

    assert state.count == 2 and state.breached is True  # counted at clock time, not year 3000
    # and the stored occurrences are stamped with the trusted clock, not the spoofed time
    assert all(r["at"].startswith("2026-01-01") for r in store.pending())


async def test_absurd_event_time_does_not_crash_snapshot(tmp_path: Path):
    # datetime.min as envelope.time used to overflow when it was the reference; it is now unused.
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=1, window_s=600, key_of=_is_anomaly, now=_Clock(_T0)
    )

    state = await agg.snapshot(_ctx(_event("e1", time="0001-01-01T00:00:00+00:00")))

    assert state.count == 1 and state.breached is True  # no crash; counted at the trusted clock


async def test_huge_window_does_not_crash(tmp_path: Path):
    # A pathologically huge window_s overflows the horizon subtraction → guarded to datetime.min.
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=1, window_s=1e18, key_of=_is_anomaly, now=_Clock(_T0)
    )

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1 and state.breached is True


# ── two-sided window + planted-record robustness (untrusted media) ───────────


async def test_planted_future_dated_record_is_excluded(tmp_path: Path):
    # A record dated AFTER the trusted reference (planted, or a backwards clock skew) must not
    # count — the window is two-sided, so a spoofed future record cannot inflate the count.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(
        store,
        {
            "idempotency_key": "future",
            "bucket": "anomaly",
            "at": (_T0 + timedelta(seconds=1000)).isoformat(),
        },
    )
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1 and state.breached is False  # only e1; the future record excluded
    assert [r["idempotency_key"] for r in store.pending()] == ["e1"]  # and pruned out


async def test_planted_record_missing_idempotency_key_is_dropped_not_crashed(tmp_path: Path):
    # A valid-JSON record lacking idempotency_key (plantable on removable media) must be dropped,
    # never dereferenced — no KeyError in the snapshot or the queue's index rebuild.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(store, {"bucket": "anomaly", "at": (_T0 - timedelta(seconds=60)).isoformat()})
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1  # only the valid e1
    assert [r["idempotency_key"] for r in store.pending()] == ["e1"]  # foreign record scrubbed


async def test_corrupt_record_timestamp_is_dropped(tmp_path: Path):
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(store, {"idempotency_key": "bad", "bucket": "anomaly", "at": "not-a-time"})
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1
    assert [r["idempotency_key"] for r in store.pending()] == ["e1"]


async def test_planted_overflowing_boundary_timestamp_is_dropped_not_crashed(tmp_path: Path):
    # A syntactically valid boundary date with a nonzero offset parses but overflows the UTC
    # conversion. It must be dropped (like any corrupt record), never crash the snapshot.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(store, {"idempotency_key": "of", "bucket": "anomaly", "at": "9999-12-31T23:59:59-14:00"})
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1  # only e1; the overflowing record dropped, no crash
    assert [r["idempotency_key"] for r in store.pending()] == ["e1"]


async def test_stored_at_variants_and_shapes(tmp_path: Path):
    # Records written with a Z suffix / a naive timestamp still count; a non-string `at` and a
    # non-string `bucket` are dropped. Exercises every branch of the timestamp parser + shape check.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(store, {"idempotency_key": "z", "bucket": "anomaly", "at": "2025-12-31T23:55:00Z"})
    _plant(store, {"idempotency_key": "naive", "bucket": "anomaly", "at": "2025-12-31T23:57:00"})
    _plant(store, {"idempotency_key": "nonstr_at", "bucket": "anomaly", "at": 99999})
    _plant(
        store,
        {"idempotency_key": "nonstr_bucket", "bucket": 123, "at": "2025-12-31T23:58:00+00:00"},
    )
    agg = ThresholdAggregator(store, threshold=5, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 3  # z + naive + e1; the non-string `at` and `bucket` records dropped
    assert sorted(r["idempotency_key"] for r in store.pending()) == ["e1", "naive", "z"]


async def test_default_clock_is_the_wall_clock(tmp_path: Path):
    # No injected `now` → the real UTC wall clock governs the window (the default). A lone event
    # counts at wall-now regardless of its own envelope.time.
    agg = ThresholdAggregator(_store(tmp_path), threshold=1, window_s=600, key_of=_is_anomaly)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1 and state.breached is True


# ── clock robustness: backward skew, naive clock ─────────────────────────────


async def test_backward_clock_step_retains_and_later_counts_the_occurrence(tmp_path: Path):
    # An NTP correction steps the wall clock BACKWARD. A just-recorded occurrence is briefly dated
    # "in the future" vs the new reference — it must be RETAINED (not deleted) and NOT counted yet,
    # then counted again once the clock recovers. (Regression for the receive-time redesign.)
    clock = _Clock(_T0)
    store = _store(tmp_path)
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    await agg.snapshot(_ctx(_event("e1")))  # e1 at T0
    clock.advance(-5)  # clock steps back 5s → e1 is now "in the future"
    mid = await agg.snapshot(_ctx(_event("e2")))

    assert mid.count == 1  # e1 not counted while it is future vs the stepped-back reference
    assert "e1" in [r["idempotency_key"] for r in store.pending()]  # but NOT deleted

    clock.advance(10)  # clock recovers past T0
    after = await agg.snapshot(_ctx(_event("e3")))

    assert after.count == 3 and after.breached is True  # e1 counts again once the clock catches up


async def test_far_future_planted_record_is_pruned_beyond_the_retention_band(tmp_path: Path):
    # A record more than one window ahead (planted / spoofed far-future) is outside the retention
    # band → pruned, so growth stays bounded and the count stays spoof-safe.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(
        store,
        {
            "idempotency_key": "far",
            "bucket": "anomaly",
            "at": (_T0 + timedelta(days=365)).isoformat(),
        },
    )
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 1 and state.breached is False
    assert [r["idempotency_key"] for r in store.pending()] == ["e1"]  # far-future record pruned


async def test_naive_injected_clock_does_not_raise_on_the_second_event(tmp_path: Path):
    # A tz-naive injected `now` is read as UTC; without normalization the aware/naive comparison
    # would raise on the second event (once the store is non-empty).
    class _NaiveClock:
        def __init__(self) -> None:
            self._t = datetime(2026, 1, 1, 0, 0, 0)  # naive

        def __call__(self) -> datetime:
            return self._t

        def advance(self, seconds: float) -> None:
            self._t += timedelta(seconds=seconds)

    clock = _NaiveClock()
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("e2")))  # would TypeError if reference stayed naive

    assert state.count == 2 and state.breached is True


# ── window edges (pinned choices) ────────────────────────────────────────────


async def test_record_exactly_at_horizon_counts(tmp_path: Path):
    # at == horizon is "not older than window_s" → inclusive lower bound.
    clock = _Clock(_T0)
    store = _store(tmp_path)
    _plant(
        store,
        {
            "idempotency_key": "edge",
            "bucket": "anomaly",
            "at": (_T0 - timedelta(seconds=600)).isoformat(),
        },
    )
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    state = await agg.snapshot(_ctx(_event("e1")))

    assert state.count == 2 and state.breached is True  # edge record + e1


# ── durability across a restart ──────────────────────────────────────────────


async def test_threshold_survives_a_fresh_aggregator_over_the_same_store(tmp_path: Path):
    clock = _Clock(_T0)
    await ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    ).snapshot(_ctx(_event("e1")))

    clock.advance(60)
    reborn = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )
    state = await reborn.snapshot(_ctx(_event("e2")))

    assert state.count == 2 and state.breached is True


# ── validation ───────────────────────────────────────────────────────────────


def test_threshold_must_be_at_least_one(tmp_path: Path):
    try:
        ThresholdAggregator(_store(tmp_path), threshold=0, window_s=600, key_of=_is_anomaly)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected threshold >= 1 to be enforced")


def test_window_must_be_positive(tmp_path: Path):
    try:
        ThresholdAggregator(_store(tmp_path), threshold=2, window_s=0, key_of=_is_anomaly)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected window_s > 0 to be enforced")


# ── integration: via the storage seam + the pipeline ─────────────────────────


async def test_aggregator_reuses_the_storage_seam_durable_store(tmp_path: Path):
    clock = _Clock(_T0)
    provider = select_storage_provider(
        DeviceCapabilityManifest(device_id="d", silicon="jetson_orin", ram_mb=8192),
        internal_root=tmp_path,
    )
    store = provider.durable_store("anomalies")
    agg = ThresholdAggregator(store, threshold=2, window_s=600, key_of=_is_anomaly, now=clock)

    await agg.snapshot(_ctx(_event("e1")))
    clock.advance(60)
    state = await agg.snapshot(_ctx(_event("e2")))

    assert isinstance(store, DurableQueue) and state.breached is True


async def test_aggregator_drives_a_deterministic_decision_in_a_pipeline(tmp_path: Path):
    from convilyn_edge.result import Ok, Result
    from convilyn_edge.spi.operator import ExecutionContext, OperatorError

    class _AlarmRule:
        def execute(
            self, state: ThresholdState, ctx: ExecutionContext
        ) -> Result[str, OperatorError]:
            return Ok("ALARM" if state.breached else "OK")

    clock = _Clock(_T0)
    agg = ThresholdAggregator(
        _store(tmp_path), threshold=2, window_s=600, key_of=_is_anomaly, now=clock
    )
    pipeline = Pipeline("wf").state(agg).decide(_AlarmRule(), bind=lambda s: s.output("state"))

    await pipeline.run(_event("e1"))
    clock.advance(60)
    result = await pipeline.run(_event("e2"))

    assert result.completed and result.output("decide") == "ALARM"
