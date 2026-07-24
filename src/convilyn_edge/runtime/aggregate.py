"""Stateful threshold aggregation — cross-event accumulation as a StateProvider.

The pipeline threads a per-event :class:`~convilyn_edge.runtime.pipeline.PipelineState`; each
event is otherwise stateless. Some deterministic decisions need *history* — "≥ N of this thing
within a time window" (two feeder jams in 10 min, three low-water readings in an hour). This
module is that history primitive, expressed as a :class:`~convilyn_edge.spi.state.StateProvider`
so it drops straight into ``Pipeline.state(...)`` and the deterministic operator downstream reads
its snapshot exactly like any other environment state.

## Durable through the storage seam — no parallel state

The accumulator is a :class:`~convilyn_edge.offline.queue.DurableQueue` handed in by the caller
(``select_storage_provider(manifest, internal_root=…).durable_store("…")``). The aggregator does
**not** open its own store, invent a second state substrate, or reimplement persistence — it reuses
the one durable, apply-once record store the storage axis already provides, so a threshold survives
a device restart and an offline window like every other durable record.

The store must be **dedicated** to one aggregator (a store *name* not shared with another writer):
each snapshot rewrites the file to just the surviving in-window occurrences (bounded retention), so
a co-tenant's unrelated records in the same store would be discarded. This mirrors the sole-writer
assumption ``DurableQueue`` already documents.

## Integrity: a trusted clock, never event content

This is a **safety-relevant counter** — ``breached`` can gate an alert or action — and its store may
sit on removable / media-store media. So the window is anchored to a **trusted clock** injected by
the composition root (``now``, default the UTC wall clock), NOT to ``envelope.time`` (which
``EventEnvelope.from_wire`` copies from the wire unvalidated). Each occurrence is stamped with the
trusted *receive* time; the count band is the strict two-sided ``[now - window_s, now]`` — a record
dated after ``now`` (planted / spoofed / clock-skewed) is NOT counted, so a bad timestamp can never
inflate the count. For a pinned-clock **replay**, inject a ``now`` that returns the replay time; the
count then reproduces exactly.

**Semantics — counts by RECEIVE time.** Because occurrences are stamped when observed, this counts
"≥ N events *received* within ``window_s``". That is the spoof-safe choice, but note the trade: a
batch of **offline-buffered** events flushed together on reconnect is counted at flush time, so
occurrences that truly happened hours apart can collapse into one window. If a workflow needs
occurrence-time fidelity across a buffered flush, drive the flush through an injected ``now`` that
returns each event's true (trusted) time; otherwise place the aggregator on the live intake path.

**Clock contract.** ``now`` should be **non-decreasing**. A backward step (an NTP correction on an
RTC-less edge device) is tolerated up to one window: the retention band is wider than the count band
(``[now - window_s, now + window_s]``), so a just-recorded occurrence briefly dated "in the future"
after a backward step is *retained* (not deleted) and counts again once the clock recovers — only a
skew larger than ``window_s`` can transiently drop it.

Records read back from the store are **shape-validated** (a string ``idempotency_key`` + ``bucket``
and a parseable in-band ``at``); anything else — a torn, foreign, or planted record — is dropped,
so a corrupt record can neither crash a snapshot nor poison the count.

## Deterministic + scenario-free

What *counts* as an occurrence, and how occurrences bucket, is the pack's injected ``key_of``
binding — the aggregator counts, it never knows "anomaly" or any scenario noun (no ``if silicon
==``, no vertical vocabulary). It never calls an LLM. Idempotency is apply-once by ``event_id``
**within the retention window** (a replay of an event still on the store is not double-counted; the
SDK's pinned-clock replay reproduces the original). ``event_id`` is required to be a unique
per-occurrence identity — the envelope contract's guarantee (uuid4-minted, pinned on replay); a
source that reuses ids collapses distinct occurrences. Stdlib only; no ``import app.*``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.spi.state import EventContext

#: The pack's binding: map an event to the bucket key it counts under, or ``None`` when the event
#: is not a countable occurrence. Two events with the SAME key accumulate toward one threshold; a
#: constant key counts everything together ("any anomaly"), a composite key
#: (``f"{device}:{kind}"``) counts per device+kind. The grouping is the pack's; the counting is the
#: aggregator's. (A ``key_of`` that reads attacker-controlled ``data`` lets a caller steer which
#: bucket increments — that is the pack's trust decision, documented here for pack authors.)
OccurrenceKey = Callable[[EventEnvelope], "str | None"]


def _utc_now() -> datetime:
    """The default trusted reference clock (UTC wall clock). Injectable for replay / tests."""
    return datetime.now(timezone.utc)


def _shift(reference: datetime, seconds: float) -> datetime:
    """``reference`` shifted by ``seconds`` (may be negative), guarded against overflow.

    A pathologically large ``window_s`` overflows the ``timedelta`` add/subtract; clamp to the
    representable bound (aware UTC) so the window degrades to unbounded rather than crashing.
    """
    try:
        return reference + timedelta(seconds=seconds)
    except OverflowError:
        bound = datetime.min if seconds < 0 else datetime.max
        return bound.replace(tzinfo=timezone.utc)


def _try_parse_utc(value: Any) -> datetime | None:
    """Parse a UTC ISO-8601 timestamp to an aware ``datetime``, or ``None`` if it can't.

    Accepts the envelope's ``+00:00`` form and a trailing ``Z`` (``fromisoformat`` learned ``Z``
    only in 3.11; the SDK floor is 3.10). A naive value is read as UTC. Any bad / non-string input
    yields ``None`` so a single corrupt record can never crash a snapshot — it is simply dropped.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text[-1:] in {"Z", "z"}:
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        # A boundary date (year 0001/9999) with a nonzero offset parses but OVERFLOWS on the
        # UTC conversion — a planted record on removable media could otherwise crash every
        # snapshot. Catch it here so a corrupt record is dropped, never fatal.
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class Occurrence:
    """One counted occurrence within the window: the event that caused it + its receive time."""

    event_id: str
    at: str


@dataclass(frozen=True)
class ThresholdState:
    """A snapshot of the accumulator for the current event's bucket.

    ``key`` is the current event's bucket (``None`` when the event did not qualify — then ``count``
    is 0 and ``breached`` is False). ``count`` is the in-window occurrence count for that bucket
    *including* the current event; ``breached`` is ``count >= threshold``. ``occurrences`` are the
    in-window occurrences the count is drawn from (event ids + receive times), for the pack's
    evidence.
    """

    key: str | None
    count: int
    threshold: int
    window_s: float
    breached: bool
    occurrences: tuple[Occurrence, ...] = field(default_factory=tuple)


class ThresholdAggregator:
    """A ``StateProvider[ThresholdState]`` that accumulates occurrences across events, durably.

    Satisfies the :class:`~convilyn_edge.spi.state.StateProvider` Protocol structurally, so it wires
    in via ``Pipeline.state(aggregator)``. Construct it with a **dedicated** durable store obtained
    from the storage seam and the pack's occurrence binding::

        store = select_storage_provider(manifest, internal_root=root).durable_store("anomalies")
        aggregator = ThresholdAggregator(store, threshold=2, window_s=600, key_of=is_anomaly)
        pipeline = Pipeline("wf").normalize(...).state(aggregator).decide(alert_rule, bind=...)

    ``snapshot`` records the triggering event (when it qualifies), prunes occurrences older than
    ``window_s`` (and drops any invalid/foreign record), and returns the current in-window count for
    its bucket. Recording during a snapshot is intentional and idempotent — the aggregator's state
    genuinely *includes* the current event's contribution, and a within-window replay is not
    double-counted. It writes durable state during ``snapshot`` (not a pure read) and is **not
    concurrency-safe**: drive one event at a time (the pipeline does).
    """

    def __init__(
        self,
        store: DurableQueue,
        *,
        threshold: int,
        window_s: float,
        key_of: OccurrenceKey,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._store = store
        self._threshold = threshold
        self._window_s = window_s
        self._key_of = key_of
        # The trusted time authority. MUST NOT be derived from event content — the window's
        # integrity depends on it (a caller passing an event-derived clock reopens the spoofing
        # hole this design closes).
        self._now = now if now is not None else _utc_now

    async def snapshot(self, ctx: EventContext) -> ThresholdState:
        """Record this event's occurrence (if it qualifies), prune the window, return the count."""
        bucket = self._key_of(ctx.envelope)
        reference = self._now()
        if reference.tzinfo is None:
            # Tolerate a naive injected clock (read as UTC) — else the aware/naive comparison in
            # the window predicates would raise on the second event, once the store is non-empty.
            reference = reference.replace(tzinfo=timezone.utc)
        horizon = _shift(reference, -self._window_s)
        # Retention band is WIDER than the count band on the future side: a record dated slightly
        # after `reference` because the wall clock stepped BACKWARD (an NTP correction) is retained
        # (up to one window ahead) rather than deleted, so a benign skew never loses a not-yet-aged
        # occurrence — it simply isn't *counted* until the clock recovers. A record further ahead
        # than that (planted / spoofed far-future) is still pruned, keeping growth bounded (≤ 2
        # windows) and the count spoof-safe.
        retain_ceiling = _shift(reference, self._window_s)

        # Read (no key_of applied — safe against a key-less record), then keep only well-shaped
        # records inside the retention band. Rewriting to `kept` bounds retention AND scrubs any
        # planted / foreign / aged-out record from the store before we enqueue (so the queue's
        # index build never dereferences a missing idempotency_key).
        records = self._store.pending()
        kept = [r for r in records if self._retained(r, horizon, retain_ceiling)]
        if len(kept) != len(records):
            self._store.replace(kept)

        if bucket is None:
            return ThresholdState(None, 0, self._threshold, self._window_s, False, ())

        new_record = {
            "idempotency_key": ctx.envelope.event_id,
            "bucket": bucket,
            "at": reference.isoformat(),
        }
        # Idempotent by event_id within the retained window: a replay of a still-present event is
        # dropped (written=False) and is already among `kept`; otherwise it is appended.
        written = self._store.enqueue(new_record)
        counted = [*kept, new_record] if written else kept

        # Count is strictly two-sided ``[horizon, reference]``: a retained-but-future (skew) record
        # is NOT counted until the clock catches up, so a spoofed/skewed future timestamp can never
        # inflate the count.
        occurrences = tuple(
            Occurrence(event_id=r["idempotency_key"], at=r["at"])
            for r in counted
            if r.get("bucket") == bucket and self._countable(r, horizon, reference)
        )
        count = len(occurrences)
        return ThresholdState(
            key=bucket,
            count=count,
            threshold=self._threshold,
            window_s=self._window_s,
            breached=count >= self._threshold,
            occurrences=occurrences,
        )

    @staticmethod
    def _retained(record: dict[str, Any], horizon: datetime, ceiling: datetime) -> bool:
        """Well-shaped AND within the (wider) retention band ``[horizon, ceiling]``.

        A string ``idempotency_key`` + ``bucket`` are required so downstream dereferences (and the
        queue's apply-once index) can never ``KeyError`` on a foreign record; the timestamp must
        parse and sit inside the band (aged-out or far-future records are dropped).
        """
        if not isinstance(record.get("idempotency_key"), str):
            return False
        if not isinstance(record.get("bucket"), str):
            return False
        at = _try_parse_utc(record.get("at"))
        return at is not None and horizon <= at <= ceiling

    @staticmethod
    def _countable(record: dict[str, Any], horizon: datetime, reference: datetime) -> bool:
        """Within the strict count band ``[horizon, reference]`` (a future record isn't counted)."""
        at = _try_parse_utc(record.get("at"))
        return at is not None and horizon <= at <= reference


__all__ = ["OccurrenceKey", "Occurrence", "ThresholdState", "ThresholdAggregator"]
