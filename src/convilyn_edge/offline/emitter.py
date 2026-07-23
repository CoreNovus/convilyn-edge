"""``EventEmitter`` — emit Device Event Envelopes, offline-buffered.

Observability is a first-class SDK surface: every device event should travel from
hardware to the cloud through one structured, correlated
channel. This emitter is that channel's device end — it sends an
:class:`~convilyn_edge.envelope.EventEnvelope` (the one envelope shape, carrying
the W3C ``traceparent``-style correlation id — the shared observability
convention, not a backend import) to a sink, and when the sink is **offline**
buffers the envelope in a :class:`~convilyn_edge.offline.queue.DurableQueue`
instead of dropping it. ``flush`` drains the buffer when connectivity returns.

Apply-once: a buffered event is keyed by its globally-unique ``event_id`` (see
:func:`event_key`), so a re-``emit`` or a partial ``flush`` never queues or sends a
duplicate; a flush that re-sends an already-delivered event on retry is
deduplicated by the server's reconcile (the point of the idempotency key). The
sink is injected (a narrow Protocol) — the package takes on no transport
dependency; the caller brings HTTP/MQTT/gRPC.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.offline.queue import DurableQueue, FlushReport

#: The exceptions that mean "the cloud is unreachable" (buffer), as opposed to
#: "the cloud rejected the event" (surface — buffering would just replay it).
#: Deliberately narrow: a bare ``OSError`` would also swallow disk-full /
#: permission / not-found from a sink that touches the filesystem, misclassifying
#: a real failure as "offline". Callers widen it explicitly if their transport
#: needs it (e.g. adding an ``httpx.TransportError``).
DEFAULT_OFFLINE_EXCEPTIONS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)


def event_key(record: Mapping[str, Any]) -> str:
    """Apply-once key for a buffered event record — its unique wire ``eventId``.

    Pass to the queue: ``DurableQueue(path, key_of=event_key)``.
    """
    return record["eventId"]


class EventSink(Protocol):
    """The transport an emitter sends envelopes over (HTTP/MQTT/gRPC/…).

    ``send`` raises one of the emitter's ``offline_exceptions`` when the cloud is
    unreachable (→ buffered), or any other exception when the event is rejected
    (→ surfaced)."""

    async def send(self, envelope: EventEnvelope) -> None:
        """Deliver ``envelope`` to the cloud, or raise."""
        ...


class EventEmitter:
    """Emit envelopes with transparent offline buffering + apply-once flush."""

    def __init__(
        self,
        sink: EventSink,
        queue: DurableQueue,
        *,
        offline_exceptions: tuple[type[BaseException], ...] = DEFAULT_OFFLINE_EXCEPTIONS,
    ) -> None:
        self._sink = sink
        self._queue = queue
        self._offline = offline_exceptions

    async def emit(self, envelope: EventEnvelope) -> bool:
        """Send ``envelope`` now, or buffer it durably if the sink is offline.

        Returns ``True`` if delivered, ``False`` if buffered. A non-offline sink
        error propagates (buffering a rejected event would just replay it)."""
        try:
            await self._sink.send(envelope)
        except self._offline:
            self._queue.enqueue(envelope.to_wire())
            return False
        return True

    async def flush(self) -> FlushReport:
        """Drain the buffer through the sink, apply-once.

        FIFO is preserved: the first *offline* send STOPS the pass (connectivity
        is lost — the current record and everything after it stay queued, in
        order), so a sink that recovers mid-drain can't deliver a later event
        before an earlier one. A record the sink *rejects* (a non-offline error)
        or that is unparseable is dropped — retrying would not help, so it must
        not block the queue behind it (head-of-line safety). Delivered records are
        removed via an atomic rewrite.
        """
        records = self._queue.pending()
        survivors: list[dict[str, Any]] = []
        applied = 0
        rejected = 0
        for index, record in enumerate(records):
            try:
                envelope = EventEnvelope.from_wire(record)
            except Exception:  # noqa: BLE001 — unparseable record: drop, don't wedge
                rejected += 1
                continue
            try:
                await self._sink.send(envelope)
            except self._offline:
                survivors = list(records[index:])  # connectivity lost — keep the rest, in order
                break
            except Exception:  # noqa: BLE001 — server rejected: drop, don't block the queue
                rejected += 1
            else:
                applied += 1
        self._queue.replace(survivors)
        return FlushReport(applied=applied, requeued=len(survivors), rejected=rejected)


__all__ = [
    "DEFAULT_OFFLINE_EXCEPTIONS",
    "event_key",
    "EventSink",
    "EventEmitter",
]
