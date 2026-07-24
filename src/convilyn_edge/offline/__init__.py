"""Offline-first — durable event/result buffering with apply-once flush.

The device keeps working when the cloud is unreachable: structured records are
appended to a :class:`DurableQueue` and flushed exactly once when connectivity
returns. Content-addressed idempotency keys (mirroring the server
reconcile contract) make re-flushing a no-op.

    from convilyn_edge.offline import DurableQueue, EventEmitter, event_key

    queue = DurableQueue(Path("edge-events.jsonl"), key_of=event_key)
    emitter = EventEmitter(my_sink, queue)   # my_sink: EventSink
    await emitter.emit(envelope)             # sent, or durably buffered if offline
    report = await emitter.flush()           # drain on reconnect; report.clean == True
"""

from __future__ import annotations

from convilyn_edge.offline.emitter import (
    DEFAULT_OFFLINE_EXCEPTIONS,
    EventEmitter,
    EventSink,
    event_key,
)
from convilyn_edge.offline.idempotency import KEY_PREFIX, derive_idempotency_key
from convilyn_edge.offline.queue import DurableQueue, FlushReport, KeyOf

__all__ = [
    # queue
    "DurableQueue",
    "FlushReport",
    "KeyOf",
    # idempotency
    "derive_idempotency_key",
    "KEY_PREFIX",
    # emitter
    "EventEmitter",
    "EventSink",
    "event_key",
    "DEFAULT_OFFLINE_EXCEPTIONS",
]
