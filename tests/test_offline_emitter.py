"""Unit tests for EventEmitter (offline buffering + apply-once flush)."""

from __future__ import annotations

from pathlib import Path

import pytest

from convilyn_edge.envelope import EventSourceRef, new_envelope
from convilyn_edge.offline.emitter import EventEmitter, event_key
from convilyn_edge.offline.queue import DurableQueue

_SOURCE = EventSourceRef("dev-1", "sim", "0.1.0")


class _FakeSink:
    def __init__(self) -> None:
        self.sent: list = []
        self.offline = False

    async def send(self, envelope) -> None:
        if self.offline:
            raise ConnectionError("cloud unreachable")
        self.sent.append(envelope)


class _FlakySink:
    """Delivers ``deliver_n`` envelopes, then goes offline for the rest."""

    def __init__(self, deliver_n: int) -> None:
        self.sent: list = []
        self._deliver_n = deliver_n

    async def send(self, envelope) -> None:
        if len(self.sent) >= self._deliver_n:
            raise ConnectionError("cloud unreachable")
        self.sent.append(envelope)


class _RejectingSink:
    """Rejects every send with a non-offline error (server validation failure)."""

    async def send(self, envelope) -> None:
        raise ValueError("server rejected the event")


def _emitter(tmp_path: Path, sink) -> tuple[EventEmitter, DurableQueue]:
    queue = DurableQueue(tmp_path / "events.jsonl", key_of=event_key)
    return EventEmitter(sink, queue), queue


def _event(seq: int = 0):
    return new_envelope(
        event_type="device.test", event_schema="s", source=_SOURCE, data={"seq": seq}
    )


# ── online emit ──────────────────────────────────────────────────────────────


async def test_emit_online_delivers_immediately(tmp_path):
    sink = _FakeSink()
    emitter, queue = _emitter(tmp_path, sink)

    delivered = await emitter.emit(_event())

    assert delivered is True and len(sink.sent) == 1 and queue.is_empty()


# ── offline emit buffers ─────────────────────────────────────────────────────


async def test_emit_offline_buffers_and_does_not_send(tmp_path):
    sink = _FakeSink()
    sink.offline = True
    emitter, queue = _emitter(tmp_path, sink)

    delivered = await emitter.emit(_event())

    assert delivered is False and sink.sent == [] and len(queue) == 1


# ── the round-trip: offline → reconnect → flush without duplicate ────────────


async def test_offline_then_reconnect_flush_delivers_once(tmp_path):
    sink = _FakeSink()
    sink.offline = True
    emitter, queue = _emitter(tmp_path, sink)
    await emitter.emit(_event())

    sink.offline = False  # reconnect
    report = await emitter.flush()

    assert report.applied == 1 and queue.is_empty() and len(sink.sent) == 1


async def test_second_flush_after_clean_is_noop(tmp_path):
    sink = _FakeSink()
    sink.offline = True
    emitter, queue = _emitter(tmp_path, sink)
    await emitter.emit(_event())
    sink.offline = False
    await emitter.flush()

    report = await emitter.flush()  # nothing left

    assert report.applied == 0 and report.clean


# ── apply-once: duplicate emit is deduped ────────────────────────────────────


async def test_reemitting_same_envelope_offline_is_deduped(tmp_path):
    sink = _FakeSink()
    sink.offline = True
    emitter, queue = _emitter(tmp_path, sink)
    envelope = _event()

    await emitter.emit(envelope)
    await emitter.emit(envelope)  # same event_id

    assert len(queue) == 1


# ── partial flush: still-offline survivors are re-queued ─────────────────────


async def test_partial_flush_requeues_survivors(tmp_path):
    sink = _FlakySink(deliver_n=1)  # delivers the first, then offline
    emitter, queue = _emitter(tmp_path, sink)
    for i in range(3):
        queue.enqueue(_event(i).to_wire())

    report = await emitter.flush()

    assert (report.applied, report.requeued, len(queue)) == (1, 2, 2)


# ── error handling: rejection / non-offline propagation ──────────────────────


async def test_emit_propagates_non_offline_error_without_buffering(tmp_path):
    emitter, queue = _emitter(tmp_path, _RejectingSink())

    with pytest.raises(ValueError):
        await emitter.emit(_event())

    assert queue.is_empty()  # a rejection is surfaced, NOT buffered


async def test_flush_drops_rejected_records_without_wedging(tmp_path):
    emitter, queue = _emitter(tmp_path, _RejectingSink())
    queue.enqueue(_event(0).to_wire())
    queue.enqueue(_event(1).to_wire())

    report = await emitter.flush()

    # Both rejected (non-offline) → dropped, not blocking the queue behind them.
    assert (report.rejected, queue.is_empty()) == (2, True)


async def test_flush_drops_unparseable_record(tmp_path):
    sink = _FakeSink()
    emitter, queue = _emitter(tmp_path, sink)
    queue.enqueue({"eventId": "bad-1"})  # has the dedup key but is not a valid envelope

    report = await emitter.flush()

    assert (report.rejected, queue.is_empty()) == (1, True)


# ── CRITICAL regression: a line-separator payload survives the round-trip ────


async def test_line_separator_payload_survives_offline_round_trip(tmp_path):
    sink = _FakeSink()
    sink.offline = True
    emitter, queue = _emitter(tmp_path, sink)
    payload_text = "scanned\u2028label\u0085fragment"
    envelope = new_envelope(
        event_type="t", event_schema="s", source=_SOURCE, data={"text": payload_text}
    )
    await emitter.emit(envelope)  # buffered while offline

    sink.offline = False
    report = await emitter.flush()

    assert report.applied == 1 and sink.sent[0].data["text"] == payload_text
