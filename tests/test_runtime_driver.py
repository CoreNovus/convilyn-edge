"""Unit tests for WorkflowDriver — event loop, offline flush, retry, stuck-abort."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.offline.emitter import EventEmitter, event_key
from convilyn_edge.offline.queue import DurableQueue, FlushReport
from convilyn_edge.runtime.driver import WorkflowDriver
from convilyn_edge.runtime.policy import CycleGuard, RetryPolicy
from convilyn_edge.spi.source import DeviceHealth, SourceContext


async def _nosleep(_seconds: float) -> None:
    return None


class _ListSource:
    """A minimal EventSource that replays a fixed list of envelopes."""

    def __init__(self, envelopes: Sequence[EventEnvelope]) -> None:
        self._envelopes = tuple(envelopes)

    def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
        async def _gen() -> AsyncIterator[EventEnvelope]:
            for envelope in self._envelopes:
                yield envelope

        return _gen()

    async def health(self) -> DeviceHealth:
        return DeviceHealth(status="connected")

    async def stop(self) -> None:
        return None


class _RejectingSink:
    """An EventSink whose send raises to simulate offline / rejection."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.calls = 0

    async def send(self, envelope: EventEnvelope) -> None:
        self.calls += 1
        raise self._exc


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


def _driver(**kwargs) -> WorkflowDriver:
    kwargs.setdefault("sleep", _nosleep)
    return WorkflowDriver(**kwargs)


# ── logic: happy loop ────────────────────────────────────────────────────────


async def test_processes_every_event():
    source = _ListSource(_events(3))

    report = await _driver().run(source, SourceContext(device_id="dev"), lambda e: _ok())

    assert report.processed == 3


async def _ok() -> str:
    return "ok"


async def test_handler_result_is_recorded():
    source = _ListSource(_events(1))

    report = await _driver().run(source, SourceContext(device_id="dev"), lambda e: _ok())

    assert report.records[0].result == "ok"


async def test_max_events_bounds_the_pass():
    source = _ListSource(_events(10))

    report = await _driver().run(
        source, SourceContext(device_id="dev"), lambda e: _ok(), max_events=2
    )

    assert len(report.records) == 2


# ── error handling: transient vs permanent ───────────────────────────────────


async def test_transient_failure_recovers_via_retry():
    state = {"n": 0}

    async def handler(_e: EventEnvelope) -> str:
        state["n"] += 1
        if state["n"] < 2:
            raise ConnectionError("blip")
        return "ok"

    report = await _driver(retry=RetryPolicy(max_attempts=3)).run(
        _ListSource(_events(1)), SourceContext(device_id="dev"), handler
    )

    assert report.processed == 1


async def test_persistent_offline_is_buffered_status():
    async def handler(_e: EventEnvelope) -> str:
        raise ConnectionError("cloud down")

    report = await _driver(retry=RetryPolicy(max_attempts=2)).run(
        _ListSource(_events(1)), SourceContext(device_id="dev"), handler
    )

    assert report.buffered == 1


async def test_non_transient_error_is_failed_status():
    async def handler(_e: EventEnvelope) -> str:
        raise ValueError("bug")

    report = await _driver().run(_ListSource(_events(1)), SourceContext(device_id="dev"), handler)

    assert report.failed == 1


async def test_failure_does_not_stop_subsequent_events():
    seen = {"n": 0}

    async def handler(_e: EventEnvelope) -> str:
        seen["n"] += 1
        if seen["n"] == 1:
            raise ValueError("first only")
        return "ok"

    report = await _driver().run(_ListSource(_events(3)), SourceContext(device_id="dev"), handler)

    assert report.failed == 1 and report.processed == 2


# ── stuck-abort ──────────────────────────────────────────────────────────────


async def test_repeated_identical_failure_aborts():
    async def handler(_e: EventEnvelope) -> str:
        raise ValueError("same every time")

    report = await _driver(cycle_guard=CycleGuard(max_consecutive=2)).run(
        _ListSource(_events(5)), SourceContext(device_id="dev"), handler
    )

    assert report.aborted is True and len(report.records) == 2


async def test_varying_failure_message_still_aborts():
    # Regression: a fault whose MESSAGE varies each iteration (a byte offset, an id,
    # a timestamp) but whose TYPE is stable must still trip the stuck-guard — the
    # signature keys on the exception type, not the formatted message.
    seen = {"n": 0}

    async def handler(_e: EventEnvelope) -> str:
        seen["n"] += 1
        raise ValueError(f"failed at offset {seen['n']}")  # different message each time

    report = await _driver(cycle_guard=CycleGuard(max_consecutive=2)).run(
        _ListSource(_events(5)), SourceContext(device_id="dev"), handler
    )

    assert report.aborted is True and len(report.records) == 2


async def test_progress_resets_stuck_guard():
    seen = {"n": 0}

    async def handler(_e: EventEnvelope) -> str:
        seen["n"] += 1
        if seen["n"] % 2 == 0:  # every other event succeeds → never 2 fails in a row
            return "ok"
        raise ValueError("mostly failing")

    report = await _driver(cycle_guard=CycleGuard(max_consecutive=2)).run(
        _ListSource(_events(4)), SourceContext(device_id="dev"), handler
    )

    assert report.aborted is False


# ── offline emitter integration ──────────────────────────────────────────────


async def test_emitter_buffers_events_when_offline(tmp_path: Path):
    queue = DurableQueue(tmp_path / "q.jsonl", key_of=event_key)
    emitter = EventEmitter(_RejectingSink(ConnectionError("offline")), queue)

    await _driver().run(
        _ListSource(_events(2)), SourceContext(device_id="dev"), lambda e: _ok(), emitter=emitter
    )

    assert len(queue.pending()) == 2  # both durably buffered


async def test_emit_rejection_does_not_stop_processing(tmp_path: Path):
    # A NON-offline emit error (cloud rejects the observability event) must be
    # swallowed so local processing still runs (offline-first).
    queue = DurableQueue(tmp_path / "q.jsonl", key_of=event_key)
    emitter = EventEmitter(_RejectingSink(ValueError("rejected")), queue)

    report = await _driver().run(
        _ListSource(_events(2)), SourceContext(device_id="dev"), lambda e: _ok(), emitter=emitter
    )

    assert report.processed == 2


async def test_flush_failure_does_not_discard_the_report(tmp_path: Path):
    # Regression: a flush failure (e.g. durable-queue disk I/O) must NOT abort the
    # run or throw away the accumulated DriveReport — offline-first. The records are
    # returned, flushed is None, and the buffer survives for a later pass.
    class _FlushBoomEmitter(EventEmitter):
        async def flush(self) -> FlushReport:
            raise OSError("disk full")

    queue = DurableQueue(tmp_path / "q.jsonl", key_of=event_key)
    emitter = _FlushBoomEmitter(_RejectingSink(ConnectionError("offline")), queue)

    report = await _driver().run(
        _ListSource(_events(2)), SourceContext(device_id="dev"), lambda e: _ok(), emitter=emitter
    )

    assert report.processed == 2
    assert report.flushed is None  # flush failure swallowed, report intact


async def test_source_is_stopped_and_stream_closed_on_bounded_exit(tmp_path: Path):
    # Regression: the driver owns the source lifecycle. On a bounded (max_events)
    # exit it must call source.stop() AND close the event stream (aclosing runs the
    # generator's finally) so real hardware handles are not leaked until GC.
    class _SpySource(_ListSource):
        def __init__(self, envelopes: Sequence[EventEnvelope]) -> None:
            super().__init__(envelopes)
            self.stopped = 0
            self.stream_closed = False

        def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
            async def _gen() -> AsyncIterator[EventEnvelope]:
                try:
                    for envelope in self._envelopes:
                        yield envelope
                finally:
                    self.stream_closed = True

            return _gen()

        async def stop(self) -> None:
            self.stopped += 1

    source = _SpySource(_events(5))

    report = await _driver().run(
        source, SourceContext(device_id="dev"), lambda e: _ok(), max_events=1
    )

    assert report.processed == 1
    assert source.stopped == 1
    assert source.stream_closed is True


async def test_flush_report_present_when_emitter_used(tmp_path: Path):
    queue = DurableQueue(tmp_path / "q.jsonl", key_of=event_key)
    emitter = EventEmitter(_RejectingSink(ConnectionError("offline")), queue)

    report = await _driver().run(
        _ListSource(_events(1)), SourceContext(device_id="dev"), lambda e: _ok(), emitter=emitter
    )

    assert report.flushed is not None
