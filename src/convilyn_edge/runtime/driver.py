"""``WorkflowDriver`` — the self-driving event loop over an ``EventSource``.

Pulls events from a :class:`~convilyn_edge.spi.source.EventSource` and drives each
through a per-event ``handler`` (typically a :meth:`Pipeline.run`), wrapping the
cross-cutting **deterministic** runtime concerns the roadmap enumerates:

* **Offline-first emission** — each event is emitted through an optional
  :class:`~convilyn_edge.offline.EventEmitter`, which durably buffers it when the
  cloud is unreachable; the buffer is drained on ``run`` completion. A cloud
  *rejection* of the observability emit never stops local processing (the device
  keeps working offline).
* **Deterministic retry/backoff** — transient handler failures retry under a
  fixed :class:`RetryPolicy` (:func:`retry_async`); no LLM decides eligibility.
* **Stuck-abort** — a :class:`CycleGuard` aborts the whole run when the same
  hard-failure signature repeats, so a source spewing an un-processable event
  cannot spin forever.

The driver owns **infrastructure** outcomes (processed / buffered / failed /
aborted); the *business* outcome (a pipeline's completed / rejected / denied /
halted) is carried untouched in ``EventRecord.result``. This split keeps the
driver generic — it encodes no scenario logic and no workflow-specific branching.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any, Literal

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.offline.emitter import DEFAULT_OFFLINE_EXCEPTIONS, EventEmitter
from convilyn_edge.offline.queue import FlushReport
from convilyn_edge.runtime.policy import DEFAULT_RETRYABLE, CycleGuard, RetryPolicy, retry_async
from convilyn_edge.runtime.watchdog import WatchdogPolicy, poll_health
from convilyn_edge.spi.source import EventSource, SourceContext

#: Infrastructure-level outcome of one event through the driver:
#: ``processed`` (handler returned — see ``result`` for the business outcome),
#: ``buffered`` (offline after retries; retained for later), ``failed`` (a
#: non-transient handler error — recorded, loop continues).
DriveStatus = Literal["processed", "buffered", "failed"]

#: A per-event handler: run the workflow for one envelope, return its outcome.
EventHandler = Callable[[EventEnvelope], Awaitable[Any]]

#: Maps a record to a stuck-abort signature, or ``None`` to signal PROGRESS
#: (resets the guard). Default: only consecutive hard ``failed`` records count.
ProgressKey = Callable[["EventRecord"], "Hashable | None"]


@dataclass(frozen=True)
class EventRecord:
    """One event's driver outcome + the handler's returned business result."""

    event_id: str
    status: DriveStatus
    result: Any = None
    detail: str | None = None
    #: Stable failure signature for the stuck-guard: the exception *type* name only
    #: (``detail`` carries the full formatted message for humans). Keying the guard
    #: on the type — not the message — means a fault whose message embeds variable
    #: data (a byte offset, a per-event id, a timestamp) still trips the guard on
    #: repetition instead of looking like fresh progress every iteration.
    failure_kind: str | None = None


def _default_progress_key(record: EventRecord) -> Hashable | None:
    """Guard only on repeated hard failures; any other outcome is progress.

    Buffering is normal offline-first behaviour (not stuck), and a processed event
    — whatever its business outcome — is progress, so neither trips the guard. The
    signature is the exception *type* (``failure_kind``), so a repeating fault with
    a varying message cannot masquerade as progress and spin forever.
    """
    if record.status == "failed":
        return ("failed", record.failure_kind)
    return None


@dataclass(frozen=True)
class DriveReport:
    """Aggregate outcome of a drive pass."""

    records: tuple[EventRecord, ...]
    flushed: FlushReport | None = None
    aborted: bool = False

    @property
    def processed(self) -> int:
        return sum(1 for record in self.records if record.status == "processed")

    @property
    def buffered(self) -> int:
        return sum(1 for record in self.records if record.status == "buffered")

    @property
    def failed(self) -> int:
        return sum(1 for record in self.records if record.status == "failed")


class WorkflowDriver:
    """Drive an ``EventSource`` through a handler with the deterministic runtime
    guarantees (retry, offline buffering, stuck-abort). Stateless across runs —
    construct once, ``run`` many sources."""

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        cycle_guard: CycleGuard | None = None,
        retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE,
        offline_on: tuple[type[BaseException], ...] = DEFAULT_OFFLINE_EXCEPTIONS,
        progress_key: ProgressKey = _default_progress_key,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._retry = retry if retry is not None else RetryPolicy()
        self._cycle_guard = cycle_guard
        self._retry_on = retry_on
        self._offline_on = offline_on
        self._progress_key = progress_key
        self._sleep = sleep

    async def run(
        self,
        source: EventSource,
        ctx: SourceContext,
        handler: EventHandler,
        *,
        emitter: EventEmitter | None = None,
        max_events: int | None = None,
        watchdog: WatchdogPolicy | None = None,
    ) -> DriveReport:
        """Consume ``source`` and drive each event through ``handler``.

        ``emitter`` (optional) carries offline-first event emission + a final
        flush. ``max_events`` bounds the pass (useful for a bounded/simulated
        source that would otherwise stream forever). ``watchdog`` (optional) runs a
        deterministic health-poll clock *beside* the event loop, so a source that goes
        silent between events (a camera that stops producing frames) is still observed
        — default ``None`` keeps the pure event-driven behaviour unchanged. Returns a
        :class:`DriveReport`; ``aborted`` is True when the stuck-guard fired.
        """
        # A fresh guard per run so a prior run's streak never leaks in. A caller
        # that injected a shared guard opts into cross-run accounting explicitly.
        guard = self._cycle_guard if self._cycle_guard is not None else CycleGuard()
        records: list[EventRecord] = []
        aborted = False
        count = 0

        # Own the source's full lifecycle: stop the watchdog, close the event stream (if it
        # exposes ``aclose``) and call ``stop()`` in ``finally`` so a real hardware source
        # (serial port, MQTT subscription, camera) is torn down on every exit path —
        # break on ``max_events``, stuck-abort, or exception — not left leaking until
        # GC. ``aclose`` is probed defensively: not every iterator is a generator.
        #
        # ``source.start(ctx)`` runs BEFORE the watchdog is scheduled: a device whose
        # start() raises (port/camera not ready — exactly what a watchdog would watch) must
        # not leave an orphaned health-poll task polling a source that never started. The
        # watchdog is created only once the stream exists, so the ``finally`` below always
        # owns whatever was scheduled.
        stream = source.start(ctx)
        stop_watch = asyncio.Event()
        # An opt-in health-poll clock running concurrently with the event loop. It shares
        # the driver's injected ``sleep`` (so tests never actually wait). asyncio is
        # cooperative-single-threaded, so the observer only runs between awaits — no data
        # race with event processing.
        watch_task = (
            asyncio.ensure_future(poll_health(source, watchdog, stop=stop_watch, sleep=self._sleep))
            if watchdog is not None
            else None
        )
        try:
            async for envelope in stream:
                if max_events is not None and count >= max_events:
                    break
                count += 1

                if emitter is not None:
                    await self._emit(emitter, envelope)

                record = await self._process(envelope, handler)
                records.append(record)

                signature = self._progress_key(record)
                if signature is None:
                    guard.reset()
                elif guard.record(signature):
                    aborted = True
                    break
        finally:
            if watch_task is not None:
                # Signal a graceful stop AND cancel, so a watchdog mid-``sleep`` wakes at
                # once instead of after its full interval; swallow the resulting cancellation.
                stop_watch.set()
                watch_task.cancel()
                with contextlib.suppress(BaseException):
                    await watch_task
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
            await source.stop()

        flushed = await self._flush(emitter)
        return DriveReport(tuple(records), flushed, aborted)

    async def _flush(self, emitter: EventEmitter | None) -> FlushReport | None:
        """Drain the offline buffer, guarded. A flush failure (e.g. durable-queue
        disk I/O) must NOT discard the accumulated DriveReport — the buffer is
        retained for a later pass. Offline-first: a cloud/storage hiccup at drain
        time is not a run failure (mirrors the per-event ``_emit`` swallow)."""
        if emitter is None:
            return None
        try:
            return await emitter.flush()
        except Exception:  # noqa: BLE001 — flush failure must not lose the report
            return None

    async def _emit(self, emitter: EventEmitter, envelope: EventEnvelope) -> None:
        """Emit for observability, offline-buffered. A cloud *rejection* is
        swallowed so it never stops local processing (offline-first)."""
        try:
            await emitter.emit(envelope)
        except Exception:  # noqa: BLE001 — emit rejection must not halt the device
            pass

    async def _process(self, envelope: EventEnvelope, handler: EventHandler) -> EventRecord:
        """Run ``handler`` under the retry policy; classify the outcome."""

        async def operation() -> Any:
            return await handler(envelope)

        try:
            result = await retry_async(
                operation, self._retry, retry_on=self._retry_on, sleep=self._sleep
            )
        except self._offline_on as exc:
            # Still offline after the retry budget — retained for a later pass.
            return EventRecord(envelope.event_id, "buffered", detail=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 — record + keep the loop alive
            return EventRecord(
                envelope.event_id,
                "failed",
                detail=f"{type(exc).__name__}: {exc}",
                failure_kind=type(exc).__name__,
            )
        return EventRecord(envelope.event_id, "processed", result=result)


__all__ = [
    "DriveStatus",
    "EventHandler",
    "ProgressKey",
    "EventRecord",
    "DriveReport",
    "WorkflowDriver",
]
