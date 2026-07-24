"""Primitive 1/7 — ``EventSource``.

A Source is where events *enter* the SDK: a scanner adapter, a serial port, an
MQTT subscription, a camera. v0.1 ships no real hardware Source — the supported
boundary (Convilyn ships SPI + simulator + reference adapter only, never
hardware drivers; real drivers are integrator work). The built-in simulator's
``SimulatedSource`` and any real ``OposScannerSource`` / ``DataWedgeSource``
an integrator writes are substitutable behind this one Protocol (LSP).

Design notes:

* **Non-generic, yields ``EventEnvelope``.** The original sketch has
  ``EventSource<T>``; we pin ``EventEnvelope`` instead so there is exactly one
  envelope shape crossing the SDK (data-flow consistency). The typed per-event
  payload lives in ``envelope.data`` and is validated by a ``Normalizer``.
* **``start()`` returns an ``AsyncIterator`` (not a coroutine)** so a caller
  writes ``async for ev in source.start(ctx)`` directly.
* ``@runtime_checkable`` — this Protocol is method-only and non-generic, so an
  ``isinstance`` structural check is honest here (unlike the generic Protocols).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from convilyn_edge.envelope import EventEnvelope

HealthStatus = Literal["connected", "disconnected", "degraded"]


@dataclass(frozen=True)
class SourceContext:
    """Ambient context handed to a Source when it starts producing.

    Kept minimal for v0.1: the device identity the Source binds to plus an
    opaque, adapter-specific ``metadata`` map (baud rate, intent filter, MQTT
    topic, …). Cancellation is expressed by breaking the ``async for`` and
    awaiting :meth:`EventSource.stop`, not by a flag here.
    """

    device_id: str
    site_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceHealth:
    """A point-in-time device health reading (device-health event).

    ``status`` is the coarse connection state a watchdog acts on; ``detail`` is a
    human/diagnostic string; ``checked_at`` is a UTC ISO-8601 timestamp.
    """

    status: HealthStatus
    detail: str | None = None
    checked_at: str | None = None


@runtime_checkable
class EventSource(Protocol):
    """Produces a stream of :class:`EventEnvelope` from one device/transport."""

    def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
        """Begin producing events. Iterate with ``async for`` until exhausted."""
        ...

    async def health(self) -> DeviceHealth:
        """Report current device health (for the watchdog / fleet telemetry)."""
        ...

    async def stop(self) -> None:
        """Stop producing and release the underlying transport."""
        ...


__all__ = ["HealthStatus", "SourceContext", "DeviceHealth", "EventSource"]
