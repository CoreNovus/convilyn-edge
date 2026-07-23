"""Device simulator — drive a workflow with no hardware.

A developer should not need a real scanner plugged in for every test — a
simulator is a first-class SDK surface.
:class:`SimulatedSource` is the first concrete :class:`~convilyn_edge.spi.source.EventSource`
implementation: it replays a JSON :class:`Scenario` as a stream of
:class:`~convilyn_edge.envelope.EventEnvelope`, so any workflow can be exercised
end to end from a file. It is the half of the boundary the SDK ships (SPI +
simulator + reference adapter — never a hardware driver).

A scenario declares a device and an ordered list of events; each event may set a
``delay_ms`` (inter-event timing) and a ``repeat`` (to simulate a duplicate scan).
Author out-of-order events by listing them out of order — the emission order is
the scenario order, so faults stay **deterministic and replayable** (the CLI's
``--no-delay`` collapses timing for a fast, deterministic replay). Stdlib only.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.spi.source import DeviceHealth, HealthStatus, SourceContext

#: The valid coarse health states — derived from the SPI literal so it never drifts.
_HEALTH_STATUSES: frozenset[str] = frozenset(get_args(HealthStatus))


@dataclass(frozen=True)
class ScenarioEvent:
    """One event the simulator will emit."""

    event_type: str
    event_schema: str
    data: Mapping[str, Any] = field(default_factory=dict)
    delay_ms: int = 0
    repeat: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, wire: Mapping[str, Any]) -> ScenarioEvent:
        return cls(
            event_type=wire["event_type"],
            event_schema=wire["event_schema"],
            data=wire.get("data", {}),
            delay_ms=int(wire.get("delay_ms", 0)),
            repeat=int(wire.get("repeat", 1)),
            metadata=wire.get("metadata", {}),
        )


@dataclass(frozen=True)
class Scenario:
    """A simulator scenario: a device + an ordered list of events."""

    source: EventSourceRef
    events: tuple[ScenarioEvent, ...]
    site_id: str | None = None
    health_status: HealthStatus = "connected"

    @classmethod
    def from_dict(cls, wire: Mapping[str, Any]) -> Scenario:
        device = wire["device"]
        health_status = wire.get("health_status", "connected")
        if health_status not in _HEALTH_STATUSES:
            raise ValueError(
                f"invalid health_status {health_status!r}: one of {sorted(_HEALTH_STATUSES)}"
            )
        return cls(
            source=EventSourceRef(
                device_id=device["device_id"],
                adapter_id=device.get("adapter_id", "sim"),
                adapter_version=device.get("adapter_version", "0.1.0"),
            ),
            events=tuple(ScenarioEvent.from_dict(event) for event in wire.get("events", [])),
            site_id=wire.get("site_id"),
            health_status=health_status,
        )

    @classmethod
    def load(cls, path: Path) -> Scenario:
        """Load a scenario from a JSON file."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class SimulatedSource:
    """Replay a :class:`Scenario` as an ``EventSource`` (no hardware).

    Implements ``EventSource`` structurally. ``no_delay`` collapses inter-event
    timing for a deterministic, fast replay.
    """

    def __init__(self, scenario: Scenario, *, no_delay: bool = False) -> None:
        self._scenario = scenario
        self._no_delay = no_delay
        self._stopped = False

    def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
        """Begin emitting the scenario's events (repeat + delay honoured)."""

        async def _generate() -> AsyncIterator[EventEnvelope]:
            for event in self._scenario.events:
                # repeat<=0 means "skip this event" (honour the author, don't clamp to 1).
                for _ in range(max(0, event.repeat)):
                    if self._stopped:
                        return
                    if event.delay_ms and not self._no_delay:
                        await asyncio.sleep(event.delay_ms / 1000)
                    if self._stopped:  # a concurrent stop() during the delay takes effect now
                        return
                    yield new_envelope(
                        event_type=event.event_type,
                        event_schema=event.event_schema,
                        source=self._scenario.source,
                        data=event.data,
                        metadata=event.metadata,
                    )

        return _generate()

    async def health(self) -> DeviceHealth:
        status: HealthStatus = "disconnected" if self._stopped else self._scenario.health_status
        return DeviceHealth(status=status)

    async def stop(self) -> None:
        self._stopped = True


__all__ = ["ScenarioEvent", "Scenario", "SimulatedSource"]
