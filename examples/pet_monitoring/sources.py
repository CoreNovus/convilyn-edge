"""Simulator ``EventSource`` adapters for the pet-monitoring sensors.

These are the reference, hardware-free half of the device-I/O layer: the integrator writes the
*real* camera / feeder / water / litter adapters against their hardware (implementing the same
:class:`~convilyn_edge.spi.source.EventSource` Protocol); Convilyn ships these **simulator**
adapters so the pack builds and is testable with no device attached. Each replays a scripted list
of that sensor's readings.

They **reuse** :class:`~convilyn_edge.simulator.SimulatedSource` rather than re-implement the event
loop — the sim adapter just fixes its ``event_type`` + adapter identity and delegates
``start`` / ``health`` / ``stop`` (so it satisfies ``EventSource`` structurally). A real adapter
would instead drive a serial port / MQTT topic / camera.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from convilyn_edge.envelope import EventEnvelope, EventSourceRef
from convilyn_edge.simulator import Scenario, ScenarioEvent, SimulatedSource
from convilyn_edge.spi.source import DeviceHealth, HealthStatus, SourceContext
from examples.pet_monitoring.events import (
    EVENT_CAMERA_FRAME,
    EVENT_FEEDER_READING,
    EVENT_LITTER_READING,
    EVENT_WATER_READING,
    SENSOR_SCHEMA,
)


class _SensorSimSource:
    """Replay scripted readings for ONE sensor as an ``EventSource`` (delegates to the simulator).

    Subclasses fix ``event_type`` + ``adapter_id``; this base builds a one-sensor
    :class:`Scenario` and forwards the ``EventSource`` surface to a
    :class:`~convilyn_edge.simulator.SimulatedSource`. ``health`` reports the configured coarse
    status (so a pack can simulate a sensor going offline, which the watchdog observes).
    """

    event_type: str = ""
    adapter_id: str = "sim"

    def __init__(
        self,
        device_id: str,
        readings: Sequence[Mapping[str, Any]],
        *,
        health: HealthStatus = "connected",
        no_delay: bool = True,
    ) -> None:
        scenario = Scenario(
            source=EventSourceRef(device_id, self.adapter_id, "0.1.0"),
            events=tuple(
                ScenarioEvent(event_type=self.event_type, event_schema=SENSOR_SCHEMA, data=dict(r))
                for r in readings
            ),
            health_status=health,
        )
        self._sim = SimulatedSource(scenario, no_delay=no_delay)

    def start(self, ctx: SourceContext) -> AsyncIterator[EventEnvelope]:
        return self._sim.start(ctx)

    async def health(self) -> DeviceHealth:
        return await self._sim.health()

    async def stop(self) -> None:
        await self._sim.stop()


class CameraSimSource(_SensorSimSource):
    """Camera frames — each reading is ``{"frame_id": ..., "motion": bool, "image_ref": ...}``."""

    event_type = EVENT_CAMERA_FRAME
    adapter_id = "camera-sim"


class FeederSimSource(_SensorSimSource):
    """Feeder readings — each is ``{"level_pct": 0-100, "jammed": bool}``."""

    event_type = EVENT_FEEDER_READING
    adapter_id = "feeder-sim"


class WaterSimSource(_SensorSimSource):
    """Water readings — each is ``{"level_ml": int}``."""

    event_type = EVENT_WATER_READING
    adapter_id = "water-sim"


class LitterSimSource(_SensorSimSource):
    """Litter readings — each is ``{"uses": int}`` since the last clean."""

    event_type = EVENT_LITTER_READING
    adapter_id = "litter-sim"


__all__ = [
    "CameraSimSource",
    "FeederSimSource",
    "WaterSimSource",
    "LitterSimSource",
]
