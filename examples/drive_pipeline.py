"""Self-drive a declarative pipeline over the simulator — the runtime hello-world.

Composes the 7-primitive edge SPI into a :class:`Pipeline` (Normalizer →
StateProvider → DeterministicOperator → ActionSink) and hands it to a
:class:`WorkflowDriver`, which pulls events from ``SimulatedSource`` and drives
each one through the pipeline with the deterministic runtime guarantees
(offline-first emission, fixed-backoff retry, stuck-abort). No hardware, no cloud,
no LLM — and it depends ONLY on ``convilyn-edge`` (never a Solution Pack), so it
is a faithful "build your own self-driving edge workflow" starting point.

    python examples/drive_pipeline.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from convilyn_edge import EventSourceRef, Ok, Result
from convilyn_edge.result import Err
from convilyn_edge.runtime import Pipeline, WorkflowDriver
from convilyn_edge.simulator import Scenario, ScenarioEvent, SimulatedSource
from convilyn_edge.spi.action import ActionAuthorization, ActionDescriptor, ActionResult, RiskLevel
from convilyn_edge.spi.normalizer import NormalizeError
from convilyn_edge.spi.operator import ExecutionContext, OperatorError
from convilyn_edge.spi.source import SourceContext
from convilyn_edge.spi.state import EventContext

SCHEMA = "convilyn://schemas/reading/v1"


# ── Tiny inline SPI impls (a real workflow injects concrete Solution-Pack ones) ─


class _IntNormalizer:
    """Raw ``{"value": "<n>"}`` → an int; a non-numeric payload is a rejection."""

    def normalize(self, raw: Mapping[str, Any]) -> Result[int, NormalizeError]:
        try:
            return Ok(int(raw["value"]))
        except (KeyError, ValueError):
            return Err(NormalizeError(code="not_a_number"))


class _ThresholdState:
    async def snapshot(self, ctx: EventContext) -> int:
        return 10  # the alarm threshold this register runs under


class _AlarmRule:
    """Deterministic: emit an 'ALARM' message iff the reading exceeds the threshold."""

    def execute(self, input: tuple[int, int], ctx: ExecutionContext) -> Result[str, OperatorError]:
        reading, threshold = input
        verdict = "ALARM" if reading > threshold else "ok"
        return Ok(f"{verdict}: reading={reading} threshold={threshold}")


class _PrintSink:
    """An R1 (reversible, low-risk) sink — the gate auto-grants it."""

    def describe(self) -> ActionDescriptor:
        return ActionDescriptor(action_id="display", risk=RiskLevel.R1, description="print")

    async def invoke(self, input: str, auth: ActionAuthorization) -> ActionResult[str]:
        print(f"    → {input}")
        return ActionResult(status="performed", output=input)


def build_pipeline() -> Pipeline:
    """Declare the DAG once; the driver self-drives every event through it."""
    return (
        Pipeline("threshold_alarm")
        .normalize(_IntNormalizer(), bind=lambda s: s.envelope.data)
        .state(_ThresholdState())
        .decide(_AlarmRule(), bind=lambda s: (s.output("normalize"), s.output("state")))
        .act(_PrintSink(), bind=lambda s: s.output("decide"))
    )


def build_source() -> SimulatedSource:
    readings = ["3", "12", "not-a-number", "25"]
    events = tuple(
        ScenarioEvent(event_type="device.reading", event_schema=SCHEMA, data={"value": r})
        for r in readings
    )
    scenario = Scenario(source=EventSourceRef("sensor-1", "sim", "0.1.0"), events=events)
    return SimulatedSource(scenario, no_delay=True)


async def main() -> None:
    pipeline = build_pipeline()
    ctx = SourceContext(device_id="sensor-1")

    print("Driving 4 readings (one is un-parseable → rejected, not crashed):")
    report = await WorkflowDriver().run(build_source(), ctx, pipeline.run)

    print(f"\nprocessed={report.processed} buffered={report.buffered} failed={report.failed}")
    for record in report.records:
        outcome = record.result  # the PipelineResult carried untouched by the driver
        print(f"  event {record.event_id[:8]} → {record.status} / {outcome.status}")


if __name__ == "__main__":
    asyncio.run(main())
