"""The anti-divergence litmus — the SDK's reason to exist, as a committed lint.

The removability question: *"delete an entire vertical Solution Pack; does the
remaining SDK still let you build another IoT AI workflow?"* If yes, this is a
general substrate; if no, a vertical product with a POC baked into its core.
These tests pin that answer at CI time:

1. The substrate carries **no scenario vocabulary in its control flow** — no
   ``if … barcode/cashier/scanner/…`` branch anywhere in ``convilyn_edge`` (the
   grep-observable of ``generic-tools-agent-orchestration.md``, promoted to a
   committed check).
2. The substrate has **no reverse dependency** on any Solution Pack — importing
   ``convilyn_edge`` never pulls in any other ``convilyn*`` package.
3. A **generic workflow builds + runs** against the simulator using ONLY
   ``convilyn_edge`` — a factory over-temperature alarm, zero scenario imports.

These run in the edge package's own test env, where no Solution Pack is installed
— so the "pack absent" precondition is real, not mocked.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from convilyn_edge import (
    ActionDescriptor,
    ActionResult,
    Err,
    ExecutionContext,
    NormalizeError,
    Ok,
    Result,
    RiskLevel,
)
from convilyn_edge.simulator import Scenario, SimulatedSource
from convilyn_edge.spi.source import SourceContext

_SRC = Path(__file__).resolve().parents[1] / "src" / "convilyn_edge"

#: Scenario proper nouns that must never drive control flow in the substrate.
_SCENARIO_BRANCH = re.compile(
    r"\bif\b.*\b(barcode|cashier|scanner|pos_stage|shopee|lazada|momo|retail)\b",
    re.IGNORECASE,
)


# ── 1. no scenario control flow in the substrate ─────────────────────────────


def test_no_scenario_branch_in_substrate():
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _SCENARIO_BRANCH.search(line):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {line.strip()}")

    detail = "scenario vocabulary drives control flow in the substrate:\n" + "\n".join(offenders)
    assert offenders == [], detail


# ── 2. no reverse dependency on a Solution Pack ──────────────────────────────


def test_substrate_does_not_import_a_solution_pack():
    # importing convilyn_edge (already done at module load) must not have pulled
    # in any solution pack or any other convilyn package.
    leaked = [
        m
        for m in sys.modules
        if m.startswith("convilyn") and not (m == "convilyn_edge" or m.startswith("convilyn_edge."))
    ]

    assert leaked == []


# ── 3. a generic workflow builds + runs with ONLY convilyn_edge ──────────────


class _TemperatureNormalizer:
    """Normalizer[dict, float] — a raw sensor payload → a temperature reading."""

    def normalize(self, raw) -> Result[float, NormalizeError]:
        value = raw.get("celsius")
        if not isinstance(value, (int, float)):
            return Err(NormalizeError(code="missing_celsius"))
        return Ok(float(value))


class _OverTemperatureOperator:
    """DeterministicOperator[float, str] — a pure threshold rule, no LLM."""

    def execute(self, input: float, ctx: ExecutionContext) -> Result[str, NormalizeError]:
        return Ok("ALARM" if input > 80.0 else "OK")


class _AlarmSink:
    """ActionSink[str, None] — records raised alarms (R1)."""

    def __init__(self) -> None:
        self.alarms: list[str] = []

    def describe(self) -> ActionDescriptor:
        return ActionDescriptor(action_id="machine.alarm", risk=RiskLevel.R1, description="alarm")

    async def invoke(self, input: str, auth) -> ActionResult[None]:
        self.alarms.append(input)
        return ActionResult(status="performed", output=None)


async def test_generic_workflow_runs_against_simulator_with_substrate_only():
    # A factory machine-alarm workflow — Source → Normalizer → DeterministicOperator
    # → ActionSink — composed from convilyn_edge alone, driven by the simulator.
    scenario = Scenario.from_dict(
        {
            "device": {"device_id": "machine-1"},
            "events": [
                {"event_type": "sensor.temp", "event_schema": "s", "data": {"celsius": 95}},
                {"event_type": "sensor.temp", "event_schema": "s", "data": {"celsius": 40}},
            ],
        }
    )
    source = SimulatedSource(scenario, no_delay=True)
    normalizer = _TemperatureNormalizer()
    operator = _OverTemperatureOperator()
    sink = _AlarmSink()

    async for envelope in source.start(SourceContext(device_id="machine-1")):
        reading = normalizer.normalize(envelope.data)
        if reading.is_err:
            continue
        outcome = operator.execute(reading.unwrap(), ExecutionContext(envelope=envelope))
        if outcome.unwrap() == "ALARM":
            await sink.invoke("over-temperature", None)

    assert sink.alarms == ["over-temperature"]  # only the 95°C event alarmed
