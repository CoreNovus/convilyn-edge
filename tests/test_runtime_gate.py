"""Unit tests for ActionGate — deterministic R0/R1 auto-grant, R2+ gating."""

from __future__ import annotations

from dataclasses import dataclass

from convilyn_edge.runtime.gate import ActionGate
from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    RiskLevel,
)


@dataclass
class _RecordingSink:
    """A fake ActionSink at a configurable risk that records whether it ran."""

    risk: RiskLevel
    requires_approval: bool = False
    invoked_with: ActionAuthorization | None = None

    def describe(self) -> ActionDescriptor:
        return ActionDescriptor(
            action_id="test.sink",
            risk=self.risk,
            description="test",
            requires_approval=self.requires_approval,
        )

    async def invoke(self, input: str, auth: ActionAuthorization) -> ActionResult[str]:
        self.invoked_with = auth
        return ActionResult(status="performed", output=input)


_APPROVED = ActionAuthorization(operator="sup-1", reason="ok", approved=True)
_UNAPPROVED = ActionAuthorization(operator="sup-1", reason="pending", approved=False)


# ── logic: R0/R1 auto-grant ──────────────────────────────────────────────────


async def test_r0_is_auto_granted_without_auth():
    sink = _RecordingSink(risk=RiskLevel.R0)

    result = await ActionGate().invoke(sink, "hi")

    assert result.status == "performed"


async def test_r1_auto_grant_synthesizes_system_authorization():
    sink = _RecordingSink(risk=RiskLevel.R1)

    await ActionGate().invoke(sink, "hi")

    assert sink.invoked_with is not None and sink.invoked_with.operator == "system"


async def test_auto_grant_prefers_supplied_auth():
    sink = _RecordingSink(risk=RiskLevel.R1)

    await ActionGate().invoke(sink, "hi", _APPROVED)

    assert sink.invoked_with is _APPROVED


# ── logic: R2+ gating ────────────────────────────────────────────────────────


async def test_r2_denied_without_auth():
    sink = _RecordingSink(risk=RiskLevel.R2)

    result = await ActionGate().invoke(sink, "hi")

    assert result.status == "denied"


async def test_r2_denied_does_not_invoke_sink():
    sink = _RecordingSink(risk=RiskLevel.R2)

    await ActionGate().invoke(sink, "hi")

    assert sink.invoked_with is None  # sink never ran


async def test_r2_performed_with_approved_auth():
    sink = _RecordingSink(risk=RiskLevel.R2)

    result = await ActionGate().invoke(sink, "hi", _APPROVED)

    assert result.status == "performed"


async def test_r3_denied_with_unapproved_auth():
    sink = _RecordingSink(risk=RiskLevel.R3)

    result = await ActionGate().invoke(sink, "hi", _UNAPPROVED)

    assert result.status == "denied"


# ── boundary: requires_approval overrides low risk ───────────────────────────


async def test_requires_approval_forces_gate_even_at_r1():
    sink = _RecordingSink(risk=RiskLevel.R1, requires_approval=True)

    result = await ActionGate().invoke(sink, "hi")

    assert result.status == "denied"


async def test_denied_reason_names_the_action_and_risk():
    sink = _RecordingSink(risk=RiskLevel.R3)

    result = await ActionGate().invoke(sink, "hi")

    assert result.reason is not None and "R3" in result.reason
