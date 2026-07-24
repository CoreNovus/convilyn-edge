"""Unit tests for the declarative branch/router primitive (#2792 PR-c)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.result import Err, Ok, Result
from convilyn_edge.runtime.gate import ActionGate
from convilyn_edge.runtime.pipeline import Pipeline, PipelineState, RouteOutcome
from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    RiskLevel,
)
from convilyn_edge.spi.operator import ExecutionContext, OperatorError


@dataclass
class _FakeOperator:
    value: str = "decided"
    ok: bool = True

    def execute(self, input: Any, ctx: ExecutionContext) -> Result[str, OperatorError]:
        return Ok(self.value) if self.ok else Err(OperatorError(code="rule_failed"))


@dataclass
class _FakeSink:
    label: str = "sink"
    risk: RiskLevel = RiskLevel.R1
    ran: bool = False

    def describe(self) -> ActionDescriptor:
        return ActionDescriptor(action_id=self.label, risk=self.risk, description="d")

    async def invoke(self, input: Any, auth: ActionAuthorization) -> ActionResult[str]:
        self.ran = True
        return ActionResult(status="performed", output=self.label)


def _envelope() -> EventEnvelope:
    return new_envelope(
        event_type="t",
        event_schema="s",
        source=EventSourceRef("dev", "sim", "0.1.0"),
        data={"code": "abc"},
    )


# ── selection: the matching branch runs, the others don't ────────────────────


async def test_selected_branch_runs():
    online = _FakeSink("online")
    offline = _FakeSink("offline")
    pipeline = Pipeline("wf").route(
        select=lambda s: "offline",
        routes={
            "online": Pipeline("online").act(online, bind=lambda s: "x"),
            "offline": Pipeline("offline").act(offline, bind=lambda s: "x"),
        },
    )

    result = await pipeline.run(_envelope())

    assert result.completed and offline.ran and not online.ran


async def test_selector_reads_upstream_state():
    # The selector picks a branch from an upstream stage's output — the branch is a declaration
    # driven by accumulated state, exactly the "offline → check connectivity, else → locate" shape.
    locate = _FakeSink("locate")
    connectivity = _FakeSink("connectivity")
    pipeline = (
        Pipeline("wf")
        .decide(_FakeOperator(value="offline"), bind=lambda s: s.envelope.data)
        .route(
            select=lambda s: s.output("decide"),
            routes={
                "online": Pipeline("online").act(locate, bind=lambda s: "x"),
                "offline": Pipeline("offline").act(connectivity, bind=lambda s: "x"),
            },
        )
    )

    result = await pipeline.run(_envelope())

    assert result.completed and connectivity.ran and not locate.ran


# ── default / unmapped key ───────────────────────────────────────────────────


async def test_default_used_for_unmapped_key():
    fallback = _FakeSink("fallback")
    pipeline = Pipeline("wf").route(
        select=lambda s: "unknown",
        routes={"fallback": Pipeline("fallback").act(fallback, bind=lambda s: "x")},
        default="fallback",
    )

    result = await pipeline.run(_envelope())

    assert result.completed and fallback.ran


async def test_unmapped_key_without_default_rejects():
    pipeline = Pipeline("wf").route(
        select=lambda s: "nope",
        routes={"a": Pipeline("a")},
    )

    result = await pipeline.run(_envelope())

    assert result.status == "rejected"


async def test_unmapped_key_records_the_attempted_route_outcome():
    # Audit symmetry: even a routing dead-end records a RouteOutcome (attempted key, no steps).
    pipeline = Pipeline("wf").route(select=lambda s: "nope", routes={"a": Pipeline("a")})

    result = await pipeline.run(_envelope())
    outcome = result.output("route")

    assert isinstance(outcome, RouteOutcome) and outcome.key == "nope" and outcome.steps == ()


async def test_default_pointing_at_missing_route_rejects():
    pipeline = Pipeline("wf").route(
        select=lambda s: "nope",
        routes={"a": Pipeline("a")},
        default="also-missing",
    )

    result = await pipeline.run(_envelope())

    assert result.status == "rejected"


# ── branch short-circuit propagates ──────────────────────────────────────────


async def test_branch_short_circuit_propagates_status():
    # A branch whose inner stage rejects makes the router (and pipeline) terminate rejected.
    after = _FakeSink("after")
    pipeline = (
        Pipeline("wf")
        .route(
            select=lambda s: "bad",
            routes={"bad": Pipeline("bad").decide(_FakeOperator(ok=False), bind=lambda s: {})},
        )
        .act(after, bind=lambda s: "x")  # must NOT run — router short-circuited
    )

    result = await pipeline.run(_envelope())

    assert result.status == "rejected" and not after.ran


async def test_branch_denied_action_propagates():
    pipeline = Pipeline("wf").route(
        select=lambda s: "r2",
        routes={"r2": Pipeline("r2").act(_FakeSink("hi", risk=RiskLevel.R2), bind=lambda s: "x")},
    )

    result = await pipeline.run(_envelope())

    assert result.status == "denied"


async def test_parent_gate_governs_actions_inside_a_branch():
    # A tightened PARENT gate (auto-grant only R0) must be enforced inside branches too — an R1
    # action in a branch is denied, not silently auto-granted by the branch's own default gate.
    r1 = _FakeSink("locate", risk=RiskLevel.R1)
    strict = Pipeline("wf", gate=ActionGate(auto_grant_max=RiskLevel.R0)).route(
        select=lambda s: "b",
        routes={"b": Pipeline("b").act(r1, bind=lambda s: "x")},
    )

    result = await strict.run(_envelope())

    assert result.status == "denied" and not r1.ran  # parent policy propagated into the branch


async def test_parent_gate_governs_nested_branch_actions():
    # The parent gate propagates recursively through nested routers.
    r1 = _FakeSink("deep", risk=RiskLevel.R1)
    inner = Pipeline("inner").route(
        select=lambda s: "d", routes={"d": Pipeline("d").act(r1, bind=lambda s: "x")}
    )
    strict = Pipeline("wf", gate=ActionGate(auto_grant_max=RiskLevel.R0)).route(
        select=lambda s: "outer", routes={"outer": inner}
    )

    result = await strict.run(_envelope())

    assert result.status == "denied" and not r1.ran


# ── state threading + RouteOutcome ───────────────────────────────────────────


async def test_branch_outputs_are_readable_downstream():
    # A branch stage's output is threaded onto the same state, so a later top-level stage reads it.
    sink = _FakeSink("act")
    pipeline = (
        Pipeline("wf")
        .route(
            select=lambda s: "b",
            routes={
                "b": Pipeline("b").decide(_FakeOperator(value="from-branch"), bind=lambda s: {})
            },
        )
        .act(sink, bind=lambda s: s.output("decide"))  # reads the branch's decide output
    )

    result = await pipeline.run(_envelope())

    assert result.completed and result.output("decide") == "from-branch"


async def test_route_outcome_records_key_and_sub_steps():
    pipeline = Pipeline("wf").route(
        select=lambda s: "b",
        routes={"b": Pipeline("b").decide(_FakeOperator(value="v"), bind=lambda s: {})},
    )

    result = await pipeline.run(_envelope())
    outcome = result.output("route")

    assert isinstance(outcome, RouteOutcome)
    assert outcome.key == "b"
    assert len(outcome.steps) == 1 and outcome.steps[0].status == "ok"


# ── nested routers ───────────────────────────────────────────────────────────


async def test_nested_routers():
    leaf = _FakeSink("leaf")
    inner = Pipeline("inner").route(
        select=lambda s: "deep",
        routes={"deep": Pipeline("deep").act(leaf, bind=lambda s: "x")},
    )
    pipeline = Pipeline("wf").route(select=lambda s: "outer", routes={"outer": inner})

    result = await pipeline.run(_envelope())

    assert result.completed and leaf.ran


# ── immutability + custom name ───────────────────────────────────────────────


async def test_route_builder_is_immutable():
    base = Pipeline("wf")

    extended = base.route(select=lambda s: "a", routes={"a": Pipeline("a")})

    assert base.steps == () and len(extended.steps) == 1


async def test_custom_route_name_keys_the_outcome():
    pipeline = Pipeline("wf").route(
        select=lambda s: "b",
        routes={"b": Pipeline("b")},
        name="branch",
    )

    result = await pipeline.run(_envelope())

    assert isinstance(result.output("branch"), RouteOutcome)


async def test_empty_branch_completes():
    # A route to an empty pipeline is a valid no-op branch (declaratively "do nothing here").
    pipeline = Pipeline("wf").route(select=lambda s: "noop", routes={"noop": Pipeline("noop")})

    result = await pipeline.run(_envelope())

    assert result.completed and result.output("route").steps == ()


def test_route_state_is_never_mutated():
    # Selector receives the live state but the primitive threads NEW states (immutability).
    state = PipelineState(envelope=_envelope()).with_output("x", 1)

    assert "route" not in state.outputs  # sanity: nothing leaks back onto the pre-route state
