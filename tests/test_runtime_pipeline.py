"""Unit tests for the declarative Pipeline (7-primitive DAG, self-driving)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.result import Err, Ok, Result
from convilyn_edge.runtime.pipeline import ModelStepOutput, Pipeline, PipelineState
from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    RiskLevel,
)
from convilyn_edge.spi.model import ModelResult
from convilyn_edge.spi.normalizer import NormalizeError
from convilyn_edge.spi.operator import ExecutionContext, OperatorError
from convilyn_edge.spi.review import ReviewOutcome, ReviewRequest
from convilyn_edge.spi.state import EventContext

# ── fakes (in-memory SPI implementations) ────────────────────────────────────


@dataclass
class _FakeNormalizer:
    ok: bool = True

    def normalize(self, raw: Mapping[str, Any]) -> Result[dict[str, Any], NormalizeError]:
        if self.ok:
            return Ok(dict(raw))
        return Err(NormalizeError(code="bad_payload"))


@dataclass
class _FakeStateProvider:
    value: str = "state-A"

    async def snapshot(self, ctx: EventContext) -> str:
        return self.value


@dataclass
class _FakeOperator:
    ok: bool = True

    def execute(self, input: Any, ctx: ExecutionContext) -> Result[str, OperatorError]:
        if self.ok:
            return Ok("decided")
        return Err(OperatorError(code="illegal_transition"))


@dataclass
class _FakeModel:
    result: ModelResult[Any]

    async def infer(
        self,
        input: Any,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: str = "auto",
    ) -> ModelResult[Any]:
        return self.result


@dataclass
class _FakeReview:
    outcome: ReviewOutcome

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        return self.outcome


@dataclass
class _FakeSink:
    risk: RiskLevel = RiskLevel.R1
    ran: bool = False

    def describe(self) -> ActionDescriptor:
        return ActionDescriptor(action_id="sink", risk=self.risk, description="d")

    async def invoke(self, input: Any, auth: ActionAuthorization) -> ActionResult[str]:
        self.ran = True
        return ActionResult(status="performed", output="shown")


def _envelope() -> EventEnvelope:
    return new_envelope(
        event_type="t",
        event_schema="s",
        source=EventSourceRef("dev", "sim", "0.1.0"),
        data={"code": "abc"},
    )


# ── PipelineState: object-state / immutability ───────────────────────────────


def test_with_output_returns_new_state():
    state = PipelineState(envelope=_envelope())

    advanced = state.with_output("a", 1)

    assert advanced is not state and "a" not in state.outputs


def test_output_reads_named_value():
    state = PipelineState(envelope=_envelope()).with_output("a", 42)

    assert state.output("a") == 42


# ── Pipeline: logic (happy path) ─────────────────────────────────────────────


async def test_full_spine_completes():
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .normalize(_FakeNormalizer(), bind=lambda s: s.envelope.data)
        .state(_FakeStateProvider())
        .decide(_FakeOperator(), bind=lambda s: s.output("normalize"))
        .act(sink, bind=lambda s: s.output("decide"))
    )

    result = await pipeline.run(_envelope())

    assert result.completed and sink.ran


def test_pipeline_builder_is_immutable():
    base = Pipeline("wf")

    extended = base.normalize(_FakeNormalizer(), bind=lambda s: s.envelope.data)

    assert base.steps == () and len(extended.steps) == 1


# ── Pipeline: short-circuit (rejected / denied / halted) ─────────────────────


async def test_normalize_error_short_circuits_rejected():
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .normalize(_FakeNormalizer(ok=False), bind=lambda s: s.envelope.data)
        .act(sink, bind=lambda s: "x")
    )

    result = await pipeline.run(_envelope())

    assert result.status == "rejected" and not sink.ran


async def test_decide_error_short_circuits_rejected():
    pipeline = (
        Pipeline("wf")
        .normalize(_FakeNormalizer(), bind=lambda s: s.envelope.data)
        .decide(_FakeOperator(ok=False), bind=lambda s: s.output("normalize"))
    )

    result = await pipeline.run(_envelope())

    assert result.status == "rejected"


async def test_review_stop_halts_before_action():
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .review(_FakeReview(ReviewOutcome(decision="stop")), bind=lambda s: ReviewRequest("t", "m"))
        .act(sink, bind=lambda s: "x")
    )

    result = await pipeline.run(_envelope())

    assert result.status == "halted" and not sink.ran


async def test_action_denied_when_r2_without_auth():
    sink = _FakeSink(risk=RiskLevel.R2)
    pipeline = Pipeline("wf").act(sink, bind=lambda s: "x")

    result = await pipeline.run(_envelope())

    assert result.status == "denied" and not sink.ran


# ── Pipeline: model dispatch ─────────────────────────────────────────────────


async def test_model_step_stores_accept_disposition():
    model = _FakeModel(
        ModelResult(
            status="success",
            model_id="m",
            model_version="1",
            latency_ms=1.0,
            output={"k": "v"},
        )
    )
    pipeline = Pipeline("wf").model(model, bind=lambda s: s.envelope.data, schema={})

    result = await pipeline.run(_envelope())
    output = result.output("model")

    assert isinstance(output, ModelStepOutput) and output.disposition == "accept"


async def test_model_step_stores_fallback_on_unavailable():
    model = _FakeModel(
        ModelResult(status="unavailable", model_id="m", model_version="1", latency_ms=1.0)
    )
    pipeline = Pipeline("wf").model(model, bind=lambda s: s.envelope.data, schema={})

    result = await pipeline.run(_envelope())

    assert result.output("model").disposition == "fallback"


# ── Pipeline: review escalate proceeds (not halted) ──────────────────────────


async def test_review_escalate_proceeds_to_action():
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .review(
            _FakeReview(ReviewOutcome(decision="escalate")),
            bind=lambda s: ReviewRequest("t", "m"),
        )
        .act(sink, bind=lambda s: "x")
    )

    result = await pipeline.run(_envelope())

    assert result.completed and sink.ran


# ── Pipeline: review expiry → default_action (declarative, replay-injectable) ─


class _NeverReview:
    """A HumanReview that never resolves — the human who does not answer in time."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


async def _instant(_seconds: float) -> None:
    return None


async def test_review_expiry_default_stop_halts_before_action():
    # expires_at is load-bearing in the DAG: an unanswered review resolves to its declared
    # default_action=stop → the pipeline halts before the action fires. Clocks injected so
    # the test is deterministic and never waits (the replay guarantee at the pipeline level).
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .review(
            _NeverReview(),
            bind=lambda s: ReviewRequest(
                "t", "m", default_action="stop", expires_at="2026-01-01T00:10:00+00:00"
            ),
            now=lambda: datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            sleep=_instant,
        )
        .act(sink, bind=lambda s: "x")
    )

    result = await pipeline.run(_envelope())

    assert result.status == "halted" and not sink.ran


async def test_review_expiry_default_continue_proceeds_but_marks_source():
    # default_action=continue → the review proceeds; the stored outcome is marked
    # expiry_default (not human) so a downstream approval gate can refuse to treat the
    # timeout as a human grant.
    sink = _FakeSink()
    pipeline = (
        Pipeline("wf")
        .review(
            _NeverReview(),
            bind=lambda s: ReviewRequest(
                "t", "m", default_action="continue", expires_at="2026-01-01T00:10:00+00:00"
            ),
            now=lambda: datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            sleep=_instant,
        )
        .act(sink, bind=lambda s: "x")
    )

    result = await pipeline.run(_envelope())

    assert result.completed and sink.ran
    assert result.output("review").outcome.decision_source == "expiry_default"
