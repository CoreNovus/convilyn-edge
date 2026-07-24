"""Unit tests for the 7-primitive SPI (``convilyn_edge.spi``).

Protocols carry no logic to exercise, so these tests assert the *contract*: the
non-generic Protocols are ``@runtime_checkable`` (isinstance is honest); the
generic ones are Protocols but deliberately NOT runtime-checkable; and the frozen
data shapes carry the semantics the reference architecture depends on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from convilyn_edge.envelope import EventSourceRef, new_envelope
from convilyn_edge.spi import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    DeterministicOperator,
    DeviceHealth,
    EventSource,
    Evidence,
    HumanReview,
    ModelOperator,
    ModelResult,
    NormalizeError,
    Normalizer,
    OperatorError,
    ReviewOutcome,
    RiskLevel,
    SourceContext,
    StateProvider,
)

_SOURCE_REF = EventSourceRef("dev-1", "sim", "0.1.0")


class _ConformingSource:
    """A structurally-valid EventSource (no inheritance — proves structural typing)."""

    def start(self, ctx: SourceContext) -> AsyncIterator:
        async def _gen():
            yield new_envelope(event_type="t", event_schema="s", source=_SOURCE_REF, data={})

        return _gen()

    async def health(self) -> DeviceHealth:
        return DeviceHealth(status="connected")

    async def stop(self) -> None:
        return None


class _ConformingReview:
    async def request(self, req) -> ReviewOutcome:
        return ReviewOutcome(decision="continue")


# ── runtime_checkable (non-generic Protocols) ────────────────────────────────


def test_conforming_source_passes_isinstance():
    assert isinstance(_ConformingSource(), EventSource)


def test_nonconforming_object_fails_source_isinstance():
    assert not isinstance(object(), EventSource)


def test_conforming_review_passes_isinstance():
    assert isinstance(_ConformingReview(), HumanReview)


# ── generic Protocols are NOT runtime_checkable (by design) ──────────────────


@pytest.mark.parametrize(
    "proto",
    [Normalizer, StateProvider, DeterministicOperator, ModelOperator],
)
def test_generic_protocol_is_a_protocol(proto):
    assert getattr(proto, "_is_protocol", False) is True


@pytest.mark.parametrize(
    "proto",
    [Normalizer, StateProvider, DeterministicOperator, ModelOperator],
)
def test_generic_protocol_rejects_isinstance(proto):
    # isinstance against a non-runtime_checkable Protocol must raise, not lie.
    with pytest.raises(TypeError):
        isinstance(object(), proto)


# ── data shapes ──────────────────────────────────────────────────────────────


def test_risk_levels_are_ordered():
    assert RiskLevel.R0 < RiskLevel.R2 < RiskLevel.R3


def test_risk_gate_comparison():
    # The policy-engine idiom: "R2 and above needs approval".
    assert (RiskLevel.R3 >= RiskLevel.R2) is True


def test_model_result_defaults_to_no_output():
    result: ModelResult[str] = ModelResult(
        status="unavailable", model_id="m", model_version="v", latency_ms=1.0
    )

    assert result.output is None


def test_model_result_carries_evidence_tuple():
    result: ModelResult[str] = ModelResult(
        status="success",
        model_id="m",
        model_version="v",
        latency_ms=1.0,
        output="x",
        evidence=(Evidence(source="jd_0", snippet="x"),),
    )

    assert result.evidence[0].source == "jd_0"


def test_action_descriptor_default_no_approval():
    desc = ActionDescriptor(action_id="display.show", risk=RiskLevel.R1, description="show")

    assert desc.requires_approval is False


def test_action_result_denied_is_not_an_error_status():
    result: ActionResult[None] = ActionResult(status="denied", reason="R2 needs approval")

    assert result.status == "denied"


def test_action_authorization_defaults_unapproved():
    auth = ActionAuthorization(operator="cashier-7", reason="restart adapter")

    assert auth.approved is False


def test_normalize_error_context_defaults_empty():
    err = NormalizeError(code="unsupported_symbology")

    assert err.context == {}


def test_operator_error_holds_machine_code():
    err = OperatorError(code="illegal_transition", detail="payment→item")

    assert err.code == "illegal_transition"
