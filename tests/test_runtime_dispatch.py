"""Unit tests for deterministic status dispatch (model + review)."""

from __future__ import annotations

from convilyn_edge.runtime.dispatch import model_disposition, review_disposition
from convilyn_edge.spi.model import ModelResult
from convilyn_edge.spi.review import ReviewOutcome


def _result(status: str, *, output: object | None = None) -> ModelResult[object]:
    return ModelResult(
        status=status,  # type: ignore[arg-type]
        model_id="m",
        model_version="1",
        latency_ms=1.0,
        output=output,
    )


# ── model_disposition: logic ─────────────────────────────────────────────────


def test_success_accepts():
    assert model_disposition(_result("success", output={"k": "v"})) == "accept"


def test_unavailable_falls_back():
    assert model_disposition(_result("unavailable")) == "fallback"


def test_invalid_falls_back():
    assert model_disposition(_result("invalid")) == "fallback"


def test_uncertain_falls_back_by_default():
    assert model_disposition(_result("uncertain", output={"k": "v"})) == "fallback"


# ── model_disposition: boundary (accept_uncertain opt-in) ────────────────────


def test_uncertain_accepts_when_opted_in_with_output():
    disposition = model_disposition(_result("uncertain", output={"k": "v"}), accept_uncertain=True)

    assert disposition == "accept"


def test_uncertain_still_falls_back_when_output_missing():
    # Opted in, but no output to trust → cannot accept.
    disposition = model_disposition(_result("uncertain", output=None), accept_uncertain=True)

    assert disposition == "fallback"


# ── review_disposition: logic ────────────────────────────────────────────────


def test_selected_passes_through():
    outcome = ReviewOutcome(decision="selected", choice_id="c1")

    assert review_disposition(outcome) == "selected"


def test_default_action_stop_passes_through():
    outcome = ReviewOutcome(decision="stop")

    assert review_disposition(outcome) == "stop"


def test_escalate_passes_through():
    outcome = ReviewOutcome(decision="escalate")

    assert review_disposition(outcome) == "escalate"
