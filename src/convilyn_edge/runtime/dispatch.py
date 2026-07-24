"""Deterministic status dispatch for the two richer primitives.

A :class:`ModelResult` and a :class:`ReviewOutcome` carry a *status/decision*
that the runtime must route on — but that routing is a fixed table, never an LLM
re-interpreting the status. These are the edge runtime's ``ModelResult.status``
and ``HumanReview`` dispatch mechanisms: pure functions from a typed result to a
coarse **disposition** the driver/pack branches on. The classifier lives here
(generic, deterministic); *what each branch does* is scenario orchestration and
stays in the Solution Pack.
"""

from __future__ import annotations

from typing import Literal

from convilyn_edge.spi.model import ModelResult
from convilyn_edge.spi.review import ReviewOutcome

#: ``accept`` — trust the model output. ``fallback`` — take the workflow's fixed
#: offline/uncertain path (the model was unavailable, invalid, or below bar).
ModelDisposition = Literal["accept", "fallback"]

#: Mirrors :data:`~convilyn_edge.spi.review.ReviewDecision` — the resolved human
#: action the workflow proceeds under.
ReviewDisposition = Literal["continue", "selected", "stop", "escalate"]


def model_disposition(
    result: ModelResult[object],
    *,
    accept_uncertain: bool = False,
) -> ModelDisposition:
    """Route a :class:`ModelResult` to ``accept`` or ``fallback`` — deterministic.

    ``success`` always accepts. ``uncertain`` accepts ONLY when the caller opted
    in (``accept_uncertain=True``) AND an ``output`` is actually present — an
    uncertain result with no output cannot be trusted. ``unavailable`` and
    ``invalid`` always fall back (offline-first). No status else exists, so the
    mapping is total.
    """
    if result.status == "success":
        return "accept"
    if result.status == "uncertain" and accept_uncertain and result.output is not None:
        return "accept"
    return "fallback"


def review_disposition(outcome: ReviewOutcome) -> ReviewDisposition:
    """Normalize a :class:`ReviewOutcome` to the action the workflow proceeds under.

    A ``selected`` decision surfaces as ``selected`` (the pack reads
    ``choice_id``); ``continue`` / ``stop`` / ``escalate`` — including the
    ``default_action`` that fired on expiry — pass through unchanged. Deterministic
    and total over :data:`ReviewDecision`.
    """
    return outcome.decision


__all__ = [
    "ModelDisposition",
    "ReviewDisposition",
    "model_disposition",
    "review_disposition",
]
