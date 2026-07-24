"""Primitive 6/7 — ``HumanReview``.

A structured human-in-the-loop request: a titled prompt, a fixed set of
``choices``, and a ``default_action`` for when nobody answers — e.g. a
"suspected duplicate event — [keep one] / [keep both] / [escalate]" card.
The response is a *typed* outcome, never free-form text
the workflow has to re-parse.

A real implementation is the device-side face of the Convilyn cloud's existing
human-in-the-loop surface (the consumer SDK's ``goals.fill_slots`` /
``goals.confirm``) — not a reimplementation.

Non-generic + method-only, so ``@runtime_checkable`` is honest here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

DefaultAction = Literal["stop", "escalate", "continue"]
ReviewDecision = Literal["stop", "escalate", "continue", "selected"]

#: Where a resolved decision came from. ``human`` — a person actually answered. ``expiry_default``
#: — nobody answered by ``expires_at`` and the runtime synthesized the request's
#: ``default_action`` (see ``runtime.watchdog.resolve_review``). This is a **structured** signal,
#: not a note substring: a pack that gates a downstream action on human approval MUST require
#: ``decision_source == "human"`` (or ``decision == "selected"``), never merely "the review did
#: not stop" — a timeout must never be mistaken for a human grant (error-routing zone 7).
DecisionSource = Literal["human", "expiry_default"]


@dataclass(frozen=True)
class ReviewChoice:
    """One selectable option in a review request."""

    id: str
    label: str


@dataclass(frozen=True)
class ReviewRequest:
    """A human decision the workflow is blocking on.

    ``default_action`` is what the runtime does if the request expires
    (``expires_at``, UTC ISO-8601) with no answer — the fail-safe / fail-open
    choice, which must be workflow policy, not hardcoded.
    """

    title: str
    message: str
    choices: Sequence[ReviewChoice] = field(default_factory=tuple)
    default_action: DefaultAction = "stop"
    expires_at: str | None = None


@dataclass(frozen=True)
class ReviewOutcome:
    """The resolved outcome of a review request.

    ``decision`` is ``"selected"`` when the operator picked a choice (``choice_id``
    set), otherwise the ``default_action`` that fired on expiry / dismissal.
    ``decision_source`` discriminates a real human answer (``"human"``, the default)
    from a runtime-synthesized timeout default (``"expiry_default"``) — a structured
    field an approval gate must consult rather than parsing ``note``.
    """

    decision: ReviewDecision
    choice_id: str | None = None
    note: str | None = None
    responded_at: str | None = None
    decision_source: DecisionSource = "human"


@runtime_checkable
class HumanReview(Protocol):
    """Ask a human to decide, and return their typed choice."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        """Present ``req`` to a human and resolve to a typed :class:`ReviewOutcome`."""
        ...


__all__ = [
    "DefaultAction",
    "ReviewDecision",
    "DecisionSource",
    "ReviewChoice",
    "ReviewRequest",
    "ReviewOutcome",
    "HumanReview",
]
