"""Primitive 6/7 — ``HumanReview``.

A structured human-in-the-loop request: a titled prompt, a fixed set of
``choices``, and a ``default_action`` for when nobody answers. In the retail POC
this is the "suspected duplicate scan — [only one] / [there are two] / [escalate]"
card. The response is a *typed* outcome, never free-form text
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
    """

    decision: ReviewDecision
    choice_id: str | None = None
    note: str | None = None
    responded_at: str | None = None


@runtime_checkable
class HumanReview(Protocol):
    """Ask a human to decide, and return their typed choice."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        """Present ``req`` to a human and resolve to a typed :class:`ReviewOutcome`."""
        ...


__all__ = [
    "DefaultAction",
    "ReviewDecision",
    "ReviewChoice",
    "ReviewRequest",
    "ReviewOutcome",
    "HumanReview",
]
