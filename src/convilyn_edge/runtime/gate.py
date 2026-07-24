"""``ActionGate`` — deterministic R0/R1 auto-grant, R2+ authorization gating.

The ONLY channel a driven workflow may act through is an ``ActionSink``, and
whether an action is *allowed* is a **deterministic** policy decision, never an
LLM one — writing to the outside world without an explicit authorization is a
security boundary, not a UX preference. This gate reads a sink's static
:class:`ActionDescriptor` (its declared risk) BEFORE invoking and applies the
fixed risk ladder:

* ``R0`` / ``R1`` — read-only / reversible low-risk → **auto-granted** (a
  synthesized ``ActionAuthorization``), matching the SDK's "Convilyn ships only
  R0/R1 reference sinks" boundary.
* ``R2`` and above, or any sink whose descriptor sets ``requires_approval`` —
  invoked ONLY when a caller-supplied authorization is present with
  ``approved=True``; otherwise the gate returns ``ActionResult(status="denied")``
  **without invoking the sink** (zero undeclared device actions).

The gate itself never actuates hardware and never authors an SOP — it only
decides *allowed / denied* from the descriptor + authorization. It imports
nothing from the backend; the risk contract is shared by shape.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionResult,
    ActionSink,
    RiskLevel,
)

Input = TypeVar("Input")
Output = TypeVar("Output")

#: The highest risk the gate auto-grants without an explicit human/rule approval.
AUTO_GRANT_MAX = RiskLevel.R1

#: The authorization synthesized for an auto-granted R0/R1 action. Frozen and
#: reused — it records *system* as the operator so an audit trail still shows an
#: authorization existed for every performed action.
_AUTO_GRANT = ActionAuthorization(
    operator="system",
    reason="auto-grant: R0/R1 within policy",
    approved=True,
)


class ActionGate(Generic[Input, Output]):
    """Gate every ``ActionSink.invoke`` behind the deterministic risk ladder."""

    def __init__(self, *, auto_grant_max: RiskLevel = AUTO_GRANT_MAX) -> None:
        self._auto_grant_max = auto_grant_max

    async def invoke(
        self,
        sink: ActionSink[Input, Output],
        input: Input,
        auth: ActionAuthorization | None = None,
    ) -> ActionResult[Output]:
        """Invoke ``sink`` iff policy allows, else return a ``denied`` result.

        For an auto-grantable risk (``<= auto_grant_max`` and no
        ``requires_approval``) a missing ``auth`` is filled with the system
        auto-grant. For anything higher, ``auth`` must be present and
        ``approved`` — otherwise the sink is NOT invoked and a ``denied`` result
        is returned with the reason, so the decision is auditable.
        """
        descriptor = sink.describe()
        auto_grantable = (
            descriptor.risk <= self._auto_grant_max and not descriptor.requires_approval
        )

        if auto_grantable:
            return await sink.invoke(input, auth if auth is not None else _AUTO_GRANT)

        if auth is None or not auth.approved:
            return ActionResult(
                status="denied",
                reason=(
                    f"action {descriptor.action_id!r} at risk {descriptor.risk.name} "
                    "requires an approved authorization"
                ),
            )
        return await sink.invoke(input, auth)


__all__ = ["AUTO_GRANT_MAX", "ActionGate"]
