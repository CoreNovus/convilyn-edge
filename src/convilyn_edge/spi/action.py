"""Primitive 7/7 — ``ActionSink``.

The ONLY channel through which a workflow may affect the outside world. Every
action is described (``describe()``) with a risk level and invoked
(``invoke()``) with an explicit :class:`ActionAuthorization`. A workflow can
**never** reach a raw shell, USB, or socket — capability + policy gate every
side effect (a workflow never controls the OS or hardware directly).

**Risk ladder.** ``RiskLevel`` orders actions so the runtime's
policy engine — deterministic, never an LLM — can gate them:

* ``R0`` read-only (read state, look up SOP) — auto.
* ``R1`` reversible low-risk (show a message, beep) — auto.
* ``R2`` operational impact (restart an adapter, pause a device) — rule or human
  approval.
* ``R3`` money / safety / person (payment, unlock, shutdown) — forced
  authorization **and** human confirmation.

**Supported boundary (locked decision).** Convilyn ships only R0/R1 *reference*
sinks (e.g. a display-an-instruction or create-a-help-request sink in a
Solution Pack).
Everything R2+ — and all real hardware actuation — is an integrator/community
adapter, never shipped by Convilyn. "denied" is a first-class outcome, not an
error, so ``ActionResult`` is its own shape, not a
``Result[T, E]``.

Generic over ``Input``/``Output``, so NOT ``@runtime_checkable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, Literal, Protocol, TypeVar

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output")

ActionStatus = Literal["performed", "denied", "failed"]


class RiskLevel(IntEnum):
    """Ordered action risk. Ordering lets a policy engine gate
    with ``risk >= RiskLevel.R2`` rather than enumerating cases."""

    R0 = 0  # read-only            → auto
    R1 = 1  # reversible low-risk  → auto
    R2 = 2  # operational impact   → rule or human approval
    R3 = 3  # money / safety       → forced authorization + human confirm


@dataclass(frozen=True)
class ActionDescriptor:
    """Static, introspectable description of what a sink does and its risk.

    A runtime reads this *before* invoking so it can enforce policy (deny an
    R2+ action lacking approval) without executing anything (zero undeclared
    device actions).
    """

    action_id: str
    risk: RiskLevel
    description: str
    requires_approval: bool = False


@dataclass(frozen=True)
class ActionAuthorization:
    """The authorization presented when invoking an action.

    For R0/R1 this is typically the auto grant. For R2+ ``approved`` must be
    True with an ``operator`` and ``reason`` on record (every action carries an
    operator, a policy, and a reason). The sink/runtime rejects an under-authorized
    call with ``ActionResult(status="denied")`` rather than performing it.
    """

    operator: str
    reason: str
    approved: bool = False
    granted_at: str | None = None


@dataclass(frozen=True)
class ActionResult(Generic[Output]):
    """Outcome of an action invocation.

    ``performed`` — the action ran; ``output`` may carry its result.
    ``denied``    — policy/authorization refused it (NOT an error).
    ``failed``    — it was attempted but errored; ``reason`` explains.
    """

    status: ActionStatus
    output: Output | None = None
    reason: str | None = None


class ActionSink(Protocol[Input, Output]):
    """A gated, described side effect — the only way a workflow acts."""

    def describe(self) -> ActionDescriptor:
        """Return this sink's static descriptor (risk, approval need). Sync —
        pure metadata, read before any invocation."""
        ...

    async def invoke(self, input: Input, auth: ActionAuthorization) -> ActionResult[Output]:
        """Perform the action under ``auth``, or return
        ``ActionResult(status="denied")`` if authorization is insufficient for
        the descriptor's risk level."""
        ...


__all__ = [
    "Input",
    "Output",
    "ActionStatus",
    "RiskLevel",
    "ActionDescriptor",
    "ActionAuthorization",
    "ActionResult",
    "ActionSink",
]
