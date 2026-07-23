"""The pet-monitoring notify sink — an R1 reference ``ActionSink``.

"Notify me" / "notify sister" is the pack's only side effect, and it is **R1** (reversible,
low-risk): sending a notification. Convilyn ships only R0/R1 *reference* sinks — anything R2+
(actuating a feeder motor, unlocking a door) is integrator/community work behind the gate. A
:class:`NotifySink` has a fixed recipient and records every notification it sends; an integrator
swaps the ``deliver`` callback for a real transport (push / SMS / LINE) without touching the
workflow.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    RiskLevel,
)

#: Collapse anything outside a safe slug set — the recipient becomes part of the policy-gating
#: ``action_id``, so it must be a stable slug, not free text with spaces / dots.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class NotifyRequest:
    """What the workflow asks to send: a titled message (the recipient is the sink's identity)."""

    title: str
    body: str = ""


@dataclass(frozen=True)
class Notification:
    """A notification the sink dispatched — the recipient + the message it delivered."""

    recipient: str
    title: str
    body: str


class NotifySink:
    """An R1 ``ActionSink`` that notifies a fixed ``recipient``.

    ``describe`` reports R1 (auto-granted by the runtime's action gate). ``invoke`` stamps the
    recipient onto the request, records the :class:`Notification`, calls the optional ``deliver``
    transport, and returns it. The recorded :attr:`sent` list makes the sink assertable in tests
    without a real transport (the reference-sink contract).
    """

    def __init__(
        self, recipient: str, *, deliver: Callable[[Notification], None] | None = None
    ) -> None:
        self._recipient = recipient
        self._deliver = deliver
        self.sent: list[Notification] = []

    @property
    def recipient(self) -> str:
        return self._recipient

    def describe(self) -> ActionDescriptor:
        slug = _UNSAFE.sub("_", self._recipient).strip("_") or "recipient"
        return ActionDescriptor(
            action_id=f"pet.notify.{slug}",
            risk=RiskLevel.R1,
            description=f"send a notification to {self._recipient}",
        )

    async def invoke(
        self, input: NotifyRequest, auth: ActionAuthorization
    ) -> ActionResult[Notification]:
        note = Notification(recipient=self._recipient, title=input.title, body=input.body)
        self.sent.append(note)
        if self._deliver is not None:
            self._deliver(note)
        return ActionResult(status="performed", output=note)


__all__ = ["NotifyRequest", "Notification", "NotifySink"]
