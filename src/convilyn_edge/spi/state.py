"""Primitive 3/7 — ``StateProvider``.

Supplies the *environment state* at the moment an event fires — the context a
deterministic rule needs but the event itself does not carry. In the retail POC
that is ``PosState`` (stage: item_entry / payment / …, recent scans, whether the
product-lookup service is up). Swap the implementation and the same primitive
serves factory machine state, vehicle telemetry, an access-control system's
state — the whole generalization thesis of this primitive.

**Async, generic.** ``snapshot`` may read a POS bus, a local store, or a socket,
so it is ``async``. Generic over the state type ``T`` and therefore NOT
``@runtime_checkable``. This is the ONLY sanctioned way a deterministic operator
obtains state — the operator itself stays sync and I/O-free (see ``operator.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from convilyn_edge.envelope import EventEnvelope

T = TypeVar("T", covariant=True)


@dataclass(frozen=True)
class EventContext:
    """The event a snapshot is being taken for, plus adapter-specific extras.

    Carries the triggering :class:`EventEnvelope` so a provider can scope its
    snapshot (e.g. "state for this transaction/session") from
    ``envelope.correlation``.
    """

    envelope: EventEnvelope
    metadata: Mapping[str, Any] = field(default_factory=dict)


class StateProvider(Protocol[T]):
    """Produce a snapshot of environment state ``T`` for a given event."""

    async def snapshot(self, ctx: EventContext) -> T:
        """Return the current environment state relevant to ``ctx``'s event."""
        ...


__all__ = ["T", "EventContext", "StateProvider"]
