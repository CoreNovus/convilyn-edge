"""The Device Event Envelope — one shell for every IoT event.

Every event that crosses the SDK — a barcode scan, a device-health change, a POS
error — travels inside the *same* envelope. Uniform identity, schema-versioning,
correlation and ordering are what make dedup, replay, audit, schema-migration and
cross-device correlation possible (replay is a first-class feature).

Data-flow consistency (a binding principle): there is exactly ONE envelope
shape. The per-scenario typed payload lives in ``EventEnvelope.data`` and is
validated by a ``Normalizer``'s ``Canonical`` type downstream — the envelope is
never parameterized per scenario. Any vertical Solution Pack — factory, medical,
access-control — rides this same shell, unchanged.

``Correlation.trace_id`` is a W3C-canonical 32-hex string taken from the active
span (the same observability convention the Convilyn cloud uses) *by shape only*
— this package imports nothing from the cloud; the convention is shared, the code
is not.

Immutability: ``data`` and ``metadata`` are **deep-copied** on construction and
exposed as read-only ``MappingProxyType`` views, so a caller mutating the
structure they passed in — at *any* nesting depth — can never retroactively
change a minted envelope. Envelopes are auditable, replayable records, so a
shallow copy would not be enough for nested JSON payloads. Treat
the exposed mappings as read-only. Note: because an envelope carries arbitrary
mapping payloads it is *not hashable* despite ``frozen=True`` — it cannot be a set
member or dict key. ``to_wire()`` re-materialises plain dicts for JSON.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

SPEC_VERSION = "1.0"


def _utc_now_iso() -> str:
    """UTC ISO-8601 timestamp, e.g. ``2026-07-18T10:15:32.240000+00:00``.

    ``timezone.utc`` (not ``datetime.UTC``, which is 3.11+) keeps the >=3.10
    floor. This is the SINGLE clock read in the module — every other function is
    a pure transform, so replay stays deterministic when ``time`` is
    pinned via direct construction / ``from_wire``.
    """
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EventSourceRef:
    """Identifies the device + adapter that produced an event."""

    device_id: str
    adapter_id: str
    adapter_version: str

    def to_wire(self) -> dict[str, str]:
        return {
            "deviceId": self.device_id,
            "adapterId": self.adapter_id,
            "adapterVersion": self.adapter_version,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> EventSourceRef:
        return cls(
            device_id=wire["deviceId"],
            adapter_id=wire["adapterId"],
            adapter_version=wire["adapterVersion"],
        )


@dataclass(frozen=True)
class Correlation:
    """Cross-event linkage: distributed trace, logical session, ordering.

    ``sequence`` + ``trace_id`` are what let a consumer dedup replays and
    reconstruct the per-transaction event chain. All fields are
    optional — a source that cannot supply a trace id still produces a valid
    envelope.
    """

    trace_id: str | None = None
    session_id: str | None = None
    sequence: int | None = None

    def to_wire(self) -> dict[str, Any]:
        """Emit only the fields that are set (a lean wire object for edge)."""
        wire: dict[str, Any] = {}
        if self.trace_id is not None:
            wire["traceId"] = self.trace_id
        if self.session_id is not None:
            wire["sessionId"] = self.session_id
        if self.sequence is not None:
            wire["sequence"] = self.sequence
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any] | None) -> Correlation:
        wire = wire or {}
        return cls(
            trace_id=wire.get("traceId"),
            session_id=wire.get("sessionId"),
            sequence=wire.get("sequence"),
        )


@dataclass(frozen=True)
class EventEnvelope:
    """The universal event shell.

    Construct directly (tests / replay pin ``event_id`` + ``time``) or via
    :func:`new_envelope` (mints both). ``data`` / ``metadata`` are read-only
    views over deep copies — see the module docstring.
    """

    event_id: str
    event_type: str
    event_schema: str
    source: EventSourceRef
    time: str
    data: Mapping[str, Any]
    spec_version: str = SPEC_VERSION
    correlation: Correlation = field(default_factory=Correlation)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Deep-copy + freeze the two open maps so nested payloads can't be mutated
        # through the caller's original reference. object.__setattr__ is the
        # sanctioned way to assign on a frozen dataclass during init.
        object.__setattr__(self, "data", MappingProxyType(copy.deepcopy(dict(self.data))))
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the camelCase wire object."""
        return {
            "specVersion": self.spec_version,
            "eventId": self.event_id,
            "eventType": self.event_type,
            "eventSchema": self.event_schema,
            "source": self.source.to_wire(),
            "time": self.time,
            "correlation": self.correlation.to_wire(),
            "data": dict(self.data),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> EventEnvelope:
        """Reconstruct from a wire object. Inverse of :meth:`to_wire`."""
        return cls(
            spec_version=wire.get("specVersion", SPEC_VERSION),
            event_id=wire["eventId"],
            event_type=wire["eventType"],
            event_schema=wire["eventSchema"],
            source=EventSourceRef.from_wire(wire["source"]),
            time=wire["time"],
            correlation=Correlation.from_wire(wire.get("correlation")),
            data=wire.get("data", {}),
            metadata=wire.get("metadata", {}),
        )


def new_envelope(
    *,
    event_type: str,
    event_schema: str,
    source: EventSourceRef,
    data: Mapping[str, Any],
    correlation: Correlation | None = None,
    metadata: Mapping[str, Any] | None = None,
    spec_version: str = SPEC_VERSION,
) -> EventEnvelope:
    """Mint a new envelope with a fresh ``event_id`` (uuid4 hex) + ``time``.

    The ONE place identity/time are minted (keyword-only to prevent
    positional-argument drift as the field set grows). Direct
    ``EventEnvelope(...)`` construction stays available for replay, where both
    must be pinned rather than regenerated.
    """
    return EventEnvelope(
        event_id=uuid.uuid4().hex,
        event_type=event_type,
        event_schema=event_schema,
        source=source,
        time=_utc_now_iso(),
        data=data,
        spec_version=spec_version,
        correlation=correlation if correlation is not None else Correlation(),
        metadata=metadata if metadata is not None else {},
    )


__all__ = [
    "SPEC_VERSION",
    "EventSourceRef",
    "Correlation",
    "EventEnvelope",
    "new_envelope",
]
