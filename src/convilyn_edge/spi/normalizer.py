"""Primitive 2/7 — ``Normalizer``.

Turns a vendor-specific raw payload into a canonical, schema-validated event.
Zebra, Honeywell and OPOS all emit different byte/intent shapes; each vendor's
Normalizer maps its ``Raw`` to the *same* ``Canonical`` (the retail pack's
``BarcodeEvent``), so everything downstream is vendor-agnostic.

**Sync + ``Result``, not async + raise.** Normalization is a pure CPU transform
(parse bytes → struct); it does no I/O, so the signature is synchronous. A bad
payload is an expected, data-driven outcome — returned as ``Err(NormalizeError)``,
never raised — so a caller cannot silently drop the failure path. Generic over
``Raw``/``Canonical``, therefore NOT ``@runtime_checkable`` (an isinstance check
would ignore the type parameters and mislead).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from convilyn_edge.result import Result

Raw = TypeVar("Raw", contravariant=True)
# Invariant: ``Canonical`` is returned inside the invariant ``Result`` container,
# so a covariant declaration would be inconsistent with its use.
Canonical = TypeVar("Canonical")


@dataclass(frozen=True)
class NormalizeError:
    """Why a raw payload could not be normalized.

    ``code`` is a stable machine token (e.g. ``"unsupported_symbology"``,
    ``"malformed_frame"``) a deterministic operator can branch on; ``detail`` is
    diagnostic; ``context`` carries redaction-safe breadcrumbs (never raw PII /
    payment bytes).
    """

    code: str
    detail: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


class Normalizer(Protocol[Raw, Canonical]):
    """Map a vendor ``Raw`` payload to a canonical event (or an error value)."""

    def normalize(self, raw: Raw) -> Result[Canonical, NormalizeError]:
        """Return ``Ok(canonical)`` on success, ``Err(NormalizeError)`` on a
        rejected/unparseable payload."""
        ...


__all__ = ["Raw", "Canonical", "NormalizeError", "Normalizer"]
