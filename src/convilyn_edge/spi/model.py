"""Primitive 5/7 — ``ModelOperator`` — the KEYSTONE.

The model is an *explicit, typed workflow node*, never a hidden agent inside the
SDK. It takes a structured input + an output JSON Schema and returns a
:class:`ModelResult` — a validated, evidence-carrying, status-tagged value —
**never a bare string of natural language**.

**One Protocol, two placements.** ``placement`` selects where inference runs:

* ``"cloud"`` — the implementation wraps the consumer SDK's cloud workflow
  (``client.goals``): the server enforces every safety check and
  re-grounds every value the model returns.
* ``"edge"`` — the implementation runs inference device-local via the
  ``client_compute`` extension point; the server still re-grounds every returned
  value before trusting it (the device is never a second source of truth).
  Productized in a later release.
* ``"auto"`` — a deterministic resolver (a table/dict keyed by role × tier ×
  device-capability) picks one. **Never** an ``if provider == ...`` branch, never
  an LLM deciding placement.

The two implementations are substitutable behind this Protocol (LSP); the
placement decision is data, not control flow baked into the Protocol.

Generic over ``Input``/``Output``, so NOT ``@runtime_checkable``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output")

Placement = Literal["edge", "cloud", "auto"]
ModelStatus = Literal["success", "uncertain", "unavailable", "invalid"]


@dataclass(frozen=True)
class Evidence:
    """A grounding citation for a model output.

    Mirrors the anchor-grounding contract (``SubstringAnchorVerifier``): a value
    the model asserts should be traceable to ``source`` text at ``offset``. Lets
    the server (and an auditor) re-ground what the device produced instead of
    trusting it — the device is never a second source of truth.
    """

    source: str
    snippet: str | None = None
    offset: int | None = None


@dataclass(frozen=True)
class ModelResult(Generic[Output]):
    """A typed model outcome.

    A model outcome is **richer than success/failure**, so this is deliberately
    NOT a ``Result[T, E]``. ``status`` distinguishes:

    * ``success``   — ``output`` is present and schema-valid.
    * ``uncertain`` — produced, but below the confidence/grounding bar; a caller
      may fall back to a fixed message (fallback) rather than trust
      it.
    * ``unavailable`` — the model could not be reached / timed out; the workflow
      takes its fixed fallback path (offline-first).
    * ``invalid``   — output failed schema validation / re-grounding; discarded.

    ``output`` is populated only for ``success`` (and possibly ``uncertain``).
    """

    status: ModelStatus
    model_id: str
    model_version: str
    latency_ms: float
    output: Output | None = None
    confidence: float | None = None
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)


class ModelOperator(Protocol[Input, Output]):
    """Run typed, schema-constrained inference at a chosen placement."""

    async def infer(
        self,
        input: Input,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "auto",
    ) -> ModelResult[Output]:
        """Return a typed :class:`ModelResult` — never raw natural language.

        ``schema`` is the JSON Schema the output must validate against.
        ``deadline_ms`` bounds the call (workflow deadlines);
        exceeding it yields ``status="unavailable"`` so the workflow can fall
        back, not raise. ``placement`` is resolved deterministically (see the
        module docstring)."""
        ...


__all__ = [
    "Input",
    "Output",
    "Placement",
    "ModelStatus",
    "Evidence",
    "ModelResult",
    "ModelOperator",
]
