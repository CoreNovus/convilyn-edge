"""Primitive 4/7 — ``DeterministicOperator``.

Runs a rule, a format check, a state-machine transition, a debounce, an
aggregation — anything whose output is a pure, reproducible function of its input
and context. In the retail POC this is the ``BarcodeStageRuleEngine`` (its rule
table maps item-entry + payment-code → "go to payment first", etc.).

**This layer must never call an LLM.** That is the whole point of separating it
from ``ModelOperator`` — it is the SDK expression of the principle that
deterministic layers never call a model. Scenario rules live in a *removable* Solution Pack, never in the
substrate, and never as ``if scenario == ...`` branches inside a shared operator.

**Sync signature — a type-level guarantee.** The
original sketch has ``execute`` as ``async``; we tighten it to sync so "no I/O, no
awaiting an external service" is enforced by the type, not by convention. Any
state the rule needs is produced upstream by the (async) ``StateProvider`` and
handed in via ``Input`` / :class:`ExecutionContext`. Generic, so NOT
``@runtime_checkable``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.result import Result

Input = TypeVar("Input", contravariant=True)
# Invariant: ``Output`` is returned inside the invariant ``Result`` container,
# so a covariant declaration would be inconsistent with its use.
Output = TypeVar("Output")


@dataclass(frozen=True)
class OperatorError:
    """Why a deterministic operation failed / rejected its input.

    ``code`` is a stable machine token (e.g. ``"illegal_transition"``,
    ``"debounced"``); ``detail`` is diagnostic.
    """

    code: str
    detail: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    """Deterministic-execution context: the triggering event + a soft deadline.

    ``deadline_ms`` is advisory budget for the runtime's watchdog (resource
    limits) — a deterministic operator is expected to be fast (p95 < 50 ms) and
    never blocks on it.
    """

    envelope: EventEnvelope
    deadline_ms: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DeterministicOperator(Protocol[Input, Output]):
    """A pure, no-LLM, no-I/O transform from ``Input`` to ``Output``."""

    def execute(self, input: Input, ctx: ExecutionContext) -> Result[Output, OperatorError]:
        """Return ``Ok(output)`` or ``Err(OperatorError)``. Must not do I/O or
        call a model — see the module docstring."""
        ...


__all__ = ["Input", "Output", "OperatorError", "ExecutionContext", "DeterministicOperator"]
