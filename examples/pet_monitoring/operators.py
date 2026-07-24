"""The pet-monitoring deterministic rule table — per-event anomaly checks, offline-first.

A :class:`~convilyn_edge.spi.operator.DeterministicOperator` maps a sensor event to an anomaly
verdict by a **fixed, pure rule table** — a feeder jam, low water, litter overuse. It is the
offline-first spine: it needs no cloud and **never** calls an LLM (the whole reason
``DeterministicOperator`` is separate from ``ModelOperator``). Scenario knowledge (what "low
water" means for a cat) lives here in the pack, as a rule table keyed by ``event_type`` — never
an ``if scenario ==`` inside the substrate.

The camera event has no deterministic verdict here — deciding "is the cat present, and where?"
from a frame is the grounded :class:`ModelOperator`'s job (a later PR); this table returns "not
anomalous" for it so the rule spine stays deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.result import Ok, Result
from convilyn_edge.spi.operator import ExecutionContext, OperatorError
from examples.pet_monitoring.events import (
    EVENT_FEEDER_READING,
    EVENT_LITTER_READING,
    EVENT_WATER_READING,
)


@dataclass(frozen=True)
class AnomalyVerdict:
    """A deterministic per-event verdict. ``anomalous`` is the branch signal; ``kind`` names the
    anomaly bucket (``feeder_jam`` / ``water_low`` / ``litter_overuse``) for the downstream
    threshold aggregator and notification, ``detail`` is diagnostic."""

    anomalous: bool
    kind: str | None = None
    detail: str | None = None


_NORMAL = AnomalyVerdict(anomalous=False)


def _is_number(value: Any) -> TypeGuard[int | float]:
    """True iff ``value`` is a real numeric reading — excludes ``bool`` (a subclass of ``int``),
    so a malformed boolean in a numeric field is treated as invalid, not coerced to 0/1. A
    ``TypeGuard`` so a passing value narrows to ``int | float`` for the threshold comparison."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class PetThresholds:
    """The pack's tunable rule thresholds (data, not code branches)."""

    feeder_low_pct: int = 10
    water_low_ml: int = 100
    litter_overuse: int = 12


def _check_feeder(data: Mapping[str, Any], t: PetThresholds) -> AnomalyVerdict:
    if data.get("jammed") is True:
        return AnomalyVerdict(True, "feeder_jam", "feeder reports jammed")
    level = data.get("level_pct")
    if _is_number(level) and level < t.feeder_low_pct:
        return AnomalyVerdict(
            True, "feeder_low", f"feeder level {level}% below {t.feeder_low_pct}%"
        )
    return _NORMAL


def _check_water(data: Mapping[str, Any], t: PetThresholds) -> AnomalyVerdict:
    level = data.get("level_ml")
    if _is_number(level) and level < t.water_low_ml:
        return AnomalyVerdict(True, "water_low", f"water {level}ml below {t.water_low_ml}ml")
    return _NORMAL


def _check_litter(data: Mapping[str, Any], t: PetThresholds) -> AnomalyVerdict:
    uses = data.get("uses")
    if _is_number(uses) and uses >= t.litter_overuse:
        return AnomalyVerdict(
            True, "litter_overuse", f"litter used {uses} times (>= {t.litter_overuse})"
        )
    return _NORMAL


#: The rule table: ``event_type`` → a pure check. Extending the pack's coverage is one entry, not
#: a new branch (OCP). An event type with no rule is deterministically "not anomalous".
_RULES: dict[str, Callable[[Mapping[str, Any], PetThresholds], AnomalyVerdict]] = {
    EVENT_FEEDER_READING: _check_feeder,
    EVENT_WATER_READING: _check_water,
    EVENT_LITTER_READING: _check_litter,
}


class PetAnomalyRules:
    """``DeterministicOperator[EventEnvelope, AnomalyVerdict]`` over the pet sensor rule table.

    Reads the triggering event's ``event_type`` + ``data`` and returns a pure verdict — no I/O, no
    LLM, offline-capable. Bind it in the pipeline with ``bind=lambda s: s.envelope``.
    """

    def __init__(self, thresholds: PetThresholds | None = None) -> None:
        self._thresholds = thresholds if thresholds is not None else PetThresholds()

    def execute(
        self, input: EventEnvelope, ctx: ExecutionContext
    ) -> Result[AnomalyVerdict, OperatorError]:
        check = _RULES.get(input.event_type)
        if check is None:
            return Ok(_NORMAL)  # e.g. a camera frame — the model node decides that one
        return Ok(check(input.data, self._thresholds))


__all__ = ["AnomalyVerdict", "PetThresholds", "PetAnomalyRules"]
