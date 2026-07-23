"""Grounding introspection — which required anchors a runner failed to ground.

A grounded extractor result degrades any anchor it could not verify against the
device's own source text to the *missing sentinel*. ``failing_fields`` surfaces
exactly those keys so a caller (or a test) can assert extraction quality —
``failing_fields_count == 0`` means every required field was grounded verbatim.
Pure functions over a :class:`ModelResult`; no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence

from convilyn_edge.clientcompute.contract import MISSING_SENTINEL
from convilyn_edge.spi.model import ModelResult


def failing_fields(
    result: ModelResult[dict[str, str]],
    required_anchors: Sequence[str],
    *,
    sentinel: str = MISSING_SENTINEL,
) -> tuple[str, ...]:
    """The required anchors the runner did NOT ground (missing/sentinel/absent).

    A ``None`` output (an ``unavailable`` result) means *every* required anchor
    failed. Otherwise a key fails when its value is the ``sentinel`` or absent.
    """
    output = result.output
    if output is None:
        return tuple(required_anchors)
    return tuple(key for key in required_anchors if output.get(key, sentinel) == sentinel)


def failing_fields_count(
    result: ModelResult[dict[str, str]],
    required_anchors: Sequence[str],
    *,
    sentinel: str = MISSING_SENTINEL,
) -> int:
    """How many required anchors the runner failed to ground (0 = all grounded)."""
    return len(failing_fields(result, required_anchors, sentinel=sentinel))


__all__ = ["failing_fields", "failing_fields_count"]
