"""Warmup — make a cold start *distinguishable* from an offline server.

``health()`` answers "is the local inference server reachable?"; it says nothing
about whether the model weights are loaded. The first inference after boot can
take tens of seconds on an edge device while the runtime loads weights — a
latency an integrator must be able to surface ("warming up…") instead of
misreading as an outage. ``warmup`` is that assertable interface: a
:class:`WarmupResult` with exactly three states:

* ``warm``         — the probe inference returned within the warm threshold; the
  model is loaded and the first real request will be fast.
* ``cold_started`` — the probe ran (or timed out *while the server was up*) and
  ``latency_ms`` carries the observed cost; the warmup call itself paid the
  cold-start so the next request won't.
* ``unreachable``  — the server cannot be reached at all. Never conflated with
  ``cold_started``: offline is a different UX (and fix) than loading.

Runners opt in by exposing ``warmup(deadline_ms=None) -> WarmupResult``;
:func:`warmup_runner` gives every runner *without* the hook a no-op
already-warm default, so existing runners need zero changes.

Stdlib only — this module imports nothing from the rest of the package, so both
the runner layer and the client-compute engine can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

WarmupState = Literal["warm", "cold_started", "unreachable"]

#: A probe inference returning within this many milliseconds means the model was
#: already loaded ("warm"); anything slower is classified ``cold_started``.
DEFAULT_WARM_THRESHOLD_MS = 1000.0


@dataclass(frozen=True)
class WarmupResult:
    """Outcome of a warmup probe — the three-state cold≠offline contract.

    ``latency_ms`` is populated whenever the probe actually ran (``warm`` /
    ``cold_started``); ``detail`` carries the human-actionable line for
    ``unreachable`` and deadline cases.
    """

    state: WarmupState
    latency_ms: float | None = None
    detail: str | None = None


def warmup_runner(runner: Any, deadline_ms: int | None = None) -> WarmupResult:
    """Warm ``runner`` up if it supports warmup; otherwise report already-warm.

    The generic entry point over a heterogeneous runner registry: a runner that
    exposes ``warmup(deadline_ms)`` is delegated to; one that doesn't (e.g. an
    in-process runner with no load cost) gets the no-op default — so adding the
    hook to a runner is opt-in and existing runners keep working unchanged.
    """
    hook = getattr(runner, "warmup", None)
    if hook is None:
        return WarmupResult(
            state="warm",
            latency_ms=0.0,
            detail="runner declares no warmup hook; treated as already warm",
        )
    return hook(deadline_ms)


__all__ = [
    "DEFAULT_WARM_THRESHOLD_MS",
    "WarmupState",
    "WarmupResult",
    "warmup_runner",
]
