"""Runner selection — a deterministic descriptor lookup, never a silicon-equality lookup.

Which concrete on-device runner serves a device is chosen by a **table keyed on
the runtime string** the device reports (the ``InstalledAsset.runtime`` /
:func:`~convilyn_edge.probe.resolve_runtime` value from the frozen device-capability matrix),
NOT by branching on silicon or scenario. Adding a runtime is one registry row
(OCP); the selector never learns a device exists — it reads the runtime as data
(model-provider-portability policy 2/3, error-routing-boundaries).

    from convilyn_edge.probe import resolve_runtime
    from convilyn_edge.runners import RunnerConfig, select_runner

    runtime = resolve_runtime(manifest.silicon)          # deterministic
    runner = select_runner(runtime, RunnerConfig(model="qwen3:4b"))
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from convilyn_edge import probe
from convilyn_edge.runners.openai_compat import OpenAICompatRunner
from convilyn_edge.runners.qnn import QNN_RUNTIME, QnnOnnxRunner

logger = logging.getLogger(__name__)

#: A runner that satisfies the ``ModelOperator`` SPI. Kept as ``Any`` here because
#: the registry is heterogeneous over runner input types; each concrete runner is
#: statically an ``ModelOperator`` at its own call site.
Runner = Any


class UnsupportedRuntimeError(KeyError):
    """Raised by :func:`select_runner` for a runtime with no registered runner."""


class RamFitError(RuntimeError):
    """Raised by :func:`select_runner` under ``strict_fit=True`` when the device's
    RAM is below the config's declared ``min_ram_mb`` — an OOM prevented *before*
    the model loads, never diagnosed after."""


@dataclass(frozen=True)
class RunnerConfig:
    """Construction inputs for a runner — the local model tag + optional endpoint.

    ``base_url`` / ``api_key`` apply to the HTTP-local runners; ``transport`` is an
    injectable stdlib-urllib replacement for tests (never hits the network).
    ``min_ram_mb`` declares the model's minimum device RAM (MiB) — a declarative
    fit field compared generically against the probed device, never a per-device
    branch. ``None`` (the default) skips the fit check entirely.
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    transport: Any = None
    min_ram_mb: int | None = None


@dataclass(frozen=True)
class RamFitReport:
    """Outcome of comparing a config's ``min_ram_mb`` against the device's RAM.

    ``fits=True`` covers both "requirement satisfied" and "no requirement
    declared" (``min_ram_mb=None``); ``message`` is the human-readable line the
    selector logs when the device falls short.
    """

    fits: bool
    min_ram_mb: int | None
    available_ram_mb: int | None
    message: str


def check_ram_fit(
    config: RunnerConfig, *, available_ram_mb: int | None = None
) -> RamFitReport:
    """Compare ``config.min_ram_mb`` against the device's RAM — a pure, generic check.

    ``available_ram_mb=None`` reads the probed physical RAM of the current host
    (the same detection :func:`convilyn_edge.probe.probe_device` uses); pass an
    explicit value to check fit for a *different* device's manifest (``ram_mb``).
    A config with no declared requirement always fits.
    """
    if config.min_ram_mb is None:
        return RamFitReport(
            fits=True,
            min_ram_mb=None,
            available_ram_mb=available_ram_mb,
            message="no min_ram_mb declared; fit check skipped",
        )
    if available_ram_mb is None:
        available_ram_mb = probe._detect_ram_mb()
    if available_ram_mb >= config.min_ram_mb:
        return RamFitReport(
            fits=True,
            min_ram_mb=config.min_ram_mb,
            available_ram_mb=available_ram_mb,
            message=(
                f"model {config.model!r} fits: requires {config.min_ram_mb} MiB, "
                f"device has {available_ram_mb} MiB"
            ),
        )
    return RamFitReport(
        fits=False,
        min_ram_mb=config.min_ram_mb,
        available_ram_mb=available_ram_mb,
        message=(
            f"model {config.model!r} may not fit: requires {config.min_ram_mb} MiB "
            f"but device reports {available_ram_mb} MiB — risk of OOM at load time"
        ),
    )


def _openai_compat_builder(config: RunnerConfig) -> Runner:
    return OpenAICompatRunner.openai_compat(
        model=config.model,
        base_url=config.base_url or "http://localhost:8080/v1",
        api_key=config.api_key,
        transport=config.transport,
    )


def _ollama_builder(config: RunnerConfig) -> Runner:
    return OpenAICompatRunner.ollama(
        model=config.model,
        base_url=config.base_url or "http://localhost:11434",
        api_key=config.api_key,
        transport=config.transport,
    )


def _qnn_builder(config: RunnerConfig) -> Runner:
    return QnnOnnxRunner(model_id=config.model)


#: Runtime token → runner builder. The two llama.cpp runtimes and the OpenAI-compat
#: alias all serve GGUF weights over an OpenAI-compatible HTTP endpoint; ``ollama``
#: is its own wire; ``onnxruntime_qnn`` is the Snapdragon NPU skeleton. A pure dict
#: — extension is one row, never a control-flow branch.
_RUNNER_BUILDERS: dict[str, Callable[[RunnerConfig], Runner]] = {
    "llama_cpp": _openai_compat_builder,
    "llama_cpp_cuda": _openai_compat_builder,
    "openai-compat": _openai_compat_builder,
    "ollama": _ollama_builder,
    QNN_RUNTIME: _qnn_builder,
}

#: The runtimes with a registered runner (for a device ``doctor`` / capability report).
SUPPORTED_RUNTIMES: frozenset[str] = frozenset(_RUNNER_BUILDERS)


def select_runner(
    runtime: str,
    config: RunnerConfig,
    *,
    available_ram_mb: int | None = None,
    strict_fit: bool = False,
) -> Runner:
    """Return the concrete runner for ``runtime`` — a deterministic table lookup.

    Raises :class:`UnsupportedRuntimeError` for an unregistered runtime (loud
    failure — a silent fallback would mask a device the platform cannot yet serve).

    When ``config.min_ram_mb`` is declared, the device's RAM is checked *before*
    the runner is built (:func:`check_ram_fit`). A shortfall logs a warning by
    default — the integrator may know better than the probe — and raises
    :class:`RamFitError` only under ``strict_fit=True``. ``available_ram_mb``
    overrides the probed host RAM (e.g. from a device manifest's ``ram_mb``).
    """
    try:
        builder = _RUNNER_BUILDERS[runtime]
    except KeyError as exc:
        raise UnsupportedRuntimeError(
            f"no runner registered for runtime {runtime!r}; "
            f"known: {', '.join(sorted(_RUNNER_BUILDERS))}"
        ) from exc
    fit = check_ram_fit(config, available_ram_mb=available_ram_mb)
    if not fit.fits:
        if strict_fit:
            raise RamFitError(fit.message)
        logger.warning(fit.message)
    return builder(config)


__all__ = [
    "Runner",
    "RunnerConfig",
    "RamFitError",
    "RamFitReport",
    "UnsupportedRuntimeError",
    "SUPPORTED_RUNTIMES",
    "check_ram_fit",
    "select_runner",
]
