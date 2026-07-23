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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from convilyn_edge.runners.openai_compat import OpenAICompatRunner
from convilyn_edge.runners.qnn import QNN_RUNTIME, QnnOnnxRunner

#: A runner that satisfies the ``ModelOperator`` SPI. Kept as ``Any`` here because
#: the registry is heterogeneous over runner input types; each concrete runner is
#: statically an ``ModelOperator`` at its own call site.
Runner = Any


class UnsupportedRuntimeError(KeyError):
    """Raised by :func:`select_runner` for a runtime with no registered runner."""


@dataclass(frozen=True)
class RunnerConfig:
    """Construction inputs for a runner — the local model tag + optional endpoint.

    ``base_url`` / ``api_key`` apply to the HTTP-local runners; ``transport`` is an
    injectable stdlib-urllib replacement for tests (never hits the network).
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    transport: Any = None


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


def select_runner(runtime: str, config: RunnerConfig) -> Runner:
    """Return the concrete runner for ``runtime`` — a deterministic table lookup.

    Raises :class:`UnsupportedRuntimeError` for an unregistered runtime (loud
    failure — a silent fallback would mask a device the platform cannot yet serve).
    """
    try:
        builder = _RUNNER_BUILDERS[runtime]
    except KeyError as exc:
        raise UnsupportedRuntimeError(
            f"no runner registered for runtime {runtime!r}; "
            f"known: {', '.join(sorted(_RUNNER_BUILDERS))}"
        ) from exc
    return builder(config)


__all__ = [
    "Runner",
    "RunnerConfig",
    "UnsupportedRuntimeError",
    "SUPPORTED_RUNTIMES",
    "select_runner",
]
