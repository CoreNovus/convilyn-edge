"""``QnnOnnxRunner`` — Protocol-conformant SKELETON for the Snapdragon NPU runtime.

The runtime the frozen device matrix binds ``snapdragon_x`` to is ONNX Runtime with
the **QNN Execution Provider** (``onnxruntime_qnn``) driving the Hexagon NPU. The
platform stages no Snapdragon ONNX LLM build yet and CI has no Snapdragon hardware,
so this ships **hardware-deferred**: a ``ModelOperator`` skeleton whose ``infer``
**fails loud** as ``status="unavailable"`` — it NEVER raises and NEVER fabricates a
result. An integrator on real hardware fills :meth:`_execute` with a real QNN
inference session (the concrete override then flows through unchanged — LSP).

:meth:`available` reports whether the QNN EP is actually present, so a device
``doctor`` can distinguish "no NPU runtime on this host" from "runtime present,
execution not yet wired". Either way ``infer`` yields ``unavailable`` in this
skeleton — the honest signal for a runner that cannot yet produce a grounded value.

No ``onnxruntime`` import at module load (it is not a dependency) — availability is
probed lazily and any absence degrades to ``unavailable``. Runner selection is a
deterministic descriptor lookup (see :mod:`convilyn_edge.runners.registry`), never
a silicon-equality branch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from convilyn_edge.clientcompute.operator import ExtractInput
from convilyn_edge.spi.model import ModelResult, Placement

#: The runtime token this runner serves (matches the matrix / backend ladder).
QNN_RUNTIME = "onnxruntime_qnn"

#: A QNN-availability probe: ``() -> bool``. The default checks ONNX Runtime's
#: available providers; tests inject a fake to exercise both branches.
QnnProbe = Callable[[], bool]


class QnnUnavailableError(RuntimeError):
    """Raised internally when the QNN Execution Provider is absent; ``infer``
    catches it and degrades to ``status="unavailable"`` (never propagates)."""


def _qnn_ep_available() -> bool:
    """True iff ONNX Runtime is importable AND exposes the QNN Execution Provider.

    ``onnxruntime`` is not a dependency of this zero-dep SDK, so an ``ImportError``
    (or any probe failure) means "no QNN here" → ``False``. Never raises.
    """
    try:
        import onnxruntime  # noqa: PLC0415 — lazy, optional, not a runtime dep

        return "QNNExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001 — any failure means the EP is not usable here
        return False


class QnnOnnxRunner:
    """A ``ModelOperator`` skeleton for ONNX Runtime + QNN EP (Snapdragon Hexagon)."""

    def __init__(
        self,
        *,
        model_id: str = "local-qnn-onnx",
        model_version: str = "0",
        provider_probe: QnnProbe = _qnn_ep_available,
    ) -> None:
        self._model_id = model_id
        self._model_version = model_version
        self._provider_probe = provider_probe

    def available(self) -> bool:
        """Whether the QNN Execution Provider is present on this host (diagnostic)."""
        return self._provider_probe()

    async def infer(
        self,
        input: ExtractInput,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "edge",
    ) -> ModelResult[dict[str, str]]:
        """Fail loud as ``unavailable`` — never raise, never fabricate a result.

        Returns ``status="unavailable"`` when the QNN EP is absent (the CI/no-NPU
        reality) or when execution is not yet wired (this skeleton). A real
        on-device override of :meth:`_execute` returns its grounded ``ModelResult``
        straight through.
        """
        started = time.perf_counter()
        try:
            if not self.available():
                raise QnnUnavailableError("QNN Execution Provider not present on this host")
            return await self._execute(input, schema=schema, deadline_ms=deadline_ms)
        except Exception:  # noqa: BLE001 — fail loud AS unavailable; never propagate
            return ModelResult(
                status="unavailable",
                model_id=self._model_id,
                model_version=self._model_version,
                latency_ms=(time.perf_counter() - started) * 1000,
                output=None,
                confidence=None,
                evidence=(),
            )

    async def _execute(
        self,
        input: ExtractInput,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None,
    ) -> ModelResult[dict[str, str]]:
        """The hardware-specific inference — deferred. An integrator overrides this
        on real Snapdragon hardware to run the QNN session and return a grounded
        ``ModelResult``. The skeleton raises so the deferral is explicit in code."""
        raise NotImplementedError(
            "QNN inference execution is hardware-deferred; override _execute() on-device."
        )


__all__ = ["QNN_RUNTIME", "QnnProbe", "QnnUnavailableError", "QnnOnnxRunner"]
