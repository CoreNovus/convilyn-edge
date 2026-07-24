"""On-device model runners — concrete ``ModelOperator`` bindings, selected by table.

A *runner* is the concrete on-device execution binding for one local runtime,
implementing the existing ``ModelOperator`` SPI (no new Protocol is introduced —
runners implement the keystone). Which runner serves a device is a **deterministic
descriptor lookup** on the reported runtime string
(:func:`~convilyn_edge.probe.resolve_runtime` → :func:`select_runner`), never an
silicon-equality branch.

    from convilyn_edge.probe import probe_device, resolve_runtime
    from convilyn_edge.runners import RunnerConfig, select_runner

    manifest = probe_device(device_id="reg-07")
    runner = select_runner(resolve_runtime(manifest.silicon), RunnerConfig(model="qwen3:4b"))
    result = await runner.infer(extract_input, schema=schema)   # a grounded ModelResult

* :class:`OpenAICompatRunner` — the productized HTTP-local runner (Ollama +
  OpenAI-compatible servers), zero-dependency.
* :class:`QnnOnnxRunner` — the Snapdragon NPU (ONNX Runtime + QNN EP) runtime, a
  hardware-deferred fail-loud skeleton (``status="unavailable"``, never fabricates).
"""

from __future__ import annotations

from convilyn_edge.runners.grounding import failing_fields, failing_fields_count
from convilyn_edge.runners.openai_compat import OpenAICompatRunner
from convilyn_edge.runners.qnn import (
    QNN_RUNTIME,
    QnnOnnxRunner,
    QnnProbe,
    QnnUnavailableError,
)
from convilyn_edge.runners.registry import (
    SUPPORTED_RUNTIMES,
    RamFitError,
    RamFitReport,
    RunnerConfig,
    UnsupportedRuntimeError,
    check_ram_fit,
    select_runner,
)
from convilyn_edge.warmup import WarmupResult, WarmupState, warmup_runner

__all__ = [
    # runners
    "OpenAICompatRunner",
    "QnnOnnxRunner",
    "QNN_RUNTIME",
    "QnnProbe",
    "QnnUnavailableError",
    # selection
    "select_runner",
    "RunnerConfig",
    "SUPPORTED_RUNTIMES",
    "UnsupportedRuntimeError",
    # device-RAM fit guard
    "check_ram_fit",
    "RamFitError",
    "RamFitReport",
    # warmup (cold ≠ offline)
    "warmup_runner",
    "WarmupResult",
    "WarmupState",
    # grounding introspection
    "failing_fields",
    "failing_fields_count",
]
