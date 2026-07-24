"""Unit tests for QnnOnnxRunner — the hardware-deferred fail-loud NPU skeleton."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from convilyn_edge.clientcompute.operator import ExtractInput
from convilyn_edge.runners import QnnOnnxRunner, failing_fields_count
from convilyn_edge.spi.model import ModelResult


def _input() -> ExtractInput:
    return ExtractInput(prompt="p", sources={"s": "text"}, required_anchors=["a"])


# ── logic: no QNN → unavailable, never raises, never fabricates ──────────────


async def test_infer_unavailable_without_qnn():
    runner = QnnOnnxRunner(provider_probe=lambda: False)

    result = await runner.infer(_input(), schema={})

    assert result.status == "unavailable"


async def test_infer_does_not_fabricate_output():
    runner = QnnOnnxRunner(provider_probe=lambda: False)

    result = await runner.infer(_input(), schema={})

    assert result.output is None  # no invented values


async def test_infer_never_raises_when_ep_absent():
    runner = QnnOnnxRunner(provider_probe=lambda: False)

    # Must not raise — the whole point of the fail-loud-as-unavailable contract.
    result = await runner.infer(_input(), schema={})

    assert result.model_id == "local-qnn-onnx"


async def test_default_probe_reports_no_ep_in_ci():
    # onnxruntime is not a dependency, so the real probe returns False here.
    assert QnnOnnxRunner().available() is False


async def test_unavailable_result_fails_all_required_fields():
    # An unavailable result (output=None) means every required anchor failed.
    runner = QnnOnnxRunner(provider_probe=lambda: False)

    result = await runner.infer(_input(), schema={})

    assert failing_fields_count(result, ["a", "b"]) == 2


# ── boundary: EP present but execution deferred → still unavailable, no raise ─


async def test_ep_present_but_unwired_is_unavailable():
    runner = QnnOnnxRunner(provider_probe=lambda: True)

    # available() is True, but _execute() is hardware-deferred (raises internally,
    # caught → unavailable). The skeleton never fabricates.
    result = await runner.infer(_input(), schema={})

    assert result.status == "unavailable"


# ── object-state: an integrator override flows through unchanged (LSP) ───────


class _WiredRunner(QnnOnnxRunner):
    async def _execute(
        self, input: ExtractInput, *, schema: Mapping[str, Any], deadline_ms: int | None
    ) -> ModelResult[dict[str, str]]:
        return ModelResult(
            status="success",
            model_id="local-qnn-onnx",
            model_version="0",
            latency_ms=1.0,
            output={"a": "text"},
        )


async def test_wired_execute_returns_success():
    runner = _WiredRunner(provider_probe=lambda: True)

    result = await runner.infer(_input(), schema={})

    assert result.status == "success" and result.output == {"a": "text"}
