"""Unit tests for runner selection — deterministic runtime → runner table."""

from __future__ import annotations

import pytest

from convilyn_edge.probe import resolve_runtime
from convilyn_edge.runners import (
    OpenAICompatRunner,
    QnnOnnxRunner,
    RunnerConfig,
    UnsupportedRuntimeError,
    select_runner,
)
from convilyn_edge.runners.registry import SUPPORTED_RUNTIMES

_CFG = RunnerConfig(model="qwen3:4b")


# ── logic: runtime → correct runner class ────────────────────────────────────


def test_llama_cpp_selects_openai_compat_runner():
    assert isinstance(select_runner("llama_cpp", _CFG), OpenAICompatRunner)


def test_llama_cpp_cuda_selects_openai_compat_runner():
    assert isinstance(select_runner("llama_cpp_cuda", _CFG), OpenAICompatRunner)


def test_ollama_selects_openai_compat_runner():
    assert isinstance(select_runner("ollama", _CFG), OpenAICompatRunner)


def test_qnn_runtime_selects_qnn_runner():
    assert isinstance(select_runner("onnxruntime_qnn", _CFG), QnnOnnxRunner)


# ── integration with the probe: silicon → runtime → runner (all deterministic) ─


def test_snapdragon_resolves_to_qnn_runner():
    runner = select_runner(resolve_runtime("snapdragon_x"), _CFG)

    assert isinstance(runner, QnnOnnxRunner)


def test_jetson_resolves_to_openai_compat_runner():
    runner = select_runner(resolve_runtime("jetson_orin"), _CFG)

    assert isinstance(runner, OpenAICompatRunner)


def test_generic_host_resolves_to_openai_compat_runner():
    runner = select_runner(resolve_runtime("generic_x86_64"), _CFG)

    assert isinstance(runner, OpenAICompatRunner)


# ── error: an unknown runtime fails loud (no silent fallback) ─────────────────


def test_unknown_runtime_raises():
    with pytest.raises(UnsupportedRuntimeError):
        select_runner("tensorrt_llm", _CFG)


def test_supported_runtimes_lists_registered_only():
    assert "onnxruntime_qnn" in SUPPORTED_RUNTIMES and "tensorrt_llm" not in SUPPORTED_RUNTIMES
