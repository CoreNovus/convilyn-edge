"""Unit tests for warmup — cold start must be distinguishable from offline."""

from __future__ import annotations

import time
import urllib.request

import pytest

from convilyn_edge.clientcompute.engine import HttpLocalExtractor
from convilyn_edge.runners import OpenAICompatRunner
from convilyn_edge.warmup import WarmupResult, warmup_runner


class _FakeResponse:
    def __init__(self, body: bytes = b"{}") -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


_OK_COMPLETION = {"message": {"content": '{"ok": true}'}}


def _reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse())


def _unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


# ── logic: the no-op default and delegation of warmup_runner ─────────────────


def test_runner_without_hook_reports_already_warm():
    class _Hookless:
        pass

    result = warmup_runner(_Hookless())

    assert result.state == "warm"


def test_runner_without_hook_reports_zero_latency():
    class _Hookless:
        pass

    result = warmup_runner(_Hookless())

    assert result.latency_ms == 0.0


def test_runner_with_hook_is_delegated_to():
    sentinel = WarmupResult(state="cold_started", latency_ms=42.0)

    class _Warmable:
        def warmup(self, deadline_ms=None):
            return sentinel

    assert warmup_runner(_Warmable()) is sentinel


# ── logic: extractor probe classifies warm vs cold by observed latency ───────


def test_fast_probe_is_warm(monkeypatch):
    _reachable(monkeypatch)
    extractor = HttpLocalExtractor(model="m", transport=lambda *a: _OK_COMPLETION)

    result = extractor.warmup(warm_threshold_ms=60_000.0)

    assert result.state == "warm"


def test_slow_probe_is_cold_started(monkeypatch):
    _reachable(monkeypatch)

    def slow_transport(*args):
        time.sleep(0.005)
        return _OK_COMPLETION

    extractor = HttpLocalExtractor(model="m", transport=slow_transport)

    result = extractor.warmup(warm_threshold_ms=0.0)

    assert result.state == "cold_started"


def test_cold_start_reports_observed_latency(monkeypatch):
    _reachable(monkeypatch)

    def slow_transport(*args):
        time.sleep(0.005)
        return _OK_COMPLETION

    extractor = HttpLocalExtractor(model="m", transport=slow_transport)

    result = extractor.warmup(warm_threshold_ms=0.0)

    assert result.latency_ms is not None and result.latency_ms > 0.0


# ── boundary: deadline propagation and deadline-exceeded-while-loading ───────


def test_deadline_ms_becomes_transport_timeout_seconds(monkeypatch):
    _reachable(monkeypatch)
    seen: list[float] = []

    def capture(url, body, headers, timeout):
        seen.append(timeout)
        return _OK_COMPLETION

    HttpLocalExtractor(model="m", transport=capture).warmup(deadline_ms=2000)

    assert seen == [2.0]


def test_deadline_exceeded_while_server_up_is_cold_not_offline(monkeypatch):
    _reachable(monkeypatch)

    def timing_out(*args):
        raise TimeoutError("probe deadline")

    extractor = HttpLocalExtractor(model="m", transport=timing_out)

    result = extractor.warmup(deadline_ms=1)

    assert result.state == "cold_started"


# ── error: unreachable is never faked as cold ────────────────────────────────


def test_unreachable_server_reports_unreachable(monkeypatch):
    _unreachable(monkeypatch)
    extractor = HttpLocalExtractor(model="m", transport=lambda *a: _OK_COMPLETION)

    result = extractor.warmup()

    assert result.state == "unreachable"


def test_unreachable_report_carries_problem_detail(monkeypatch):
    _unreachable(monkeypatch)
    extractor = HttpLocalExtractor(model="m", transport=lambda *a: _OK_COMPLETION)

    result = extractor.warmup()

    assert result.detail is not None and "unreachable" in result.detail


def test_server_dropping_mid_probe_is_unreachable(monkeypatch):
    calls = {"n": 0}

    def flaky_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("server died")
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)

    def failing_transport(*args):
        raise OSError("connection reset")

    result = HttpLocalExtractor(model="m", transport=failing_transport).warmup()

    assert result.state == "unreachable"


# ── object-state: the runner delegates and the result is frozen ──────────────


def test_openai_compat_runner_delegates_warmup(monkeypatch):
    _reachable(monkeypatch)
    runner = OpenAICompatRunner.ollama(model="m", transport=lambda *a: _OK_COMPLETION)

    result = runner.warmup()

    assert result.state == "warm"


def test_warmup_result_is_immutable():
    result = WarmupResult(state="warm", latency_ms=1.0)

    with pytest.raises(AttributeError):
        result.state = "cold_started"  # type: ignore[misc]
