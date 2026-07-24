"""Unit tests for the device-RAM fit guard — warn before OOM, never after."""

from __future__ import annotations

import logging

import pytest

from convilyn_edge import probe
from convilyn_edge.runners import (
    OpenAICompatRunner,
    RamFitError,
    RunnerConfig,
    check_ram_fit,
    select_runner,
)

_FIT_CFG = RunnerConfig(model="qwen3:4b", min_ram_mb=4096)
_NO_REQ_CFG = RunnerConfig(model="qwen3:4b")


# ── logic: declared requirement vs available RAM ─────────────────────────────


def test_fit_when_available_exceeds_requirement():
    report = check_ram_fit(_FIT_CFG, available_ram_mb=8192)

    assert report.fits is True


def test_no_requirement_always_fits_even_on_tiny_device():
    report = check_ram_fit(_NO_REQ_CFG, available_ram_mb=1)

    assert report.fits is True


def test_autodetect_reads_probed_host_ram(monkeypatch):
    monkeypatch.setattr(probe, "_detect_ram_mb", lambda: 2048)

    report = check_ram_fit(_FIT_CFG)

    assert report.available_ram_mb == 2048


# ── boundary: exact fit passes, one MiB short fails ──────────────────────────


def test_exact_ram_boundary_fits():
    report = check_ram_fit(_FIT_CFG, available_ram_mb=4096)

    assert report.fits is True


def test_one_mib_short_does_not_fit():
    report = check_ram_fit(_FIT_CFG, available_ram_mb=4095)

    assert report.fits is False


# ── error: strict_fit raises instead of warning ──────────────────────────────


def test_strict_fit_raises_on_shortfall():
    with pytest.raises(RamFitError):
        select_runner("llama_cpp", _FIT_CFG, available_ram_mb=1024, strict_fit=True)


def test_default_shortfall_warns_but_still_builds_runner(caplog):
    with caplog.at_level(logging.WARNING, logger="convilyn_edge.runners.registry"):
        runner = select_runner("llama_cpp", _FIT_CFG, available_ram_mb=1024)

    assert isinstance(runner, OpenAICompatRunner)


def test_shortfall_warning_is_assertable(caplog):
    with caplog.at_level(logging.WARNING, logger="convilyn_edge.runners.registry"):
        select_runner("llama_cpp", _FIT_CFG, available_ram_mb=1024)

    assert "may not fit" in caplog.text


def test_fit_emits_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="convilyn_edge.runners.registry"):
        select_runner("llama_cpp", _FIT_CFG, available_ram_mb=8192)

    assert caplog.text == ""


# ── object-state: the report carries the numbers the integrator surfaces ─────


def test_shortfall_report_carries_both_numbers():
    report = check_ram_fit(_FIT_CFG, available_ram_mb=1024)

    assert (report.min_ram_mb, report.available_ram_mb) == (4096, 1024)


def test_shortfall_message_names_the_model():
    report = check_ram_fit(_FIT_CFG, available_ram_mb=1024)

    assert "qwen3:4b" in report.message
