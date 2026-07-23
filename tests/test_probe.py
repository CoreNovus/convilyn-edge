"""Unit tests for the device-capability probe (Epic #2556 — Gap 2).

The probe detects the host it runs on and produces a reportable
``DeviceCapabilityManifest``. Its detection sources are module-level (a tegra
file, cpuinfo, ``nvidia-smi``) so a test can monkeypatch them without real
hardware. Covers:

* **Logic** — a Jetson file present → ``jetson_orin`` (NPU defaults False: the
  Orin family splits Nano/NX on DLA and the probe can't disambiguate); a Qualcomm
  cpuinfo marker → ``snapdragon_x`` (+ NPU); neither → ``generic_<machine>`` (no NPU).
* **Boundary** — ``nvidia-smi`` present → ``vram_mb`` parsed; absent → ``None``.
* **Error** — every probe degrades gracefully (an OSError reading a source, a
  failed ``nvidia-smi``) and ``probe_device`` never raises.
* **Object state** — ``to_wire()`` is the exact snake_case shape the backend's
  pydantic ``DeviceCapabilityManifest`` deserializes (the SDK↔backend contract),
  and ``device_id`` / ``installed_assets`` pass through unchanged.

No network, no real hardware, stdlib only.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from convilyn_edge import (
    DeviceCapabilityManifest,
    InstalledAsset,
    probe_device,
)
from convilyn_edge import probe as probe_mod


def _completed(stdout: str) -> subprocess.CompletedProcess:
    """A successful ``nvidia-smi`` CompletedProcess with the given stdout."""
    return subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=stdout, stderr="")


@pytest.fixture(autouse=True)
def _neutral_host(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Default every probe source to 'absent' so each test sets only what it needs.

    Points the file sources at non-existent paths, forces a known machine string,
    and makes ``nvidia-smi`` absent — a deterministic generic/CPU-only baseline.
    Mocks the OS/subprocess boundary (not the probe helpers) so the real detection
    code runs.
    """
    monkeypatch.setattr(probe_mod, "_TEGRA_RELEASE", tmp_path / "no_tegra")
    monkeypatch.setattr(probe_mod, "_CPUINFO", tmp_path / "no_cpuinfo")
    monkeypatch.setattr(probe_mod.platform, "machine", lambda: "x86_64")

    def _no_nvidia_smi(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(probe_mod.subprocess, "run", _no_nvidia_smi)


# ── 1. Logic — silicon detection ─────────────────────────────────────────


def test_jetson_release_file_present_detects_jetson(monkeypatch, tmp_path):
    tegra = tmp_path / "nv_tegra_release"
    tegra.write_text("# R35 (release), REVISION: 4.1\n", encoding="utf-8")
    monkeypatch.setattr(probe_mod, "_TEGRA_RELEASE", tegra)

    manifest = probe_device(device_id="dev-1")

    assert manifest.silicon == "jetson_orin"


def test_jetson_detected_defaults_no_npu(monkeypatch, tmp_path):
    # Tegra can't distinguish an Orin Nano (no DLA) from an NX (2× DLA), so has_npu
    # conservatively defaults to False; a DLA-bearing Orin sets it on its own manifest.
    tegra = tmp_path / "nv_tegra_release"
    tegra.write_text("R35\n", encoding="utf-8")
    monkeypatch.setattr(probe_mod, "_TEGRA_RELEASE", tegra)

    assert probe_device(device_id="dev-1").has_npu is False


def test_qualcomm_cpuinfo_detects_snapdragon(monkeypatch, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("Hardware\t: Qualcomm Technologies, Inc SC8280XP\n", encoding="utf-8")
    monkeypatch.setattr(probe_mod, "_CPUINFO", cpuinfo)

    manifest = probe_device(device_id="dev-2")

    assert manifest.silicon == "snapdragon_x"


def test_no_marker_falls_back_to_generic_machine():
    # Neither source present (autouse fixture) + machine forced to x86_64.
    manifest = probe_device(device_id="dev-3")

    assert manifest.silicon == "generic_x86_64"


def test_generic_silicon_reports_no_npu():
    assert probe_device(device_id="dev-3").has_npu is False


# ── 2. Boundary — vram_mb via nvidia-smi ─────────────────────────────────


def test_nvidia_smi_present_parses_vram(monkeypatch):
    monkeypatch.setattr(probe_mod.subprocess, "run", lambda *a, **k: _completed("8192\n"))

    assert probe_device(device_id="dev-4").vram_mb == 8192


def test_nvidia_smi_absent_yields_none_vram():
    # autouse fixture makes nvidia-smi (subprocess.run) raise FileNotFoundError.
    assert probe_device(device_id="dev-5").vram_mb is None


def test_ram_mb_is_always_positive():
    # os.sysconf is absent on non-POSIX hosts; the probe must still report ram_mb > 0.
    assert probe_device(device_id="dev-6").ram_mb > 0


# ── storage fields (#2791) ───────────────────────────────────────────────────


def test_disk_gb_reports_total_in_gib(monkeypatch):
    monkeypatch.setattr(
        probe_mod.shutil, "disk_usage", lambda _: SimpleNamespace(total=100 * 1024**3)
    )

    assert probe_device(device_id="dev-disk").disk_gb == 100


def test_to_wire_emits_non_default_storage_triple():
    manifest = DeviceCapabilityManifest(
        device_id="d",
        silicon="jetson_orin",
        ram_mb=8192,
        disk_gb=64,
        removable=True,
        media_store="/mnt/sd",
    )

    wire = manifest.to_wire()

    assert (wire["disk_gb"], wire["removable"], wire["media_store"]) == (64, True, "/mnt/sd")


def test_disk_gb_none_when_disk_usage_fails(monkeypatch):
    def _boom(_):
        raise OSError("no such path")

    monkeypatch.setattr(probe_mod.shutil, "disk_usage", _boom)

    assert probe_device(device_id="dev-disk").disk_gb is None


def test_removable_and_media_store_come_from_args():
    manifest = probe_device(device_id="dev-sd", removable=True, media_store="/mnt/sdcard")

    assert (manifest.removable, manifest.media_store) == (True, "/mnt/sdcard")


def test_storage_fields_default_when_not_supplied(monkeypatch):
    monkeypatch.setattr(probe_mod.shutil, "disk_usage", lambda _: SimpleNamespace(total=1024**3))

    manifest = probe_device(device_id="dev-def")

    assert (manifest.removable, manifest.media_store) == (False, None)


def test_ram_mb_uses_sysconf_when_available(monkeypatch):
    # Exercise the POSIX success path: 4 KiB pages × 2 Mi pages = 8192 MiB.
    sizes = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2 * 1024 * 1024}
    monkeypatch.setattr(probe_mod.os, "sysconf", lambda name: sizes[name], raising=False)

    assert probe_mod._detect_ram_mb() == 8192


def test_ram_mb_falls_back_when_sysconf_reports_zero(monkeypatch):
    # A degenerate sysconf (0 pages) must not report ram_mb=0 → use the floor.
    monkeypatch.setattr(probe_mod.os, "sysconf", lambda _name: 0, raising=False)

    assert probe_mod._detect_ram_mb() == probe_mod._FALLBACK_RAM_MB


# ── 3. Error — graceful degradation, never raises ────────────────────────


def test_unreadable_source_degrades_without_raising(monkeypatch):
    class _Boom:
        def exists(self) -> bool:
            raise OSError("permission denied")

    monkeypatch.setattr(probe_mod, "_TEGRA_RELEASE", _Boom())

    # The OSError from the source probe is swallowed → generic silicon, no raise.
    assert probe_device(device_id="dev-7").silicon == "generic_x86_64"


def test_nvidia_smi_nonzero_exit_yields_none(monkeypatch):
    def _fail(*_a, **_k):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(probe_mod.subprocess, "run", _fail)

    assert probe_device(device_id="dev-8").vram_mb is None


def test_nvidia_smi_unparseable_output_yields_none(monkeypatch):
    monkeypatch.setattr(probe_mod.subprocess, "run", lambda *a, **k: _completed("no GPU here\n"))

    assert probe_device(device_id="dev-9").vram_mb is None


def test_nvidia_smi_empty_output_yields_none(monkeypatch):
    monkeypatch.setattr(probe_mod.subprocess, "run", lambda *a, **k: _completed("\n"))

    assert probe_device(device_id="dev-9b").vram_mb is None


# ── 4. Object state — the SDK↔backend wire contract + pass-through ────────


def test_to_wire_is_the_backend_snake_case_shape():
    manifest = DeviceCapabilityManifest(
        device_id="dev-serial-777",
        silicon="jetson_orin",
        ram_mb=8192,
        vram_mb=None,
        has_npu=True,
        installed_assets=(
            InstalledAsset(
                capability="llm",
                model="local-qwen3-4b",
                quality_tier="standard",
                runtime="llama_cpp_cuda",
            ),
        ),
    )

    assert manifest.to_wire() == {
        "device_id": "dev-serial-777",
        "silicon": "jetson_orin",
        "ram_mb": 8192,
        "vram_mb": None,
        "has_npu": True,
        "disk_gb": None,
        "removable": False,
        "media_store": None,
        "installed_assets": [
            {
                "capability": "llm",
                "model": "local-qwen3-4b",
                "quality_tier": "standard",
                "adapter": None,
                "runtime": "llama_cpp_cuda",
            }
        ],
    }


def test_probe_device_passes_through_device_id_and_assets():
    asset = InstalledAsset(capability="ocr", model="local-paddleocr-v4")
    manifest = probe_device(device_id="dev-10", installed_assets=(asset,))

    assert manifest.device_id == "dev-10"
    assert manifest.installed_assets == (asset,)
