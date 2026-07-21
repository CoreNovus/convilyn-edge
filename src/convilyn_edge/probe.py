"""Device-capability probe — let a device REPORT what it can run.

Multi-device IoT readiness has two halves. The backend half lets the bundle-build
resolver bind against a *device-reported* ``DeviceCapabilityManifest`` instead of a
server-hardcoded profile table. This module is the device half: an SDK-side probe
that inspects the host it runs on and produces that manifest, so any device — a
Jetson, a Snapdragon AI-PC, or a plain x86-64 box that has never been enumerated
server-side — plugs in as *data it reports*, never a new server-side profile row.

## The wire shape is snake_case (deliberately, unlike the event envelope)

:meth:`DeviceCapabilityManifest.to_wire` emits **snake_case** keys
(``device_id`` / ``ram_mb`` / ``has_npu`` / ``installed_assets`` / ``quality_tier``
…). This is NOT the camelCase convention :mod:`convilyn_edge.envelope` uses — it is
chosen so the JSON deserializes **straight into the backend's pydantic
``DeviceCapabilityManifest``** (``extra="forbid"``, no field aliases). The wire
shape is dictated by the model that consumes it. These frozen dataclasses *mirror*
that model's field names by value only — this package imports nothing from the
backend (zero runtime dependencies; the "no reverse dependency" removability check).

## Detection is file-existence + best-effort, never a scenario/silicon branch

``silicon`` is produced by **file-existence probes** (a Jetson tegra-release file,
a Qualcomm marker in cpuinfo) that yield a *string*; everything downstream is a
table lookup, never an ``if silicon == …`` control-flow branch. Every probe is
**graceful**: a missing file, an absent ``nvidia-smi``, or a non-POSIX host
degrades to a sensible default (unknown silicon → ``generic_<machine>``; no
discrete GPU → ``vram_mb=None``) and **never raises**. Stdlib only.
"""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Vocabulary mirrored (by value only) from the backend's provider_descriptor. Kept
# as typing Literals — zero runtime cost, no dependency — so an author gets the
# same fail-fast on a typo the backend enforces at parse time.
Capability = Literal["llm", "ocr", "object_detection", "tts", "stt", "embeddings"]
QualityTier = Literal["basic", "standard", "premium"]

# ── Detection sources (module-level so a test can monkeypatch them) ───────────

#: Present on NVIDIA Jetson (Tegra) devices — the reference edge silicon.
_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
#: CPU info; a Qualcomm marker here identifies a Snapdragon host.
_CPUINFO = Path("/proc/cpuinfo")

#: Silicon → whether it carries a neural accelerator. A table (OCP), never an
#: ``if silicon ==`` branch: a new accelerator device is one row. Unknown/generic
#: silicon defaults to ``False`` (a device that knows better can construct the
#: manifest explicitly).
_SILICON_NPU: dict[str, bool] = {
    "jetson_orin": True,
    "snapdragon_x": True,
}

#: Conservative positive floor for ``ram_mb`` when the host cannot report physical
#: memory (a non-POSIX host with no ``os.sysconf``, or a restricted sandbox). Keeps
#: the wire manifest's "ram_mb > 0" invariant satisfied; a device that knows its
#: real memory should construct the manifest directly.
_FALLBACK_RAM_MB = 1024


@dataclass(frozen=True)
class InstalledAsset:
    """One model/asset installed on the device, mirroring the backend shape.

    ``model`` lines up with a provider registry entry so the resolver can bind a
    device asset to a candidate; ``adapter`` names a hot-swappable LoRA over a
    shared base; ``runtime`` names the per-silicon engine that loads it.
    """

    capability: Capability
    model: str
    quality_tier: QualityTier = "standard"
    adapter: str | None = None
    runtime: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The snake_case object the backend's ``InstalledAsset`` deserializes."""
        return {
            "capability": self.capability,
            "model": self.model,
            "quality_tier": self.quality_tier,
            "adapter": self.adapter,
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class DeviceCapabilityManifest:
    """What a device reports it can run locally (mirrors the backend model).

    An immutable snapshot the device produces and the backend resolver reads.
    Construct it directly for full control, or via :func:`probe_device` to detect
    the hardware fields automatically.
    """

    device_id: str
    silicon: str
    ram_mb: int
    vram_mb: int | None = None
    has_npu: bool = False
    installed_assets: tuple[InstalledAsset, ...] = field(default_factory=tuple)

    def to_wire(self) -> dict[str, Any]:
        """The snake_case object the backend's ``DeviceCapabilityManifest`` accepts.

        Every field is emitted (including ``None``s) — the backend model has
        ``extra="forbid"`` but accepts explicit ``None`` for its optional fields, so
        this round-trips exactly into it.
        """
        return {
            "device_id": self.device_id,
            "silicon": self.silicon,
            "ram_mb": self.ram_mb,
            "vram_mb": self.vram_mb,
            "has_npu": self.has_npu,
            "installed_assets": [asset.to_wire() for asset in self.installed_assets],
        }


def _detect_silicon() -> str:
    """Best-effort silicon family from file-existence probes → a string.

    Jetson (tegra release file) → ``"jetson_orin"``; a Qualcomm marker in cpuinfo →
    ``"snapdragon_x"``; otherwise ``"generic_<platform.machine()>"``. Any I/O error
    degrades to the generic branch — never raises.
    """
    try:
        if _TEGRA_RELEASE.exists():
            return "jetson_orin"
        if _CPUINFO.exists() and "qualcomm" in _CPUINFO.read_text(encoding="utf-8").lower():
            return "snapdragon_x"
    except OSError:
        pass
    return f"generic_{platform.machine() or 'unknown'}"


def _detect_ram_mb() -> int:
    """Physical RAM in MiB via ``os.sysconf`` (POSIX); a positive floor otherwise.

    ``os.sysconf`` is absent on non-POSIX hosts (raises ``AttributeError``) and a
    key may be unsupported (``ValueError``/``OSError``) — all degrade to
    :data:`_FALLBACK_RAM_MB` so the returned value is always ``> 0``.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_mb = (page_size * phys_pages) // (1024 * 1024)
        if total_mb > 0:
            return total_mb
    except (AttributeError, ValueError, OSError):
        pass
    return _FALLBACK_RAM_MB


def _run_nvidia_smi() -> str:
    """Query total GPU memory (MiB) via ``nvidia-smi``; return its raw stdout.

    Raises when ``nvidia-smi`` is absent / fails — the caller turns that into
    ``vram_mb=None``. Factored out so a test can monkeypatch it without a subprocess.
    """
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return completed.stdout


def _detect_vram_mb() -> int | None:
    """Discrete-GPU VRAM in MiB via ``nvidia-smi``; ``None`` when unavailable.

    Any failure — no ``nvidia-smi`` binary (``OSError``), a non-zero exit / timeout
    (``subprocess.SubprocessError``), or unparseable output — yields ``None`` (a
    device on shared/unified memory, e.g. Jetson, or a CPU-only host).
    """
    try:
        raw = _run_nvidia_smi()
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return int(lines[0].split()[0])
    except (ValueError, IndexError):
        return None


def probe_device(
    *,
    device_id: str,
    installed_assets: Iterable[InstalledAsset] = (),
) -> DeviceCapabilityManifest:
    """Detect this host's capabilities and return a reportable manifest.

    Args:
        device_id: the stable device identifier (serial / provisioning id) the
            manifest is keyed by. The probe cannot invent this — it is the one field
            the integrator supplies.
        installed_assets: the model assets actually installed on the device. The
            probe cannot enumerate installed weights (that is the packaging step's
            knowledge), so the caller passes them; defaults to none.

    Returns:
        A :class:`DeviceCapabilityManifest` with ``silicon`` / ``ram_mb`` /
        ``vram_mb`` / ``has_npu`` filled from the host. ``.to_wire()`` on it is the
        exact snake_case body the backend's ``device_manifest`` field accepts.

    Never raises: every hardware probe degrades gracefully (unknown → generic).
    """
    silicon = _detect_silicon()
    return DeviceCapabilityManifest(
        device_id=device_id,
        silicon=silicon,
        ram_mb=_detect_ram_mb(),
        vram_mb=_detect_vram_mb(),
        has_npu=_SILICON_NPU.get(silicon, False),
        installed_assets=tuple(installed_assets),
    )


__all__ = [
    "Capability",
    "QualityTier",
    "InstalledAsset",
    "DeviceCapabilityManifest",
    "probe_device",
]
