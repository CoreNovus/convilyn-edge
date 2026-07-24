"""Parity tests: resolve_runtime + the frozen #2771 manifest wire-field contract.

These PIN the SDK to the canonical device matrix
(``docs/reference/edge_device_matrix.md``, issue #2771): the per-silicon runtime
table (§2) and the frozen additive-only `DeviceCapabilityManifest` field set (§1a).
If either drifts, a test fails loudly — the cross-lane contract cannot diverge
silently.
"""

from __future__ import annotations

from convilyn_edge import DeviceCapabilityManifest, InstalledAsset
from convilyn_edge.probe import resolve_runtime

# ── resolve_runtime parity with the matrix §2 (_RUNTIME_BY_SILICON) ──────────


def test_jetson_orin_runtime():
    assert resolve_runtime("jetson_orin") == "llama_cpp_cuda"


def test_jetson_xavier_runtime():
    assert resolve_runtime("jetson_xavier") == "llama_cpp_cuda"


def test_snapdragon_runtime():
    assert resolve_runtime("snapdragon_x") == "onnxruntime_qnn"


def test_unknown_silicon_defaults_to_llama_cpp():
    assert resolve_runtime("generic_aarch64") == "llama_cpp"


def test_intel_aipc_falls_through_to_default():
    # §5 forward-looking scaffold silicon has no _RUNTIME_BY_SILICON row yet.
    assert resolve_runtime("aipc_npu_intel") == "llama_cpp"


# ── frozen manifest wire-field contract (§1a) — additive-only pin ────────────

#: The EXACT field set the frozen #2771 `DeviceCapabilityManifest` allows. The
#: backend model is `extra="forbid"`, so emitting any key outside this set breaks
#: ingestion. Adding a field is a matrix-first change (Lane C), never an SDK-only one.
_FROZEN_MANIFEST_FIELDS = {
    "device_id",
    "silicon",
    "ram_mb",
    "vram_mb",
    "has_npu",
    "disk_gb",
    "removable",
    "media_store",
    "installed_assets",
}
_FROZEN_ASSET_FIELDS = {"capability", "model", "quality_tier", "adapter", "runtime"}


def test_manifest_wire_field_set_is_frozen():
    manifest = DeviceCapabilityManifest(device_id="d", silicon="jetson_orin", ram_mb=8192)

    assert set(manifest.to_wire()) == _FROZEN_MANIFEST_FIELDS


def test_installed_asset_wire_field_set_is_frozen():
    asset = InstalledAsset(capability="llm", model="local-qwen3-4b", runtime="llama_cpp_cuda")

    assert set(asset.to_wire()) == _FROZEN_ASSET_FIELDS
