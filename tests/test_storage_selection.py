"""Unit tests for deterministic storage-tier selection (#2791 PR③)."""

from __future__ import annotations

from pathlib import Path

from convilyn_edge.probe import DeviceCapabilityManifest
from convilyn_edge.storage.provider import LocalStorageProvider
from convilyn_edge.storage.selection import (
    _TIER_PRIORITY,
    _available_roots,
    select_storage_provider,
)


def _manifest(**overrides) -> DeviceCapabilityManifest:
    base = {"device_id": "d", "silicon": "jetson_orin", "ram_mb": 8192}
    base.update(overrides)
    return DeviceCapabilityManifest(**base)


# ── priority: media_store > removable > internal ─────────────────────────────


def test_media_store_is_preferred(tmp_path):
    provider = select_storage_provider(_manifest(media_store="/mnt/nvme"), internal_root=tmp_path)

    assert provider.tier == "media_store"


def test_media_store_is_rooted_at_its_declared_path(tmp_path):
    provider = select_storage_provider(_manifest(media_store="/mnt/nvme"), internal_root=tmp_path)

    assert provider.root == Path("/mnt/nvme")


def test_media_store_beats_an_opted_in_removable(tmp_path):
    manifest = _manifest(media_store="/mnt/nvme", removable=True)

    provider = select_storage_provider(manifest, internal_root=tmp_path, removable_root="/mnt/sd")

    assert provider.tier == "media_store"


def test_removable_is_selected_when_opted_in(tmp_path):
    provider = select_storage_provider(
        _manifest(removable=True), internal_root=tmp_path, removable_root="/mnt/sd"
    )

    assert (provider.tier, provider.root) == ("removable", Path("/mnt/sd"))


def test_removable_is_ignored_without_a_supplied_root(tmp_path):
    # removable=True but the caller did not opt in with a removable_root → not selectable.
    provider = select_storage_provider(_manifest(removable=True), internal_root=tmp_path)

    assert provider.tier == "internal"


def test_internal_is_the_floor(tmp_path):
    provider = select_storage_provider(_manifest(), internal_root=tmp_path)

    assert (provider.tier, provider.root) == ("internal", tmp_path)


def test_internal_is_always_available(tmp_path):
    # The always-returns guarantee: internal is seeded regardless of the profile, so the
    # priority walk can never fall through (the trailing AssertionError is unreachable).
    roots = _available_roots(_manifest(), internal_root=tmp_path, removable_root=None)

    assert "internal" in roots


def test_internal_root_accepts_a_str(tmp_path):
    provider = select_storage_provider(_manifest(), internal_root=str(tmp_path))

    assert provider.root == tmp_path


# ── shape + determinism ──────────────────────────────────────────────────────


def test_returns_a_local_storage_provider(tmp_path):
    assert isinstance(
        select_storage_provider(_manifest(), internal_root=tmp_path), LocalStorageProvider
    )


def test_selection_is_deterministic(tmp_path):
    manifest = _manifest(media_store="/mnt/nvme")

    first = select_storage_provider(manifest, internal_root=tmp_path)
    second = select_storage_provider(manifest, internal_root=tmp_path)

    assert first.tier == second.tier == "media_store"


def test_tier_priority_table_is_capacity_first():
    assert _TIER_PRIORITY == ("media_store", "removable", "internal")
