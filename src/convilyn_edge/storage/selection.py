"""Deterministic storage-tier **selection** — device profile → a provider.

Closes the storage-selection axis: given a device's reported storage profile — the
:class:`~convilyn_edge.probe.DeviceCapabilityManifest` **tier** signals ``removable`` /
``media_store`` (``disk_gb`` is capacity, for an install-time fit check, not a tier) —
and the mount roots the caller can offer, pick which tier durable state lives on and
return the :class:`~convilyn_edge.storage.provider.LocalStorageProvider` for it (PR②).

## A fixed priority table, never an ``if silicon ==`` branch

Selection walks :data:`_TIER_PRIORITY` — a constant, most-capable-first table — and
returns a provider on the first tier the device actually has. It is a **pure function**
of the reported profile + the supplied roots: no silicon equality, no workflow
vocabulary. Extending the tier set is a one-token edit to the table + the profile map,
never a new branch (OCP).

## The caller controls the durability trade-off

``media_store`` (a device-provisioned, typically large fixed mount — its path is in the
manifest) is preferred, then ``removable``, then the always-present ``internal`` floor.
A ``removable`` medium is ejectable, so it is only ever selected when the caller **opts
in** by supplying a ``removable_root``: a bulk-capacity caller supplies it (extra space),
a state-critical caller omits it and falls through to a fixed tier — the same table
serves both. Stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from convilyn_edge.storage.provider import LocalStorageProvider, StorageTier

if TYPE_CHECKING:  # pragma: no cover - typing only
    from convilyn_edge.probe import DeviceCapabilityManifest

#: Tier priority, most-capable-first: a device-provisioned media store, then a removable
#: medium, then the always-present internal root. A fixed descriptor table the selection
#: reads — never an ``if silicon ==`` branch. A new tier is one entry here + its
#: availability line in :func:`_available_roots` (plus a keyword root arg if, like
#: ``removable``, its mount path is not carried on the manifest) — an OCP edit, not a branch.
_TIER_PRIORITY: tuple[StorageTier, ...] = ("media_store", "removable", "internal")


def _available_roots(
    manifest: DeviceCapabilityManifest,
    *,
    internal_root: Path,
    removable_root: Path | None,
) -> dict[StorageTier, Path]:
    """Map each tier the device actually has to its mount root.

    ``internal`` is always present. ``media_store`` is present iff the manifest reports
    one (its field carries the path). ``removable`` is present iff the manifest reports a
    removable medium AND the caller opted in with a ``removable_root`` (the manifest
    carries only the boolean, not a path).
    """
    roots: dict[StorageTier, Path] = {"internal": internal_root}
    if manifest.media_store:
        roots["media_store"] = Path(manifest.media_store)
    if manifest.removable and removable_root is not None:
        roots["removable"] = removable_root
    return roots


def select_storage_provider(
    manifest: DeviceCapabilityManifest,
    *,
    internal_root: Path | str,
    removable_root: Path | str | None = None,
) -> LocalStorageProvider:
    """Pick the device's durable-storage tier deterministically → a provider.

    Returns a :class:`LocalStorageProvider` on the highest-priority tier
    (:data:`_TIER_PRIORITY`) the device has: ``media_store`` when the manifest reports one
    (rooted at its declared path), else ``removable`` when the manifest reports a removable
    medium AND ``removable_root`` is supplied, else ``internal`` (rooted at
    ``internal_root``). Deterministic and scenario-free.

    ``internal_root`` is required, so a provider is always returned. The store *names* are
    traversal-proofed by :class:`LocalStorageProvider`; the roots
    (``internal_root`` / ``removable_root`` / the manifest's ``media_store``) are trusted
    device configuration — the mount points the device runs on, not attacker input.
    """
    roots = _available_roots(
        manifest,
        internal_root=Path(internal_root),
        removable_root=Path(removable_root) if removable_root is not None else None,
    )
    for tier in _TIER_PRIORITY:
        root = roots.get(tier)
        if root is not None:
            return LocalStorageProvider(root, tier=tier)
    raise AssertionError("internal tier is always available")  # pragma: no cover


__all__ = ["select_storage_provider"]
