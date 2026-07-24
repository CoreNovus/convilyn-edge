"""``convilyn_edge.storage`` — the device storage-selection axis.

A device persists durable state on one of several tiers (internal disk / a removable
medium / a media store mount). This subpackage is the seam that abstracts *where*:

* :mod:`~convilyn_edge.storage.provider` — the :class:`StorageProvider` seam + the
  filesystem-rooted :class:`LocalStorageProvider`, which hands out a named durable
  record store (the reused :class:`~convilyn_edge.offline.queue.DurableQueue`) on a
  chosen tier.
* :mod:`~convilyn_edge.storage.selection` — :func:`select_storage_provider`, the
  deterministic tier-selection table (device storage profile → a provider).

Which tier a device uses is a deterministic function of its reported storage profile
(the manifest ``removable`` / ``media_store`` tier signals; ``disk_gb`` is capacity, not a
tier) — no scenario branch. Zero runtime dependencies.
"""

from __future__ import annotations

from convilyn_edge.storage.provider import (
    LocalStorageProvider,
    StorageProvider,
    StorageTier,
)
from convilyn_edge.storage.selection import select_storage_provider

__all__ = [
    "StorageTier",
    "StorageProvider",
    "LocalStorageProvider",
    "select_storage_provider",
]
