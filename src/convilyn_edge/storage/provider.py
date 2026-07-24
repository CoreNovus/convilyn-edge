"""The device **storage-selection** seam — durable persistence by name, on a tier.

A device has more than one place to keep durable state: internal disk, a removable
medium (SD / USB), or a large-asset / media store mount. This module is the seam that
lets a caller persist *without* knowing which tier it landed on — the tier is chosen
once (deterministically, from the device's reported storage profile) and
handed to the caller as a :class:`StorageProvider`; the caller then asks it for a named
durable store and reads / writes records.

## Reuses ``DurableQueue`` — does not reinvent a store

The durable store a provider hands out IS the existing
:class:`~convilyn_edge.offline.queue.DurableQueue` (``offline/queue.py``) — the SDK's
JSONL-backed, idempotent, apply-once record store — placed at a tier-appropriate path.
The seam adds *tier placement* over that primitive; it does not fork it. So an offline
audit log, the stateful threshold aggregator, or any device state persist
through one durable-store contract, on whichever tier the device warrants.

## Scenario-free (the substrate rule)

The seam knows tiers, not scenarios. There is no ``if silicon == …`` here and no
vertical vocabulary — a pet monitor and a POS both persist the same way; which tier a
device uses is a deterministic function of its manifest, never a branch on
what the workflow is. Stdlib only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from convilyn_edge.offline.queue import DurableQueue, KeyOf

#: The device storage tiers a provider may be rooted on. A closed vocabulary that
#: mirrors the :class:`~convilyn_edge.probe.DeviceCapabilityManifest` storage fields
#: (``disk_gb`` → ``internal``, ``removable`` → ``removable``, ``media_store`` →
#: ``media_store``). Extending it is a one-token OCP edit, never a scenario branch.
StorageTier = Literal["internal", "removable", "media_store"]

#: A store name is turned into a single safe file-name component: anything outside
#: ``[A-Za-z0-9_.-]`` collapses to ``_``. So a name can never contain a path separator,
#: ``..``, or an absolute prefix — a store always lands *inside* the provider's root.
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Windows reserved device names — even with an extension these resolve to a device, so a
#: store named one of them is prefixed. Edge targets are Linux; this is a dev-host belt.
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _safe_store_filename(name: str) -> str:
    """A traversal-proof, collision-free ``<name>.jsonl`` for a store within a root.

    ``name`` is reduced to a single ``[A-Za-z0-9_.-]`` component (so no separator, ``..``,
    or absolute prefix survives — a store always lands *inside* the provider root). When
    that reduction is **lossy** (any char collapsed / stripped / truncated), a short stable
    digest of the ORIGINAL name is appended, so two distinct names can never silently
    collide onto one file (which would corrupt a shared store). A clean, short name keeps
    its readable form; a Windows reserved device name is prefixed.
    """
    slug = _UNSAFE_NAME_RE.sub("_", name).strip("_.")
    safe = slug[:48] or "store"
    if safe != name:  # lossy reduction → disambiguate so distinct names never collide
        safe = f"{safe}-{hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]}"
    if safe.lower() in _WINDOWS_RESERVED:
        safe = f"_{safe}"
    return f"{safe}.jsonl"


@runtime_checkable
class StorageProvider(Protocol):
    """Hand out a named durable record store, on a device storage tier.

    The injectable seam over device durable storage. :attr:`tier` names which tier this
    provider is rooted on (for audit / selection transparency); :meth:`durable_store`
    returns a durable, apply-once record store for a given logical ``name`` — the same
    :class:`DurableQueue` contract regardless of tier. Implementations MUST keep every
    store *inside* their own root (a ``name`` can never escape it) and MUST return the
    *same* store for the same ``name`` (stable placement = idempotent persistence).
    """

    @property
    def tier(self) -> StorageTier:
        """The device storage tier this provider persists on."""
        ...

    def durable_store(self, name: str, *, key_of: KeyOf | None = None) -> DurableQueue:
        """A durable, apply-once record store named ``name`` on this provider's tier."""
        ...


class LocalStorageProvider:
    """A filesystem-rooted :class:`StorageProvider` — one ``DurableQueue`` file per store.

    The concrete the tier selector constructs once it has picked a tier and
    its root directory. Each :meth:`durable_store` is a
    :class:`~convilyn_edge.offline.queue.DurableQueue` at ``<root>/<safe name>.jsonl``
    (the queue creates the file + parent lazily on first write), so this reuses the
    existing durable store rather than reimplementing one. A general capability — it
    branches on no scenario.
    """

    def __init__(self, root: Path | str, *, tier: StorageTier) -> None:
        self._root = Path(root)
        self._tier = tier
        # Memoize one live store per name: DurableQueue's O(1) dedup assumes it is the
        # SOLE writer of its file, so handing out a fresh instance per call would let two
        # live handles on one file break apply-once. Same name → same live store.
        self._stores: dict[str, DurableQueue] = {}

    @property
    def tier(self) -> StorageTier:
        return self._tier

    @property
    def root(self) -> Path:
        """The directory this provider's stores live under."""
        return self._root

    def durable_store(self, name: str, *, key_of: KeyOf | None = None) -> DurableQueue:
        """The one :class:`DurableQueue` for ``name`` on this tier (traversal-proof).

        Returns the **same live store** for the same ``name`` (the apply-once sole-writer
        invariant — never a fresh handle on an already-open file). ``name`` is reduced to a
        single safe file-name component, so a hostile name (``"../../etc/x"``, an absolute
        path, a separator) can never place the store outside :attr:`root`, and distinct
        names never collide on one file. ``key_of`` overrides the queue's apply-once key
        derivation (e.g. an events store keyed by ``eventId``) and takes effect only on the
        FIRST open of a ``name`` — a later ``key_of`` for an already-open name is ignored
        (the first call fixes the store); omitted uses the queue default.
        """
        cached = self._stores.get(name)
        if cached is not None:
            return cached
        path = self._root / _safe_store_filename(name)
        store = (
            DurableQueue(path=path) if key_of is None else DurableQueue(path=path, key_of=key_of)
        )
        self._stores[name] = store
        return store


__all__ = [
    "StorageTier",
    "StorageProvider",
    "LocalStorageProvider",
]
