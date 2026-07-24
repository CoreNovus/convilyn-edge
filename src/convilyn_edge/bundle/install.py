"""Install a verified bundle into the device store — atomic, idempotent, additive.

The final Path B step. Given a :class:`~convilyn_edge.bundle.verify.VerifiedBundle`
(everything already digest-checked or exempted, PR②), this lands each artifact's
bytes into a device-local store and reports the model assets to record on the
device's :class:`~convilyn_edge.probe.DeviceCapabilityManifest`.

## Landing (honours the verify → install contract)

Each staged file is moved into the store with :func:`os.replace` — **atomic** and,
on the same filesystem, **inode-preserving**, so the exact bytes `verify` hashed are
what land (no re-read, no re-fetch, no TOCTOU gap). The store and the staging scratch
dir MUST therefore share a filesystem; a cross-device move fails loud rather than
silently degrading to a re-reading copy. The store is **hash-addressed** — an
artifact lands at ``sha256(key)`` (sharded) — so a tampered ``key`` cannot direct a
write outside the store (no path traversal), and the same artifact always lands at
the same stable path (the basis for idempotency).

Only a **known** artifact kind is installed: an unrecognised kind fails loud
(:data:`_INSTALLABLE_KINDS`), honouring the verify → install split — `verify`
classifies an unknown kind liberally as ``out_of_band`` (forward-compat), and this
layer is the fail-closed allowlist of what may actually land.

## Idempotent re-run (mirrors the server asset_stager)

Re-installing the same bundle is a no-op — an artifact already at its target is
``skipped_present`` (GB-scale weights are never re-written). The skip test is
integrity-aware without re-hashing large files: a **byte-verified** artifact
(``workflow_spec``) is skipped only if the *stored* file still hashes to its pinned
``content_hash`` — so a corrupted small spec is re-installed (self-healing) — while an
**out-of-band / deferred** artifact (a GB weight; no wire-pin over its bytes) is
skipped on presence alone, since the resolver key identifies the asset and
:func:`os.replace` lands it atomically (a present file is always complete).

## Additive manifest (aligned to the frozen device-capability matrix)

Each installed ``role_asset`` / ``tool_asset`` carries its :class:`InstalledAsset`,
reconstructed from its resolver key **at install time** (the sole point that mirrors
the server's ``artifact_keys._asset_key`` composition — parity-pinned by a golden
vector in the tests). :meth:`InstallReport.installed_assets` is then a pure read, and
:func:`merge_installed_assets` folds them into a manifest **additively** (never
dropping or reshaping existing entries). Stdlib only.
"""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from convilyn_edge.bundle.verify import VerifiedArtifact, VerifiedBundle
from convilyn_edge.probe import DeviceCapabilityManifest, InstalledAsset

#: Streaming chunk for hashing a stored file (memory-bounded).
_HASH_CHUNK_SIZE = 65536
#: Sentinel an ``_asset_key`` uses for an absent optional field — mirrors
#: ``app.services.edge.push.artifact_keys._EMPTY`` (by value; parity-pinned).
_EMPTY_SEGMENT = "-"
#: Number of ``/``-segments in a role/tool ``_asset_key``:
#: ``{tenant}/{role|tool}/{binding}/{capability}/{model}/{runtime}/{quality_tier}/{adapter}``.
_ASSET_KEY_ARITY = 8
#: Kinds whose installed artifact maps to a reportable :class:`InstalledAsset`.
_ASSET_KINDS = frozenset({"role_asset", "tool_asset"})
#: The closed set of artifact kinds this layer may land — the fail-closed allowlist
#: `verify` delegates here. Mirrors the manifest's reference kinds one-for-one; an
#: unknown kind fails loud rather than landing unrecognised bytes.
_INSTALLABLE_KINDS = frozenset(
    {"workflow_spec", "role_asset", "tool_asset", "runtime", "trusted_core"}
)
#: Kinds verified over their fetched bytes — the only ones whose *stored* file can be
#: (and is) re-checked against its pinned digest on an idempotent skip.
_BYTE_VERIFIED_STATUS = "verified"

InstallAction = Literal["installed", "skipped_present"]


class InstallError(Exception):
    """A verified artifact could not be installed into the device store."""


@dataclass(frozen=True)
class InstalledArtifact:
    """One artifact after landing — the verified artifact, its store path, action.

    ``installed_asset`` is the reportable :class:`InstalledAsset` for a
    ``role_asset`` / ``tool_asset`` (reconstructed from its key at install time),
    or ``None`` for any other kind.
    """

    verified: VerifiedArtifact
    path: Path
    action: InstallAction
    installed_asset: InstalledAsset | None = None


@dataclass(frozen=True)
class InstallReport:
    """The outcome of installing a whole bundle — provenance, landings, assets."""

    bundle_id: str
    workflow_spec_id: str
    artifacts: tuple[InstalledArtifact, ...] = ()

    @property
    def installed_count(self) -> int:
        """Artifacts newly written to the store (not skipped-present)."""
        return sum(1 for a in self.artifacts if a.action == "installed")

    @property
    def skipped_count(self) -> int:
        """Artifacts already present at the same digest (idempotent no-ops)."""
        return sum(1 for a in self.artifacts if a.action == "skipped_present")

    @property
    def installed_assets(self) -> tuple[InstalledAsset, ...]:
        """The model assets to record on the device manifest (deduped, in order).

        The :class:`InstalledAsset` of each installed ``role_asset`` / ``tool_asset``
        (parsed at install time), deduped so two identical bindings — or a re-run —
        never inflate the manifest. A pure read; no I/O and never raises.
        """
        seen: set[InstalledAsset] = set()
        assets: list[InstalledAsset] = []
        for installed in self.artifacts:
            asset = installed.installed_asset
            if asset is not None and asset not in seen:
                seen.add(asset)
                assets.append(asset)
        return tuple(assets)


def _sha256_file(path: Path) -> str:
    """Streaming, no-follow ``sha256:…`` digest of a stored file.

    Opens ``O_NOFOLLOW`` (POSIX) for parity with staging's / verify's read posture,
    and wraps any I/O error as :class:`InstallError` so a vanished/unreadable file is
    a typed install failure, not a bare ``OSError``.
    """
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise InstallError(
            f"could not read an artifact for install ({type(exc).__name__})"
        ) from None
    return "sha256:" + digest.hexdigest()


def _store_path(store_dir: Path, key: str) -> Path:
    """Hash-addressed, traversal-proof store path for an artifact ``key``.

    ``sha256(key)`` (hex) sharded by its first two chars — a stable path per key
    (idempotency) that a tampered ``key`` cannot use to escape ``store_dir`` (the
    name is pure hex, never the key's own segments).
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return store_dir / digest[:2] / f"{digest}.artifact"


def _already_installed(verified: VerifiedArtifact, target: Path) -> bool:
    """Is ``target`` already the correct install of ``verified``? (idempotent skip).

    A byte-verified artifact is skippable only if the **stored** file still hashes to
    its pinned ``content_hash`` — so a corrupted small spec is re-installed
    (self-heal), and a GB weight is never re-hashed. An out-of-band / deferred
    artifact has no wire pin over its bytes, so presence alone is idempotent (same
    resolver key == same asset; ``os.replace`` lands it atomically → always complete).
    """
    if verified.status != _BYTE_VERIFIED_STATUS:
        return True
    expected = verified.staged.artifact.content_hash
    if not expected:  # a "verified" artifact always carries a validated pin (verify.py)
        raise InstallError("a byte-verified artifact is missing its content_hash")
    return _sha256_file(target) == expected.strip().lower()


def _install_one(verified: VerifiedArtifact, store_dir: Path) -> InstalledArtifact:
    """Land one verified artifact into ``store_dir`` (idempotent, atomic)."""
    artifact = verified.staged.artifact
    if artifact.kind not in _INSTALLABLE_KINDS:
        raise InstallError(f"refusing to install unrecognised artifact kind {artifact.kind!r}")
    # Reconstruct the reportable asset up front — a malformed asset key fails loud
    # here, before anything is written (no partial-install from a bad key).
    asset = _installed_asset_from_key(artifact.key) if artifact.kind in _ASSET_KINDS else None
    target = _store_path(store_dir, artifact.key)
    if target.exists() and _already_installed(verified, target):
        return InstalledArtifact(
            verified=verified, path=target, action="skipped_present", installed_asset=asset
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(verified.staged.path, target)
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.EXDEV:
            raise InstallError(
                "could not atomically install artifact; the store and the staging "
                "scratch dir must share a filesystem"
            ) from None
        raise InstallError(f"could not install artifact ({type(exc).__name__})") from None
    return InstalledArtifact(
        verified=verified, path=target, action="installed", installed_asset=asset
    )


def install_verified_bundle(
    verified: VerifiedBundle,
    store_dir: Path | str,
) -> InstallReport:
    """Install every artifact of a :class:`VerifiedBundle` into ``store_dir``.

    Creates ``store_dir`` private (``0o700`` at birth) and lands each artifact
    atomically and idempotently (see the module docstring). Returns an
    :class:`InstallReport` whose :attr:`~InstallReport.installed_assets` are ready to
    fold into the device manifest via :func:`merge_installed_assets`. A
    cross-filesystem store, a vanished/unreadable staged file, an unrecognised kind,
    or a malformed asset key fails loud (:class:`InstallError`) — a partially-installed
    bundle is never silently reported as complete. A mid-install failure may leave
    earlier artifacts already landed; because each landing is per-artifact idempotent,
    re-running the pull completes the install without re-writing what is already
    present (no all-or-nothing rollback is attempted, as a device cannot atomically
    stage GB-scale weights across many files).

    Only a verified bundle should reach here: it ran back-to-back with
    :func:`~convilyn_edge.bundle.verify.verify_staged_bundle` over the same un-mutated
    scratch (the verify → install contract).
    """
    store = Path(store_dir)
    # 0o700 at creation (umask-safe) closes the mkdir→chmod window; the chmod
    # self-heals the perms of a store dir that already existed.
    store.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(store, 0o700)
    installed = tuple(_install_one(a, store) for a in verified.artifacts)
    return InstallReport(
        bundle_id=verified.bundle_id,
        workflow_spec_id=verified.workflow_spec_id,
        artifacts=installed,
    )


def _installed_asset_from_key(key: str) -> InstalledAsset:
    """Reconstruct an :class:`InstalledAsset` from a role/tool ``_asset_key``.

    Mirrors ``app.services.edge.push.artifact_keys._asset_key`` **by value** (the SDK
    imports nothing from the backend): the fixed-arity key
    ``{tenant}/{role|tool}/{binding}/{capability}/{model}/{runtime}/{quality_tier}/{adapter}``
    yields the five asset fields, with the ``"-"`` sentinel mapped back to ``None``. A
    field whose real value is literally ``"-"`` is indistinguishable from the sentinel
    — an inherited limitation of the backend's ``safe_optional_segment`` scheme, not
    an SDK bug. A golden-vector parity test pins this against the server composition.
    """
    parts = key.split("/")
    if len(parts) != _ASSET_KEY_ARITY:
        raise InstallError(
            f"asset key has unexpected arity ({len(parts)}, expected {_ASSET_KEY_ARITY})"
        )
    _tenant, _prefix, _binding, capability, model, runtime, quality_tier, adapter = parts
    return InstalledAsset(
        capability=capability,  # typed Capability; a plain str at runtime (mirror by value)
        model=model,
        quality_tier=quality_tier,  # typed QualityTier; plain str at runtime
        adapter=None if adapter == _EMPTY_SEGMENT else adapter,
        runtime=None if runtime == _EMPTY_SEGMENT else runtime,
    )


def merge_installed_assets(
    manifest: DeviceCapabilityManifest,
    assets: tuple[InstalledAsset, ...],
) -> DeviceCapabilityManifest:
    """Return a new manifest with ``assets`` folded into ``installed_assets`` — additive.

    Purely additive and idempotent: existing entries are never dropped or reshaped,
    and an asset already present is not duplicated (order preserved: existing first,
    then the new ones in order). The manifest is frozen, so a new instance is returned;
    when nothing new is added, the original instance is returned unchanged.
    """
    seen = set(manifest.installed_assets)
    ordered_new: list[InstalledAsset] = []
    for asset in assets:
        if asset not in seen:
            seen.add(asset)
            ordered_new.append(asset)
    if not ordered_new:
        return manifest
    return dataclasses.replace(
        manifest, installed_assets=manifest.installed_assets + tuple(ordered_new)
    )


__all__ = [
    "InstallAction",
    "InstallError",
    "InstalledArtifact",
    "InstallReport",
    "install_verified_bundle",
    "merge_installed_assets",
]
