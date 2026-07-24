"""Per-artifact **digest verification** of a staged bundle — fail-loud, no landing.

After :func:`~convilyn_edge.bundle.staging.stage_bundle` has downloaded a payload
to scratch, this layer proves each staged file's integrity against the digest the
server pinned in the manifest — **before** anything is installed. It is the gate
between "downloaded" and "trusted": a single mismatch raises and the caller
discards the whole staging (nothing lands), because a bundle carrying one tampered
artifact is entirely untrusted.

## Which artifacts are verified here, and how (the digest domain)

The push payload's ``content_hash`` does not mean the same thing for every kind
(confirmed against the server's ``bundle_producer.py`` / ``artifact_keys.py``):

* ``workflow_spec`` — ``content_hash`` is ``sha256`` over the canonical compiled-spec
  JSON, i.e. over **exactly the fetched bytes** (``model_dump_json()``). Verified
  **here** by recomputing the digest of the staged file. Any future kind that
  starts carrying a ``content_hash`` is treated the same way (fail-closed default:
  a pinned digest is checked, never silently skipped).
* ``trusted_core`` — ``content_hash`` is a digest over the device's **own S2 source
  tree** (the shipped gate modules), NOT over the fetched artifact bytes. Byte-hashing
  the fetched object would falsely mismatch, so it is **deferred** to the Trusted-Core
  fail-closed gate (a separate mechanism); it is never byte-hashed here.
* ``role_asset`` / ``tool_asset`` / ``runtime`` — carry **no** ``content_hash`` on the
  wire (integrity pinned out of band). Nothing to check here.

A kind that MUST carry a digest (``workflow_spec`` / ``trusted_core``) but arrives
without one — ``None`` **or** empty/whitespace — fails loud, closing the "strip the
digest to skip verification" downgrade; a blank must never be read as "nothing to
verify". :func:`verify_staged_bundle` also requires the bundle to contain a
``workflow_spec`` at all, closing the "drop the whole spec artifact" variant.

## Contract for the install layer (verify → install)

Verification is point-in-time: a returned :class:`VerifiedBundle` states each staged
file hashed correctly *at verify time*. The install layer MUST therefore

* run back-to-back with verify in the same process — no re-fetch, no yield to
  untrusted code, nothing else writing the scratch dir in between;
* land each file by ``os.rename`` within the same filesystem (atomic, preserves the
  verified inode) — never copy-by-path or re-download;
* reject / skip any artifact ``kind`` it does not recognise. This layer classifies an
  unknown digest-less kind as ``out_of_band`` (forward-compat); the closed set of
  *installable* kinds is the install layer's fail-closed allowlist, not this gate's.

The single-owner scratch dir (``0o700``/``0o600``, created no-follow by
:mod:`~convilyn_edge.bundle.staging`) is what makes that point-in-time guarantee
usable. Read-only here: this layer hashes staged files (also no-follow) and raises;
it never moves, deletes, or installs.

## Logging note

Error messages embed the wire-supplied ``kind`` (via ``!r``, which escapes control
chars) and *validated* digests only — never the signed URL (absent from this layer),
the resolver key, or the scratch path. Log them structured (JSON) all the same.
Stdlib only.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from convilyn_edge.bundle.staging import StagedArtifact, StagedBundle

#: Streaming chunk for the file digest — bounds peak memory (assets are GB-scale).
_DIGEST_CHUNK_SIZE = 65536
#: The only digest algorithm the manifest pins (``sha256:…`` — see
#: ``bundle_producer._content_hash``). An unknown prefix fails closed.
_DIGEST_PREFIX = "sha256:"
#: Exact shape of a supported pin: ``sha256:`` + 64 lowercase hex. Validating the
#: shape (not just the prefix) gives a precise ``UnsupportedDigestError`` for a
#: malformed pin and guarantees the digest embedded in an error message is clean.
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")

#: Kinds that MUST carry a ``content_hash``; one arriving without fails loud
#: (a stripped digest must never silently downgrade to "nothing to verify").
_DIGEST_REQUIRED_KINDS = frozenset({"workflow_spec", "trusted_core"})
#: Kinds whose ``content_hash`` is a digest over the device's own S2 source tree
#: (not the fetched bytes) — verified by the Trusted-Core fail-closed gate, never
#: byte-hashed here.
_DEVICE_SOURCE_KINDS = frozenset({"trusted_core"})

# Load-bearing invariant: a device-source kind (whose digest is verified elsewhere)
# MUST also be digest-required — else a null/blank content_hash on it would fall
# through to ``out_of_band`` (a silent bypass) instead of failing loud. A plain
# ``assert`` is stripped under ``python -O``, so raise explicitly at import to keep
# this fail-closed guarantee even in optimised runs.
if not (_DEVICE_SOURCE_KINDS <= _DIGEST_REQUIRED_KINDS):  # pragma: no cover - config invariant
    raise RuntimeError(
        "bundle.verify invariant broken: a device-source kind is not digest-required "
        "(a blank content_hash could bypass verification)"
    )

#: Every bundle packages exactly one compiled workflow, so a verifiable bundle must
#: contain this kind; its absence is the "drop the spec" tamper.
_WORKFLOW_SPEC_KIND = "workflow_spec"

#: Per-artifact verification outcome:
#: ``verified`` — its fetched bytes were digest-checked and matched;
#: ``deferred_trusted_core`` — carries a source-tree digest, checked by the
#: Trusted-Core gate, not here; ``out_of_band`` — no wire digest, pinned elsewhere.
VerificationStatus = Literal["verified", "deferred_trusted_core", "out_of_band"]


class VerificationError(Exception):
    """A staged bundle failed integrity verification — do NOT install it."""


class DigestMismatchError(VerificationError):
    """A staged artifact's recomputed digest did not match the pinned one."""


class UnsupportedDigestError(VerificationError):
    """A pinned ``content_hash`` is not a supported (``sha256:<64 hex>``) digest."""


class MissingDigestError(VerificationError):
    """A kind that must carry a ``content_hash`` arrived without one (fail-closed)."""


class StagedArtifactUnreadableError(VerificationError):
    """A staged file could not be read to hash it — treated as a verification failure."""


class IncompleteBundleError(VerificationError):
    """A staged bundle is missing a required part (no ``workflow_spec``)."""


@dataclass(frozen=True)
class VerifiedArtifact:
    """One staged artifact plus the outcome of verifying it."""

    staged: StagedArtifact
    status: VerificationStatus


@dataclass(frozen=True)
class VerifiedBundle:
    """A staged bundle whose every artifact passed (or was exempt from) verification.

    Reaching this object at all means no digest mismatch occurred and the bundle is
    structurally complete (has a ``workflow_spec``) — the install layer may land
    these files. Per-artifact :attr:`VerifiedArtifact.status` records *how* each was
    cleared (byte-hash verified / deferred / out-of-band), so install and audit know
    what was actually integrity-checked.
    """

    bundle_id: str
    workflow_spec_id: str
    artifacts: tuple[VerifiedArtifact, ...] = ()

    @property
    def verified_count(self) -> int:
        """How many artifacts were digest-verified over their fetched bytes."""
        return sum(1 for a in self.artifacts if a.status == "verified")


def _sha256_file(path: Path) -> str:
    """Streaming, no-follow ``sha256:…`` digest of ``path`` (memory-bounded).

    Opens ``O_NOFOLLOW`` (POSIX) for parity with staging's write posture — a
    scratch file swapped for a symlink after staging is not read through. Any I/O
    error (vanished / unreadable file) becomes a :class:`StagedArtifactUnreadableError`
    so "cannot integrity-check it" is uniformly a :class:`VerificationError`.
    """
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            while chunk := handle.read(_DIGEST_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        # `from None`: keep the scratch path out of the surfaced message.
        raise StagedArtifactUnreadableError(
            f"staged artifact could not be read for verification ({type(exc).__name__})"
        ) from None
    return _DIGEST_PREFIX + digest.hexdigest()


def _verify_fetched_bytes(staged: StagedArtifact, expected: str) -> None:
    """Recompute the staged file's digest and fail loud on any mismatch."""
    normalized = expected.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise UnsupportedDigestError(
            f"{staged.artifact.kind!r} content_hash is not a supported 'sha256:<64 hex>' digest"
        )
    actual = _sha256_file(staged.path)
    if actual != normalized:
        # Digests are public integrity pins (not secrets), and both are validated /
        # canonical, so safe to surface; the url/key/path are never here.
        raise DigestMismatchError(
            f"{staged.artifact.kind!r} digest mismatch: expected {normalized}, got {actual}"
        )


def verify_staged_artifact(staged: StagedArtifact) -> VerificationStatus:
    """Verify one staged artifact; return its status or raise (fail-loud).

    Classifies the artifact by its digest domain and, for a fetched-bytes digest,
    recomputes and compares. Raises :class:`MissingDigestError` for a digest-required
    kind whose ``content_hash`` is absent/blank, :class:`UnsupportedDigestError` for a
    non-sha256 pin, :class:`DigestMismatchError` on a content mismatch, and
    :class:`StagedArtifactUnreadableError` if the staged file cannot be read.
    """
    kind = staged.artifact.kind
    content_hash = staged.artifact.content_hash
    # Absent OR blank is "no digest": str(None) is not None, and from_wire can emit
    # an empty string, so a whitespace-only pin must not slip past this gate.
    if content_hash is None or not content_hash.strip():
        if kind in _DIGEST_REQUIRED_KINDS:
            raise MissingDigestError(f"{kind!r} artifact must carry a non-empty content_hash")
        return "out_of_band"
    if kind in _DEVICE_SOURCE_KINDS:
        # Its digest is over the device's own source tree, not these bytes — the
        # Trusted-Core fail-closed gate checks it, not this layer.
        return "deferred_trusted_core"
    _verify_fetched_bytes(staged, content_hash)
    return "verified"


def verify_staged_bundle(staged: StagedBundle) -> VerifiedBundle:
    """Verify every staged artifact; return a :class:`VerifiedBundle` or raise.

    Fail-loud and all-or-nothing: the first bad artifact (mismatch / unsupported /
    missing / unreadable) raises, and the bundle must contain a ``workflow_spec``
    (else :class:`IncompleteBundleError`) — so a tampered *or* truncated bundle never
    yields a :class:`VerifiedBundle`, and the caller discards the scratch staging.
    On success every artifact is present with its :class:`VerificationStatus`.
    """
    verified = tuple(
        VerifiedArtifact(staged=artifact, status=verify_staged_artifact(artifact))
        for artifact in staged.artifacts
    )
    if not any(a.staged.artifact.kind == _WORKFLOW_SPEC_KIND for a in verified):
        raise IncompleteBundleError(
            "bundle contains no workflow_spec artifact — refusing to verify as installable"
        )
    return VerifiedBundle(
        bundle_id=staged.bundle_id,
        workflow_spec_id=staged.workflow_spec_id,
        artifacts=verified,
    )


__all__ = [
    "VerificationStatus",
    "VerificationError",
    "DigestMismatchError",
    "UnsupportedDigestError",
    "MissingDigestError",
    "StagedArtifactUnreadableError",
    "IncompleteBundleError",
    "VerifiedArtifact",
    "VerifiedBundle",
    "verify_staged_artifact",
    "verify_staged_bundle",
]
