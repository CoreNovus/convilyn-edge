"""Stage a whole push payload to a scratch directory — download, don't install.

:func:`stage_bundle` walks a :class:`~convilyn_edge.bundle.payload.DeviceInstallablePayload`
and fetches every artifact (via an injected
:class:`~convilyn_edge.bundle.fetch.ArtifactFetcher`) into a **caller-owned scratch
directory**, returning a :class:`StagedBundle` of on-disk paths.

Staging is deliberately *not* installing:

* **No verification here.** Digest verification (``content_hash`` + asset digest,
  fail-loud) is the next layer's job — staged bytes are untrusted until verified.
* **No landing here.** The device model store and
  ``DeviceCapabilityManifest.installed_assets`` are updated only by the install
  layer, and only from *verified* staged files. A scratch directory (typically a
  :class:`tempfile.TemporaryDirectory`) is not the install location.

This separation is what makes "verify before install, never land an unverified or
partial artifact" enforceable downstream: the pull writes only to scratch, verify
gates the move, install performs it. Streaming keeps peak memory bounded no matter
how large a model asset is.

Scratch-dir hardening: the directory is created private (``0o700``) and each file
is opened **no-follow** where the OS supports it, so a pre-planted symlink in a
shared/predictable ``dest_dir`` cannot redirect a write to an arbitrary path. Each
artifact is also bounded in size — by an explicit ``max_artifact_bytes`` or, absent
that, by the server-advertised ``size_bytes`` — so a runaway/mis-declared stream
cannot fill the device disk. Stdlib only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from convilyn_edge.bundle.fetch import ArtifactFetcher, FetchError
from convilyn_edge.bundle.payload import DeviceInstallablePayload, ResolvedArtifact

#: Non-alphanumerics collapse to ``_`` when deriving a staging file name from an
#: artifact ``kind`` — the file name is cosmetic (identity travels on the carried
#: :class:`ResolvedArtifact`); the numeric index prefix guarantees uniqueness.
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _staging_name(index: int, kind: str) -> str:
    """A collision-free, filesystem-safe staging file name for one artifact.

    ``f"{index:04d}-{slug(kind)}.artifact"``. The zero-padded index makes it unique
    even when two artifacts share a kind (or carry an unslugifiable one); the slug
    is a readability aid only, never used for identity or routing.
    """
    slug = _UNSAFE_NAME_RE.sub("_", kind).strip("_")[:32] or "artifact"
    return f"{index:04d}-{slug}.artifact"


def _open_no_follow(path: Path) -> BinaryIO:
    """Open ``path`` for binary write, refusing to follow a symlink at the leaf.

    Uses ``O_NOFOLLOW`` (POSIX) so a pre-existing symlink at ``path`` — e.g. one a
    local attacker planted in a shared scratch dir pointing at ``~/.ssh/authorized_keys``
    — makes the open fail (``ELOOP``) instead of writing through the link to an
    arbitrary file. The flag is absent on Windows (``getattr(..., 0)``), where this
    degrades to a plain write; ``O_BINARY`` (also Windows-only) keeps the stream raw.
    New files are created ``0o600``.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb")


class _BoundedSink:
    """A write-only wrapper that fails loud once more than ``limit`` bytes arrive.

    Wraps the real file sink so a per-artifact size ceiling is enforced *as the
    fetcher streams*, before the bytes hit disk — the fetcher only ever calls
    ``write`` on its sink, so this thin proxy is all that is needed. ``limit=None``
    means unbounded (the caller opted out of a ceiling for this artifact).
    """

    def __init__(self, sink: BinaryIO, limit: int | None) -> None:
        self._sink = sink
        self._limit = limit
        self._written = 0

    def write(self, data: bytes) -> int:
        self._written += len(data)
        if self._limit is not None and self._written > self._limit:
            raise FetchError(f"artifact exceeded its {self._limit}-byte staging ceiling")
        return self._sink.write(data)


@dataclass(frozen=True)
class StagedArtifact:
    """One fetched-but-unverified artifact on disk.

    Pairs the resolved reference (which carries ``content_hash`` for the verify
    layer and ``kind`` / ``key`` for the install layer) with the scratch ``path``
    the bytes landed at and the ``size_bytes`` actually written.
    """

    artifact: ResolvedArtifact
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class StagedBundle:
    """The result of staging a whole payload — provenance plus on-disk paths.

    Carries the bundle provenance (``bundle_id`` / ``workflow_spec_id``) and one
    :class:`StagedArtifact` per fetched reference, in payload order.
    """

    bundle_id: str
    workflow_spec_id: str
    artifacts: tuple[StagedArtifact, ...] = ()

    @property
    def artifact_count(self) -> int:
        """Number of staged artifacts (pure)."""
        return len(self.artifacts)


def stage_bundle(
    payload: DeviceInstallablePayload,
    fetcher: ArtifactFetcher,
    dest_dir: Path | str,
    *,
    max_artifact_bytes: int | None = None,
) -> StagedBundle:
    """Fetch every artifact in ``payload`` into ``dest_dir``; return the staging.

    Creates ``dest_dir`` if absent (private, ``0o700``), then streams each artifact
    to its own no-follow file via ``fetcher``.

    **Size-bounded.** Each artifact is capped at ``max_artifact_bytes`` when given,
    otherwise at its server-advertised ``size_bytes`` (the payload carries it for
    exactly this pre-flight purpose). An artifact whose stream exceeds its ceiling
    fails loud. Only an artifact with *no* advertised size and no explicit override
    is unbounded here — pass ``max_artifact_bytes`` (or a capped fetcher) to bound
    those too when the payload is untrusted.

    **Fail-loud and torn-file-safe.** If any fetch raises (a rejected scheme, an
    expired URL, a transport error, an exceeded ceiling), that artifact's partial
    file is removed and the exception propagates — no half-written file is ever left
    masquerading as a complete artifact. Already-staged files from earlier artifacts
    remain in ``dest_dir``; the caller owns the scratch directory and discards it
    (or retries the whole pull) on failure, per the push contract's "re-request,
    don't retry-forever" rule.

    Nothing is verified or installed here — see the module docstring.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    os.chmod(dest, 0o700)  # private scratch: staged (unverified) bytes are not world-readable
    staged: list[StagedArtifact] = []
    for index, artifact in enumerate(payload.artifacts):
        target = dest / _staging_name(index, artifact.kind)
        limit = max_artifact_bytes if max_artifact_bytes is not None else artifact.size_bytes
        wrote = False
        try:
            with _open_no_follow(target) as raw:
                size = fetcher.fetch(artifact.url, _BoundedSink(raw, limit))
            staged.append(StagedArtifact(artifact=artifact, path=target, size_bytes=size))
            wrote = True
        finally:
            if not wrote:
                target.unlink(missing_ok=True)
    return StagedBundle(
        bundle_id=payload.bundle_id,
        workflow_spec_id=payload.workflow_spec_id,
        artifacts=tuple(staged),
    )


__all__ = [
    "StagedArtifact",
    "StagedBundle",
    "stage_bundle",
]
