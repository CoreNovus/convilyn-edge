"""``convilyn_edge.bundle`` — Path B device bundle consumption (pull → install).

The device half of the chat-built-workflow → on-device path. The cloud compiles a
``uw_*`` workflow into a bundle and the push endpoint resolves it into a
:class:`DeviceInstallablePayload` (references + short-TTL fetch locations, never
bytes). This subpackage consumes that payload end-to-end:

* :mod:`~convilyn_edge.bundle.payload` — parse the payload; URL-free provenance.
* :mod:`~convilyn_edge.bundle.fetch` — the https-only, TLS-verified fetch seam.
* :mod:`~convilyn_edge.bundle.staging` — download a whole payload to scratch.
* :mod:`~convilyn_edge.bundle.verify` — digest-verify staged files (fail-loud, no landing).
* :mod:`~convilyn_edge.bundle.install` — land a verified bundle (atomic, idempotent) +
  the additive device-manifest assets.

Zero runtime dependencies.
"""

from __future__ import annotations

from convilyn_edge.bundle.fetch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_TOTAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ArtifactFetcher,
    FetchError,
    UrllibArtifactFetcher,
    fetch_to_bytes,
)
from convilyn_edge.bundle.install import (
    InstallAction,
    InstalledArtifact,
    InstallError,
    InstallReport,
    install_verified_bundle,
    merge_installed_assets,
)
from convilyn_edge.bundle.payload import (
    ArtifactKind,
    DeviceInstallablePayload,
    ResolvedArtifact,
)
from convilyn_edge.bundle.staging import (
    StagedArtifact,
    StagedBundle,
    stage_bundle,
)
from convilyn_edge.bundle.verify import (
    DigestMismatchError,
    IncompleteBundleError,
    MissingDigestError,
    StagedArtifactUnreadableError,
    UnsupportedDigestError,
    VerificationError,
    VerificationStatus,
    VerifiedArtifact,
    VerifiedBundle,
    verify_staged_artifact,
    verify_staged_bundle,
)

__all__ = [
    # payload
    "ArtifactKind",
    "ResolvedArtifact",
    "DeviceInstallablePayload",
    # fetch
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_TOTAL_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "FetchError",
    "ArtifactFetcher",
    "UrllibArtifactFetcher",
    "fetch_to_bytes",
    # staging
    "StagedArtifact",
    "StagedBundle",
    "stage_bundle",
    # verify
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
    # install
    "InstallAction",
    "InstallError",
    "InstalledArtifact",
    "InstallReport",
    "install_verified_bundle",
    "merge_installed_assets",
]
