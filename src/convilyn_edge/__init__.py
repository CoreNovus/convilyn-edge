"""Convilyn Edge — the Edge AI Workflow SDK (``convilyn-edge``).

The Device Data Plane + Edge Runtime SPI: seven typed Protocols that let a
developer compose device events, environment state, deterministic rules, models,
human confirmation and gated actions into an auditable, offline-capable edge AI
workflow — without the substrate ever encoding one vertical scenario.

Quick start — define a Source and drive it::

    from convilyn_edge import EventEnvelope, EventSource, new_envelope

    async for event in my_source.start(ctx):   # my_source: EventSource
        ...   # event is an EventEnvelope

Design invariants (the binding principles, expressed in code):

* The device is never a second source of truth — the server holds the 7
  server-enforced safety checks and re-grounds every device value. The edge SPI
  *inherits* that contract.
* One envelope shape, one Result type, one observability convention.
* No LLM in the deterministic operators; no ``if scenario == ...`` in the
  substrate. Scenario logic lives in a removable Solution Pack.

``convilyn_edge`` (this top level) is the stable public surface — everything here
follows an independent-versioning rule for the SDK surface. Nothing
in this package imports the Convilyn backend or any other SDK at import time;
runtime dependencies are zero.
"""

from __future__ import annotations

from convilyn_edge._version import __version__
from convilyn_edge.bundle import (
    ArtifactFetcher,
    ArtifactKind,
    DeviceInstallablePayload,
    DigestMismatchError,
    FetchError,
    IncompleteBundleError,
    InstallAction,
    InstalledArtifact,
    InstallError,
    InstallReport,
    MissingDigestError,
    ResolvedArtifact,
    StagedArtifact,
    StagedArtifactUnreadableError,
    StagedBundle,
    UnsupportedDigestError,
    UrllibArtifactFetcher,
    VerificationError,
    VerificationStatus,
    VerifiedArtifact,
    VerifiedBundle,
    fetch_to_bytes,
    install_verified_bundle,
    merge_installed_assets,
    stage_bundle,
    verify_staged_artifact,
    verify_staged_bundle,
)
from convilyn_edge.envelope import (
    SPEC_VERSION,
    Correlation,
    EventEnvelope,
    EventSourceRef,
    new_envelope,
)
from convilyn_edge.probe import (
    Capability,
    DeviceCapabilityManifest,
    InstalledAsset,
    QualityTier,
    probe_device,
    resolve_runtime,
)
from convilyn_edge.result import Err, Ok, Result
from convilyn_edge.spi import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    ActionSink,
    ActionStatus,
    DecisionSource,
    DefaultAction,
    DeterministicOperator,
    DeviceHealth,
    EventContext,
    EventSource,
    Evidence,
    ExecutionContext,
    HealthStatus,
    HumanReview,
    ModelOperator,
    ModelResult,
    ModelStatus,
    NormalizeError,
    Normalizer,
    OperatorError,
    Placement,
    ReviewChoice,
    ReviewDecision,
    ReviewOutcome,
    ReviewRequest,
    RiskLevel,
    SourceContext,
    StateProvider,
)
from convilyn_edge.storage import (
    LocalStorageProvider,
    StorageProvider,
    StorageTier,
    select_storage_provider,
)

__all__ = [
    "__version__",
    # Result
    "Ok",
    "Err",
    "Result",
    # Envelope
    "EventEnvelope",
    "EventSourceRef",
    "Correlation",
    "new_envelope",
    "SPEC_VERSION",
    # Bundle consumption (Path B: pull → stage → verify)
    "DeviceInstallablePayload",
    "ResolvedArtifact",
    "ArtifactKind",
    "ArtifactFetcher",
    "UrllibArtifactFetcher",
    "FetchError",
    "fetch_to_bytes",
    "StagedArtifact",
    "StagedBundle",
    "stage_bundle",
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
    "InstallAction",
    "InstallError",
    "InstalledArtifact",
    "InstallReport",
    "install_verified_bundle",
    "merge_installed_assets",
    # Storage-selection axis (device durable-store tiers)
    "StorageTier",
    "StorageProvider",
    "LocalStorageProvider",
    "select_storage_provider",
    # Device-capability probe (multi-device IoT readiness)
    "probe_device",
    "resolve_runtime",
    "DeviceCapabilityManifest",
    "InstalledAsset",
    "Capability",
    "QualityTier",
    # SPI — 1. Source
    "EventSource",
    "SourceContext",
    "DeviceHealth",
    "HealthStatus",
    # SPI — 2. Normalizer
    "Normalizer",
    "NormalizeError",
    # SPI — 3. StateProvider
    "StateProvider",
    "EventContext",
    # SPI — 4. DeterministicOperator
    "DeterministicOperator",
    "ExecutionContext",
    "OperatorError",
    # SPI — 5. ModelOperator
    "ModelOperator",
    "ModelResult",
    "Evidence",
    "Placement",
    "ModelStatus",
    # SPI — 6. HumanReview
    "HumanReview",
    "ReviewRequest",
    "ReviewChoice",
    "ReviewOutcome",
    "DefaultAction",
    "ReviewDecision",
    "DecisionSource",
    # SPI — 7. ActionSink
    "ActionSink",
    "ActionDescriptor",
    "ActionAuthorization",
    "ActionResult",
    "RiskLevel",
    "ActionStatus",
]
