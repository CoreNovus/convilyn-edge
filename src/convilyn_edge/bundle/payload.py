"""Device-side mirror of the server's edge **push payload** (Path B consumption).

A chat-built ``uw_*`` workflow is compiled server-side into a bundle; the push
endpoint (``POST /api/v1/edge/push``) resolves that bundle into a
:class:`DeviceInstallablePayload` — the bundle id, the workflow it packages, and
every reference resolved to a short-TTL, single-object **fetch location**. The
server never proxies the bytes; the device pulls each object itself.

This module is the SDK's zero-dependency mirror of that payload's wire shape. It
**confirms-and-consumes** the contract — the authoritative model lives server-side
(``app/services/edge/push/payload.py``); these frozen dataclasses reproduce its
field names *by value only* and import nothing from the backend (the "no reverse
dependency" removability rule). :meth:`DeviceInstallablePayload.from_wire` parses
the JSON body the push endpoint returns; :attr:`artifact_fingerprint` and
:meth:`log_summary` reproduce the server's URL-free provenance digest so a re-pull
is observably idempotent and a URL never reaches a log.

Forward-compatibility: the wire body may grow *additional* keys in a future minor
(the device tolerates them in :attr:`~ResolvedArtifact.extra`), but the fetch
essentials (``kind`` / ``key`` / ``url``) are hard-required and fail loud when
absent. Pure stdlib; no I/O — fetching lives in :mod:`convilyn_edge.bundle.fetch`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

#: The kinds of reference a bundle can carry, mirroring the server's
#: ``PushArtifactKind`` one-for-one (by value). Documented vocabulary — the
#: :attr:`ResolvedArtifact.kind` field itself is a plain ``str`` so an *older*
#: device tolerates a *new* kind a future server adds (forward-compatible),
#: deferring the "can I install this kind?" decision to the installer.
ArtifactKind = Literal[
    "workflow_spec",
    "role_asset",
    "tool_asset",
    "runtime",
    "trusted_core",
]


def _parse_size_bytes(raw: Any) -> int | None:
    """Coerce a wire ``size_bytes`` to a non-negative int, or ``None``.

    A hint only (pre-flight storage / progress), but kept honest against the
    server's ``size_bytes: int = Field(ge=0)``: an explicit ``null`` → ``None``; a
    negative value fails loud rather than silently mirroring a malformed payload.
    """
    if raw is None:
        return None
    size = int(raw)
    if size < 0:
        raise ValueError("size_bytes must be >= 0")
    return size


@dataclass(frozen=True)
class ResolvedArtifact:
    """One bundle reference resolved to a fetch location (device-side view).

    Mirrors the server's ``ResolvedArtifact``. ``content_hash`` is the integrity
    pin (``sha256:…``) where the reference carries one (workflow spec, trusted
    core) and ``None`` where it does not (asset / runtime references pin integrity
    out of band). ``url`` is a short-TTL, single-object signed GET URL — the only
    credential in the flow: **never persist, log, or forward it** (see
    :meth:`DeviceInstallablePayload.log_summary`).
    """

    #: Reference kind. A known value is one of :data:`ArtifactKind`; typed ``str``
    #: to tolerate a forward-compatible new kind rather than reject the payload.
    kind: str
    #: Stable, resolver-facing identity of this artifact (tenant-prefixed). Not a
    #: secret, but resolver-internal — kept out of logs alongside the URL.
    key: str
    #: Short-TTL, single-object signed GET URL. Ephemeral credential — see above.
    url: str
    #: Integrity digest (``sha256:…``) when the reference carries one, else ``None``.
    content_hash: str | None = None
    #: Size in bytes when the resolver knew it (pre-flight storage / progress),
    #: else ``None``.
    size_bytes: int | None = None
    #: Any forward-compatible keys the wire carried that this version does not
    #: model — preserved rather than rejected.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> ResolvedArtifact:
        """Parse one wire artifact object (snake_case) into a :class:`ResolvedArtifact`.

        ``kind`` / ``key`` / ``url`` are hard-required — the device cannot fetch or
        route an artifact without them, so a malformed reference fails loud:
        ``KeyError`` on absence, ``ValueError`` on an explicit ``null`` or an empty
        string (checked *before* ``str()`` coercion, since ``str(None) == "None"``
        is truthy and would slip a null through). ``content_hash`` / ``size_bytes``
        default to ``None`` (an explicit wire ``null`` is honoured; a negative
        ``size_bytes`` is rejected, mirroring the server's ``ge=0``); unknown keys
        are preserved in :attr:`extra`.
        """
        known = {"kind", "key", "url", "content_hash", "size_bytes"}
        raw_kind, raw_key, raw_url = wire["kind"], wire["key"], wire["url"]  # KeyError on absence
        if raw_kind is None or raw_key is None or raw_url is None:
            raise ValueError("artifact requires non-null 'kind', 'key', and 'url'")
        kind, key, url = str(raw_kind), str(raw_key), str(raw_url)
        if not (kind and key and url):
            raise ValueError("artifact requires non-empty 'kind', 'key', and 'url'")
        raw_hash = wire.get("content_hash")
        return cls(
            kind=kind,
            key=key,
            url=url,
            content_hash=(str(raw_hash) if raw_hash is not None else None),
            size_bytes=_parse_size_bytes(wire.get("size_bytes")),
            extra={k: v for k, v in wire.items() if k not in known},
        )


@dataclass(frozen=True)
class DeviceInstallablePayload:
    """A whole bundle resolved to fetch locations — the device runner's input.

    Mirrors the server's ``DeviceInstallablePayload``: the bundle id, the compiled
    workflow spec it installs, and every reference resolved to a location. The
    device installs from this; it never sees the manifest's un-resolved references.
    """

    #: Provenance / dedupe key, carried through from the manifest.
    bundle_id: str
    #: The compiled workflow spec id this payload installs.
    workflow_spec_id: str
    #: Every reference, resolved to a fetch location (server's stable order).
    artifacts: tuple[ResolvedArtifact, ...] = ()
    #: Forward-compatible payload-level keys this version does not model.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> DeviceInstallablePayload:
        """Parse the push endpoint's JSON body into a :class:`DeviceInstallablePayload`.

        ``bundle_id`` / ``workflow_spec_id`` are hard-required and non-empty;
        ``artifacts`` defaults to empty and each element is parsed by
        :meth:`ResolvedArtifact.from_wire` (in wire order — the artifact order is
        load-bearing for :attr:`artifact_fingerprint`). Unknown top-level keys are
        preserved in :attr:`extra`.
        """
        known = {"bundle_id", "workflow_spec_id", "artifacts"}
        # `wire[...]` (not `.get`) so an absent key fails loud as KeyError.
        raw_bundle_id, raw_spec_id = wire["bundle_id"], wire["workflow_spec_id"]
        if raw_bundle_id is None or raw_spec_id is None:
            raise ValueError("payload requires non-null 'bundle_id' and 'workflow_spec_id'")
        bundle_id, workflow_spec_id = str(raw_bundle_id), str(raw_spec_id)
        if not (bundle_id and workflow_spec_id):
            raise ValueError("payload requires non-empty 'bundle_id' and 'workflow_spec_id'")
        # `... or ()` (not `.get(k, ())`) so an explicit wire ``null`` degrades to
        # "no artifacts" rather than blowing up as ``for a in None``.
        artifacts = tuple(ResolvedArtifact.from_wire(a) for a in (wire.get("artifacts") or ()))
        return cls(
            bundle_id=bundle_id,
            workflow_spec_id=workflow_spec_id,
            artifacts=artifacts,
            extra={k: v for k, v in wire.items() if k not in known},
        )

    @property
    def artifact_count(self) -> int:
        """Number of resolved artifacts in this payload (pure)."""
        return len(self.artifacts)

    @property
    def artifact_fingerprint(self) -> str:
        """Deterministic, **URL-free** content digest of the resolved artifacts.

        Byte-for-byte identical to the server's
        ``DeviceInstallablePayload.artifact_fingerprint`` (parity anchor): hashes
        each artifact's stable content identity — ``(kind, key, content_hash)`` in
        wire order — and deliberately EXCLUDES the volatile ``url`` / ``size_bytes``.
        So two pulls of the same bundle (each minting fresh short-TTL URLs) produce
        the **same** fingerprint — re-pull idempotency is observable, and the digest
        can be cross-checked against the server's push-audit line without a URL ever
        being logged. A pure function of the payload's content identity.
        """
        material = "\n".join(
            f"{artifact.kind}\t{artifact.key}\t{artifact.content_hash or ''}"
            for artifact in self.artifacts
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def log_summary(self) -> dict[str, Any]:
        """A URL-free, key-free summary safe to log / audit.

        The resolved payload carries short-lived signed fetch URLs (and a
        resolver-facing key per artifact); neither must ever reach a log. This is
        the ONLY sanctioned way to reference a payload in a log line — it exposes
        provenance and shape (bundle id, workflow spec id, artifact count, a
        per-kind tally, and the URL-free fingerprint) and structurally cannot carry
        a URL or a key. Builds a fresh dict each call (no mutation of ``self``).

        Serialize the return with :func:`json.dumps` (or an equivalent structured
        emitter), never a bare ``%s``/f-string: ``artifact_kinds`` keys are the raw
        wire ``kind`` values, so a tampered payload could smuggle a newline/CR into
        one and forge a log line if it were string-interpolated. JSON encoding
        neutralises that.
        """
        kind_tally: dict[str, int] = {}
        for artifact in self.artifacts:
            kind_tally[artifact.kind] = kind_tally.get(artifact.kind, 0) + 1
        return {
            "bundle_id": self.bundle_id,
            "workflow_spec_id": self.workflow_spec_id,
            "artifact_count": self.artifact_count,
            "artifact_kinds": kind_tally,
            "artifact_fingerprint": self.artifact_fingerprint,
        }


__all__ = [
    "ArtifactKind",
    "ResolvedArtifact",
    "DeviceInstallablePayload",
]
