"""Unit tests for the device-side push-payload mirror (parse + provenance)."""

from __future__ import annotations

import json

import pytest

from convilyn_edge.bundle.payload import DeviceInstallablePayload, ResolvedArtifact

_WIRE_ARTIFACT = {
    "kind": "workflow_spec",
    "key": "tenant-x/uw_abc123/spec.json",
    "url": "https://bucket.example/tenant-x/uw_abc123/spec.json?sig=abc",
    "content_hash": "sha256:aaa",
    "size_bytes": 1234,
}

_WIRE_PAYLOAD = {
    "bundle_id": "bundle-1",
    "workflow_spec_id": "uw_abc123",
    "artifacts": [
        {
            "kind": "workflow_spec",
            "key": "tenant-x/uw_abc123/spec.json",
            "url": "https://bucket.example/a?sig=1",
            "content_hash": "sha256:aaa",
            "size_bytes": 1234,
        },
        {
            "kind": "trusted_core",
            "key": "tenant-x/core/core.tar",
            "url": "https://bucket.example/b?sig=2",
            "content_hash": "sha256:bbb",
        },
        {
            "kind": "role_asset",
            "key": "tenant-x/assets/qwen3.gguf",
            "url": "https://bucket.example/c?sig=3",
        },
    ],
}

#: Parity anchor — must equal the server's
#: ``DeviceInstallablePayload.artifact_fingerprint`` for the same 3 artifacts.
_GOLDEN_FINGERPRINT = "sha256:1d4285182889af70447c317b73584fcb6a1306f6b48f6cbf096226445bf6a943"


# ── ResolvedArtifact.from_wire ───────────────────────────────────────────────


def test_resolved_artifact_from_wire_parses_all_fields():
    artifact = ResolvedArtifact.from_wire(_WIRE_ARTIFACT)

    assert artifact.content_hash == "sha256:aaa"


def test_resolved_artifact_from_wire_defaults_optional_fields_to_none():
    wire = {"kind": "role_asset", "key": "k", "url": "https://h/o"}

    artifact = ResolvedArtifact.from_wire(wire)

    assert (artifact.content_hash, artifact.size_bytes) == (None, None)


def test_resolved_artifact_from_wire_honours_explicit_null_content_hash():
    wire = {**_WIRE_ARTIFACT, "content_hash": None}

    artifact = ResolvedArtifact.from_wire(wire)

    assert artifact.content_hash is None


def test_resolved_artifact_from_wire_coerces_size_bytes_to_int():
    wire = {**_WIRE_ARTIFACT, "size_bytes": "4096"}

    artifact = ResolvedArtifact.from_wire(wire)

    assert artifact.size_bytes == 4096


def test_resolved_artifact_from_wire_preserves_forward_compatible_extra():
    wire = {**_WIRE_ARTIFACT, "future_key": {"x": 1}}

    artifact = ResolvedArtifact.from_wire(wire)

    assert artifact.extra == {"future_key": {"x": 1}}


def test_resolved_artifact_from_wire_missing_url_raises():
    wire = {k: v for k, v in _WIRE_ARTIFACT.items() if k != "url"}

    with pytest.raises(KeyError):
        ResolvedArtifact.from_wire(wire)


def test_resolved_artifact_from_wire_empty_key_raises():
    wire = {**_WIRE_ARTIFACT, "key": ""}

    with pytest.raises(ValueError, match="non-empty"):
        ResolvedArtifact.from_wire(wire)


def test_resolved_artifact_from_wire_explicit_null_kind_raises():
    # str(None) == "None" is truthy, so the null must be caught BEFORE coercion.
    wire = {**_WIRE_ARTIFACT, "kind": None}

    with pytest.raises(ValueError, match="non-null"):
        ResolvedArtifact.from_wire(wire)


def test_resolved_artifact_from_wire_negative_size_bytes_raises():
    wire = {**_WIRE_ARTIFACT, "size_bytes": -1}

    with pytest.raises(ValueError, match="size_bytes"):
        ResolvedArtifact.from_wire(wire)


# ── DeviceInstallablePayload.from_wire ───────────────────────────────────────


def test_payload_from_wire_parses_artifacts_in_order():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    assert [a.kind for a in payload.artifacts] == ["workflow_spec", "trusted_core", "role_asset"]


def test_payload_from_wire_defaults_artifacts_to_empty():
    wire = {"bundle_id": "b", "workflow_spec_id": "uw_x"}

    payload = DeviceInstallablePayload.from_wire(wire)

    assert payload.artifacts == ()


def test_payload_from_wire_tolerates_explicit_null_artifacts():
    wire = {"bundle_id": "b", "workflow_spec_id": "uw_x", "artifacts": None}

    payload = DeviceInstallablePayload.from_wire(wire)

    assert payload.artifacts == ()


def test_payload_from_wire_preserves_forward_compatible_extra():
    wire = {**_WIRE_PAYLOAD, "expires_at": "2026-01-01T00:00:00Z"}

    payload = DeviceInstallablePayload.from_wire(wire)

    assert payload.extra == {"expires_at": "2026-01-01T00:00:00Z"}


def test_payload_from_wire_missing_bundle_id_raises():
    wire = {k: v for k, v in _WIRE_PAYLOAD.items() if k != "bundle_id"}

    with pytest.raises(KeyError):
        DeviceInstallablePayload.from_wire(wire)


def test_payload_from_wire_empty_workflow_spec_id_raises():
    wire = {**_WIRE_PAYLOAD, "workflow_spec_id": ""}

    with pytest.raises(ValueError, match="non-empty"):
        DeviceInstallablePayload.from_wire(wire)


def test_payload_from_wire_explicit_null_bundle_id_raises():
    wire = {**_WIRE_PAYLOAD, "bundle_id": None}

    with pytest.raises(ValueError, match="non-null"):
        DeviceInstallablePayload.from_wire(wire)


def test_payload_artifact_count():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    assert payload.artifact_count == 3


# ── artifact_fingerprint (URL-free parity anchor) ────────────────────────────


def test_artifact_fingerprint_matches_server_golden_vector():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    assert payload.artifact_fingerprint == _GOLDEN_FINGERPRINT


def test_artifact_fingerprint_is_url_free_stable_across_fresh_urls():
    # A re-pull mints fresh short-TTL URLs; the fingerprint must not move.
    repull = {
        **_WIRE_PAYLOAD,
        "artifacts": [
            {**art, "url": f"https://bucket.example/rotated?sig={i}"}
            for i, art in enumerate(_WIRE_PAYLOAD["artifacts"])
        ],
    }

    assert DeviceInstallablePayload.from_wire(repull).artifact_fingerprint == _GOLDEN_FINGERPRINT


def test_artifact_fingerprint_changes_when_content_hash_changes():
    tampered = {
        **_WIRE_PAYLOAD,
        "artifacts": [
            {**_WIRE_PAYLOAD["artifacts"][0], "content_hash": "sha256:tampered"},
            *_WIRE_PAYLOAD["artifacts"][1:],
        ],
    }

    assert DeviceInstallablePayload.from_wire(tampered).artifact_fingerprint != _GOLDEN_FINGERPRINT


# ── log_summary (URL-free, key-free) ─────────────────────────────────────────


def test_log_summary_tallies_kinds():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    assert payload.log_summary()["artifact_kinds"] == {
        "workflow_spec": 1,
        "trusted_core": 1,
        "role_asset": 1,
    }


def test_log_summary_never_carries_a_url():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    dumped = json.dumps(payload.log_summary())

    assert "bucket.example" not in dumped


def test_log_summary_never_carries_a_resolver_key():
    payload = DeviceInstallablePayload.from_wire(_WIRE_PAYLOAD)

    dumped = json.dumps(payload.log_summary())

    assert "tenant-x/" not in dumped
