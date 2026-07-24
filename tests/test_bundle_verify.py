"""Unit tests for staged-bundle digest verification (fail-loud, no landing)."""

from __future__ import annotations

import pytest

from convilyn_edge.bundle.payload import ResolvedArtifact
from convilyn_edge.bundle.staging import StagedArtifact, StagedBundle
from convilyn_edge.bundle.verify import (
    DigestMismatchError,
    IncompleteBundleError,
    MissingDigestError,
    StagedArtifactUnreadableError,
    UnsupportedDigestError,
    VerificationError,
    VerifiedBundle,
    verify_staged_artifact,
    verify_staged_bundle,
)

# sha256(b"hello") — the golden digest of the bytes every staged file below carries.
_HELLO_DIGEST = "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
# A well-formed sha256 digest that does NOT match b"hello" (a tamper / mismatch pin).
_WRONG_DIGEST = "sha256:" + "0" * 64


def _staged(tmp_path, *, kind, content_hash, data=b"hello", name="0000.artifact"):
    path = tmp_path / name
    path.write_bytes(data)
    artifact = ResolvedArtifact(kind=kind, key="k", url="https://h/a", content_hash=content_hash)
    return StagedArtifact(artifact=artifact, path=path, size_bytes=len(data))


# ── fetched-bytes digest (workflow_spec + fail-closed default) ───────────────


def test_workflow_spec_matching_digest_is_verified(tmp_path):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST)

    assert verify_staged_artifact(staged) == "verified"


def test_workflow_spec_digest_mismatch_raises(tmp_path):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=_WRONG_DIGEST)

    with pytest.raises(DigestMismatchError, match="mismatch"):
        verify_staged_artifact(staged)


def test_digest_comparison_is_case_insensitive(tmp_path):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST.upper())

    assert verify_staged_artifact(staged) == "verified"


def test_unknown_kind_carrying_a_digest_is_verified_fail_closed(tmp_path):
    # Forward-compat: a future kind that starts pinning a digest is checked, not
    # silently skipped.
    staged = _staged(tmp_path, kind="future_kind", content_hash=_HELLO_DIGEST)

    assert verify_staged_artifact(staged) == "verified"


def test_non_sha256_digest_is_rejected(tmp_path):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash="md5:abcdef")

    with pytest.raises(UnsupportedDigestError, match="sha256"):
        verify_staged_artifact(staged)


# ── deferred trusted_core (S2 source-tree digest, not byte-hashed here) ──────


def test_trusted_core_is_deferred_not_byte_hashed(tmp_path):
    # Its content_hash is over the device's own source tree, so byte-hashing the
    # fetched object would mismatch — it must NOT be checked here, and a wrong
    # (for these bytes) digest must not raise.
    staged = _staged(tmp_path, kind="trusted_core", content_hash=_WRONG_DIGEST)

    assert verify_staged_artifact(staged) == "deferred_trusted_core"


# ── out-of-band assets / runtimes (no wire digest) ──────────────────────────


@pytest.mark.parametrize("kind", ["role_asset", "tool_asset", "runtime"])
def test_asset_without_digest_is_out_of_band(tmp_path, kind):
    staged = _staged(tmp_path, kind=kind, content_hash=None)

    assert verify_staged_artifact(staged) == "out_of_band"


def test_unknown_kind_without_digest_is_out_of_band(tmp_path):
    staged = _staged(tmp_path, kind="future_kind", content_hash=None)

    assert verify_staged_artifact(staged) == "out_of_band"


# ── fail-closed on a stripped required digest ────────────────────────────────


def test_workflow_spec_missing_digest_fails_closed(tmp_path):
    # A null content_hash on a digest-required kind must NOT downgrade to "nothing
    # to verify" — that would be the "strip the digest to skip verification" attack.
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=None)

    with pytest.raises(MissingDigestError, match="content_hash"):
        verify_staged_artifact(staged)


def test_trusted_core_missing_digest_fails_closed(tmp_path):
    staged = _staged(tmp_path, kind="trusted_core", content_hash=None)

    with pytest.raises(MissingDigestError):
        verify_staged_artifact(staged)


# ── streaming digest correctness (multi-chunk) ───────────────────────────────


def test_large_multichunk_file_hashes_correctly(tmp_path, monkeypatch):
    import convilyn_edge.bundle.verify as verify_module

    monkeypatch.setattr(verify_module, "_DIGEST_CHUNK_SIZE", 4)  # force many chunks
    data = b"hello"
    expected = "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=expected, data=data)

    assert verify_staged_artifact(staged) == "verified"


# ── verify_staged_bundle (all-or-nothing) ────────────────────────────────────


def _bundle(*staged):
    return StagedBundle(bundle_id="b", workflow_spec_id="uw_x", artifacts=tuple(staged))


def test_bundle_all_pass_returns_verified_bundle(tmp_path):
    bundle = _bundle(
        _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST, name="0.artifact"),
        _staged(tmp_path, kind="role_asset", content_hash=None, name="1.artifact"),
        _staged(tmp_path, kind="trusted_core", content_hash=_WRONG_DIGEST, name="2.artifact"),
    )

    result = verify_staged_bundle(bundle)

    assert isinstance(result, VerifiedBundle)


def test_bundle_reports_per_artifact_status(tmp_path):
    bundle = _bundle(
        _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST, name="0.artifact"),
        _staged(tmp_path, kind="role_asset", content_hash=None, name="1.artifact"),
        _staged(tmp_path, kind="trusted_core", content_hash=_WRONG_DIGEST, name="2.artifact"),
    )

    statuses = [a.status for a in verify_staged_bundle(bundle).artifacts]

    assert statuses == ["verified", "out_of_band", "deferred_trusted_core"]


def test_bundle_verified_count(tmp_path):
    bundle = _bundle(
        _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST, name="0.artifact"),
        _staged(tmp_path, kind="role_asset", content_hash=None, name="1.artifact"),
    )

    assert verify_staged_bundle(bundle).verified_count == 1


def test_bundle_one_mismatch_raises_whole_bundle(tmp_path):
    bundle = _bundle(
        _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST, name="0.artifact"),
        _staged(tmp_path, kind="future_kind", content_hash=_WRONG_DIGEST, name="1.artifact"),
    )

    with pytest.raises(DigestMismatchError):
        verify_staged_bundle(bundle)


def test_bundle_carries_provenance(tmp_path):
    bundle = _bundle(_staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST))

    result = verify_staged_bundle(bundle)

    assert (result.bundle_id, result.workflow_spec_id) == ("b", "uw_x")


# ── completeness: a bundle must contain a workflow_spec ───────────────────────


def test_bundle_without_workflow_spec_is_incomplete(tmp_path):
    # The "drop the whole spec artifact" tamper: only out-of-band assets remain.
    bundle = _bundle(_staged(tmp_path, kind="role_asset", content_hash=None))

    with pytest.raises(IncompleteBundleError, match="workflow_spec"):
        verify_staged_bundle(bundle)


def test_empty_bundle_is_incomplete(tmp_path):
    with pytest.raises(IncompleteBundleError):
        verify_staged_bundle(_bundle())


# ── empty / whitespace content_hash fails closed (not just None) ──────────────


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_digest_on_required_kind_fails_closed(tmp_path, blank):
    for kind in ("workflow_spec", "trusted_core"):
        staged = _staged(tmp_path, kind=kind, content_hash=blank, name=f"{kind}.artifact")
        with pytest.raises(MissingDigestError):
            verify_staged_artifact(staged)


def test_blank_digest_on_asset_is_out_of_band(tmp_path):
    staged = _staged(tmp_path, kind="role_asset", content_hash="")

    assert verify_staged_artifact(staged) == "out_of_band"


# ── malformed digest shape → UnsupportedDigestError (not "mismatch") ─────────


@pytest.mark.parametrize(
    "bad", ["sha256:xyz", "sha256:0", "sha256:" + "a" * 63, "sha1:" + "a" * 40]
)
def test_malformed_digest_shape_is_unsupported(tmp_path, bad):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=bad)

    with pytest.raises(UnsupportedDigestError):
        verify_staged_artifact(staged)


# ── unreadable staged file → VerificationError (uniform contract) ────────────


def test_vanished_staged_file_raises_verification_error(tmp_path):
    staged = _staged(tmp_path, kind="workflow_spec", content_hash=_HELLO_DIGEST)
    staged.path.unlink()  # gone between stage and verify

    with pytest.raises(StagedArtifactUnreadableError):
        verify_staged_artifact(staged)


# ── error hierarchy + no credential leakage (public contract pins) ───────────


@pytest.mark.parametrize(
    "err",
    [
        DigestMismatchError,
        UnsupportedDigestError,
        MissingDigestError,
        StagedArtifactUnreadableError,
        IncompleteBundleError,
    ],
)
def test_all_errors_subclass_verification_error(err):
    assert issubclass(err, VerificationError)


def test_mismatch_message_carries_no_url_or_key(tmp_path):
    artifact = ResolvedArtifact(
        kind="workflow_spec",
        key="tenant-x/resolver-secret-key",
        url="https://bucket.example/signed?sig=SECRET",
        content_hash=_WRONG_DIGEST,
    )
    path = tmp_path / "s.artifact"
    path.write_bytes(b"hello")
    staged = StagedArtifact(artifact=artifact, path=path, size_bytes=5)

    try:
        verify_staged_artifact(staged)
    except DigestMismatchError as exc:
        message = str(exc)
        assert "bucket.example" not in message and "resolver-secret-key" not in message


def test_error_message_escapes_control_chars_in_kind(tmp_path):
    # A tampered `kind` with a newline must not forge a second log line — !r escapes
    # it. `kind` is what the error interpolates, so the newline lives there.
    staged = _staged(tmp_path, kind="ev\nil", content_hash="not-a-digest")

    with pytest.raises(UnsupportedDigestError) as excinfo:
        verify_staged_artifact(staged)

    assert "\n" not in str(excinfo.value)
