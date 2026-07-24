"""Unit tests for installing a verified bundle into the device store."""

from __future__ import annotations

import errno
import hashlib
import sys
from pathlib import Path

import pytest

from convilyn_edge.bundle.install import (
    InstalledArtifact,
    InstallError,
    InstallReport,
    _installed_asset_from_key,
    install_verified_bundle,
    merge_installed_assets,
)
from convilyn_edge.bundle.payload import ResolvedArtifact
from convilyn_edge.bundle.staging import StagedArtifact
from convilyn_edge.bundle.verify import VerifiedArtifact, VerifiedBundle
from convilyn_edge.probe import DeviceCapabilityManifest, InstalledAsset

_HELLO = b"hello"
_HELLO_DIGEST = "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
# A well-formed role _asset_key, mirroring the server's artifact_keys._asset_key:
# tenant/role/binding/capability/model/runtime/quality_tier/adapter. Source-of-truth
# anchor: backend tests/unit/services/edge/test_push_artifact_keys.py (shape
# "acme/role/extractor/llm/qwen3:1.7b/-/standard/-").
_ASSET_KEY = "tenant/role/reason/llm/qwen3_4b/llama_cpp_cuda/standard/-"


def _verified(scratch, *, kind, key, status, content_hash=None, data=_HELLO):
    name = hashlib.sha256(key.encode()).hexdigest()[:12] + ".artifact"
    path = scratch / name
    path.write_bytes(data)
    artifact = ResolvedArtifact(kind=kind, key=key, url="https://h/a", content_hash=content_hash)
    staged = StagedArtifact(artifact=artifact, path=path, size_bytes=len(data))
    return VerifiedArtifact(staged=staged, status=status)


def _bundle(*verified):
    return VerifiedBundle(bundle_id="b", workflow_spec_id="uw_x", artifacts=tuple(verified))


def _spec(scratch):
    return _verified(
        scratch,
        kind="workflow_spec",
        key="tenant/workflow/uw_x/1",
        status="verified",
        content_hash=_HELLO_DIGEST,
    )


def _asset(scratch, *, key=_ASSET_KEY, kind="role_asset"):
    return _verified(scratch, kind=kind, key=key, status="out_of_band")


# ── landing ──────────────────────────────────────────────────────────────────


def test_install_lands_artifact_bytes_in_store(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert report.artifacts[0].path.read_bytes() == _HELLO


def test_install_marks_new_artifact_installed(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert report.artifacts[0].action == "installed"


def test_install_store_path_is_traversal_proof(tmp_path):
    # A tampered key with ".." must not escape the store — the path is hash-addressed.
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    evil = _verified(
        scratch,
        kind="workflow_spec",
        key="../../../etc/passwd",
        status="verified",
        content_hash=_HELLO_DIGEST,
    )
    report = install_verified_bundle(_bundle(evil, _spec(scratch)), store)

    assert store in report.artifacts[0].path.parents


def test_install_rejects_unrecognised_kind(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    mystery = _verified(scratch, kind="mystery_kind", key="t/x", status="out_of_band")

    with pytest.raises(InstallError, match="unrecognised"):
        install_verified_bundle(_bundle(mystery), store)


def test_trusted_core_and_runtime_kinds_land(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    core = _verified(
        scratch,
        kind="trusted_core",
        key="t/trusted_core/1",
        status="deferred_trusted_core",
        content_hash=_HELLO_DIGEST,
    )
    runtime = _verified(
        scratch, kind="runtime", key="t/runtime/jetson/gguf/q4/-", status="out_of_band"
    )
    report = install_verified_bundle(_bundle(_spec(scratch), core, runtime), store)

    assert report.installed_count == 3


@pytest.mark.skipif(sys.platform == "win32", reason="0o700 chmod is POSIX")
def test_store_dir_is_created_private(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    install_verified_bundle(_bundle(_spec(scratch)), store)

    assert (store.stat().st_mode & 0o777) == 0o700


def test_empty_bundle_installs_nothing(tmp_path):
    store = tmp_path / "store"
    report = install_verified_bundle(_bundle(), store)

    assert report.artifacts == ()


# ── idempotency (skip-on-presence; self-heal a corrupted verified file) ──────


def test_reinstall_same_bundle_is_skipped_present(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    install_verified_bundle(_bundle(_spec(scratch)), store)  # installs (consumes staged)
    report = install_verified_bundle(_bundle(_spec(scratch)), store)  # fresh stage, same spec

    assert report.artifacts[0].action == "skipped_present"


def test_reinstall_does_not_rewrite_the_stored_file(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    first = install_verified_bundle(_bundle(_spec(scratch)), store)
    mtime = first.artifacts[0].path.stat().st_mtime_ns
    install_verified_bundle(_bundle(_spec(scratch)), store)

    assert first.artifacts[0].path.stat().st_mtime_ns == mtime


def test_verified_artifact_with_corrupted_store_is_reinstalled(tmp_path):
    # A byte-verified kind is skipped only if the STORED file still matches its pin;
    # a corrupted spec is re-installed (self-healing).
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    first = install_verified_bundle(_bundle(_spec(scratch)), store)
    first.artifacts[0].path.write_bytes(b"tampered")  # corrupt the stored spec
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert report.artifacts[0].action == "installed"


def test_corrupted_store_is_healed_to_correct_bytes(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    first = install_verified_bundle(_bundle(_spec(scratch)), store)
    first.artifacts[0].path.write_bytes(b"tampered")
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert report.artifacts[0].path.read_bytes() == _HELLO


def test_verified_artifact_without_content_hash_fails_loud_on_skip(tmp_path):
    # A (contract-violating) verified artifact with no pin installs once, but the
    # idempotent skip-check has no digest to re-verify against -> fail loud.
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    bad = _verified(
        scratch, kind="workflow_spec", key="t/w/x/1", status="verified", content_hash=None
    )
    install_verified_bundle(_bundle(bad), store)  # 1st: target missing -> installs

    with pytest.raises(InstallError, match="content_hash"):
        install_verified_bundle(_bundle(bad), store)  # 2nd: skip-check hits the guard


def test_out_of_band_asset_skips_on_presence_not_digest(tmp_path):
    # An out-of-band asset (no wire pin) is skipped on presence alone — even if its
    # stored bytes were changed (integrity for these is transport-pinned, not here).
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    first = install_verified_bundle(_bundle(_spec(scratch), _asset(scratch)), store)
    asset_path = first.artifacts[1].path
    asset_path.write_bytes(b"different weights")
    report = install_verified_bundle(_bundle(_spec(scratch), _asset(scratch)), store)

    assert report.artifacts[1].action == "skipped_present"


# ── report counts + installed_assets on a skipped re-run ─────────────────────


def test_report_counts_installed_and_skipped(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    install_verified_bundle(_bundle(_spec(scratch)), store)
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert (report.installed_count, report.skipped_count) == (0, 1)


def test_installed_assets_reported_even_on_skipped_rerun(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    install_verified_bundle(_bundle(_spec(scratch), _asset(scratch)), store)
    report = install_verified_bundle(_bundle(_spec(scratch), _asset(scratch)), store)

    assert [a.capability for a in report.installed_assets] == ["llm"]


# ── cross-filesystem / atomic move failure fails loud ────────────────────────


def test_cross_filesystem_move_raises_install_error(tmp_path, monkeypatch):
    import convilyn_edge.bundle.install as install_module

    def _exdev(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(install_module.os, "replace", _exdev)
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()

    with pytest.raises(InstallError, match="filesystem"):
        install_verified_bundle(_bundle(_spec(scratch)), store)


def test_other_move_error_raises_generic_install_error(tmp_path, monkeypatch):
    import convilyn_edge.bundle.install as install_module

    def _perm(src, dst):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(install_module.os, "replace", _perm)
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()

    with pytest.raises(InstallError, match="could not install"):
        install_verified_bundle(_bundle(_spec(scratch)), store)


def test_unreadable_stored_file_on_skip_raises_install_error(tmp_path, monkeypatch):
    import convilyn_edge.bundle.install as install_module

    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    install_verified_bundle(_bundle(_spec(scratch)), store)

    def _boom(path, flags):
        raise OSError("cannot open")

    monkeypatch.setattr(install_module.os, "open", _boom)  # stored re-hash fails
    with pytest.raises(InstallError, match="could not read"):
        install_verified_bundle(_bundle(_spec(scratch)), store)


# ── installed_assets (key → InstalledAsset parity) ───────────────────────────


def test_installed_asset_from_key_parity_no_adapter():
    # Golden vector pinned against artifact_keys._asset_key composition.
    asset = _installed_asset_from_key(_ASSET_KEY)

    assert asset == InstalledAsset(
        capability="llm",
        model="qwen3_4b",
        quality_tier="standard",
        adapter=None,
        runtime="llama_cpp_cuda",
    )


def test_installed_asset_from_key_maps_dash_adapter_and_runtime_to_none():
    asset = _installed_asset_from_key("t/role/reason/llm/m/-/premium/lora_x")

    assert (asset.runtime, asset.adapter) == (None, "lora_x")


def test_installed_asset_from_key_bad_arity_raises():
    with pytest.raises(InstallError, match="arity"):
        _installed_asset_from_key("too/few/segments")


def test_install_of_malformed_asset_key_fails_loud(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    bad = _verified(scratch, kind="role_asset", key="too/few", status="out_of_band")

    with pytest.raises(InstallError, match="arity"):
        install_verified_bundle(_bundle(_spec(scratch), bad), store)


def test_report_installed_assets_reconstructs_role_and_tool(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    role = _asset(scratch, key=_ASSET_KEY, kind="role_asset")
    tool = _asset(scratch, key="t/tool/ocr:run/ocr/paddle/-/standard/-", kind="tool_asset")
    report = install_verified_bundle(_bundle(_spec(scratch), role, tool), store)

    assert [a.capability for a in report.installed_assets] == ["llm", "ocr"]


def test_report_installed_assets_dedups_identical_bindings():
    # Two artifacts carrying the same InstalledAsset yield ONE entry.
    artifact = ResolvedArtifact(kind="role_asset", key=_ASSET_KEY, url="https://h/a")
    staged = StagedArtifact(artifact=artifact, path=Path("x"), size_bytes=1)
    asset = _installed_asset_from_key(_ASSET_KEY)
    landed = InstalledArtifact(
        verified=VerifiedArtifact(staged=staged, status="out_of_band"),
        path=Path("y"),
        action="installed",
        installed_asset=asset,
    )
    report = InstallReport(bundle_id="b", workflow_spec_id="uw", artifacts=(landed, landed))

    assert len(report.installed_assets) == 1


def test_report_installed_assets_excludes_non_asset_kinds(tmp_path):
    scratch, store = tmp_path / "s", tmp_path / "store"
    scratch.mkdir()
    report = install_verified_bundle(_bundle(_spec(scratch)), store)

    assert report.installed_assets == ()


# ── merge_installed_assets (additive, dedup) ─────────────────────────────────


def _manifest(*assets):
    return DeviceCapabilityManifest(
        device_id="d", silicon="jetson_orin", ram_mb=8192, installed_assets=tuple(assets)
    )


_ASSET_A = InstalledAsset(capability="llm", model="a", runtime="llama_cpp")
_ASSET_B = InstalledAsset(capability="ocr", model="b", runtime="onnx")


def test_merge_appends_new_asset_additively():
    merged = merge_installed_assets(_manifest(_ASSET_A), (_ASSET_B,))

    assert merged.installed_assets == (_ASSET_A, _ASSET_B)


def test_merge_dedups_an_already_present_asset():
    merged = merge_installed_assets(_manifest(_ASSET_A), (_ASSET_A,))

    assert merged.installed_assets == (_ASSET_A,)


def test_merge_mixes_new_and_duplicate_additions_order_stable():
    merged = merge_installed_assets(_manifest(_ASSET_A), (_ASSET_A, _ASSET_B))

    assert merged.installed_assets == (_ASSET_A, _ASSET_B)


def test_merge_of_no_new_assets_returns_manifest_unchanged():
    manifest = _manifest(_ASSET_A)

    assert merge_installed_assets(manifest, (_ASSET_A,)) is manifest


def test_merge_preserves_other_manifest_fields():
    merged = merge_installed_assets(_manifest(_ASSET_A), (_ASSET_B,))

    assert (merged.device_id, merged.silicon, merged.ram_mb) == ("d", "jetson_orin", 8192)
