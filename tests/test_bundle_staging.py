"""Unit tests for staging a whole payload to scratch (download, not install)."""

from __future__ import annotations

import os

import pytest

from convilyn_edge.bundle.fetch import FetchError
from convilyn_edge.bundle.payload import DeviceInstallablePayload, ResolvedArtifact
from convilyn_edge.bundle.staging import _staging_name, stage_bundle


def _one_artifact_payload(*, size_bytes: int | None = None) -> DeviceInstallablePayload:
    return DeviceInstallablePayload(
        bundle_id="b",
        workflow_spec_id="uw_x",
        artifacts=(
            ResolvedArtifact(kind="role_asset", key="k", url="https://h/a", size_bytes=size_bytes),
        ),
    )


class _MapFetcher:
    """A fake fetcher: writes each URL's canned bytes, optionally failing on one."""

    def __init__(self, contents: dict[str, bytes], fail_on: str | None = None):
        self._contents = contents
        self._fail_on = fail_on
        self.calls: list[str] = []

    def fetch(self, url: str, sink) -> int:
        self.calls.append(url)
        if url == self._fail_on:
            sink.write(b"partial")  # a torn write that must not survive
            raise FetchError("boom")
        data = self._contents[url]
        sink.write(data)
        return len(data)


def _payload() -> DeviceInstallablePayload:
    return DeviceInstallablePayload(
        bundle_id="bundle-1",
        workflow_spec_id="uw_x",
        artifacts=(
            ResolvedArtifact(kind="workflow_spec", key="k1", url="https://h/a"),
            ResolvedArtifact(kind="role_asset", key="k2", url="https://h/b"),
        ),
    )


# ── _staging_name ─────────────────────────────────────────────────────────────


def test_staging_name_slugs_kind_and_pads_index():
    assert _staging_name(2, "workflow_spec") == "0002-workflow_spec.artifact"


def test_staging_name_falls_back_when_kind_is_unslugifiable():
    assert _staging_name(0, "///").startswith("0000-artifact")


def test_staging_name_is_unique_per_index_for_same_kind():
    assert _staging_name(0, "role_asset") != _staging_name(1, "role_asset")


# ── stage_bundle happy path ──────────────────────────────────────────────────


def test_stage_bundle_writes_each_artifact_to_disk(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"spec-bytes", "https://h/b": b"asset-bytes"})

    staged = stage_bundle(_payload(), fetcher, tmp_path)

    assert staged.artifacts[0].path.read_bytes() == b"spec-bytes"


def test_stage_bundle_records_written_size(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"spec-bytes", "https://h/b": b"asset-bytes"})

    staged = stage_bundle(_payload(), fetcher, tmp_path)

    assert staged.artifacts[1].size_bytes == len(b"asset-bytes")


def test_stage_bundle_preserves_payload_order(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"x", "https://h/b": b"y"})

    staged = stage_bundle(_payload(), fetcher, tmp_path)

    assert [s.artifact.kind for s in staged.artifacts] == ["workflow_spec", "role_asset"]


def test_stage_bundle_carries_bundle_provenance(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"x", "https://h/b": b"y"})

    staged = stage_bundle(_payload(), fetcher, tmp_path)

    assert (staged.bundle_id, staged.workflow_spec_id) == ("bundle-1", "uw_x")


def test_stage_bundle_creates_missing_dest_dir(tmp_path):
    dest = tmp_path / "nested" / "scratch"
    fetcher = _MapFetcher({"https://h/a": b"x", "https://h/b": b"y"})

    staged = stage_bundle(_payload(), fetcher, dest)

    assert staged.artifact_count == 2


# ── stage_bundle fail-loud / torn-file safety ────────────────────────────────


def test_stage_bundle_propagates_fetch_failure(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"x"}, fail_on="https://h/b")

    with pytest.raises(FetchError, match="boom"):
        stage_bundle(_payload(), fetcher, tmp_path)


def test_stage_bundle_removes_the_partial_file_on_failure(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"x"}, fail_on="https://h/b")

    with pytest.raises(FetchError):
        stage_bundle(_payload(), fetcher, tmp_path)

    assert not (tmp_path / "0001-role_asset.artifact").exists()


def test_stage_bundle_keeps_already_staged_files_on_later_failure(tmp_path):
    fetcher = _MapFetcher({"https://h/a": b"x"}, fail_on="https://h/b")

    with pytest.raises(FetchError):
        stage_bundle(_payload(), fetcher, tmp_path)

    assert (tmp_path / "0000-workflow_spec.artifact").read_bytes() == b"x"


def test_stage_bundle_of_empty_payload_is_a_noop(tmp_path):
    payload = DeviceInstallablePayload(bundle_id="b", workflow_spec_id="uw_x")

    staged = stage_bundle(payload, _MapFetcher({}), tmp_path)

    assert staged.artifacts == ()


# ── size ceiling (advertised size_bytes / explicit override) ─────────────────


def test_stage_bundle_bounds_artifact_by_advertised_size(tmp_path):
    payload = _one_artifact_payload(size_bytes=3)  # advertised 3, fetcher streams 5
    fetcher = _MapFetcher({"https://h/a": b"12345"})

    with pytest.raises(FetchError, match="ceiling"):
        stage_bundle(payload, fetcher, tmp_path)


def test_stage_bundle_removes_partial_on_ceiling_breach(tmp_path):
    payload = _one_artifact_payload(size_bytes=3)
    fetcher = _MapFetcher({"https://h/a": b"12345"})

    with pytest.raises(FetchError):
        stage_bundle(payload, fetcher, tmp_path)

    assert not (tmp_path / "0000-role_asset.artifact").exists()


def test_stage_bundle_within_advertised_size_succeeds(tmp_path):
    payload = _one_artifact_payload(size_bytes=5)  # exactly 5 == the 5 streamed bytes
    fetcher = _MapFetcher({"https://h/a": b"12345"})

    staged = stage_bundle(payload, fetcher, tmp_path)

    assert staged.artifacts[0].size_bytes == 5


def test_stage_bundle_max_artifact_bytes_overrides_advertised_size(tmp_path):
    payload = _one_artifact_payload(size_bytes=1000)  # generous advertised size…
    fetcher = _MapFetcher({"https://h/a": b"12345"})

    with pytest.raises(FetchError, match="ceiling"):  # …but the explicit cap of 2 wins
        stage_bundle(payload, fetcher, tmp_path, max_artifact_bytes=2)


def test_stage_bundle_unbounded_when_no_size_and_no_override(tmp_path):
    payload = _one_artifact_payload(size_bytes=None)
    fetcher = _MapFetcher({"https://h/a": b"unbounded-ok"})

    staged = stage_bundle(payload, fetcher, tmp_path)

    assert staged.artifacts[0].size_bytes == len(b"unbounded-ok")


# ── scratch-dir hardening (M4) ───────────────────────────────────────────────


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX-only")
def test_stage_bundle_refuses_to_write_through_a_symlink(tmp_path):
    dest = tmp_path / "scratch"
    dest.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = dest / "0000-role_asset.artifact"
    try:
        os.symlink(victim, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink in this environment")
    fetcher = _MapFetcher({"https://h/a": b"attacker-bytes"})

    with pytest.raises((FetchError, OSError)):
        stage_bundle(_one_artifact_payload(), fetcher, dest)

    assert victim.read_text() == "original"  # the write never followed the link
