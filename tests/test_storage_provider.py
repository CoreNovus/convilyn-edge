"""Unit tests for the device storage-selection seam (#2791 PR②)."""

from __future__ import annotations

from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.storage.provider import (
    LocalStorageProvider,
    StorageProvider,
    _safe_store_filename,
)

# ── _safe_store_filename (traversal-proof + collision-free) ──────────────────


def test_clean_short_name_keeps_its_readable_form():
    assert _safe_store_filename("audit") == "audit.jsonl"


def test_lossy_name_gets_a_disambiguating_digest():
    # A separator makes the reduction lossy → a hash suffix disambiguates it.
    name = _safe_store_filename("a/b")

    assert name.startswith("a_b-") and "/" not in name and name.endswith(".jsonl")


def test_traversal_name_is_reduced_to_a_single_in_root_component():
    name = _safe_store_filename("../../etc/passwd")

    assert "/" not in name and ".." not in name and name.endswith(".jsonl")


def test_unslugifiable_name_falls_back_to_store():
    assert _safe_store_filename("...").startswith("store-")


def test_overlong_name_is_bounded():
    # Truncated → lossy → digest-suffixed; the slug body is capped at 48.
    name = _safe_store_filename("x" * 100)

    assert name.startswith("x" * 48 + "-") and len(name.removesuffix(".jsonl")) <= 61


def test_distinct_lossy_names_do_not_collide():
    assert _safe_store_filename("x" * 100) != _safe_store_filename("x" * 64 + "y" * 40)


def test_strip_collision_is_disambiguated():
    # ".hidden" strips to "hidden" — must not share a file with a real "hidden".
    assert _safe_store_filename(".hidden") != _safe_store_filename("hidden")


def test_windows_reserved_name_is_prefixed():
    assert _safe_store_filename("con").startswith("_con")


# ── LocalStorageProvider ─────────────────────────────────────────────────────


def test_provider_reports_its_tier(tmp_path):
    assert LocalStorageProvider(tmp_path, tier="removable").tier == "removable"


def test_root_property_exposes_the_directory(tmp_path):
    assert LocalStorageProvider(tmp_path, tier="internal").root == tmp_path


def test_durable_store_is_a_durable_queue(tmp_path):
    store = LocalStorageProvider(tmp_path, tier="internal").durable_store("audit")

    assert isinstance(store, DurableQueue)


def test_durable_store_lands_under_root(tmp_path):
    store = LocalStorageProvider(tmp_path, tier="internal").durable_store("audit")

    assert store.path == tmp_path / "audit.jsonl"


def test_durable_store_is_traversal_proof(tmp_path):
    store = LocalStorageProvider(tmp_path, tier="internal").durable_store("../../etc/passwd")

    assert tmp_path in store.path.parents


def test_same_name_returns_the_same_live_store(tmp_path):
    # Memoized — a fresh handle on the same file would break DurableQueue's apply-once.
    provider = LocalStorageProvider(tmp_path, tier="media_store")

    assert provider.durable_store("s") is provider.durable_store("s")


def test_distinct_long_names_get_distinct_stores(tmp_path):
    provider = LocalStorageProvider(tmp_path, tier="internal")

    assert provider.durable_store("x" * 100).path != provider.durable_store("x" * 64 + "y").path


def test_durable_store_does_not_eagerly_create_root(tmp_path):
    root = tmp_path / "not-yet"
    LocalStorageProvider(root, tier="internal").durable_store("s")  # builds path, no write

    assert not root.exists()


def test_durable_store_round_trips_records(tmp_path):
    store = LocalStorageProvider(tmp_path, tier="internal").durable_store("audit")

    store.enqueue({"idempotency_key": "k1", "value": 42})

    assert store.pending() == [{"idempotency_key": "k1", "value": 42}]


def test_durable_store_honours_a_custom_key_of(tmp_path):
    store = LocalStorageProvider(tmp_path, tier="internal").durable_store(
        "events", key_of=lambda r: r["eventId"]
    )

    store.enqueue({"eventId": "e1"})
    store.enqueue({"eventId": "e1"})  # same key → idempotent drop

    assert len(store.pending()) == 1


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_local_provider_satisfies_the_protocol(tmp_path):
    assert isinstance(LocalStorageProvider(tmp_path, tier="internal"), StorageProvider)


def test_object_is_not_a_storage_provider():
    assert not isinstance(object(), StorageProvider)
