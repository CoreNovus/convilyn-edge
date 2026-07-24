"""Unit tests for DurableQueue (durable, idempotent offline buffer)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from convilyn_edge.offline.queue import DurableQueue, FlushReport


def _queue(tmp_path: Path) -> DurableQueue:
    return DurableQueue(tmp_path / "q.jsonl", key_of=lambda r: r["id"])


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW is POSIX-only")
def test_enqueue_refuses_to_append_through_a_symlink(tmp_path):
    # A planted symlink at the queue path (e.g. on a shared / removable mount) must not
    # let the durable append write THROUGH it to an arbitrary file.
    victim = tmp_path / "victim"
    victim.write_text("original")
    qpath = tmp_path / "q.jsonl"
    try:
        os.symlink(victim, qpath)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink in this environment")
    queue = DurableQueue(qpath, key_of=lambda r: r["id"])

    with pytest.raises(OSError):
        queue.enqueue({"id": "a"})

    assert victim.read_text() == "original"  # the append never followed the link


# ── logic ────────────────────────────────────────────────────────────────────


def test_enqueue_persists_record(tmp_path):
    q = _queue(tmp_path)

    q.enqueue({"id": "a", "v": 1})

    assert q.pending() == [{"id": "a", "v": 1}]


def test_pending_preserves_fifo_order(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})
    q.enqueue({"id": "b"})

    assert [r["id"] for r in q.pending()] == ["a", "b"]


def test_enqueue_returns_true_when_written(tmp_path):
    q = _queue(tmp_path)

    assert q.enqueue({"id": "a"}) is True


# ── idempotency (apply-once) ─────────────────────────────────────────────────


def test_duplicate_key_is_dropped(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a", "v": 1})

    q.enqueue({"id": "a", "v": 2})  # same key

    assert len(q) == 1


def test_duplicate_enqueue_returns_false(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})

    assert q.enqueue({"id": "a"}) is False


def test_extend_counts_only_new(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})

    written = q.extend([{"id": "a"}, {"id": "b"}, {"id": "c"}])

    assert written == 2


# ── boundary / object-state ──────────────────────────────────────────────────


def test_default_key_of_uses_idempotency_key_field(tmp_path):
    # The default keys on the sync-contract "idempotency_key" field.
    q = DurableQueue(tmp_path / "q.jsonl")
    q.enqueue({"idempotency_key": "idem:1", "v": 1})

    assert q.enqueue({"idempotency_key": "idem:1", "v": 2}) is False


def test_absent_file_is_empty(tmp_path):
    q = _queue(tmp_path)

    assert q.is_empty() is True


def test_clear_empties_queue(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})

    q.clear()

    assert q.pending() == []


def test_keys_returns_current_key_set(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})
    q.enqueue({"id": "b"})

    assert q.keys() == {"a", "b"}


# ── replace (atomic rewrite) ─────────────────────────────────────────────────


def test_replace_rewrites_to_survivors(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})
    q.enqueue({"id": "b"})

    q.replace([{"id": "b"}])

    assert [r["id"] for r in q.pending()] == ["b"]


def test_replace_empty_clears(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})

    q.replace([])

    assert q.is_empty() is True


def test_replace_leaves_no_temp_file(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})

    q.replace([{"id": "a"}])

    assert list(tmp_path.glob("*.tmp")) == []


# ── parse robustness (CRITICAL: line-separator poisoning + torn writes) ──────


def test_unicode_line_separators_in_payload_survive(tmp_path):
    # U+2028/U+2029/U+0085 are legal (unescaped) inside a JSON string with
    # ensure_ascii=False; splitlines() would shred the record, split('\n')
    # keeps it whole. Built via escapes so the source stays ASCII.
    text = "line1\u2028line2\u2029end\u0085x"
    q = _queue(tmp_path)
    q.enqueue({"id": "a", "text": text})

    assert q.pending()[0]["text"] == text


def test_line_separator_record_is_not_split_into_two(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a", "text": "x\u2028y"})

    assert len(q.pending()) == 1


def test_torn_trailing_line_is_skipped_not_fatal(tmp_path):
    q = _queue(tmp_path)
    q.enqueue({"id": "a"})
    # Simulate a crash mid-append: a partial, unparseable trailing line.
    with (tmp_path / "q.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id": "b", "partial')

    assert [r["id"] for r in q.pending()] == ["a"]  # valid record survives


# ── FlushReport ──────────────────────────────────────────────────────────────


def test_flush_report_clean_when_nothing_requeued():
    assert FlushReport(applied=3, requeued=0).clean is True


def test_flush_report_not_clean_with_requeue():
    assert FlushReport(applied=1, requeued=2).clean is False
