"""``DurableQueue`` — a device-side durable, idempotent, offline buffer.

The offline-first substrate: local rules keep working while the network is down,
unsent events enter a durable queue, and on reconnect they are re-delivered per
policy. When the cloud is unreachable, structured
records (an emitted event, a captured workflow, an offline result) are appended to
a JSONL file; when connectivity returns the caller drains the queue and applies
each record exactly once.

Apply-once alignment with the server reconcile contract: ``enqueue`` is
**idempotent** — a record whose key is already queued is dropped — so a producer
that re-queues (or a partial flush that re-runs) cannot inflate the queue or apply
a record twice. The key is content-addressed (see
:func:`~convilyn_edge.offline.idempotency.derive_idempotency_key` for terminal
results) so the *same* record always yields the *same* key. This mirrors the
server-side ``OfflineResultQueue`` contract (the durable form of which is exactly
this file); the server independently re-derives the key and rejects a mismatch, so
the device is never the sole authority.

The queue is the pure durable store — ``enqueue`` / ``pending`` / ``clear`` — and
leaves the flush *orchestration* (sync or async, one sink or many) to the caller
(see :class:`~convilyn_edge.offline.emitter.EventEmitter` for the async-sink form).
JSONL append is deliberately simple for a single-process producer; swap the store
for SQLite when multi-process producers appear. Stdlib only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Extract the apply-once key from a record. Default: the frozen ``idempotency_key``
#: field (terminal-result records); events pass ``lambda r: r["eventId"]``.
KeyOf = Callable[[Mapping[str, Any]], str]


def _default_key_of(record: Mapping[str, Any]) -> str:
    return record["idempotency_key"]


@dataclass
class DurableQueue:
    """A JSONL-backed, idempotent FIFO queue of pending records.

    ``path`` is the queue file (created on first ``enqueue``). ``key_of`` derives
    each record's apply-once key for dedup.

    Single-writer: an in-memory key index makes ``enqueue`` O(1) instead of
    re-reading the whole file, so buffering N events during an offline window is
    O(N) not O(N²). This assumes THIS instance is the sole writer of ``path``
    (the steady state for one device process); if the file is mutated
    out-of-band, construct a fresh queue.
    """

    path: Path
    key_of: KeyOf = _default_key_of
    _index: set[str] | None = field(default=None, init=False, repr=False, compare=False)

    def _read_lines(self) -> list[dict[str, Any]]:
        """Parse the queue file. Split on ``"\\n"`` ONLY (never ``splitlines()``,
        which also breaks on U+2028/U+2029/U+0085 — legal *unescaped* characters
        inside a JSON string, so a single such char in a payload would otherwise
        shred a valid record and poison the whole queue). A line that fails to
        parse (a torn final append from a crash) is skipped so one bad write
        can't lose every buffered record — bounded blast radius, not total loss.
        """
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn/corrupt line — skip, keep the rest
        return records

    def _keys_index(self) -> set[str]:
        """The cached apply-once key set, built lazily from disk on first use."""
        if self._index is None:
            self._index = {self.key_of(record) for record in self._read_lines()}
        return self._index

    def pending(self) -> list[dict[str, Any]]:
        """Every queued record, in FIFO order. Empty if the file is absent."""
        return self._read_lines()

    def keys(self) -> set[str]:
        """The set of apply-once keys currently queued."""
        return set(self._keys_index())

    def __len__(self) -> int:
        return len(self._read_lines())

    def is_empty(self) -> bool:
        return not self.path.exists() or len(self) == 0

    def enqueue(self, record: Mapping[str, Any]) -> bool:
        """Append ``record`` durably. Idempotent: a record whose key is already
        queued is dropped. Returns ``True`` if it was written, ``False`` if it was
        a duplicate."""
        key = self.key_of(record)
        if key in self._keys_index():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._keys_index().add(key)
        return True

    def extend(self, records: Iterable[Mapping[str, Any]]) -> int:
        """Enqueue many records; return the count actually written (dedup applied)."""
        return sum(1 for record in records if self.enqueue(record))

    def clear(self) -> None:
        """Remove the queue file (truncate to empty)."""
        self.path.unlink(missing_ok=True)
        self._index = set()

    def replace(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Atomically rewrite the queue to exactly ``records`` (FIFO order).

        Written to a temp file, ``fsync``-ed, then ``os.replace``-d over the queue
        — so a crash (or power loss) mid-flush leaves the ORIGINAL queue intact
        rather than a half-written one. The flush re-sends already-sent records on
        retry, which the server's apply-once reconcile deduplicates (that is what
        the idempotency key is for). An empty ``records`` clears the queue.
        """
        materialised = list(records)
        if not materialised:
            self.clear()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in materialised)
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        self._index = {self.key_of(record) for record in materialised}


@dataclass(frozen=True)
class FlushReport:
    """Outcome of a flush pass over the queue.

    ``applied`` — delivered and removed. ``requeued`` — still offline, kept for a
    later flush (FIFO order preserved). ``rejected`` — the sink refused the record
    (a non-offline error) or it was unparseable; retrying would not help, so it is
    dropped rather than blocking the queue behind it (head-of-line safety).
    """

    applied: int
    requeued: int
    rejected: int = 0

    @property
    def clean(self) -> bool:
        """True when everything delivered — nothing re-queued or rejected."""
        return self.requeued == 0 and self.rejected == 0


__all__ = ["KeyOf", "DurableQueue", "FlushReport"]
