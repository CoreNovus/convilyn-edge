"""Content-addressed idempotency keys for offline terminal results.

The device buffers an offline terminal result and flushes it when connectivity
returns; the cloud reconciles it **exactly once**. This module derives the
apply-once key the reconcile is keyed on — content-addressed, so the *same*
result always yields the *same* key (a retried flush is a no-op) and two distinct
results never collide.

**This mirrors the server's ``derive_idempotency_key`` by shape, byte-for-byte**
(the canonical JSON: ``sort_keys=True``, compact ``(",", ":")`` separators,
``ensure_ascii=False``; SHA-256; the ``"idem:"`` prefix; the **device clock is
deliberately excluded**). The SDK imports nothing from the backend — the
observability/reconcile convention is shared, the code is not — but the derivation
must stay identical: the server re-derives the key from the result's OWN content
and rejects a mismatch, so a device key that drifts from this formula is rejected.
Contract source of truth: the server's edge-reconcile contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

KEY_PREFIX = "idem:"


def derive_idempotency_key(
    *,
    job_spec_id: str,
    run_status: str,
    terminal_reason: str | None,
    payload: Mapping[str, Any],
) -> str:
    """Return the content-addressed apply-once key for one offline terminal result.

    Deterministic SHA-256 over the canonical ``(job_spec_id, run_status,
    terminal_reason, payload)``. A digest, not a secret. The device clock is
    excluded on purpose so re-flushing reconciles to a no-op.
    """
    canonical = json.dumps(
        {
            "job_spec_id": job_spec_id,
            "run_status": run_status,
            "terminal_reason": terminal_reason,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        # Fail loud on NaN/Infinity: bare non-finite tokens are invalid JSON, so a
        # strict server re-derivation would format them differently → silent key
        # drift, the exact failure this key guards against.
        allow_nan=False,
    )
    return KEY_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["KEY_PREFIX", "derive_idempotency_key"]
