"""Unit tests for derive_idempotency_key (the apply-once reconcile contract).

These pin the derivation the SERVER independently re-derives and reconciles
against — a drift here (key format, canonicalisation, prefix) breaks apply-once,
so a golden digest guards it byte-for-byte.
"""

from __future__ import annotations

import pytest

from convilyn_edge.offline.idempotency import KEY_PREFIX, derive_idempotency_key


def _key(**overrides) -> str:
    base = dict(
        job_spec_id="job-1",
        run_status="completed",
        terminal_reason=None,
        payload={"b": 2, "a": 1},
    )
    base.update(overrides)
    return derive_idempotency_key(**base)  # type: ignore[arg-type]


# ── logic / contract pin ─────────────────────────────────────────────────────


def test_golden_digest_pins_server_agreement():
    # Frozen digest for a fixed input — if the canonicalisation drifts from the
    # server's, this fails (and apply-once would silently break in production).
    assert _key() == "idem:653ee599a3866c38bdf6579fc00e62543e20d822f81768f3849d859a14b40369"


def test_key_carries_idem_prefix():
    assert _key().startswith(KEY_PREFIX)


def test_same_result_yields_same_key():
    assert _key() == _key()


def test_payload_key_order_does_not_change_key():
    # sort_keys canonicalisation: {"b":2,"a":1} == {"a":1,"b":2}
    assert _key(payload={"b": 2, "a": 1}) == _key(payload={"a": 1, "b": 2})


# ── boundary / distinctness ──────────────────────────────────────────────────


def test_different_status_yields_different_key():
    assert _key(run_status="completed") != _key(run_status="failed")


def test_different_payload_yields_different_key():
    assert _key(payload={"a": 1}) != _key(payload={"a": 2})


def test_terminal_reason_participates():
    assert _key(terminal_reason=None) != _key(terminal_reason="budget_exceeded")


def test_non_finite_float_payload_fails_loud():
    # allow_nan=False: NaN/Infinity are invalid JSON and would drift the key vs a
    # strict server re-derivation, so they raise rather than produce a bad key.
    with pytest.raises(ValueError):
        _key(payload={"ratio": float("inf")})
