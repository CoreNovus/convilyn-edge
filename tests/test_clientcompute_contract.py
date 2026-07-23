"""Unit tests for the client-compute contract (parse + self-verify)."""

from __future__ import annotations

import pytest

from convilyn_edge.clientcompute.contract import (
    DEFAULT_MAX_TOTAL_CHARS,
    MISSING_SENTINEL,
    AnchorsContract,
    ClientComputeRequest,
    build_resume_answer,
    ground_anchors,
)

_FROZEN_PAYLOAD = {
    "nonce": "n-1",
    "required_anchors": ["title", "company"],
    "extractor_prompt": "Extract fields.",
    "extractor_prompt_id": "p-3",
    "file_ids": ["file-a"],
    "anchors_contract": {
        "max_value_chars": 2000,
        "max_total_chars": 20000,
        "value_type": "string",
        "missing_sentinel": "Not specified",
    },
    "strategy": "client_delegated",
}


# ── ClientComputeRequest.from_payload ────────────────────────────────────────


def test_from_payload_parses_required_anchors():
    request = ClientComputeRequest.from_payload(_FROZEN_PAYLOAD)

    assert request.required_anchors == ("title", "company")


def test_from_payload_tolerates_forward_compatible_extra_keys():
    payload = {**_FROZEN_PAYLOAD, "future_key": {"x": 1}}

    request = ClientComputeRequest.from_payload(payload)

    assert request.extra == {"future_key": {"x": 1}}


def test_from_payload_does_not_surface_role_model_map_as_frozen():
    # role_model_map is not device-facing in v1; if a server leaks it, it lands in
    # `extra`, never as a first-class field the SDK depends on.
    payload = {**_FROZEN_PAYLOAD, "role_model_map": {"extractor": "local-qwen3-8b"}}

    request = ClientComputeRequest.from_payload(payload)

    assert "role_model_map" in request.extra


def test_from_payload_missing_frozen_key_raises():
    payload = {k: v for k, v in _FROZEN_PAYLOAD.items() if k != "nonce"}

    with pytest.raises(KeyError):
        ClientComputeRequest.from_payload(payload)


def test_anchors_contract_defaults_when_absent():
    contract = AnchorsContract.from_wire(None)

    assert contract.missing_sentinel == MISSING_SENTINEL


def test_anchors_contract_hardens_explicit_null():
    # An explicit null in the wire must fall back to the default, not become
    # int(None) (raises) or the literal "None" sentinel (silently corrupting).
    contract = AnchorsContract.from_wire({"max_value_chars": None, "missing_sentinel": None})

    assert (contract.max_value_chars, contract.missing_sentinel) == (2000, MISSING_SENTINEL)


# ── ground_anchors: verbatim substring grounding ─────────────────────────────

_SOURCES = {"file-a": "The role is Staff Engineer at Acme Corp in Taipei."}
_CONTRACT = AnchorsContract()


def test_verbatim_substring_is_grounded():
    raw = {"title": "Staff Engineer"}

    grounded = ground_anchors(raw, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == "Staff Engineer"


def test_non_substring_degrades_to_sentinel():
    raw = {"title": "Principal Engineer"}  # not in the source

    grounded = ground_anchors(raw, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == MISSING_SENTINEL


def test_whitespace_collapsed_substring_still_grounds():
    raw = {"title": "Staff   Engineer"}  # extra spaces vs source

    grounded = ground_anchors(raw, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == "Staff   Engineer"


def test_result_is_total_over_required_keys():
    grounded = ground_anchors({}, ["title", "company"], _SOURCES, _CONTRACT)

    assert set(grounded) == {"title", "company"}


# ── ground_anchors: boundary / error ─────────────────────────────────────────


def test_non_string_value_degrades():
    grounded = ground_anchors({"title": 123}, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == MISSING_SENTINEL


def test_empty_value_degrades():
    grounded = ground_anchors({"title": "   "}, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == MISSING_SENTINEL


def test_over_value_cap_degrades():
    tight = AnchorsContract(max_value_chars=5)
    raw = {"title": "Staff Engineer"}  # grounded but 14 > 5 cap

    grounded = ground_anchors(raw, ["title"], _SOURCES, tight)

    assert grounded["title"] == MISSING_SENTINEL


def test_explicit_sentinel_passes_through():
    grounded = ground_anchors({"title": MISSING_SENTINEL}, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == MISSING_SENTINEL


def test_control_chars_are_stripped_before_grounding():
    raw = {"title": "Staff\x00 Engineer"}  # NUL stripped → "Staff Engineer"

    grounded = ground_anchors(raw, ["title"], _SOURCES, _CONTRACT)

    assert grounded["title"] == "Staff Engineer"


# ── ground_anchors: total-cap degradation ────────────────────────────────────


def test_total_cap_degrades_longest_first():
    # Values must exceed the 13-char sentinel for degradation to reduce the total.
    long_val, short_val = "A" * 30, "B" * 20
    sources = {"f": f"{long_val} {short_val}"}
    raw = {"long": long_val, "short": short_val}
    contract = AnchorsContract(max_total_chars=40)  # 30+20=50 > 40; drop "long" → 13+20=33

    grounded = ground_anchors(raw, ["long", "short"], sources, contract)

    assert (grounded["long"], grounded["short"]) == (MISSING_SENTINEL, short_val)


def test_total_cap_skips_degradation_that_would_not_help():
    # Both values are shorter than the sentinel, so degrading can't reduce the
    # total — the values are kept as-is rather than swelled to sentinels.
    sources = {"f": "AAAA BB"}
    raw = {"long": "AAAA", "short": "BB"}
    contract = AnchorsContract(max_total_chars=3)  # unsatisfiable (< one sentinel)

    grounded = ground_anchors(raw, ["long", "short"], sources, contract)

    assert (grounded["long"], grounded["short"]) == ("AAAA", "BB")


def test_total_cap_not_triggered_under_default():
    grounded = ground_anchors({"title": "Staff Engineer"}, ["title"], _SOURCES, AnchorsContract())

    assert len("".join(grounded.values())) <= DEFAULT_MAX_TOTAL_CHARS


# ── build_resume_answer ──────────────────────────────────────────────────────


def test_build_resume_answer_shape():
    answer = build_resume_answer({"title": "X"}, "n-1")

    assert answer == {"anchors": {"title": "X"}, "nonce": "n-1"}


def test_build_resume_answer_copies_anchors():
    anchors = {"title": "X"}
    answer = build_resume_answer(anchors, "n-1")

    anchors["title"] = "Y"  # mutate the caller's dict after

    assert answer["anchors"]["title"] == "X"
