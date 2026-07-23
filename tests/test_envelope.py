"""Unit tests for the Device Event Envelope (``convilyn_edge.envelope``)."""

from __future__ import annotations

import re

import pytest

from convilyn_edge.envelope import (
    SPEC_VERSION,
    Correlation,
    EventEnvelope,
    EventSourceRef,
    new_envelope,
)

_SOURCE = EventSourceRef("scanner-8f-03", "opos-scanner", "0.3.1")


def _make(**overrides) -> EventEnvelope:
    base = dict(
        event_id="evt-1",
        event_type="device.barcode.scan.received",
        event_schema="convilyn://schemas/barcode-scan/v1",
        source=_SOURCE,
        time="2026-07-18T10:15:32.240000+00:00",
        data={"scanData": "4711234567890"},
    )
    base.update(overrides)
    return EventEnvelope(**base)  # type: ignore[arg-type]


# ── logic ────────────────────────────────────────────────────────────────────


def test_to_wire_uses_camelcase_keys():
    wire = _make().to_wire()

    assert set(wire) == {
        "specVersion",
        "eventId",
        "eventType",
        "eventSchema",
        "source",
        "time",
        "correlation",
        "data",
        "metadata",
    }


def test_source_ref_wire_is_camelcase():
    wire = _make().to_wire()

    assert wire["source"] == {
        "deviceId": "scanner-8f-03",
        "adapterId": "opos-scanner",
        "adapterVersion": "0.3.1",
    }


def test_round_trip_is_identity():
    env = _make(correlation=Correlation(trace_id="abc", sequence=43))

    assert EventEnvelope.from_wire(env.to_wire()) == env


def test_correlation_wire_omits_none_fields():
    wire = Correlation(trace_id="abc").to_wire()

    assert wire == {"traceId": "abc"}


def test_correlation_wire_includes_all_set_fields():
    wire = Correlation(trace_id="abc", session_id="txn-1", sequence=7).to_wire()

    assert wire == {"traceId": "abc", "sessionId": "txn-1", "sequence": 7}


def test_correlation_from_wire_handles_missing_object():
    assert Correlation.from_wire(None) == Correlation()


# ── boundary ─────────────────────────────────────────────────────────────────


def test_default_spec_version_applied():
    assert _make().spec_version == SPEC_VERSION


def test_from_wire_defaults_missing_spec_version():
    wire = _make().to_wire()
    del wire["specVersion"]

    assert EventEnvelope.from_wire(wire).spec_version == SPEC_VERSION


def test_empty_data_round_trips():
    env = _make(data={})

    assert EventEnvelope.from_wire(env.to_wire()).data == {}


# ── error ────────────────────────────────────────────────────────────────────


def test_from_wire_missing_required_key_raises():
    wire = _make().to_wire()
    del wire["eventId"]

    with pytest.raises(KeyError):
        EventEnvelope.from_wire(wire)


# ── object-state (immutability) ──────────────────────────────────────────────


def test_data_is_read_only_view():
    env = _make(data={"a": 1})

    with pytest.raises(TypeError):  # MappingProxyType rejects item assignment
        env.data["a"] = 2  # type: ignore[index]


def test_data_is_defensively_copied_from_caller_dict():
    caller_dict = {"a": 1}
    env = _make(data=caller_dict)

    caller_dict["a"] = 999  # mutate the ORIGINAL after construction

    assert env.data["a"] == 1


def test_nested_data_is_deep_copied_from_caller():
    # A shallow copy would share the nested dict; deep-copy decouples it fully.
    caller_dict = {"outer": {"inner": 1}}
    env = _make(data=caller_dict)

    caller_dict["outer"]["inner"] = 999  # mutate a NESTED value after construction

    assert env.data["outer"]["inner"] == 1


def test_envelope_field_reassignment_is_frozen():
    env = _make()

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        env.event_id = "other"  # type: ignore[misc]


# ── new_envelope factory ─────────────────────────────────────────────────────


def test_new_envelope_mints_hex_event_id():
    env = new_envelope(
        event_type="t",
        event_schema="s",
        source=_SOURCE,
        data={},
    )

    assert re.fullmatch(r"[0-9a-f]{32}", env.event_id)


def test_new_envelope_sets_utc_iso_time():
    env = new_envelope(event_type="t", event_schema="s", source=_SOURCE, data={})

    assert env.time.endswith("+00:00")


def test_new_envelope_mints_unique_ids():
    a = new_envelope(event_type="t", event_schema="s", source=_SOURCE, data={})
    b = new_envelope(event_type="t", event_schema="s", source=_SOURCE, data={})

    assert a.event_id != b.event_id


def test_new_envelope_defaults_empty_correlation():
    env = new_envelope(event_type="t", event_schema="s", source=_SOURCE, data={})

    assert env.correlation == Correlation()
