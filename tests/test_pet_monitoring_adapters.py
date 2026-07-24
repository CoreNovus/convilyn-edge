"""Tests for the pet-monitoring pack adapters (#2793 PR-1) — sources, sink, rules, registry.

The pack lives in ``examples/`` (outside the wheel, removable); these tests import it to prove it
builds and runs against the simulator with only ``convilyn_edge``.
"""

from __future__ import annotations

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.spi.action import ActionAuthorization, RiskLevel
from convilyn_edge.spi.operator import ExecutionContext
from convilyn_edge.spi.source import SourceContext
from examples.pet_monitoring import (
    AdapterRegistry,
    AnomalyVerdict,
    CameraSimSource,
    FeederSimSource,
    LitterSimSource,
    Notification,
    NotifyRequest,
    NotifySink,
    PetAnomalyRules,
    WaterSimSource,
)
from examples.pet_monitoring.events import (
    EVENT_CAMERA_FRAME,
    EVENT_FEEDER_READING,
    EVENT_WATER_READING,
)
from examples.pet_monitoring.operators import PetThresholds

_AUTH = ActionAuthorization(operator="system", reason="test", approved=True)


async def _drain(source, device_id: str = "dev") -> list[EventEnvelope]:
    out: list[EventEnvelope] = []
    async for envelope in source.start(SourceContext(device_id=device_id)):
        out.append(envelope)
    await source.stop()
    return out


# ── simulator sources ────────────────────────────────────────────────────────


async def test_feeder_source_emits_typed_events():
    source = FeederSimSource("dev", [{"level_pct": 80, "jammed": False}, {"level_pct": 5}])

    events = await _drain(source)

    assert [e.event_type for e in events] == [EVENT_FEEDER_READING, EVENT_FEEDER_READING]
    assert events[0].data["level_pct"] == 80 and events[0].source.adapter_id == "feeder-sim"


async def test_each_sensor_source_tags_its_own_event_type():
    cam = await _drain(CameraSimSource("dev", [{"frame_id": "f1", "motion": True}]))
    water = await _drain(WaterSimSource("dev", [{"level_ml": 250}]))
    litter = await _drain(LitterSimSource("dev", [{"uses": 3}]))

    assert cam[0].event_type == EVENT_CAMERA_FRAME
    assert water[0].event_type == EVENT_WATER_READING
    assert litter[0].data["uses"] == 3


async def test_source_health_reports_configured_status():
    source = WaterSimSource("dev", [{"level_ml": 250}], health="degraded")

    health = await source.health()

    assert health.status == "degraded"


async def test_source_health_goes_disconnected_after_stop():
    source = CameraSimSource("dev", [{"frame_id": "f1", "motion": True}])
    await source.stop()

    assert (await source.health()).status == "disconnected"


# ── notify sink (R1 reference) ───────────────────────────────────────────────


def test_notify_sink_is_r1():
    assert NotifySink("me").describe().risk == RiskLevel.R1


async def test_notify_sink_records_and_delivers():
    delivered: list[Notification] = []
    sink = NotifySink("sister", deliver=delivered.append)

    result = await sink.invoke(NotifyRequest("Alert", "2 anomalies"), _AUTH)

    assert result.status == "performed"
    assert result.output == Notification("sister", "Alert", "2 anomalies")
    assert sink.sent == [Notification("sister", "Alert", "2 anomalies")]
    assert delivered == sink.sent  # the deliver transport saw it too


async def test_two_sinks_have_distinct_recipients_and_action_ids():
    me, sister = NotifySink("me"), NotifySink("sister")

    assert me.describe().action_id == "pet.notify.me"
    assert sister.describe().action_id == "pet.notify.sister"


# ── deterministic rule table (offline-first, no LLM) ─────────────────────────


def _event(event_type: str, data: dict) -> EventEnvelope:
    return new_envelope(
        event_type=event_type,
        event_schema="pet.sensor.v1",
        source=EventSourceRef("dev", "sim", "0.1.0"),
        data=data,
    )


def _verdict(event: EventEnvelope, rules: PetAnomalyRules | None = None) -> AnomalyVerdict:
    rules = rules or PetAnomalyRules()
    return rules.execute(event, ExecutionContext(envelope=event)).unwrap()


def test_feeder_jam_is_flagged():
    assert _verdict(_event(EVENT_FEEDER_READING, {"jammed": True})) == AnomalyVerdict(
        True, "feeder_jam", "feeder reports jammed"
    )


def test_low_water_is_flagged():
    v = _verdict(_event(EVENT_WATER_READING, {"level_ml": 40}))
    assert v.anomalous and v.kind == "water_low"


def test_healthy_reading_is_not_anomalous():
    assert _verdict(_event(EVENT_WATER_READING, {"level_ml": 500})).anomalous is False


def test_camera_event_has_no_deterministic_verdict():
    # the model node decides the camera event; the rule table returns "not anomalous" for it
    assert (
        _verdict(_event(EVENT_CAMERA_FRAME, {"frame_id": "f1", "motion": True})).anomalous is False
    )


def test_unknown_event_type_is_not_anomalous():
    assert _verdict(_event("pet.unknown", {"x": 1})).anomalous is False


def test_thresholds_are_tunable_data():
    strict = PetAnomalyRules(PetThresholds(water_low_ml=1000))

    assert _verdict(_event(EVENT_WATER_READING, {"level_ml": 500}), strict).kind == "water_low"


def test_malformed_reading_does_not_crash():
    # a non-numeric level is simply not "below threshold" — never raises
    assert _verdict(_event(EVENT_WATER_READING, {"level_ml": "n/a"})).anomalous is False


def test_bool_reading_is_not_treated_as_a_number():
    # bool is a subclass of int; a boolean in a numeric field must NOT coerce to 0/1 and
    # spuriously flag an anomaly (the classic bool-subclass gotcha, closed for the reference).
    assert _verdict(_event(EVENT_WATER_READING, {"level_ml": False})).anomalous is False


def test_recipient_slug_sanitizes_the_action_id():
    assert NotifySink("my sister").describe().action_id == "pet.notify.my_sister"


def test_pet_thresholds_re_exported_at_package_level():
    from examples.pet_monitoring import PetThresholds as Exported

    assert Exported is PetThresholds


# ── adapter registry (rule 2: bind by key, never a brand branch) ─────────────


def test_registry_resolves_by_key():
    registry = AdapterRegistry().register("notify.primary", lambda: NotifySink("me"))

    sink = registry.resolve("notify.primary")

    assert isinstance(sink, NotifySink) and sink.recipient == "me"


def test_registry_is_immutable():
    base = AdapterRegistry()
    extended = base.register("camera", CameraSimSource)

    assert "camera" not in base and "camera" in extended


def test_registry_passes_args_to_factory():
    registry = AdapterRegistry().register("camera", CameraSimSource)

    source = registry.resolve("camera", "dev", [{"frame_id": "f1", "motion": True}])

    assert isinstance(source, CameraSimSource)


def test_registry_unknown_key_raises_with_known_keys():
    registry = AdapterRegistry().register("a", lambda: None)

    try:
        registry.resolve("missing")
    except KeyError as exc:
        assert "missing" in str(exc) and "'a'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for an unregistered capability")
