"""Unit tests for the device simulator (the first EventSource SPI impl)."""

from __future__ import annotations

import asyncio
import json

import pytest

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.simulator import Scenario, ScenarioEvent, SimulatedSource
from convilyn_edge.spi.source import EventSource, SourceContext

_SCENARIO = {
    "device": {"device_id": "scanner-1", "adapter_id": "sim", "adapter_version": "0.1.0"},
    "site_id": "store-a",
    "events": [
        {"event_type": "device.scan", "event_schema": "s/v1", "data": {"scan": "471"}, "repeat": 2},
        {"event_type": "device.scan", "event_schema": "s/v1", "data": {"scan": "888"}},
    ],
}


def _ctx() -> SourceContext:
    return SourceContext(device_id="scanner-1")


async def _collect(source: SimulatedSource) -> list[EventEnvelope]:
    return [envelope async for envelope in source.start(_ctx())]


# ── scenario parsing ─────────────────────────────────────────────────────────


def test_scenario_parses_device():
    scenario = Scenario.from_dict(_SCENARIO)

    assert scenario.source.device_id == "scanner-1"


def test_scenario_parses_events():
    scenario = Scenario.from_dict(_SCENARIO)

    assert len(scenario.events) == 2


def test_scenario_event_defaults():
    event = ScenarioEvent.from_dict({"event_type": "t", "event_schema": "s"})

    assert (event.delay_ms, event.repeat) == (0, 1)


def test_scenario_load_from_file(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(_SCENARIO), encoding="utf-8")

    scenario = Scenario.load(path)

    assert scenario.site_id == "store-a"


# ── SimulatedSource conforms to the SPI ──────────────────────────────────────


def test_simulated_source_is_an_event_source():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO))

    assert isinstance(source, EventSource)


# ── emission (async) ─────────────────────────────────────────────────────────


async def test_repeat_is_honoured():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO), no_delay=True)

    envelopes = await _collect(source)

    assert len(envelopes) == 3  # first event repeat=2, second repeat=1


async def test_events_are_emitted_in_scenario_order():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO), no_delay=True)

    envelopes = await _collect(source)

    assert [e.data["scan"] for e in envelopes] == ["471", "471", "888"]


async def test_emits_event_envelopes():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO), no_delay=True)

    envelopes = await _collect(source)

    assert isinstance(envelopes[0], EventEnvelope)


async def test_no_delay_still_applies_delay_when_off():
    # A delayed event with no_delay=True must not stall the test (delay ignored).
    scenario = Scenario.from_dict(
        {
            "device": {"device_id": "d"},
            "events": [{"event_type": "t", "event_schema": "s", "delay_ms": 5000}],
        }
    )
    source = SimulatedSource(scenario, no_delay=True)

    envelopes = await _collect(source)

    assert len(envelopes) == 1


async def test_delay_is_applied_when_not_suppressed():
    # delay_ms with no_delay=False actually sleeps (1ms keeps the test fast).
    scenario = Scenario.from_dict(
        {
            "device": {"device_id": "d"},
            "events": [{"event_type": "t", "event_schema": "s", "delay_ms": 1}],
        }
    )
    source = SimulatedSource(scenario, no_delay=False)

    envelopes = await _collect(source)

    assert len(envelopes) == 1


# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_stop_before_start_emits_nothing():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO), no_delay=True)
    await source.stop()

    envelopes = await _collect(source)

    assert envelopes == []


async def test_health_connected_by_default():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO))

    health = await source.health()

    assert health.status == "connected"


async def test_health_disconnected_after_stop():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO))
    await source.stop()

    health = await source.health()

    assert health.status == "disconnected"


async def test_stop_mid_stream_ends_early():
    source = SimulatedSource(Scenario.from_dict(_SCENARIO), no_delay=True)  # 3 events
    seen = []
    async for envelope in source.start(_ctx()):
        seen.append(envelope)
        await source.stop()  # stop after the first

    assert len(seen) == 1


async def test_concurrent_stop_during_delay_suppresses_event():
    # A watchdog task calls stop() while the generator is parked in the delay;
    # the post-sleep guard must suppress the event (S1).
    scenario = Scenario.from_dict(
        {
            "device": {"device_id": "d"},
            "events": [{"event_type": "t", "event_schema": "s", "delay_ms": 100}],
        }
    )
    source = SimulatedSource(scenario, no_delay=False)
    task = asyncio.create_task(_collect(source))
    await asyncio.sleep(0.01)  # let the generator enter its 100ms delay
    await source.stop()

    assert await task == []


async def test_degraded_health_status_propagates():
    scenario = Scenario.from_dict(
        {"device": {"device_id": "d"}, "events": [], "health_status": "degraded"}
    )

    health = await SimulatedSource(scenario).health()

    assert health.status == "degraded"


# ── validation / boundary ────────────────────────────────────────────────────


def test_missing_device_raises_key_error():
    with pytest.raises(KeyError):
        Scenario.from_dict({"events": []})


def test_missing_event_type_raises_key_error():
    with pytest.raises(KeyError):
        ScenarioEvent.from_dict({"event_schema": "s"})


def test_invalid_health_status_raises():
    with pytest.raises(ValueError, match="health_status"):
        Scenario.from_dict({"device": {"device_id": "d"}, "health_status": "bogus"})


async def test_repeat_zero_skips_event():
    scenario = Scenario.from_dict(
        {
            "device": {"device_id": "d"},
            "events": [{"event_type": "t", "event_schema": "s", "repeat": 0}],
        }
    )
    source = SimulatedSource(scenario, no_delay=True)

    envelopes = await _collect(source)

    assert envelopes == []
