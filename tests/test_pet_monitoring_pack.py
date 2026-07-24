"""Acceptance tests for the pet-monitoring pack assembly (#2793 PR-3) — the charter's three rules.

The declarative DAG is driven end-to-end against the simulator; the assertions pin the moat
charter's three iron rules:
  ① the pack is a DECLARATION (branches run via the router primitive, not hand-wired if/else),
  ② device capabilities bind via the capability registry + a capability-negotiated resolver,
  ③ the decision path consumes a grounded ModelOperator.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from convilyn_edge.envelope import EventEnvelope, EventSourceRef, new_envelope
from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.probe import DeviceCapabilityManifest, InstalledAsset
from convilyn_edge.runtime import Pipeline, ThresholdAggregator
from convilyn_edge.runtime.pipeline import RouteOutcome
from convilyn_edge.spi.review import ReviewOutcome, ReviewRequest
from examples.pet_monitoring import (
    AdapterRegistry,
    CatLocatorModel,
    ConnectivityProvider,
    PetAnomalyRules,
    WaterSimSource,
    anomaly_bucket,
    assemble_pet_alert_pipeline,
    default_alert_review,
    resolve_placement,
)
from examples.pet_monitoring.events import EVENT_WATER_READING, SENSOR_SCHEMA
from examples.pet_monitoring.sinks import NotifySink

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_OD_ASSET = InstalledAsset(capability="object_detection", model="yolo-nano")


class _Clock:
    def __init__(self, start: datetime) -> None:
        self._t = start

    def __call__(self) -> datetime:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += timedelta(seconds=seconds)


class _AckReview:
    """A human that answers (acks) — disposition 'selected'."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        return ReviewOutcome(decision="selected", choice_id="ack")


class _SilentReview:
    """A human that never answers (only reached if there is no expiry)."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:  # pragma: no cover
        raise AssertionError("review should not be consulted when the deadline has passed")


class _NeverReview:
    """A human that blocks forever — the review is raced against the (injected) deadline timer."""

    async def request(self, req: ReviewRequest) -> ReviewOutcome:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


async def _instant(_seconds: float) -> None:
    """A sleep that returns immediately — lets the deadline timer win without a real wait."""
    return None


def _manifest(*, silicon: str = "generic_x86_64", assets=()) -> DeviceCapabilityManifest:
    return DeviceCapabilityManifest(
        device_id="d", silicon=silicon, ram_mb=8192, installed_assets=assets
    )


def _anomaly_event() -> EventEnvelope:
    # a low-water reading → the deterministic rules flag it as an anomaly
    return new_envelope(
        event_type=EVENT_WATER_READING,
        event_schema=SENSOR_SCHEMA,
        source=EventSourceRef("dev", "water-sim", "0.1.0"),
        data={"level_ml": 40},
    )


def _build(
    tmp_path: Path,
    *,
    online: bool = True,
    review=None,
    expires_at: str | None = None,
    manifest: DeviceCapabilityManifest | None = None,
    sleep=None,
):
    clock = _Clock(_T0)
    rules = PetAnomalyRules()
    store = DurableQueue(tmp_path / "anomalies.jsonl")
    aggregator = ThresholdAggregator(
        store, threshold=2, window_s=600, key_of=anomaly_bucket(rules), now=clock
    )
    manifest = manifest if manifest is not None else _manifest(assets=(_OD_ASSET,))
    cat_locator = CatLocatorModel(placement=resolve_placement(manifest))
    notify_me, notify_sister = NotifySink("me"), NotifySink("sister")
    pipeline = assemble_pet_alert_pipeline(
        aggregator=aggregator,
        connectivity=ConnectivityProvider(online),
        cat_locator=cat_locator,
        review=review if review is not None else _AckReview(),
        notify_primary=notify_me,
        notify_escalation=notify_sister,
        review_request=default_alert_review(expires_at=expires_at),
        sleep=sleep,
    )
    return pipeline, clock, notify_me, notify_sister, cat_locator


# ── end-to-end: threshold → locate → alert ───────────────────────────────────


async def test_below_threshold_is_quiet(tmp_path: Path):
    pipeline, _clock, me, sister, _ = _build(tmp_path)

    result = await pipeline.run(_anomaly_event())  # first anomaly — count 1, not breached

    assert result.output("alert").key == "quiet"
    assert me.sent == [] and sister.sent == []  # no notification below the threshold


async def test_breach_locates_and_notifies_primary(tmp_path: Path):
    pipeline, clock, me, sister, _ = _build(tmp_path)  # AckReview + no expiry → human answers

    await pipeline.run(_anomaly_event())  # count 1
    clock.advance(60)
    result = await pipeline.run(_anomaly_event())  # count 2 → breached → locate branch

    assert result.output("alert").key == "locate"
    assert len(me.sent) == 1 and sister.sent == []  # human acked → primary notified, not escalated
    assert "cat located" in me.sent[0].body  # the grounded model output is on the alert message


async def test_expiry_escalates_to_the_sister(tmp_path: Path):
    # A past deadline → the review expires immediately → default_action=escalate → notify sister.
    pipeline, clock, me, sister, _ = _build(
        tmp_path, review=_SilentReview(), expires_at="2020-01-01T00:00:00+00:00"
    )

    await pipeline.run(_anomaly_event())
    clock.advance(60)
    result = await pipeline.run(_anomaly_event())

    assert result.output("dispatch").key == "escalate"
    assert len(sister.sent) == 1 and me.sent == []  # escalated to the sister
    assert "ESCALATED" in sister.sent[0].title


async def test_future_deadline_escalates_via_the_injected_sleep(tmp_path: Path):
    # A FUTURE expires_at + a silent human: the injected instant `sleep` lets the deadline timer
    # win the race without a real 10-minute wait — proving the review-expiry clock seam works and
    # that a cloner should inject `sleep` (never block on a real deadline).
    pipeline, clock, me, sister, _ = _build(
        tmp_path,
        review=_NeverReview(),
        expires_at="2099-01-01T00:00:00+00:00",
        sleep=_instant,
    )

    await pipeline.run(_anomaly_event())
    clock.advance(60)
    result = await pipeline.run(_anomaly_event())

    assert result.output("dispatch").key == "escalate"
    assert len(sister.sent) == 1 and me.sent == []  # timer fired → escalated without waiting


async def test_offline_breach_notifies_locally_without_locating(tmp_path: Path):
    pipeline, clock, me, sister, locator = _build(tmp_path, online=False)

    await pipeline.run(_anomaly_event())
    clock.advance(60)
    result = await pipeline.run(_anomaly_event())

    assert result.output("alert").key == "offline"
    assert len(me.sent) == 1 and "check connectivity" in me.sent[0].body
    # the offline branch never runs the model (no "locate" output on the state)
    try:
        result.output("locate")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("offline branch must not run the model node")


# ── charter rule ① — the pack is a DECLARATION (router-driven, not hand if/else) ─


async def test_rule1_branches_run_via_the_router_primitive(tmp_path: Path):
    pipeline, clock, _me, _sister, _ = _build(tmp_path)

    quiet = await pipeline.run(_anomaly_event())
    clock.advance(60)
    alert = await pipeline.run(_anomaly_event())

    # Each path is a declarative RouteOutcome the router recorded — no imperative branch drove it.
    assert isinstance(quiet.output("alert"), RouteOutcome) and quiet.output("alert").key == "quiet"
    assert isinstance(alert.output("alert"), RouteOutcome) and alert.output("alert").key == "locate"


def test_rule1_assembly_returns_an_immutable_declarative_pipeline(tmp_path: Path):
    pipeline, *_ = _build(tmp_path)

    assert isinstance(pipeline, Pipeline)
    # the top level is state → state → route (a declaration), and route is the branching primitive
    assert [type(s).__name__ for s in pipeline.steps][-1] == "_RouterStep"


# ── charter rule ② — capabilities bind via registry + capability-negotiated resolver ─


def test_rule2_placement_is_capability_negotiated(tmp_path: Path):
    _, _, _, _, edge_locator = _build(tmp_path, manifest=_manifest(assets=(_OD_ASSET,)))
    _, _, _, _, cloud_locator = _build(tmp_path, manifest=_manifest())  # no local detector

    assert edge_locator.placement == "edge"
    assert cloud_locator.placement == "cloud"


def test_rule2_adapters_bind_through_the_registry():
    registry = (
        AdapterRegistry()
        .register("source.water", WaterSimSource)
        .register("sink.primary", lambda: NotifySink("me"))
    )

    source = registry.resolve("source.water", "dev", [{"level_ml": 40}])
    sink = registry.resolve("sink.primary")

    assert isinstance(source, WaterSimSource) and isinstance(sink, NotifySink)


# ── charter rule ③ — the decision path consumes a grounded ModelOperator ─────


async def test_rule3_grounded_model_is_on_the_decision_path(tmp_path: Path):
    pipeline, clock, me, _sister, _ = _build(tmp_path)

    await pipeline.run(_anomaly_event())
    clock.advance(60)
    result = await pipeline.run(_anomaly_event())  # breach → locate branch runs the model

    model_output = result.output("locate")  # the grounded model ran ON the alert path
    assert model_output.result.evidence  # a grounded result (evidence ties output → source frame)
    assert model_output.result.output is not None  # a typed CatLocation, not natural language
