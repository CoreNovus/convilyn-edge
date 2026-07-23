"""The declarative assembly — the pet-monitoring workflow as a DECLARATION (charter rule 1).

This is the capstone: the pack's control flow — time-window threshold aggregation, an
offline-vs-locate branch, a grounded model node, a human review that escalates on expiry — is
expressed by composing the **generic runtime primitives** (#2792 + #2791) into an immutable
:class:`~convilyn_edge.runtime.Pipeline`. There is **no hand-wired ``async if``/``else``**: every
branch is a declarative ``.route(...)``, the "≥ N anomalies in a window" is a
:class:`~convilyn_edge.runtime.ThresholdAggregator`, and "escalate after 10 min of silence" is a
``ReviewRequest.expires_at`` the runtime's timer enforces. Hand-rolled integrator orchestration is
exactly the lock-in leak the charter (rule 1) rejects — the pack is data the Convilyn runtime
drives.

The shape (charter §5 journey ③), as a DAG::

    state:  connectivity  → state:  anomaly threshold
      → route(alert):
          quiet   (below threshold)     → do nothing
          offline (breached, no net)    → notify me: "offline — check connectivity"
          locate  (breached, online)    → model: cat-locate (grounded)
                                          → review: expires_at=10min → default escalate
                                          → route(dispatch):
                                                primary  → notify me
                                                escalate → notify sister

Every step is a generic primitive with a pack-supplied *binding*; the scenario knowledge lives in
the bindings + the adapters, never in the substrate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.runtime import OccurrenceKey, Pipeline, PipelineState, ThresholdAggregator
from convilyn_edge.spi.operator import ExecutionContext
from convilyn_edge.spi.review import HumanReview, ReviewChoice, ReviewRequest
from convilyn_edge.spi.state import EventContext
from examples.pet_monitoring.model_node import CameraFrame, CatLocation, CatLocatorModel
from examples.pet_monitoring.operators import PetAnomalyRules
from examples.pet_monitoring.sinks import NotifyRequest, NotifySink

#: The output JSON Schema the grounded model validates against (a real edge/cloud impl enforces it).
LOCATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "zone": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
    },
    "required": ["present"],
}


@dataclass(frozen=True)
class ConnectivityState:
    """Whether the cloud / camera path is reachable — the offline-vs-locate branch signal."""

    online: bool


class ConnectivityProvider:
    """A ``StateProvider[ConnectivityState]`` reporting reachability (the reference reads a flag).

    ``online`` may be a bool or a zero-arg callable (a real provider pings connectivity); the sim
    passes a fixed value or a lambda so a test drives the offline branch deterministically.
    """

    def __init__(self, online: bool | Callable[[], bool] = True) -> None:
        self._online = online

    async def snapshot(self, ctx: EventContext) -> ConnectivityState:
        online = self._online() if callable(self._online) else self._online
        return ConnectivityState(online=bool(online))


def anomaly_bucket(rules: PetAnomalyRules, *, per_kind: bool = False) -> OccurrenceKey:
    """The threshold aggregator's ``key_of``: run the deterministic rules and return the count
    bucket for an anomaly, or ``None`` for a non-anomaly.

    Default (``per_kind=False``) buckets ALL anomalies together (a constant key), so the threshold
    is "≥ N anomalies of any kind in the window" — the charter's "≥ 2 anomalies". ``per_kind=True``
    keys by the anomaly kind instead ("≥ N feeder jams"). Either way the aggregator counts
    *anomalies*, not raw events, deriving that from the SAME pure rule table (no duplicated logic,
    no LLM).
    """

    def _key(envelope: EventEnvelope) -> str | None:
        verdict = rules.execute(envelope, ExecutionContext(envelope=envelope)).unwrap()
        if not verdict.anomalous:
            return None
        return verdict.kind if per_kind else "anomaly"

    return _key


# ── run-time bindings (the pack's scenario glue for the generic stages) ───────


def _frame_for_alert(state: PipelineState) -> CameraFrame:
    """Derive the frame to locate against. The reference captures on the alert event; a real pack
    would grab a live camera frame here."""
    return CameraFrame(frame_id=state.envelope.event_id, motion=True)


def _describe(location: CatLocation | None) -> str:
    if location is None:
        return "camera unavailable"
    if not location.present:
        return "cat not located"
    return f"cat located at {location.zone or 'unknown zone'}"


def _breach_count(state: PipelineState) -> int:
    return state.output("anomalies").count


def _locate_summary(state: PipelineState) -> str:
    """Describe the located cat, but ONLY trust the model when its deterministic disposition is
    ``accept`` — an ``uncertain`` / ``unavailable`` result (disposition ``fallback``) is not
    rendered as fact (the rule-3 grounding contract: don't trust a below-bar model output)."""
    output = state.output("locate")  # a ModelStepOutput (result + deterministic disposition)
    if output.disposition != "accept":
        return "cat not confidently located"
    return _describe(output.result.output)


def _primary_notice(state: PipelineState) -> NotifyRequest:
    return NotifyRequest(
        title="Pet alert",
        body=f"{_breach_count(state)} anomalies in the window; {_locate_summary(state)}.",
    )


def _escalation_notice(state: PipelineState) -> NotifyRequest:
    return NotifyRequest(
        title="Pet alert — ESCALATED",
        body=(
            f"No response in time — escalating. {_breach_count(state)} anomalies; "
            f"{_locate_summary(state)}."
        ),
    )


def _offline_notice(state: PipelineState) -> NotifyRequest:
    return NotifyRequest(
        title="Pet alert (offline)",
        body=(
            f"{_breach_count(state)} anomalies, but the camera/cloud is unreachable — "
            "check connectivity."
        ),
    )


def default_alert_review(*, expires_at: str | None) -> Callable[[PipelineState], ReviewRequest]:
    """Build the review-request binding: a titled check-in whose ``default_action`` is ``escalate``.

    ``expires_at`` (a UTC ISO-8601 deadline) is what makes "nobody answered in 10 min → notify the
    sister" a declaration the runtime's timer enforces — pass ``None`` to wait indefinitely.
    """

    def _request(state: PipelineState) -> ReviewRequest:
        return ReviewRequest(
            title="Pet anomaly — can you check in?",
            message=f"{_breach_count(state)} anomalies detected; {_locate_summary(state)}.",
            choices=(ReviewChoice(id="ack", label="I'll handle it"),),
            default_action="escalate",
            expires_at=expires_at,
        )

    return _request


def _alert_route(state: PipelineState) -> str:
    """Pure selector → branch key. NOT imperative control flow: it computes a key the declarative
    router dispatches on (below-threshold → quiet; breached + offline → offline; else → locate).

    "offline" here means the camera / cloud path is unreachable — NOT a lack of local inference
    capability. A pack whose device carries an on-device detector (``placement="edge"``) could add
    a route that still locates offline; this reference keeps the charter §5 journey's two branches.
    """
    if not state.output("anomalies").breached:
        return "quiet"
    return "locate" if state.output("net").online else "offline"


def _dispatch_route(state: PipelineState) -> str:
    """Pure selector for the escalation dispatch: the review's disposition → recipient key."""
    return "escalate" if state.output("review").disposition == "escalate" else "primary"


def assemble_pet_alert_pipeline(
    *,
    aggregator: ThresholdAggregator,
    connectivity: ConnectivityProvider,
    cat_locator: CatLocatorModel,
    review: HumanReview,
    notify_primary: NotifySink,
    notify_escalation: NotifySink,
    review_request: Callable[[PipelineState], ReviewRequest] | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    name: str = "pet_alert",
) -> Pipeline:
    """Compose the pet-monitoring workflow as a declarative :class:`Pipeline` (charter rule 1).

    Every argument is a generic runtime primitive or SPI adapter; this function only *wires* them
    with pack bindings — it contains no imperative branching of its own. Returns an immutable
    pipeline ``run`` once per sensor event.
    """
    build_review = (
        review_request if review_request is not None else default_alert_review(expires_at=None)
    )

    locate_branch = (
        Pipeline("locate")
        .model(cat_locator, bind=_frame_for_alert, schema=LOCATION_SCHEMA, name="locate")
        .review(review, bind=build_review, now=now, sleep=sleep, name="review")
        .route(
            select=_dispatch_route,
            routes={
                "primary": Pipeline("primary").act(
                    notify_primary, bind=_primary_notice, name="notify"
                ),
                "escalate": Pipeline("escalate").act(
                    notify_escalation, bind=_escalation_notice, name="notify"
                ),
            },
            name="dispatch",
        )
    )
    offline_branch = Pipeline("offline").act(notify_primary, bind=_offline_notice, name="notify")

    return (
        Pipeline(name)
        .state(connectivity, name="net")
        .state(aggregator, name="anomalies")
        .route(
            select=_alert_route,
            routes={
                "quiet": Pipeline("quiet"),
                "offline": offline_branch,
                "locate": locate_branch,
            },
            default="quiet",
            name="alert",
        )
    )


__all__ = [
    "LOCATION_SCHEMA",
    "ConnectivityState",
    "ConnectivityProvider",
    "anomaly_bucket",
    "default_alert_review",
    "assemble_pet_alert_pipeline",
]
