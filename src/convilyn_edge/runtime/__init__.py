"""Edge runtime — the zero-dependency, self-driving workflow DAG driver.

Composes the seven SPI primitives into a running edge workflow and drives it with
the **deterministic** runtime guarantees: an event loop over an ``EventSource``,
offline-first buffering + flush, fixed-backoff retry, a stuck-abort cycle-guard,
R0/R1 action gating, and ``ModelResult.status`` / ``HumanReview`` dispatch.

This layer is deliberately THIN — it *drives*, it does not think. It replicates
none of the cloud agent's intelligence and moves none of the seven server-side
safety checks onto the device: retry/backoff, action gating, and status dispatch
are fixed tables here, never LLM-driven control flow, and the hard zones stay
server-authoritative. Scenario logic lives in a removable Solution Pack via the
pipeline's per-stage binding functions, never in this substrate.

    from convilyn_edge.runtime import Pipeline, WorkflowDriver

    pipeline = (
        Pipeline("guide_scan")
        .normalize(normalizer, bind=lambda s: s.envelope.data)
        .state(pos_provider)
        .decide(rule_engine, bind=_scan_ctx)
        .act(display_sink, bind=lambda s: s.output("decide"))
    )
    report = await WorkflowDriver().run(source, ctx, pipeline.run, emitter=emitter)
"""

from __future__ import annotations

from convilyn_edge.runtime.aggregate import (
    Occurrence,
    OccurrenceKey,
    ThresholdAggregator,
    ThresholdState,
)
from convilyn_edge.runtime.dispatch import (
    ModelDisposition,
    ReviewDisposition,
    model_disposition,
    review_disposition,
)
from convilyn_edge.runtime.driver import (
    DriveReport,
    DriveStatus,
    EventHandler,
    EventRecord,
    ProgressKey,
    WorkflowDriver,
)
from convilyn_edge.runtime.gate import AUTO_GRANT_MAX, ActionGate
from convilyn_edge.runtime.pipeline import (
    ModelStepOutput,
    Pipeline,
    PipelineResult,
    PipelineState,
    PipelineStatus,
    ReviewStepOutput,
    RouteOutcome,
    Step,
    StepResult,
    StepStatus,
)
from convilyn_edge.runtime.policy import (
    DEFAULT_RETRYABLE,
    CycleGuard,
    RetryPolicy,
    full_jitter,
    retry_async,
)
from convilyn_edge.runtime.watchdog import (
    HealthObserver,
    WatchdogPolicy,
    poll_health,
    resolve_review,
    seconds_until_expiry,
)

__all__ = [
    # policy (deterministic gates)
    "RetryPolicy",
    "CycleGuard",
    "retry_async",
    "full_jitter",
    "DEFAULT_RETRYABLE",
    # gate
    "ActionGate",
    "AUTO_GRANT_MAX",
    # dispatch
    "model_disposition",
    "review_disposition",
    "ModelDisposition",
    "ReviewDisposition",
    # pipeline (declarative DAG)
    "Pipeline",
    "PipelineResult",
    "PipelineState",
    "PipelineStatus",
    "StepResult",
    "StepStatus",
    "Step",
    "ModelStepOutput",
    "ReviewStepOutput",
    "RouteOutcome",
    # driver (event loop)
    "WorkflowDriver",
    "DriveReport",
    "DriveStatus",
    "EventRecord",
    "EventHandler",
    "ProgressKey",
    # watchdog (timer / health clock)
    "resolve_review",
    "seconds_until_expiry",
    "WatchdogPolicy",
    "poll_health",
    "HealthObserver",
    # aggregate (stateful threshold accumulation)
    "ThresholdAggregator",
    "ThresholdState",
    "Occurrence",
    "OccurrenceKey",
]
