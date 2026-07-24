"""The declarative edge pipeline — the 7-primitive DAG made self-driving.

A :class:`Pipeline` is an ordered, immutable composition of the SPI primitives in
their natural edge topology::

    Source → Normalizer → StateProvider → Deterministic/ModelOperator
           → HumanReview → ActionSink

Each stage wraps ONE injected SPI primitive plus a small **binding function**
that maps the accumulated :class:`PipelineState` to that primitive's input. The
binding functions are where per-scenario mapping lives — they belong to the
Solution Pack, never to this substrate (generic capability vs scenario
special-case). The pipeline itself contains **zero** scenario knowledge and
**zero** LLM control flow: it folds the stages, short-circuits deterministically
on a rejecting/denying/halting stage, and threads an immutable state.

Extension is by *composition* (OCP): a new workflow wires existing stages; the
substrate does not grow a branch per scenario. The SPI is frozen at seven
primitives, so the stage vocabulary is closed by design.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol, TypeVar

from convilyn_edge.envelope import EventEnvelope
from convilyn_edge.runtime.dispatch import (
    ModelDisposition,
    ReviewDisposition,
    model_disposition,
    review_disposition,
)
from convilyn_edge.runtime.gate import ActionGate
from convilyn_edge.runtime.watchdog import resolve_review
from convilyn_edge.spi.action import ActionAuthorization, ActionResult, ActionSink
from convilyn_edge.spi.model import ModelOperator, ModelResult
from convilyn_edge.spi.operator import DeterministicOperator, ExecutionContext
from convilyn_edge.spi.review import HumanReview, ReviewOutcome, ReviewRequest
from convilyn_edge.spi.state import EventContext as StateEventContext
from convilyn_edge.spi.state import StateProvider

Raw = TypeVar("Raw")
Canonical = TypeVar("Canonical")

#: A stage outcome. ``ok`` continues the fold; the other three are terminal and
#: STOP the pipeline (deterministically — no LLM decides to keep going):
#: ``rejected`` (a Normalizer/Operator ``Err``), ``denied`` (the action gate
#: refused), ``halted`` (a human ``stop``, or an action that errored).
StepStatus = Literal["ok", "rejected", "denied", "halted"]

#: The four terminal pipeline outcomes. ``completed`` = every stage returned ``ok``.
PipelineStatus = Literal["completed", "rejected", "denied", "halted"]


@dataclass(frozen=True)
class PipelineState:
    """Immutable carrier threaded through the stages.

    ``outputs`` accumulates each stage's result keyed by stage name. It is
    intentionally ``Any``-valued — the pipeline is heterogeneous by construction
    (a Normalizer's ``Canonical`` ≠ a ModelResult) — with type-safety recovered at
    each stage's binding function, which knows the concrete shapes it reads. Every
    ``with_output`` returns a NEW state (never mutates), so a stage can never
    retroactively alter an earlier one.
    """

    envelope: EventEnvelope
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def with_output(self, name: str, value: Any) -> PipelineState:
        return replace(self, outputs={**self.outputs, name: value})

    def output(self, name: str) -> Any:
        return self.outputs[name]


@dataclass(frozen=True)
class StepResult:
    """One stage's outcome: its ``status``, the (possibly-advanced) state, detail."""

    step: str
    status: StepStatus
    state: PipelineState
    detail: str | None = None


@dataclass(frozen=True)
class ModelStepOutput:
    """What a ``model`` stage stores: the typed result + its deterministic
    disposition (so a downstream binding branches on ``accept`` / ``fallback``
    without re-deriving the status)."""

    result: ModelResult[Any]
    disposition: ModelDisposition


@dataclass(frozen=True)
class ReviewStepOutput:
    """What a ``review`` stage stores: the typed outcome + its disposition."""

    outcome: ReviewOutcome
    disposition: ReviewDisposition


@dataclass(frozen=True)
class RouteOutcome:
    """What a ``route`` stage stores: which branch the selector picked + that branch's step
    outcomes. Lets a downstream binding (and an auditor) see the chosen path and read its stages'
    outputs from the threaded state."""

    key: str
    steps: tuple[StepResult, ...]


class Step(Protocol):
    """One pipeline stage: a named, self-contained async transform of the state."""

    name: str

    async def run(self, state: PipelineState) -> StepResult: ...


# ── Concrete stages (each wraps ONE SPI primitive + a binding function) ───────


@dataclass(frozen=True)
class _NormalizeStep:
    name: str
    normalizer: Any  # Normalizer[Raw, Canonical]
    bind: Callable[[PipelineState], Any]

    async def run(self, state: PipelineState) -> StepResult:
        result = self.normalizer.normalize(self.bind(state))
        if result.is_err:
            return StepResult(self.name, "rejected", state, result.error.code)
        return StepResult(self.name, "ok", state.with_output(self.name, result.value))


@dataclass(frozen=True)
class _StateStep:
    name: str
    provider: StateProvider[Any]
    bind: Callable[[PipelineState], StateEventContext]

    async def run(self, state: PipelineState) -> StepResult:
        snapshot = await self.provider.snapshot(self.bind(state))
        return StepResult(self.name, "ok", state.with_output(self.name, snapshot))


@dataclass(frozen=True)
class _DecideStep:
    name: str
    operator: DeterministicOperator[Any, Any]
    bind: Callable[[PipelineState], Any]
    ctx: Callable[[PipelineState], ExecutionContext]

    async def run(self, state: PipelineState) -> StepResult:
        result = self.operator.execute(self.bind(state), self.ctx(state))
        if result.is_err:
            return StepResult(self.name, "rejected", state, result.error.code)
        return StepResult(self.name, "ok", state.with_output(self.name, result.value))


@dataclass(frozen=True)
class _ModelStep:
    name: str
    operator: ModelOperator[Any, Any]
    bind: Callable[[PipelineState], Any]
    schema: Mapping[str, Any]
    accept_uncertain: bool
    deadline_ms: int | None

    async def run(self, state: PipelineState) -> StepResult:
        result = await self.operator.infer(
            self.bind(state), schema=self.schema, deadline_ms=self.deadline_ms
        )
        disposition = model_disposition(result, accept_uncertain=self.accept_uncertain)
        output = ModelStepOutput(result=result, disposition=disposition)
        # Never short-circuits: a ``fallback`` disposition is a valid, expected
        # offline-first path the downstream binding handles — not a failure.
        return StepResult(self.name, "ok", state.with_output(self.name, output))


@dataclass(frozen=True)
class _ReviewStep:
    name: str
    review: HumanReview
    bind: Callable[[PipelineState], ReviewRequest]
    #: Injectable clocks for the expiry timer — ``None`` uses the real UTC clock /
    #: ``asyncio.sleep``. A replay/test pins them so an ``expires_at`` review is reproducible
    #: and never actually waits (mirrors ``WorkflowDriver(sleep=…)``).
    now: Callable[[], datetime] | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None

    async def run(self, state: PipelineState) -> StepResult:
        # Route through the watchdog's expiry timer so ReviewRequest.expires_at is honoured:
        # a request the human never answers by its deadline resolves to the workflow's
        # declared default_action (deterministic, no LLM). An expiry-less request awaits the
        # human exactly as before — the primitive is additive.
        outcome = await resolve_review(
            self.review, self.bind(state), now=self.now, sleep=self.sleep
        )
        disposition = review_disposition(outcome)
        advanced = state.with_output(
            self.name, ReviewStepOutput(outcome=outcome, disposition=disposition)
        )
        # A human ``stop`` halts the pipeline before any action fires (a
        # deterministic gate). ``continue`` / ``selected`` / ``escalate`` proceed —
        # the downstream binding routes on the stored disposition.
        if disposition == "stop":
            return StepResult(self.name, "halted", advanced, "review:stop")
        return StepResult(self.name, "ok", advanced)


@dataclass(frozen=True)
class _ActionStep:
    name: str
    sink: ActionSink[Any, Any]
    bind: Callable[[PipelineState], Any]
    gate: ActionGate[Any, Any]
    auth: Callable[[PipelineState], ActionAuthorization | None]

    async def run(self, state: PipelineState) -> StepResult:
        result: ActionResult[Any] = await self.gate.invoke(
            self.sink, self.bind(state), self.auth(state)
        )
        advanced = state.with_output(self.name, result)
        if result.status == "denied":
            return StepResult(self.name, "denied", advanced, result.reason)
        if result.status == "failed":
            return StepResult(self.name, "halted", advanced, result.reason)
        return StepResult(self.name, "ok", advanced)


async def _drive_steps(
    steps: tuple[Step, ...], state: PipelineState
) -> tuple[StepStatus, tuple[StepResult, ...], PipelineState]:
    """Fold ``steps`` over ``state``, short-circuiting on the first terminal (non-``ok``) status.

    The one place the linear fold lives — shared by :meth:`Pipeline.run` (the whole pipeline) and
    :class:`_RouterStep` (a selected branch), so a branch runs by exactly the same deterministic,
    short-circuiting rules as the top level. Returns the terminal status, every step outcome so
    far, and the (possibly-advanced) state.
    """
    outcomes: list[StepResult] = []
    for step in steps:
        outcome = await step.run(state)
        outcomes.append(outcome)
        state = outcome.state
        if outcome.status != "ok":
            return outcome.status, tuple(outcomes), state
    return "ok", tuple(outcomes), state


@dataclass(frozen=True)
class _RouterStep:
    """A declarative conditional: a pure selector picks ONE named branch to run.

    ``select`` maps the accumulated state to a branch key (the pack's binding — deterministic, no
    LLM); the chosen branch (a :class:`Pipeline`) is folded over the CURRENT state, so it sees every
    upstream output and its own stages accumulate onto the same threaded state. This is how a
    Solution Pack expresses "offline → check connectivity, else → locate" as a **declaration** the
    runtime drives, never a hand-wired ``if``/``else`` in integrator code. If ``select`` returns a
    key with no route, ``default`` is used when set + present, otherwise the step terminates
    ``rejected`` (a deterministic dead-end, never a silent pass-through)."""

    name: str
    select: Callable[[PipelineState], str]
    routes: Mapping[str, Pipeline]
    default: str | None

    async def run(self, state: PipelineState) -> StepResult:
        key = self.select(state)
        if key not in self.routes:
            if self.default is None or self.default not in self.routes:
                # Deterministic dead-end. Record the attempted key (steps=()) so the RouteOutcome
                # is present downstream/for audit on every path a router took, not only success.
                dead = state.with_output(self.name, RouteOutcome(key=key, steps=()))
                return StepResult(self.name, "rejected", dead, f"no route for {key!r}")
            key = self.default
        status, sub_outcomes, advanced = await _drive_steps(self.routes[key].steps, state)
        advanced = advanced.with_output(self.name, RouteOutcome(key=key, steps=sub_outcomes))
        if status == "ok":
            return StepResult(self.name, "ok", advanced)
        tail = sub_outcomes[-1].detail if sub_outcomes else None
        return StepResult(self.name, status, advanced, f"{key}:{tail}" if tail else key)


def _with_gate(pipeline: Pipeline, gate: ActionGate[Any, Any]) -> Pipeline:
    """Return a copy of ``pipeline`` whose action steps — and every nested router's branches —
    enforce ``gate``.

    The action gate is a security boundary (R2+ always needs approval), so ONE gate must govern a
    whole composition rather than each branch silently keeping its own: an integrator who tightens
    the top-level gate expects it everywhere. ``route()`` calls this to rebind each branch to the
    parent gate; because a nested router is rebound recursively, the outermost pipeline's gate wins
    throughout the tree. A branch built with the default gate is unchanged when the parent is also
    default (the common case).
    """
    rebound: list[Step] = []
    for step in pipeline.steps:
        if isinstance(step, _ActionStep):
            rebound.append(replace(step, gate=gate))
        elif isinstance(step, _RouterStep):
            rebound.append(
                replace(step, routes={k: _with_gate(p, gate) for k, p in step.routes.items()})
            )
        else:
            rebound.append(step)
    return replace(pipeline, steps=tuple(rebound), gate=gate)


def _default_state_ctx(state: PipelineState) -> StateEventContext:
    return StateEventContext(envelope=state.envelope)


def _default_exec_ctx(state: PipelineState) -> ExecutionContext:
    return ExecutionContext(envelope=state.envelope)


def _no_auth(_state: PipelineState) -> ActionAuthorization | None:
    return None


@dataclass(frozen=True)
class PipelineResult:
    """The terminal outcome of one event through a pipeline."""

    pipeline: str
    status: PipelineStatus
    steps: tuple[StepResult, ...]
    state: PipelineState

    @property
    def completed(self) -> bool:
        """True iff every stage returned ``ok``."""
        return self.status == "completed"

    def output(self, name: str) -> Any:
        """The named stage output (KeyError if that stage did not run)."""
        return self.state.output(name)


@dataclass(frozen=True)
class Pipeline:
    """An immutable, declarative composition of pipeline stages.

    Build fluently — every builder method returns a NEW ``Pipeline`` (immutable):

        pipeline = (
            Pipeline("guide_scan")
            .normalize(barcode_normalizer, bind=lambda s: s.envelope.data)
            .state(pos_provider)
            .decide(rule_engine, bind=_scan_ctx)
            .act(display_sink, bind=lambda s: s.output("decide"))
        )
        result = await pipeline.run(envelope)
    """

    name: str
    steps: tuple[Step, ...] = ()
    gate: ActionGate[Any, Any] = field(default_factory=ActionGate)

    def then(self, step: Step) -> Pipeline:
        """Append an arbitrary custom stage (escape hatch for integrator stages)."""
        return replace(self, steps=(*self.steps, step))

    def normalize(
        self, normalizer: Any, *, bind: Callable[[PipelineState], Any], name: str = "normalize"
    ) -> Pipeline:
        return self.then(_NormalizeStep(name, normalizer, bind))

    def state(
        self,
        provider: StateProvider[Any],
        *,
        bind: Callable[[PipelineState], StateEventContext] = _default_state_ctx,
        name: str = "state",
    ) -> Pipeline:
        return self.then(_StateStep(name, provider, bind))

    def decide(
        self,
        operator: DeterministicOperator[Any, Any],
        *,
        bind: Callable[[PipelineState], Any],
        ctx: Callable[[PipelineState], ExecutionContext] = _default_exec_ctx,
        name: str = "decide",
    ) -> Pipeline:
        return self.then(_DecideStep(name, operator, bind, ctx))

    def model(
        self,
        operator: ModelOperator[Any, Any],
        *,
        bind: Callable[[PipelineState], Any],
        schema: Mapping[str, Any],
        accept_uncertain: bool = False,
        deadline_ms: int | None = None,
        name: str = "model",
    ) -> Pipeline:
        return self.then(_ModelStep(name, operator, bind, schema, accept_uncertain, deadline_ms))

    def review(
        self,
        review: HumanReview,
        *,
        bind: Callable[[PipelineState], ReviewRequest],
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        name: str = "review",
    ) -> Pipeline:
        # ``now``/``sleep`` inject the expiry timer's clock (default: real) so a review with an
        # ``expires_at`` deadline is replay-reproducible, consistent with the driver's clock.
        return self.then(_ReviewStep(name, review, bind, now, sleep))

    def act(
        self,
        sink: ActionSink[Any, Any],
        *,
        bind: Callable[[PipelineState], Any],
        auth: Callable[[PipelineState], ActionAuthorization | None] = _no_auth,
        name: str = "act",
    ) -> Pipeline:
        return self.then(_ActionStep(name, sink, bind, self.gate, auth))

    def route(
        self,
        *,
        select: Callable[[PipelineState], str],
        routes: Mapping[str, Pipeline],
        default: str | None = None,
        name: str = "route",
    ) -> Pipeline:
        """Append a declarative conditional: ``select`` picks ONE of ``routes`` to run inline.

        The selected branch (a :class:`Pipeline`) is folded over the current state — it sees every
        upstream output and threads its own onto the same state, and short-circuits by the same
        rules as the top level. ``default`` runs when ``select`` returns an unmapped key; with no
        usable default an unmapped key terminates the pipeline ``rejected``. The branch's outputs
        and chosen key are readable downstream via the stored :class:`RouteOutcome` (under ``name``)
        and by the sub-stages' own output names. This is the substrate's only branching primitive —
        scenario branching stays a *declaration* here, never a hand-wired ``if`` in the pack.

        ``select`` MUST be a pure, total function: it signals "can't decide" by returning an
        unmapped key (→ ``default`` / ``rejected``), never by raising. The parent pipeline's
        :class:`ActionGate` governs the whole composition — it is propagated into every branch
        (recursively through nested routers), so a tightened action policy is enforced inside
        branches too, not silently bypassed. Give nested routers distinct ``name=`` values so their
        :class:`RouteOutcome` entries don't overwrite each other in the outputs map."""
        bound = {k: _with_gate(p, self.gate) for k, p in routes.items()}
        return self.then(_RouterStep(name, select, bound, default))

    async def run(self, envelope: EventEnvelope) -> PipelineResult:
        """Fold the stages over ``envelope``, short-circuiting on the first
        terminal status. Returns the terminal :class:`PipelineResult`."""
        status, outcomes, state = await _drive_steps(self.steps, PipelineState(envelope=envelope))
        return PipelineResult(self.name, "completed" if status == "ok" else status, outcomes, state)


__all__ = [
    "StepStatus",
    "PipelineStatus",
    "PipelineState",
    "StepResult",
    "ModelStepOutput",
    "ReviewStepOutput",
    "RouteOutcome",
    "Step",
    "PipelineResult",
    "Pipeline",
]
