"""The 7-primitive Edge SPI.

Each primitive is one narrow, single-responsibility Protocol (ISP/DIP) plus its
frozen dataclasses. Adapters and Solution Packs depend on these Protocols, never
on a concrete runtime — that inversion is what keeps the substrate reusable and
the vertical logic removable.

    1. Source                 — events enter the SDK          (source.py)
    2. Normalizer             — raw → canonical event         (normalizer.py)
    3. StateProvider          — environment state at event    (state.py)
    4. DeterministicOperator  — pure, no-LLM rules            (operator.py)
    5. ModelOperator          — typed inference (KEYSTONE)    (model.py)
    6. HumanReview            — structured HITL               (review.py)
    7. ActionSink             — gated side effects            (action.py)
"""

from __future__ import annotations

from convilyn_edge.spi.action import (
    ActionAuthorization,
    ActionDescriptor,
    ActionResult,
    ActionSink,
    ActionStatus,
    RiskLevel,
)
from convilyn_edge.spi.model import (
    Evidence,
    ModelOperator,
    ModelResult,
    ModelStatus,
    Placement,
)
from convilyn_edge.spi.normalizer import NormalizeError, Normalizer
from convilyn_edge.spi.operator import (
    DeterministicOperator,
    ExecutionContext,
    OperatorError,
)
from convilyn_edge.spi.review import (
    DecisionSource,
    DefaultAction,
    HumanReview,
    ReviewChoice,
    ReviewDecision,
    ReviewOutcome,
    ReviewRequest,
)
from convilyn_edge.spi.source import (
    DeviceHealth,
    EventSource,
    HealthStatus,
    SourceContext,
)
from convilyn_edge.spi.state import EventContext, StateProvider

__all__ = [
    # 1. Source
    "EventSource",
    "SourceContext",
    "DeviceHealth",
    "HealthStatus",
    # 2. Normalizer
    "Normalizer",
    "NormalizeError",
    # 3. StateProvider
    "StateProvider",
    "EventContext",
    # 4. DeterministicOperator
    "DeterministicOperator",
    "ExecutionContext",
    "OperatorError",
    # 5. ModelOperator
    "ModelOperator",
    "ModelResult",
    "Evidence",
    "Placement",
    "ModelStatus",
    # 6. HumanReview
    "HumanReview",
    "ReviewRequest",
    "ReviewChoice",
    "ReviewOutcome",
    "DefaultAction",
    "ReviewDecision",
    "DecisionSource",
    # 7. ActionSink
    "ActionSink",
    "ActionDescriptor",
    "ActionAuthorization",
    "ActionResult",
    "RiskLevel",
    "ActionStatus",
]
