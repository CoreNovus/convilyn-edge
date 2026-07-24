"""Pet-monitoring — the flagship reference **Solution Pack** for ``convilyn-edge``.

The moat charter (``backend-api/docs/engineering/edge_solution_pack_moat_and_lockin.md``)
makes live device monitoring an *out-of-moat* scenario delivered as a **removable Solution
Pack**: the integrator builds the vertical on the edge SDK's 7 SPI primitives + the generic
runtime primitives, on their own hardware. This package is Convilyn's reference build of that
pack — SPI adapters (simulated), an R1 notify sink, a deterministic rule table, a grounded
model node, and a **declarative** assembly — that an integrator clones and points at real
hardware.

It is deliberately **removable**: it imports :mod:`convilyn_edge`; nothing in
``convilyn_edge`` imports it, and it lives in ``examples/`` (outside the shipped wheel). Delete
it and the SDK still builds any other IoT AI workflow — the substrate carries zero pet
vocabulary. All the pet/cat/feeder nouns live *here*, in the pack, where scenario knowledge
belongs (``generic-tools-agent-orchestration.md``).

The pack's pieces:

* :mod:`~examples.pet_monitoring.sources` — camera / feeder / water / litter simulator
  ``EventSource`` adapters.
* :mod:`~examples.pet_monitoring.sinks` — an R1 ``NotifySink`` ("notify me" / "notify sister").
* :mod:`~examples.pet_monitoring.operators` — a deterministic, offline-first anomaly rule table
  (never an LLM).
* :mod:`~examples.pet_monitoring.registry` — an adapter registry so a device adapter binds by
  a capability **key**, swappable as a registry entry, never an ``if brand ==`` branch
  (charter rule 2, adapter half).
* :mod:`~examples.pet_monitoring.model_node` — the grounded "is the cat present, and where?"
  ``ModelOperator``, with capability-negotiated ``edge`` / ``cloud`` placement (charter rules
  2 + 3).
* :mod:`~examples.pet_monitoring.pack` — the **declarative assembly**
  (:func:`~examples.pet_monitoring.pack.assemble_pet_alert_pipeline`) that composes the above via
  the generic runtime primitives (threshold aggregation, branch/router, review-expiry) into an
  immutable ``Pipeline`` — the pack as a DECLARATION the runtime drives, never hand-wired
  ``async if``/``else`` (charter rule 1).

All three charter iron rules are embodied: ① the assembly is declarative, ② capabilities bind via
the adapter registry + capability-negotiated placement, ③ the decision path consumes the grounded
``ModelOperator``.
"""

from __future__ import annotations

from examples.pet_monitoring.contract_node import (
    PET_CONTRACT_PATH,
    ContractCatLocator,
    ScriptedContractExtractor,
    build_cat_locator,
    describe_frame,
)
from examples.pet_monitoring.events import (
    EVENT_CAMERA_FRAME,
    EVENT_FEEDER_READING,
    EVENT_LITTER_READING,
    EVENT_WATER_READING,
    SENSOR_SCHEMA,
)
from examples.pet_monitoring.model_node import (
    CameraFrame,
    CatDetector,
    CatLocation,
    CatLocatorModel,
    resolve_placement,
    scripted_detector,
)
from examples.pet_monitoring.operators import AnomalyVerdict, PetAnomalyRules, PetThresholds
from examples.pet_monitoring.pack import (
    ConnectivityProvider,
    ConnectivityState,
    anomaly_bucket,
    assemble_pet_alert_pipeline,
    default_alert_review,
)
from examples.pet_monitoring.registry import AdapterRegistry
from examples.pet_monitoring.sinks import Notification, NotifyRequest, NotifySink
from examples.pet_monitoring.sources import (
    CameraSimSource,
    FeederSimSource,
    LitterSimSource,
    WaterSimSource,
)

__all__ = [
    # events
    "EVENT_CAMERA_FRAME",
    "EVENT_FEEDER_READING",
    "EVENT_WATER_READING",
    "EVENT_LITTER_READING",
    "SENSOR_SCHEMA",
    # sources
    "CameraSimSource",
    "FeederSimSource",
    "WaterSimSource",
    "LitterSimSource",
    # sink
    "NotifySink",
    "NotifyRequest",
    "Notification",
    # operator
    "PetAnomalyRules",
    "PetThresholds",
    "AnomalyVerdict",
    # grounded model node (rules 2 + 3)
    "CatLocatorModel",
    "PET_CONTRACT_PATH",
    "ContractCatLocator",
    "ScriptedContractExtractor",
    "build_cat_locator",
    "describe_frame",
    "CameraFrame",
    "CatLocation",
    "CatDetector",
    "resolve_placement",
    "scripted_detector",
    # registry
    "AdapterRegistry",
    # declarative assembly (rule 1)
    "assemble_pet_alert_pipeline",
    "ConnectivityProvider",
    "ConnectivityState",
    "anomaly_bucket",
    "default_alert_review",
]
