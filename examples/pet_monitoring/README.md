# Pet-monitoring — the flagship reference Solution Pack

A complete, **removable** edge Solution Pack built on `convilyn-edge`. It shows how an
integrator delivers an out-of-moat scenario (live device monitoring) on the SDK's 7 SPI
primitives + the generic runtime primitives — on their own hardware, offline-capable — while
staying **locked into** Convilyn's runtime, grounding, and deploy chain.

> Governed by `backend-api/docs/engineering/edge_solution_pack_moat_and_lockin.md`. Lives in
> `examples/` (outside the shipped wheel) and is imported by nothing in `convilyn_edge` — delete it
> and the SDK still builds any other IoT workflow. All pet/cat/feeder vocabulary lives here.

## One-command demo — "Doudou's day" (integrator quickstart)

See the whole pack run end to end — a full day of a cat's monitoring, fully **offline**,
**deterministic**, and **uninterrupted** — then read a plain-language business-value report. No
model, no cloud, no config required (it uses a deterministic stub if no local model is set).

```bash
# 1. install the SDK
pip install convilyn-edge
# 2. get the pack (clone the repo, or copy the examples/pet_monitoring/ folder next to your code)
git clone https://github.com/CoreNovus/convilyn && cd convilyn/sdk/edge-python
# 3. run the day (stub mode — no Ollama needed)
PYTHONPATH=src python -m examples.pet_monitoring.run_e2e
```

**Time to first run: under a minute** (three steps, one command; the escalation timer is injected
so the "10-minute" wait is instant). It prints a narrated timeline (08:00 breakfast → 11:55 low
water → 11:57 litter anomaly → ≥2 breach → cat-locate → 10-min review → **escalate to the sister**),
plus the primary-response and offline variants, and exits `0` when the day ran correctly.

**Run it against a real local model** (`placement=edge`): point it at your on-device inference
server and re-run — same command, real grounded inference.

```bash
# Ollama serving qwen3:4b:
CONVILYN_EDGE_MODEL_URL=http://localhost:11434 PYTHONPATH=src python -m examples.pet_monitoring.run_e2e
# or any OpenAI-compatible server (llama.cpp / LM Studio / vLLM):
CONVILYN_EDGE_MODEL_URL=http://localhost:8080/v1 CONVILYN_EDGE_MODEL=qwen3:4b PYTHONPATH=src \
  python -m examples.pet_monitoring.run_e2e
```

The routing (threshold → branch → escalate) is identical in both modes — the model only enriches
the alert; the flow is deterministic either way.

## The three iron rules (this pack's acceptance)

| Rule | Where |
|---|---|
| **① A pack is a DECLARATION, not imperative control-flow.** | `pack.py::assemble_pet_alert_pipeline` composes the workflow from the generic runtime primitives — threshold aggregation, `.route(...)` branches, review-expiry — into an immutable `Pipeline`. No hand-wired `async if`/`else`. |
| **② Device capabilities bind via a capability registry + a deterministic resolver.** | `registry.py::AdapterRegistry` (adapters by key, never `if brand ==`) + `model_node.py::resolve_placement` (edge/cloud from the device's reported `object_detection` asset, never `if silicon ==`). |
| **③ The decision path consumes a grounded `ModelOperator`.** | `model_node.py::CatLocatorModel` returns a typed, evidence-carrying `ModelResult` on the alert branch; the server re-grounds it — the device is never a second source of truth. |

## The workflow (charter §5 journey ③)

```
state: connectivity → state: anomaly threshold (≥2 in a 10-min window)
  → route(alert):
      quiet   (below threshold)   → do nothing
      offline (breached, no net)  → notify me: "offline — check connectivity"
      locate  (breached, online)  → model: cat-locate (grounded)
                                    → review: expires_at=10min → default_action=escalate
                                    → route(dispatch):
                                          primary  → notify me
                                          escalate → notify sister
```

Each stage is a **generic** primitive with a pack-supplied *binding*: the threshold is a
`ThresholdAggregator` (durable through the storage seam), the branches are the declarative
`route` primitive, and "escalate after 10 minutes of silence" is a `ReviewRequest.expires_at`
the runtime's timer enforces — none of it is orchestration the integrator hand-wrote.

## Pieces

| Module | What |
|---|---|
| `events.py` | the pack's `event_type` / schema vocabulary |
| `sources.py` | camera / feeder / water / litter **simulator** `EventSource` adapters (a real pack swaps in hardware) |
| `operators.py` | `PetAnomalyRules` — a deterministic, offline-first anomaly rule table (never an LLM) |
| `sinks.py` | `NotifySink` — an R1 reference `ActionSink` ("notify me" / "notify sister") |
| `model_node.py` | `CatLocatorModel` — the grounded cat-locate node + capability-negotiated placement |
| `registry.py` | `AdapterRegistry` — bind adapters by capability key |
| `pack.py` | `assemble_pet_alert_pipeline` — the declarative DAG |

## Run it

```python
from pathlib import Path

from convilyn_edge.offline.queue import DurableQueue
from convilyn_edge.runtime import ThresholdAggregator
from convilyn_edge.probe import probe_device
from examples.pet_monitoring import (
    PetAnomalyRules, CatLocatorModel, ConnectivityProvider, NotifySink,
    anomaly_bucket, assemble_pet_alert_pipeline, default_alert_review, resolve_placement,
)

my_review = ...        # your device-side HumanReview adapter (the on-device HITL surface)
deadline_iso = ...     # UTC ISO-8601: 10 minutes from the alert; None to wait indefinitely

rules = PetAnomalyRules()
store = DurableQueue(Path("anomalies.jsonl"))                 # durable via the storage seam
aggregator = ThresholdAggregator(store, threshold=2, window_s=600, key_of=anomaly_bucket(rules))
manifest = probe_device(device_id="jetson-01")               # capability negotiation
pipeline = assemble_pet_alert_pipeline(
    aggregator=aggregator,
    connectivity=ConnectivityProvider(online=True),
    cat_locator=CatLocatorModel(placement=resolve_placement(manifest)),  # edge iff object_detection installed
    review=my_review,
    notify_primary=NotifySink("me"),
    notify_escalation=NotifySink("sister"),
    review_request=default_alert_review(expires_at=deadline_iso),
)
# then drive each sensor event: result = await pipeline.run(envelope)
# (a WorkflowDriver + the SimulatedSource adapters wire the event loop — see the tests)
```

See `tests/test_pet_monitoring_pack.py` for the end-to-end flow (threshold → locate → escalate)
driven against the simulator, and `tests/test_pet_monitoring_*.py` for the per-piece tests.
