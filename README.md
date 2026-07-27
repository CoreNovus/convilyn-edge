# convilyn-edge

The **Convilyn Edge AI Workflow SDK** — the Device Data Plane + Edge Runtime SPI
for building auditable, offline-capable edge/IoT AI workflows.

> **Public mirror** of Convilyn's primary repository (the source of truth). Contributions are
> welcome and land in the shipped package — see
> **[CONTRIBUTING.md](CONTRIBUTING.md)** (fork → PR → upstreamed, authorship preserved).

> **Alpha (`0.1.0b` series).** v0.1 ships the seven typed SPI Protocols + the Event
> Envelope + a `Result` type — with **zero runtime dependencies** — plus, landed
> across the 0.1 beta series: the `client_compute` on-device model keystone
> (`convilyn_edge.clientcompute`), the durable offline queue + emitter
> (`convilyn_edge.offline`), the device simulator + `convilyn-edge` CLI
> (`convilyn_edge.simulator` / `.cli`), and a capability `probe`. Vertical
> scenario logic ships as removable Solution Packs built on this SPI — never
> inside this package. Each section below documents the module as it exists in
> this release.

## What this is (and is not)

Convilyn splits an edge AI product into three planes:

| Plane | Home |
|---|---|
| **AI Workflow Plane** — SOP lookup, explain, re-ground, HITL, escalate, gated tools + the 7 server-enforced safety checks | the Convilyn cloud service |
| **Device Data Plane + Edge Runtime + adapter/provider SPI** | **this package (`convilyn-edge`)** |
| **Vertical logic** — scenario rules, device state, workflows | a *removable* Solution Pack (its own package, built on this SPI) |

Convilyn ships the **SPI + a simulator + reference adapters only** — never
hardware drivers or action connectors. Real OPOS/.NET, Zebra/Kotlin, serial /
MQTT / camera adapters and any device actuation beyond R0/R1 are integrator /
community work. That boundary — what the SDK ships vs. integrator work — is the
anti-divergence guarantee.

**Build once, run anywhere.** Workflows are authored in Convilyn's chat-driven
**Builder** — a shared, client-agnostic capability in the AI Workflow Plane, *not*
part of this SDK. Every client (web, desktop, or device) then *runs* that same
compiled workflow (`uw_…`); the Edge SDK **consumes** workflows via its
`ModelOperator` (cloud placement wraps the consumer SDK's `client.goals.run`; edge
placement runs a local model), it never builds one.

## The 7 primitives (`convilyn_edge.spi`)

Each is one narrow Protocol — depend on the interface, not a runtime (DIP/ISP).

| # | Primitive | Essence |
|---|---|---|
| 1 | `EventSource` | events enter the SDK → `AsyncIterator[EventEnvelope]` |
| 2 | `Normalizer[Raw, Canonical]` | raw vendor payload → canonical event (`Result`, sync) |
| 3 | `StateProvider[T]` | environment state at event time (`async`) |
| 4 | `DeterministicOperator[In, Out]` | pure, **no-LLM** rules (`Result`, **sync**) |
| 5 | `ModelOperator[In, Out]` | typed inference — `edge`/`cloud`/`auto` (**keystone**) |
| 6 | `HumanReview` | structured human-in-the-loop → typed `ReviewOutcome` |
| 7 | `ActionSink[In, Out]` | gated side effects, risk R0–R3 |

Everything crosses the SDK inside one **`EventEnvelope`** (uniform id / schema
version / correlation / ordering — the basis for dedup, replay, and audit).

```python
from convilyn_edge import new_envelope, EventSourceRef, Ok, Err

env = new_envelope(
    event_type="device.barcode.scan.received",
    event_schema="convilyn://schemas/barcode-scan/v1",
    source=EventSourceRef("scanner-8f-03", "opos-scanner", "0.3.1"),
    data={"scanData": "4711234567890", "symbology": "EAN13"},
)
wire = env.to_wire()                    # camelCase JSON object
assert EventEnvelope.from_wire(wire) == env
```

## Client-compute — the on-device keystone (`convilyn_edge.clientcompute`)

When a cloud workflow routes the extractor role to the device, it pauses with a
`client_compute` interrupt and hands the device a **content-free** delegation
request (files by reference only). The device runs a local model over its **own**
copy of the file and returns grounded anchors; the server re-grounds them before
trusting them. `convilyn-edge` **confirms-and-consumes** that frozen contract:

```python
import os
from convilyn import AsyncConvilyn
from convilyn_edge.clientcompute import (
    ClientComputeBridge, EdgeModelOperator, HttpLocalExtractor,
)

# A local inference server (Ollama / any OpenAI-compatible endpoint), chosen by env.
operator = EdgeModelOperator(HttpLocalExtractor.from_env(os.environ))
# `resolver.resolve(file_id) -> local text` — the device reads its OWN file copy.
bridge = ClientComputeBridge(operator, resolver)

async with AsyncConvilyn() as client:
    job = await client.goals.wait(job_id)
    # If the cloud delegated an extract step, fulfil it locally and resume:
    updated = await bridge.handle_if_present(client.goals, job)
```

The consumer SDK is *injected* (a narrow `GoalClientPort` Protocol), never
imported — so `convilyn-edge` itself stays dependency-free. Values that aren't a
verbatim substring of the local source degrade to `"Not specified"` **on the
device**, exactly as the server would degrade them — an ungrounded (possibly
injected) string never crosses the boundary.

## Offline-first (`convilyn_edge.offline`)

The device keeps working when the cloud is unreachable — structured events buffer
durably and flush exactly once on reconnect:

```python
from pathlib import Path
from convilyn_edge.offline import DurableQueue, EventEmitter, event_key

queue = DurableQueue(Path("edge-events.jsonl"), key_of=event_key)
emitter = EventEmitter(sink, queue)          # sink: EventSink (your HTTP/MQTT transport)

await emitter.emit(envelope)                 # delivered, or durably buffered if offline
report = await emitter.flush()               # drain on reconnect; report.clean == True
```

Enqueue is **idempotent** (keyed by the envelope's unique `event_id`), and
`derive_idempotency_key` reproduces the server's content-addressed reconcile key
byte-for-byte — so a retried flush is a no-op, never a duplicate.

## The manufactured contract (`convilyn_edge.authored`)

A workflow authored on the Convilyn platform compiles its decision-critical AI
node into a **grounded contract**: the prompt, the typed output fields, and a
deterministic grounding rule per field. The artifact ships to the device inside
a `uw_*` bundle; on-device you mount it in one line:

```python
from convilyn_edge.authored import ContractModelOperator

operator = ContractModelOperator.for_contract("installed/pet_cat_locate.uw.json")
result = await operator.infer({"scene": scene_text}, schema={})
```

`for_contract` loads the artifact, builds the reference HTTP-local extractor
from the environment (`EDGE_LLM_URL` → OpenAI-compatible, else Ollama), resolves
the local model tag from the contract's `model_binding`, and wires the
`closed_set` steering below — all overridable (`extractor=`, `model=`, `env=`).
The multi-step assembly (`load_contract` + your own runner) remains for full
control. A supplied `extractor=` still composes with steering: a guidance-empty
`HttpLocalExtractor` gets the contract's guidance injected automatically, an
extractor that can't carry guidance triggers a loud `UserWarning`, and
`steering="caller"` declares steering caller-managed (silent, as-is).

### Environment contract (`from_env`)

These are the canonical variables the reference extractor reads — packs should
map their own configuration onto these names rather than invent parallel ones:

| Variable | Meaning | Default |
|---|---|---|
| `EDGE_LLM_URL` | OpenAI-compatible base URL; **setting it selects the `openai-compat` backend** | unset (→ Ollama) |
| `EDGE_LLM_API_KEY` | Bearer token for the OpenAI-compatible server | unset |
| `EDGE_LLM_MODEL` | Model tag (overridden by an explicit `model=` / contract binding) | `qwen3:4b` |
| `OLLAMA_BASE` | Ollama base URL when `EDGE_LLM_URL` is unset | `http://localhost:11434` |

### Generation params & honest degradation (reasoning models)

Reasoning models can silently spend their whole token budget thinking —
producing an empty answer that a naive integration reads as "server down".
Both are first-class now:

```python
operator = ContractModelOperator.for_contract(
    "installed/pet_cat_locate.uw.json",
    max_tokens=4096,        # openai-compat max_tokens / ollama num_predict
    reasoning=False,        # ollama think:false; openai-compat chat_template_kwargs
    extra_body={"reasoning_effort": "low"},  # vendor passthrough, merged last
)

result = await operator.infer({"scene": scene_text}, schema={})
if result.status == "unavailable":
    result.degrade_reason   # "server_unreachable" | "deadline_exceeded"
                            # | "output_unparseable" | "error"
```

`degrade_reason` is the difference between *"the server is offline"* and *"the
model ran and produced nothing parseable"* — assert on it in your load-bearing
test tier (below) so a green fallback can't hide a dead model.

### The two per-field weapons: `closed_set` and `field_guidance`

Every contract field carries one of two deterministic grounding modes:

- **`verbatim`** — the value must appear (whitespace-collapsed) in the device's
  own source text. For extraction: quotes, names, readings. Anything else
  degrades to the missing sentinel — blank over fabrication.
- **`closed_set`** — the value must normalise into the field's **authored**
  `allowed_values`, and the grounded output is always the authored canonical
  label, never the model's raw string. For classification and derived answers
  ("is the cat present" → `{"true","false"}`) — values that legitimately never
  appear verbatim in the source.

`closed_set` fields need one more thing: an unguided model is steered by the
blanket "answer verbatim from the source" rule, which a closed-set answer can
never satisfy. `guidance_from_contract(contract)` renders each `closed_set`
field's authored labels as a per-field answer rule; pass it as the extractor's
`field_guidance` and each such field is steered toward its own label set
(`for_contract` does this automatically):

```python
from convilyn_edge.authored import guidance_from_contract, load_contract
from convilyn_edge.clientcompute.engine import HttpLocalExtractor

contract = load_contract("installed/pet_cat_locate.uw.json")
extractor = HttpLocalExtractor(
    model="qwen3:4b", field_guidance=guidance_from_contract(contract)
)
```

Steering is advisory; grounding is enforced regardless. A model that answers
off-set is degraded to the sentinel — never trusted.

### Determinism-by-design ≠ model correctness (tiered testing)

The SDK's guarantees are **structural**, and it's important to test at the
right tier:

- **T0 — structure & schema (the SDK guarantees this).** Every `infer` returns
  a total, schema-shaped field dict; `closed_set` answers are always authored
  labels; `verbatim` answers always appear in your sources; failures degrade to
  the sentinel instead of fabricating. You don't need to test any of that —
  it's pinned by the SDK's own suite.
- **T1 — model output quality (you must test this).** Whether *your* local
  model on *your* hardware answers *your* scenes correctly is not something the
  SDK can promise. Build a small fixed eval set and assert per-case:

```python
CASES = [  # (scene text, expected grounded fields) — grow this from real traffic
    ("Cat curled on the sofa by the window.", {"present": "true", "zone": "sofa"}),
    ("Empty room, feeder untouched.", {"present": "false"}),
]

async def evaluate(operator) -> float:
    passed = 0
    for scene, expected in CASES:
        result = await operator.infer({"scene": scene}, schema={})
        got = result.output or {}
        passed += all(got.get(k) == v for k, v in expected.items())
    return passed / len(CASES)  # gate your rollout on a floor, e.g. >= 0.9
```

Run it against every model/quantisation/prompt change and compare to the last
score before swapping anything in production — a candidate model that scores
below your floor never ships, no matter how good one demo answer looked.

## CLI — simulate with no hardware (`convilyn-edge`)

A developer shouldn't need a real scanner to test a workflow. Replay a JSON
scenario through the built-in simulator:

```bash
convilyn-edge simulate scenario.json --no-delay    # prints one wire-JSON envelope per event
convilyn-edge init adapter my-sensor               # scaffold a device adapter
convilyn-edge init workflow my-workflow            # scaffold a workflow (workflow.py Pipeline skeleton)
python -m convilyn_edge.cli --help                 # same CLI on vendored/no-pip installs
```

A scenario declares a device and an ordered list of events (each with an optional
`delay_ms` / `repeat`); `SimulatedSource` — the first concrete `EventSource` —
replays it as an `EventEnvelope` stream. (`dev run` / `trace replay` land in v0.2
with the workflow executor; `simulate --no-delay` is the deterministic replay.)

## Device-RAM fit guard (`convilyn_edge.runners`)

Declare a model's minimum device RAM on its `RunnerConfig` and the selector
warns *before* the model loads — never an OOM diagnosed after:

```python
from convilyn_edge.runners import RunnerConfig, select_runner

config = RunnerConfig(model="qwen3:4b", min_ram_mb=4096)
runner = select_runner("llama_cpp", config)            # logs a warning if short
runner = select_runner("llama_cpp", config, strict_fit=True)  # raises RamFitError instead
```

By default a shortfall only warns (you may know your device better than the
probe); `strict_fit=True` turns it into a hard `RamFitError`. `check_ram_fit()`
returns the underlying `RamFitReport` (both numbers + message) for your own UX,
and `available_ram_mb=...` checks fit against another device's manifest instead
of the local host. Omit `min_ram_mb` and nothing changes — the check is opt-in.

## Health vs warmup — cold start ≠ offline (`convilyn_edge.warmup`)

`health()` answers *"is the local inference server reachable?"* — it says
nothing about whether the model weights are loaded. The first inference after
boot can take tens of seconds on an edge device; without a warmup probe that
latency is indistinguishable from an outage. `warmup()` is the assertable
three-state answer:

```python
from convilyn_edge.runners import RunnerConfig, select_runner, warmup_runner

runner = select_runner("ollama", RunnerConfig(model="qwen3:4b"))
report = runner.warmup(deadline_ms=30_000)   # pay the cold start before opening
if report.state == "unreachable":
    show_offline_banner(report.detail)       # a different fix than "loading…"
elif report.state == "cold_started":
    log_startup_latency(report.latency_ms)   # the next request will be fast
# "warm" → the model was already loaded; nothing to do
```

`unreachable` is never faked as a slow cold start (reachability is checked
first), and a probe that exceeds its deadline while the server stays up reports
`cold_started` — the model is loading, not offline. Runners without a warmup
hook are handled generically: `warmup_runner(runner)` returns already-`warm`
for them, so the call is always safe. `ContractModelOperator` forwards both
hooks (`operator.warmup(...)` / `operator.health()`) — no second extractor
needed just to warm the manufactured-contract path.

One more doctor check: `health()` proves the *server* is up, not that it can
serve *your* model. `model_available()` answers the binding question:

```python
report = operator.model_available()   # also on HttpLocalExtractor / OpenAICompatRunner
report.state   # "available" | "missing" | "unreachable" | "unknown"
```

`missing` means the server listed its models and the bound tag isn't there
(keep your banner honest); an empty or unrecognizable listing is `unknown`,
never a false `missing`. Doctor-surface data — never gate execution on it.

## Design principles (enforced in code, not just docs)

- **The device is never a second source of truth.** The server holds the 7
  server-enforced safety checks and re-grounds every device value; the edge SPI
  inherits that contract.
- **No LLM in `DeterministicOperator`** — a *sync* signature makes "no I/O, no
  model" a type-level guarantee. Scenario rules live in a removable pack.
- **One envelope, one `Result`, one observability convention.** No parallel
  transports; no `if provider == ...`.

## The removability check

> Delete an entire vertical Solution Pack. Does the remaining SDK still let you
> build another IoT AI workflow?

If yes, this is a general SDK — not a vertical wearing an SDK costume. That
question is a committed CI lint in the package.

## Install

```bash
uv add --prerelease=allow convilyn-edge   # or: pip install --pre convilyn-edge
```

Python ≥ 3.10. Zero runtime dependencies. Runnable examples live in
[`examples/`](./examples/) — start with `examples/drive_pipeline.py`. The
examples (including the `pet_monitoring` reference Solution Pack) travel in the
**source distribution and the public mirror, not the wheel** — grab them with
`pip download --no-binary :all: convilyn-edge` or from the repository.

## License

Apache-2.0.
