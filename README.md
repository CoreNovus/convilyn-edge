# convilyn-edge

The **Convilyn Edge AI Workflow SDK** — the Device Data Plane + Edge Runtime SPI
for building auditable, offline-capable edge/IoT AI workflows.

> **Public mirror** of Convilyn's primary repository (the source of truth). Contributions are
> welcome and land in the shipped package — see
> **[CONTRIBUTING.md](CONTRIBUTING.md)** (fork → PR → upstreamed, authorship preserved).

> **Alpha (`0.1.0b3`).** v0.1 ships the seven typed SPI Protocols + the Event
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

## CLI — simulate with no hardware (`convilyn-edge`)

A developer shouldn't need a real scanner to test a workflow. Replay a JSON
scenario through the built-in simulator:

```bash
convilyn-edge simulate scenario.json --no-delay    # prints one wire-JSON envelope per event
convilyn-edge init adapter my-sensor               # scaffold a device adapter
convilyn-edge init workflow my-workflow            # scaffold a workflow
```

A scenario declares a device and an ordered list of events (each with an optional
`delay_ms` / `repeat`); `SimulatedSource` — the first concrete `EventSource` —
replays it as an `EventEnvelope` stream. (`dev run` / `trace replay` land in v0.2
with the workflow executor; `simulate --no-delay` is the deterministic replay.)

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
[`examples/`](./examples/) — start with `examples/drive_pipeline.py`.

## License

Apache-2.0.
