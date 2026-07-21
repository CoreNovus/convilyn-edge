# Changelog

All notable changes to `convilyn-edge` are documented here. The SDK's public
surface follows Semantic Versioning; pre-1.0 minor/patch may still adjust
surface under the alpha label.

## 0.1.0b6 — 2026-07-21

- Install and development instructions are now uv-first across the README and examples (pip remains a documented alternative). No API or behaviour change.

## 0.1.0b5 — 2026-07-21

- Documentation polish across the public surface: docstrings, guides, examples, and this changelog now use plain product language throughout (internal shorthand and tracker references removed). No API or behaviour change.
- Package metadata now points at the package's own public repository instead of the consumer SDK's.

## 0.1.0b4 — 2026-07-20

Documentation-only release. No API or behaviour change.

### Changed

- Cleaned the shipped package surface: removed internal engineering citations
  from source docstrings and the README, and refreshed the changelog. The public
  Protocols, types, CLI, and wheel contents are byte-identical in behaviour to
  0.1.0b3 (`Requires-Dist` remains empty — zero runtime dependencies).

## 0.1.0b3 — 2026-07-20

Drive a workflow with no hardware, and scaffold new adapters/workflows.

### Added

- **`convilyn_edge.simulator`** — `SimulatedSource`, the first concrete
  `EventSource` implementation: replay a JSON `Scenario` (device + ordered
  events, with per-event `delay_ms` / `repeat`) as an `EventEnvelope` stream. The
  simulator half of the boundary the SDK ships; zero dependencies.
- **The `convilyn-edge` CLI** (stdlib `argparse`, zero deps):
  - `simulate <scenario.json> [--no-delay]` — replay a scenario, printing each
    envelope as a wire-JSON line (the v0.1 device-plane dev-run; `--no-delay`
    gives a deterministic replay).
  - `init adapter|workflow <name> [--path DIR]` — scaffold a conventional
    adapter / workflow skeleton.

### Deferred to v0.2

- `dev run` / `trace replay` — they need the workflow executor + a trace-export
  format that are out of v0.1 scope. `simulate --no-delay` is the deterministic
  replay subset for now.

## 0.1.0b2 — offline queue + Event Envelope emitter

Offline-first buffering: the device keeps working when the cloud is unreachable,
and flushes exactly once when it returns.

### Added

- **`convilyn_edge.offline`** — zero runtime dependencies:
  - `DurableQueue` — a JSONL-backed, **idempotent** FIFO queue: a record whose
    apply-once key is already queued is dropped, so a re-queue never duplicates a
    pending item. `pending` / `enqueue` / `clear` / `replace` (atomic rewrite).
  - `derive_idempotency_key` — the content-addressed apply-once key, a byte-exact
    mirror of the server reconcile contract (device clock excluded), so the
    server re-derives the same key and a re-flush is a no-op.
  - `EventEmitter` + `EventSink` — emit an `EventEnvelope` to an injected sink;
    when the sink is offline the envelope is buffered durably (keyed by its unique
    `event_id`) instead of dropped, and `flush()` drains the buffer apply-once.
    Reuses the `traceparent`-style correlation convention; the transport is
    injected, so the package takes on no HTTP/MQTT dependency.

## 0.1.0b1 — client_compute keystone

The on-device extractor keystone — `convilyn-edge` now **confirms-and-consumes**
the frozen `client_compute` interrupt contract v1, so a cloud workflow can
delegate the extractor role to the device and the device answers with grounded
anchors (the server re-grounds before trusting them).

### Added

- **`convilyn_edge.clientcompute`** — the client-compute round-trip, zero runtime
  dependencies:
  - `contract` — parse the frozen v1 interrupt payload (`ClientComputeRequest`,
    `AnchorsContract`), `build_resume_answer`, and `ground_anchors` — the device
    self-verify that mirrors the server's substring re-grounding (verbatim
    substring or the `"Not specified"` sentinel; per-value + total caps enforced).
  - `engine` — `LocalExtractor` Protocol + `HttpLocalExtractor` (Ollama and
    OpenAI-compatible wires over stdlib `urllib`; transport injectable for tests).
  - `operator` — `EdgeModelOperator`, an `edge`-placement `ModelOperator` that
    runs the local model + grounds and returns a typed `ModelResult`.
  - `bridge` — `ClientComputeBridge`, which drives one interrupt to resume via
    narrow injected Protocols (`GoalClientPort`, `FileTextResolver`) — the
    consumer SDK is injected, never imported (the core stays dependency-free).

## 0.1.0b0 — SPI scaffold

Initial scaffold of the Edge AI Workflow SDK — the SPI only, zero runtime
dependencies.

### Added

- **Event Envelope** (`convilyn_edge.envelope`): `EventEnvelope`,
  `EventSourceRef`, `Correlation`, and the `new_envelope(...)` factory, with
  `to_wire()` / `from_wire()` camelCase serialization.
- **`Result[T, E]`** (`convilyn_edge.result`): an `Ok` / `Err` tagged union for
  exception-free, statically-narrowable fallible outcomes.
- **The 7-primitive SPI** (`convilyn_edge.spi`): `EventSource`, `Normalizer`,
  `StateProvider`, `DeterministicOperator`, `ModelOperator`, `HumanReview`,
  `ActionSink` — typed Protocols + their frozen dataclasses.
- Package scaffold: `src/` layout, `py.typed`, dynamic version, `dependencies = []`.

### Notes

- No implementations ship in b0 — Protocols + data shapes only. The
  `client_compute` `ModelOperator`, offline queue + emitter, simulator + CLI, and
  the removable retail Solution Pack follow in the subsequent betas.
