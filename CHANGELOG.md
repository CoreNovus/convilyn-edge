# Changelog

All notable changes to `convilyn-edge` are documented here. The SDK's public
surface follows Semantic Versioning; pre-1.0 minor/patch may still adjust
surface under the alpha label.

## 0.1.0b20 — 2026-07-24

The integrator-feedback DX round: warm-up, fit-guard, the manufactured-contract
reference story, and pit-of-success discoverability. Additive; no breaking changes.

### Added

- `Runner.warmup(deadline_ms=None) -> WarmupResult` — first-inference latency is now
  distinguishable from unreachability: `warm | cold_started(elapsed) | unreachable`
  three-state result. Default implementation is a no-op (`already warm`), so existing
  runners are unaffected. `OpenAICompatRunner` and `HttpLocalExtractor` implement a
  real probe (health first, then timed minimal inference; `unreachable` is never
  reported as a cold start).
- Declarative device-RAM fit guard: `min_ram_mb` on runner config + fit check at
  runner selection — warns before OOM (opt-in `strict_fit=True` raises instead).
  No device-name branching; purely declarative comparison against the probed profile.
- `ContractModelOperator.for_contract(...)` — one-call assembly: auto
  `load_contract`, environment-based extractor selection, model-binding resolution,
  and `closed_set` steering.

### Changed

- The pet-monitoring example's cat-locate node now mounts the cloud-manufactured
  grounded contract (`authored/*.uw.json`) through `ContractModelOperator`; the
  scripted-sim path remains as the zero-asset offline fallback.

### Docs

- README: `closed_set` / `field_guidance` discoverability section and the
  tiered-testing guidance (T0 structure/schema is SDK-guaranteed;
  T1 model-output quality is the integrator's own harness).

## 0.1.0b19 — 2026-07-23

Closes the Jetson integrator-audit gaps: the **manufactured grounded-contract** consumption path
(the device half of "author once, grounded everywhere" — pet-guardian plan §4 path 1 / §9 gap 2),
grounded **classification** outputs, an enforced schema, and a bounded inference executor.
Additive, stdlib-only, no LLM in any grounding rule.

### Added

- `convilyn_edge.authored` — load + execute a platform-manufactured grounded contract on-device:
  - `GroundedContract` / `GroundedField` — the typed wire mirror of the compiled contract
    (prompt + fields + per-field grounding rule; forward-compatible `extra`; unknown grounding
    modes REJECTED at parse — the device never runs a rule it cannot enforce). Parses either a
    bare contract object or a `uw_*` spec embedding one under `"grounded_contract"`.
  - **Two deterministic grounding modes**: `verbatim` (the existing anchor-substring rule,
    shared with `ground_anchors`) and `closed_set` (value must normalise to one of the
    **authored** labels; the output is always the canonical label). `closed_set` is how a
    classification / derived output ("is the cat present" → `{"true","false"}`, a normalized
    zone label) stays grounded without pretending to be a verbatim substring.
  - `ground_fields` — the total, per-field self-verify (blank-over-fabrication degradation).
  - `load_contract(path)` — read the contract from an installed (digest-verified) bundle
    artifact or a pack's committed `authored/*.uw.json`.
  - `ContractModelOperator` — a generic `ModelOperator[Mapping[str, str], dict[str, str]]`
    that executes the manufactured prompt over any `LocalExtractor`-shaped runner and grounds
    every field. **`schema` is enforced, not decorative**: the effective schema derives from
    the contract (`GroundedContract.schema()`), and a caller schema naming undeclared fields
    raises.
- `EdgeModelOperator(..., executor=)` + `.close()` — blocking extraction now runs on a
  per-operator **bounded** single-thread executor by default (shared pool no longer used), so a
  deadline-abandoned call on a slow local model (Jetson cold-load ~70 s) leaks at most one
  thread and later calls apply honest backpressure instead of stacking threads. Pass a shared
  `executor` to pool across operators.

### Changed

- sdist now ships `examples/` (the pet-monitoring reference Solution Pack) — the integrator's
  scaffold template travels with the source distribution. Wheels stay lean by design; the SDK
  is consumed from source (editable install) per the pet-guardian plan §8.
- `build_extract_messages(..., field_guidance=)` + `HttpLocalExtractor.field_guidance` — generic
  per-key answer-rule overrides in the extractor message; `guidance_from_contract(contract)`
  renders them for `closed_set` fields so the reference runner steers toward the authored labels
  instead of the (contradictory) verbatim-only rule. With no guidance the message stays
  byte-identical to v1 — the client-compute interrupt flow is unchanged.

### Review hardening (post-audit code review)

- Integral floats (`3.0`) coerce to the integer label text (`"3"`) in closed_set matching.
- Explicit-null / wrong-type wire values (`contract_id: null`, a bare-string `allowed_values`)
  now fail loud at parse instead of coercing to `"None"` / per-character labels.
- An authored closed_set label colliding with the missing sentinel is rejected at construction
  (a degraded answer would be indistinguishable from a grounded one).
- Schema guard also rejects a caller `enum` that drifted from the authored `allowed_values`.
- Both operators are (a)sync context managers; `close()` releases the owned bounded executor.

## 0.1.0b18 — 2026-07-23

Adds the **declarative conditional branch/router** primitive — the pipeline's only
branching construct. Additive, stdlib-only, no LLM.

### Added

- `Pipeline.route(*, select, routes, default=None, name="route")` — a pure `select(state) -> key`
  picks ONE named branch (each a `Pipeline`) to run inline. The chosen branch is folded over the
  current state (sees every upstream output, threads its own onto the same state) and short-circuits
  by the same deterministic rules as the top level. `default` runs on an unmapped key; with no usable
  default an unmapped key terminates the pipeline `rejected` (never a silent pass-through). This is
  how a Solution Pack expresses "offline → check connectivity, else → locate" as a **declaration** the
  runtime drives — not a hand-wired `async if`/`else` in integrator code (the lock-in-charter rule 1).
- `convilyn_edge.runtime.RouteOutcome` — the stored `route`-stage value (chosen `key` + the branch's
  step outcomes), for downstream bindings and audit.

### Notes

- Deterministic + scenario-free: `select` is the pack's binding; the substrate holds no scenario
  vocabulary and no LLM. The linear fold is now shared (`_drive_steps`) by `Pipeline.run` and the
  router, so a branch runs by identical rules to the top level. Nested routers compose.

## 0.1.0b17 — 2026-07-23

Adds the **stateful threshold aggregator** — cross-event accumulation as a
`StateProvider`, durable through the storage seam. Additive, stdlib-only, no LLM.

### Added

- `convilyn_edge.runtime.ThresholdAggregator(store, *, threshold, window_s, key_of, now=None)`
  — a `StateProvider[ThresholdState]` that counts qualifying occurrences within a sliding time
  window and reports `breached` when the count reaches `threshold`. Wires straight into
  `Pipeline.state(aggregator)`; the deterministic operator downstream reads the snapshot.
  - **Durable via the storage seam:** `store` is the `DurableQueue` from
    `select_storage_provider(manifest, internal_root=…).durable_store(name)` — the aggregator
    reuses that one apply-once record store (survives restart / offline window); it does **not**
    open a parallel state substrate. The store must be **dedicated** (sole-owner), since each
    snapshot rewrites it to the surviving in-window occurrences.
  - **Integrity — trusted clock, never event content:** the window is anchored to an injected
    trusted clock (`now`, default UTC wall clock), and occurrences are stamped with the trusted
    receive time — NOT `envelope.time` (attacker-settable via the wire). The count band is strictly
    two-sided, so a planted/spoofed future-dated record is not counted. Records read back are
    shape-validated (string `idempotency_key`/`bucket` + parseable `at`), so a corrupt/foreign
    record on removable media is dropped, never crashing a snapshot (incl. boundary-date offsets
    that overflow the UTC conversion).
  - **Semantics + clock contract:** counts by *receive* time (an offline-flush burst is counted at
    flush time — inject a per-event historical `now` for occurrence-time fidelity). `now` should be
    non-decreasing; a backward step (NTP) up to one window is tolerated — a just-recorded occurrence
    is retained (not deleted) and re-counts once the clock recovers.
  - **Replay-stable within the window:** recording is idempotent by `event_id` within the
    retained window; a pinned-clock replay reproduces exactly. `event_id` must be a unique
    per-occurrence identity (the envelope contract); a source that reuses ids collapses occurrences.
  - **Scenario-free:** `key_of(envelope) -> str | None` is the pack's binding — a constant key
    counts everything together, a composite key counts per bucket; the aggregator counts, it
    never knows the scenario. No `if silicon ==`, no LLM.
- `convilyn_edge.runtime.ThresholdState` / `Occurrence` / `OccurrenceKey` — the snapshot value,
  a counted occurrence (event id + time), and the binding type.

### Fixed

- `runtime.watchdog.seconds_until_expiry` now degrades to "no deadline" on a boundary-date
  `expires_at` (year 0001/9999 with an offset) that overflows the UTC conversion, instead of
  raising an uncaught `OverflowError` — same parse-hardening as the aggregator's record reader.

## 0.1.0b16 — 2026-07-23

Adds the **timer / watchdog** runtime primitives — the deterministic clock the
purely event-driven driver + pipeline lacked. Additive, stdlib-only, no LLM.

### Added

- `convilyn_edge.runtime.resolve_review(review, request, *, now=…, sleep=…)` — awaits a
  human decision but honours `ReviewRequest.expires_at`: when the absolute UTC deadline
  elapses with no answer, it resolves to the request's declared `default_action`
  (`stop` / `escalate` / `continue`). Deterministic (injectable clock), never an LLM.
  An expiry-less request awaits the human exactly as before (additive). The pipeline's
  review stage now routes through it, so `expires_at` is load-bearing in the declarative DAG
  (e.g. "escalate to the sister after 10 min" is a declaration the runtime enforces).
- `convilyn_edge.runtime.seconds_until_expiry(expires_at, *, now)` — the pure deadline helper
  (accepts the `Z` suffix; an unparseable/absent deadline → `None` = "no deadline").
- `convilyn_edge.runtime.WatchdogPolicy` + `poll_health(source, policy, *, stop, sleep=…)` —
  a fixed-interval `source.health()` poll so a device that goes silent *between* events is
  still observed; each reading is handed to a deterministic `on_health` observer
  (`transitions_only` de-duplicates steady state). Offline-first: a failing poll/observer is
  swallowed, never crashing the driver.
- `WorkflowDriver.run(..., watchdog=None)` — an opt-in health-poll clock running beside the
  event loop, torn down on every exit path. Default `None` keeps the pure event-driven
  behaviour byte-for-byte unchanged.
- `Pipeline.review(..., now=None, sleep=None)` — the review stage's expiry clock is
  injectable, so an `expires_at` review is replay-reproducible (consistent with the driver).
- `ReviewOutcome.decision_source: DecisionSource` (`"human"` default | `"expiry_default"`) —
  a **structured** discriminator (not a note substring) so an approval gate can refuse to
  treat a synthesized timeout default as a human grant. Additive (defaults to `"human"`).

### Notes

- Scenario-free: the watchdog *observes*; the removable Solution Pack *decides* what an
  offline transition means (`on_health` is the pack's hook, never a branch in the substrate).
- Fail-safe: `default_action` defaults to `stop`; expiry cannot bypass the `ActionGate`
  (R2+ still needs an explicit approval), and a one-time `on_health` failure retries the
  transition rather than losing the offline edge.

## 0.1.0b15 — 2026-07-23

Completes the **storage-selection axis**: the deterministic tier-selection
table. Additive, stdlib-only.

### Added

- `convilyn_edge.select_storage_provider(manifest, *, internal_root, removable_root=None)`
  — picks the device's durable-storage tier deterministically from its reported profile
  and returns the `LocalStorageProvider` for it. A fixed priority table
  (`media_store > removable > internal`), pick-first-available; no `if silicon == …`.
  `media_store` uses the manifest's declared path; `removable` is opt-in (the caller
  supplies its mount root, so a state-critical caller omits it and falls to a fixed tier
  while a capacity caller supplies it); `internal` is the always-present floor.

## 0.1.0b14 — 2026-07-23

Adds the **storage-selection seam** (`convilyn_edge.storage`) — *where* a
device persists durable state, abstracted from *what* persists. Additive, stdlib-only.

### Added

- `convilyn_edge.StorageProvider` — the injectable seam: `durable_store(name)` hands out
  a named durable, apply-once record store on the provider's tier; `tier` names which
  storage tier (`StorageTier = internal | removable | media_store`, mirroring the
  manifest storage fields).
- `convilyn_edge.LocalStorageProvider` — a filesystem-rooted concrete: each store is the
  **reused** `DurableQueue` (`offline/queue.py`) at `<root>/<safe name>.jsonl` — the seam
  adds *tier placement* over the existing durable store, it does not reinvent one. A
  store `name` is reduced to a single safe file-name component, so a hostile name
  (`"../.."`, a separator, an absolute path) can never escape the provider root.

### Notes

- Scenario-free: the seam knows tiers, not workflows (no `if silicon == …`). Which tier a
  device uses is a deterministic function of its manifest storage profile — that selection
  table lands in the tier-selection step, and it must precede the threshold aggregator that
  persists through this seam.

## 0.1.0b13 — 2026-07-23

Adds the **storage-selection axis** to `DeviceCapabilityManifest` — additive
to the frozen device-capability matrix. The storage-tier seam (next PRs) selects a target
from what the device reports.

### Added

- `DeviceCapabilityManifest.disk_gb: int | None` — on-device storage in GiB (`None`
  when unreported); `probe_device` auto-detects it via `shutil.disk_usage` (graceful,
  never raises).
- `DeviceCapabilityManifest.removable: bool` — a removable medium (SD/USB) is present.
- `DeviceCapabilityManifest.media_store: str | None` — path/id of a large-asset / media
  store mount, or `None`. `probe_device` gains `removable=` / `media_store=` kwargs (the
  integrator supplies these — not reliably auto-detectable).

### Notes

- Frozen-matrix discipline: the fields were added to `docs/reference/edge_device_matrix.md`
  §1a first, then the SDK probe + the backend model in lockstep; both parity pins
  (`tests/test_probe_runtime.py` + backend `test_edge_device_matrix_parity.py`) updated.
  The SDK `to_wire()` still round-trips exactly into the backend's `extra="forbid"` model.

## 0.1.0b12 — 2026-07-22

Adds **install** (`convilyn_edge.bundle.install`) — the final Path B step that lands a
verified bundle on-device. With it the SDK completes pull → stage → verify → install.
Additive, stdlib-only.

### Added

- `convilyn_edge.bundle.install_verified_bundle` → `InstallReport` / `InstalledArtifact`.
  Lands each artifact of a `VerifiedBundle` into a **hash-addressed** device store
  (`sha256(key)` sharded — a tampered key cannot escape the store), via atomic
  `os.replace` honouring the verify→install same-inode contract (store and staging
  scratch must share a filesystem; a cross-device move fails loud as `InstallError`).
  A fail-closed **installable-kinds allowlist** rejects an unrecognised kind.
  **Idempotent, integrity-aware:** a byte-verified artifact (`workflow_spec`) is
  `skipped_present` only if the *stored* file still hashes to its pinned digest — a
  corrupted spec self-heals — while an out-of-band GB weight is skipped on presence
  alone (no re-hash). GB weights are never re-written.
- `InstallReport.installed_assets` reconstructs the `InstalledAsset` for each installed
  `role_asset` / `tool_asset` from its resolver key (the sole point mirroring the
  server's `artifact_keys._asset_key`, parity-pinned by a golden vector).
- `merge_installed_assets(manifest, assets)` folds them into a
  `DeviceCapabilityManifest` **additively** (existing entries never dropped/reshaped,
  duplicates ignored) — aligned to the frozen device-capability matrix.

### Notes

- Idempotency digest reuses a byte-verified artifact's already-checked `content_hash`
  (O(1)); an out-of-band / deferred artifact's file is hashed once at install.
- CLI verb + the authenticated cloud payload-fetch (`POST /edge/push`) are a separate
  follow-up; this PR delivers the install mechanism (LANE roadmap PR③).

## 0.1.0b11 — 2026-07-22

Adds **digest verification** (`convilyn_edge.bundle.verify`) — the gate between a
staged bundle and install. Fail-loud, read-only, no landing. Additive, stdlib-only.

### Added

- `convilyn_edge.bundle.verify_staged_bundle` / `verify_staged_artifact` →
  `VerifiedBundle` / `VerifiedArtifact` (+ `VerificationStatus`). For each staged
  artifact it recomputes the `sha256:…` digest of the fetched bytes and compares to
  the manifest's `content_hash`, per the digest's domain:
  - **`workflow_spec`** (and any future kind that pins a digest) → verified here by
    recomputing over the fetched bytes (a stripped required digest fails closed, so
    a null can't downgrade to "nothing to verify").
  - **`trusted_core`** → `content_hash` is a digest over the device's own S2 source
    tree, not the fetched bytes → **deferred** to the Trusted-Core fail-closed gate,
    never byte-hashed here.
  - **`role_asset` / `tool_asset` / `runtime`** → no wire digest → `out_of_band`.
- `VerificationError` + `DigestMismatchError` / `UnsupportedDigestError` /
  `MissingDigestError`. The first failure raises and no `VerifiedBundle` is produced
  — the caller discards the scratch staging, so nothing tampered can be installed.

### Notes

- Digest domains verified against the server's `bundle_producer._content_hash`
  (`sha256:` over `model_dump_json()`) and `artifact_keys.iter_push_artifacts`.
- Read-only: this layer hashes staged files and raises; it never moves, deletes, or
  installs (landing is the next PR's install layer, over a bundle this returned).

## 0.1.0b10 — 2026-07-22

Adds the **Path B bundle consumption** surface (`convilyn_edge.bundle`) — the
device half of "a chat-built workflow imported straight onto the device". This PR
is the *pull → stage* layer (fetch only); digest-verify and install land next.
Additive and stdlib-only; no existing surface changes.

### Added

- `convilyn_edge.bundle.DeviceInstallablePayload` / `ResolvedArtifact` — the
  zero-dependency mirror of the server's push payload (`from_wire` parse,
  forward-compatible `extra`, hard-required `kind`/`key`/`url`). `artifact_fingerprint`
  reproduces the server's **URL-free** provenance digest byte-for-byte (a re-pull
  is observably idempotent); `log_summary` is the only sanctioned way to log a
  payload — it structurally cannot carry a signed URL or a resolver key.
- `convilyn_edge.bundle.UrllibArtifactFetcher` + the `ArtifactFetcher` seam — the
  network egress boundary: **https-only** (refuses `file:`/`http:`/`ftp:`/`data:`
  before a socket opens, and the dangerous handlers are not even loaded),
  TLS-verified, **no-redirect**, streaming (peak memory independent of asset size).
  Bounded by a per-read idle timeout **and** a whole-transfer wall-clock deadline
  (both on by default) plus an optional `max_bytes` ceiling. The signed URL never
  reaches an error message *or a chained traceback* (`raise … from None`).
  `fetch_to_bytes` is the small-artifact convenience.
- `convilyn_edge.bundle.stage_bundle` → `StagedBundle` / `StagedArtifact` — download
  a whole payload to a caller-owned scratch directory, in payload order, torn-file
  safe (a failed fetch removes its partial and fails loud). The scratch dir is
  created private (`0o700`) and each file is opened **no-follow** (POSIX) so a
  planted symlink can't redirect a write; each artifact is size-bounded by an
  explicit `max_artifact_bytes` or its server-advertised `size_bytes`. Staging is
  **not** installing: nothing is verified and nothing lands in the device store here.

### Notes

- Guards the zero-dependency discipline: `convilyn_edge.bundle` imports only
  stdlib and nothing from the backend — it mirrors the server's payload shape by
  value, and the parity anchor is pinned by a golden `artifact_fingerprint` vector
  (`tests/test_bundle_payload.py`).
- Trust-boundary hardening from code-review + security-review (the download/parse
  path): credential-leak via the exception `__cause__` chain, unbounded-by-default
  transfers (disk-fill / OOM / slow-drip), the SSRF scheme guard as a single point
  of failure, and symlink-write in a shared scratch dir were all closed in this PR.

## 0.1.0b9 — 2026-07-22

Post-review robustness hardening of `WorkflowDriver` (no surface change). All three
gaps lived in previously-untested branches.

### Fixed
- **Source lifecycle teardown.** `WorkflowDriver.run` now closes the event stream
  (defensive `aclose`) and always calls `source.stop()` in a `finally`, so a real
  hardware source (serial port, MQTT subscription, camera) is torn down on every
  exit path (bounded `max_events`, stuck-abort, exception) instead of leaking until GC.
- **Guarded terminal flush.** A failure draining the offline buffer at run end
  (e.g. durable-queue disk I/O) is swallowed (`flushed=None`) instead of discarding
  the accumulated `DriveReport` — offline-first: the buffer survives for a later pass.
- **Stuck-guard keys on the exception type**, not the formatted message, so a fault
  whose message embeds variable data (offset/id/timestamp) still trips the cycle
  guard on repetition. New `EventRecord.failure_kind` carries the stable signature.

## 0.1.0b8 — 2026-07-22

Adds the **on-device model runners** (`convilyn_edge.runners`) and a matrix-aligned
runtime resolver. Additive and stdlib-only; no existing surface changes.

### Added

- `convilyn_edge.runners.OpenAICompatRunner` — the productized HTTP-local model
  runner (Ollama + any OpenAI-compatible server), a `ModelOperator` over the
  grounded extractor. Zero runtime dependencies.
- `convilyn_edge.runners.QnnOnnxRunner` — a Protocol-conformant **skeleton** for
  the Snapdragon NPU runtime (ONNX Runtime + QNN EP). Hardware-deferred: `infer`
  fails loud as `status="unavailable"` — never raises, never fabricates. An
  integrator overrides `_execute` on real hardware (LSP).
- `convilyn_edge.runners.select_runner` + `RunnerConfig` — deterministic runner
  selection by a **runtime-string table lookup** (never an `if silicon ==`
  branch); `SUPPORTED_RUNTIMES`; `UnsupportedRuntimeError` on an unknown runtime.
- `convilyn_edge.runners.failing_fields` / `failing_fields_count` — grounding
  introspection over a runner's `ModelResult`.
- `convilyn_edge.resolve_runtime(silicon)` — the deterministic per-silicon runtime
  map, mirroring the frozen device matrix by value (zero backend import).

### Notes

- The `DeviceCapabilityManifest` wire schema is **unchanged** — an earlier freeze pinned it as
  additive-only, so runtime/format/quantization ride the existing
  `InstalledAsset.runtime` field, not new manifest keys. A parity pin test
  (`tests/test_probe_runtime.py`) guards the frozen field set.

## 0.1.0b7 — 2026-07-22

Adds the **zero-dependency runtime driver** — the `convilyn_edge.runtime`
subpackage that composes the seven SPI primitives into a self-driving edge
workflow. Additive and stdlib-only; no change to any existing surface.

### Added

- `convilyn_edge.runtime.Pipeline` — a declarative, immutable composition of the
  primitives in their natural edge topology (Normalizer → StateProvider →
  Deterministic/ModelOperator → HumanReview → ActionSink). Per-stage binding
  functions keep scenario mapping in the Solution Pack; the substrate holds none.
- `convilyn_edge.runtime.WorkflowDriver` — the event loop over an `EventSource`
  with offline-first emission + flush, deterministic fixed-backoff retry, and a
  stuck-abort cycle-guard. Infrastructure outcomes (processed/buffered/failed)
  are separated from a pipeline's business outcome.
- Deterministic runtime gates: `RetryPolicy` / `CycleGuard` / `retry_async`
  (retry & backoff, replay-safe — no jitter unless injected), `ActionGate`
  (R0/R1 auto-grant, R2+ authorization gating), and `model_disposition` /
  `review_disposition` (`ModelResult.status` / `HumanReview` dispatch). None are
  LLM-driven — the seven server-side safety checks stay server-authoritative.
- Example `examples/drive_pipeline.py` — a self-driving pipeline over the
  simulator (no hardware, no cloud, no Solution Pack).

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
  `client_compute` `ModelOperator`, offline queue + emitter, and simulator + CLI
  follow in the subsequent betas.
