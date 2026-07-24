# `convilyn-edge` examples

Runnable, hardware-free examples for the Edge AI Workflow SDK. Each depends **only**
on `convilyn-edge` (zero third-party deps) — none imports a Solution Pack, so they
are honest "build your own edge workflow" starting points.

```bash
pip install --pre convilyn-edge
```

## `drive_pipeline.py` — the hello-world

Author a scenario in code and stream it through the built-in device simulator
(`SimulatedSource`, the first concrete `EventSource`), then fold the events through a
`Pipeline`. Prints each `EventEnvelope` as the exact wire-JSON a real device adapter
would emit — no hardware needed.

```bash
python examples/drive_pipeline.py
```

## Scaffolding your own adapter / workflow

The CLI generates skeletons that implement the SPI Protocols:

```bash
convilyn-edge init adapter my-sensor       # a device adapter (EventSource)
convilyn-edge init workflow my-workflow    # a workflow skeleton
```

## Building a full vertical

An end-to-end vertical composes exactly these primitives: an `EventSource` adapter
feeding envelopes into workflows that write to `ActionSink`s. Ship that composition as
its own **removable** Solution Pack package that depends on `convilyn-edge` — never as
scenario logic inside the SPI. The flagship reference Solution Pack is **pet-monitoring**
(see `backend-api/docs/engineering/edge_solution_pack_moat_and_lockin.md`).
