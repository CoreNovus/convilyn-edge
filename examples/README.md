# `convilyn-edge` examples

Runnable, hardware-free examples for the Edge AI Workflow SDK. Each depends **only**
on `convilyn-edge` (zero third-party deps) — none imports a Solution Pack, so they
are honest "build your own edge workflow" starting points.

```bash
pip install --pre convilyn-edge
```

## `simulate_barcode.py` — the hello-world

Author a barcode-scan scenario in code and stream it through the built-in device
simulator (`SimulatedSource`, the first concrete `EventSource`). Prints each
`EventEnvelope` as the exact wire-JSON a real scanner adapter would emit.

```bash
python examples/simulate_barcode.py
```

## `barcode_scenario.json` — the CLI equivalent

The same scans as a scenario file, replayed through the `convilyn-edge` CLI (no
Python glue needed):

```bash
convilyn-edge simulate examples/barcode_scenario.json --no-delay
```

## Scaffolding your own adapter / workflow

The CLI generates skeletons that implement the SPI Protocols:

```bash
convilyn-edge init adapter zebra-datawedge     # a device adapter (EventSource)
convilyn-edge init workflow cashier-guidance   # a workflow skeleton
```

## Building a full vertical

For an end-to-end workflow composition — a barcode scanner driving three retail
workflows into POS action sinks — see the **retail Solution Pack** and its
`examples/three_stage_retail.py`, which builds on this SPI:

```bash
pip install --pre convilyn-solution-retail-cashier
```
