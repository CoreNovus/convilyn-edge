"""Drive the built-in device simulator with no hardware — the `convilyn-edge` hello-world.

Runs a barcode-scan `Scenario` through `SimulatedSource` (the first concrete
`EventSource`) and prints each `EventEnvelope` as wire-JSON — exactly what a real
scanner adapter would emit. Depends ONLY on `convilyn-edge` (zero third-party
deps); it never imports a Solution Pack, so it is a faithful "build your own edge
workflow" starting point.

    python examples/simulate_barcode.py

For the CLI equivalent over a JSON scenario file, see `examples/barcode_scenario.json`:

    convilyn-edge simulate examples/barcode_scenario.json --no-delay
"""

from __future__ import annotations

import asyncio
import json

from convilyn_edge import EventSourceRef
from convilyn_edge.simulator import Scenario, ScenarioEvent, SimulatedSource
from convilyn_edge.spi.source import SourceContext

BARCODE_SCHEMA = "convilyn://schemas/barcode-scan/v1"


def build_scenario() -> Scenario:
    """Three scans a register would see — authored in code (no scanner required)."""
    scans = [
        {"data": "4710088412345", "symbology": "ean13"},  # a normal product
        {"data": "PAY:LINEPAY", "symbology": "qr"},  # a payment code
        {"data": "MEMBER:0912345678", "symbology": "code128"},  # a member card
    ]
    events = tuple(
        ScenarioEvent(
            event_type="device.barcode.scan.received",
            event_schema=BARCODE_SCHEMA,
            data={"device_id": "reg-07", **scan},
        )
        for scan in scans
    )
    return Scenario(
        source=EventSourceRef("reg-07", "sim-scanner", "0.1.0"),
        events=events,
        site_id="demo-store",
    )


async def main() -> None:
    source = SimulatedSource(build_scenario(), no_delay=True)
    ctx = SourceContext(device_id="reg-07", site_id="demo-store")
    count = 0
    async for envelope in source.start(ctx):
        # `to_wire()` is the exact camelCase JSON that crosses the SDK boundary.
        print(json.dumps(envelope.to_wire(), ensure_ascii=False))
        count += 1
    print(f"\nsimulated {count} scan event(s) with no hardware.")


if __name__ == "__main__":
    asyncio.run(main())
