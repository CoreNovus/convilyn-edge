"""The ``convilyn-edge`` CLI (stdlib ``argparse`` — zero dependencies).

Commands:

* ``simulate <scenario.json> [--no-delay]`` — replay a scenario through the
  :class:`~convilyn_edge.simulator.SimulatedSource`, printing each
  ``EventEnvelope`` as a wire-JSON line (drive a workflow with no hardware).
* ``init adapter|workflow <name> [--path DIR]`` — scaffold a device-adapter or
  workflow skeleton.

``dev run`` and ``trace replay`` are deferred to v0.2 (they need the workflow
executor + a trace-export format that are out of the v0.1 scope); ``simulate``
is the v0.1 device-plane dev-run, and ``--no-delay`` gives a deterministic replay.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from convilyn_edge._version import __version__
from convilyn_edge.cli.scaffold import scaffold_adapter, scaffold_workflow
from convilyn_edge.simulator import Scenario, SimulatedSource
from convilyn_edge.spi.source import SourceContext


async def _drive(scenario: Scenario, no_delay: bool) -> int:
    source = SimulatedSource(scenario, no_delay=no_delay)
    ctx = SourceContext(device_id=scenario.source.device_id, site_id=scenario.site_id)
    count = 0
    async for envelope in source.start(ctx):
        print(json.dumps(envelope.to_wire(), ensure_ascii=False))
        count += 1
    return count


def cmd_simulate(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    if not scenario_path.is_file():
        print(f"no such scenario file: {scenario_path}", file=sys.stderr)
        return 2
    try:
        scenario = Scenario.load(scenario_path)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"malformed scenario {scenario_path.name}: {exc}", file=sys.stderr)
        return 2
    count = asyncio.run(_drive(scenario, args.no_delay))
    print(f"simulated {count} event(s) from {scenario_path.name}", file=sys.stderr)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path) if args.path else Path.cwd()
    scaffold = scaffold_adapter if args.kind == "adapter" else scaffold_workflow
    try:
        created = scaffold(root, args.name)
    except (ValueError, FileExistsError) as exc:
        print(f"cannot scaffold {args.kind} '{args.name}': {exc}", file=sys.stderr)
        return 2
    print(f"scaffolded {args.kind} '{args.name}' at {created}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convilyn-edge",
        description="Convilyn Edge AI Workflow SDK — device simulator + scaffolds.",
    )
    parser.add_argument("--version", action="version", version=f"convilyn-edge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="replay a scenario through the simulator")
    simulate.add_argument("scenario", help="path to a scenario JSON file")
    simulate.add_argument(
        "--no-delay", action="store_true", help="ignore per-event delays (deterministic replay)"
    )
    simulate.set_defaults(func=cmd_simulate)

    init = sub.add_parser("init", help="scaffold an adapter or workflow skeleton")
    init.add_argument("kind", choices=["adapter", "workflow"])
    init.add_argument("name", help="the adapter / workflow name")
    init.add_argument("--path", default=None, help="root directory (default: cwd)")
    init.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
