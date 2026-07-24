"""Unit tests for the convilyn-edge CLI."""

from __future__ import annotations

import json

import pytest

from convilyn_edge.cli.main import build_parser, main

_SCENARIO = {
    "device": {"device_id": "scanner-1"},
    "events": [
        {"event_type": "device.scan", "event_schema": "s/v1", "data": {"scan": "471"}},
        {"event_type": "device.scan", "event_schema": "s/v1", "data": {"scan": "888"}},
    ],
}


def _scenario_file(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(_SCENARIO), encoding="utf-8")
    return path


# ── parser ───────────────────────────────────────────────────────────────────


def test_parser_routes_simulate():
    args = build_parser().parse_args(["simulate", "x.json"])

    assert args.command == "simulate"


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0


# ── simulate ─────────────────────────────────────────────────────────────────


def test_simulate_returns_zero(tmp_path):
    code = main(["simulate", str(_scenario_file(tmp_path)), "--no-delay"])

    assert code == 0


def test_simulate_prints_one_wire_line_per_event(tmp_path, capsys):
    main(["simulate", str(_scenario_file(tmp_path)), "--no-delay"])

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2


def test_simulate_emits_valid_wire_json(tmp_path, capsys):
    main(["simulate", str(_scenario_file(tmp_path)), "--no-delay"])

    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["eventType"] == "device.scan"


def test_simulate_missing_file_returns_two(tmp_path):
    code = main(["simulate", str(tmp_path / "nope.json"), "--no-delay"])

    assert code == 2


def test_simulate_malformed_json_returns_two(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not valid json", encoding="utf-8")

    assert main(["simulate", str(path), "--no-delay"]) == 2


def test_simulate_missing_required_key_returns_two(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"events": []}), encoding="utf-8")  # no "device"

    assert main(["simulate", str(path), "--no-delay"]) == 2


# ── init scaffolds ───────────────────────────────────────────────────────────


def test_init_adapter_creates_manifest(tmp_path):
    main(["init", "adapter", "zebra", "--path", str(tmp_path)])

    assert (tmp_path / "adapters" / "zebra" / "adapter.yaml").is_file()


def test_init_adapter_creates_src_dir(tmp_path):
    main(["init", "adapter", "zebra", "--path", str(tmp_path)])

    assert (tmp_path / "adapters" / "zebra" / "src").is_dir()


def test_init_workflow_creates_manifest(tmp_path):
    main(["init", "workflow", "cashier", "--path", str(tmp_path)])

    assert (tmp_path / "workflows" / "cashier" / "workflow.yaml").is_file()


def test_init_workflow_refuses_to_clobber_existing(tmp_path):
    main(["init", "workflow", "cashier", "--path", str(tmp_path)])

    assert main(["init", "workflow", "cashier", "--path", str(tmp_path)]) == 2


def test_init_adapter_returns_zero(tmp_path):
    code = main(["init", "adapter", "zebra", "--path", str(tmp_path)])

    assert code == 0


# ── init path-safety (W1/W2 regressions) ─────────────────────────────────────


def test_init_rejects_traversal_name(tmp_path):
    code = main(["init", "adapter", "../evil", "--path", str(tmp_path)])

    assert code == 2 and not (tmp_path.parent / "evil").exists()


def test_init_rejects_absolute_name(tmp_path):
    victim = tmp_path / "victim"
    code = main(["init", "adapter", str(victim), "--path", str(tmp_path)])

    assert code == 2 and not victim.exists()


def test_init_refuses_to_clobber_existing(tmp_path):
    main(["init", "adapter", "zebra", "--path", str(tmp_path)])

    second = main(["init", "adapter", "zebra", "--path", str(tmp_path)])

    assert second == 2
