"""Tests for the one-shot E2E harness (#2793 harness) — deterministic report + exit code.

Pins the harness's business-value report (six items) and its pass/exit-code semantics in the
offline **stub** mode, so `python -m examples.pet_monitoring.run_e2e` is a reproducible green.
"""

from __future__ import annotations

from pathlib import Path

from examples.pet_monitoring import run_e2e as e2e

# ── the value report: all six items green, deterministic ─────────────────────


async def test_report_all_six_value_items_are_green(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path)  # env={} → offline stub mode

    assert report.passed
    assert report.offline_capable
    assert report.deterministic
    assert report.noise_reduced
    assert report.grounded_decision
    assert report.auto_escalation
    assert report.capability_negotiated
    assert "示範模型" in report.mode  # offline stub mode label
    assert report.escalation_grace_s == 600


async def test_the_three_demos_take_the_expected_paths(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path)
    escalation, ack, offline = report.demos

    # escalation: below-threshold quiet ×2, then breach → locate → escalate → sister
    assert escalation.route_keys == ["quiet", "quiet", "locate"]
    assert (escalation.me_notified, escalation.sister_notified) == (0, 1)
    # ack: the human answers → primary (me), not the sister
    assert (ack.me_notified, ack.sister_notified) == (1, 0)
    # offline: breach but offline → notify me locally, the model never runs
    assert offline.route_keys[-1] == "offline"
    assert (offline.me_notified, offline.sister_notified) == (1, 0)
    assert offline.located is False


async def test_run_is_deterministic(tmp_path: Path):
    r1 = await e2e.run(env={}, tmp=tmp_path / "a")
    r2 = await e2e.run(env={}, tmp=tmp_path / "b")

    assert [d.route_keys for d in r1.demos] == [d.route_keys for d in r2.demos]
    assert [(d.me_notified, d.sister_notified) for d in r1.demos] == [
        (d.me_notified, d.sister_notified) for d in r2.demos
    ]


# ── the rendered output + exit code ──────────────────────────────────────────


def test_main_returns_zero_and_prints_all_green(monkeypatch, capsys):
    monkeypatch.delenv("CONVILYN_EDGE_MODEL_URL", raising=False)

    code = e2e.main()

    out = capsys.readouterr().out
    assert code == 0
    assert "全部綠燈" in out and "[ 紅燈 ]" not in out
    assert "通知妹妹" in out  # the personalized escalation line is shown to the end user


def test_assess_fails_and_exits_nonzero_on_a_wrong_outcome():
    # A broken escalation demo (notified me instead of the sister) must NOT pass → exit 1 material.
    esc = e2e.DemoResult(
        name="escalation",
        route_keys=["quiet", "quiet", "locate"],
        me_notified=1,  # wrong: should have escalated to the sister
        sister_notified=0,
        placement="edge",
    )
    ack = e2e.DemoResult(
        name="ack",
        route_keys=["quiet", "quiet", "locate"],
        me_notified=1,
        located=True,
        placement="edge",
    )
    offline = e2e.DemoResult(
        name="offline",
        route_keys=["quiet", "quiet", "offline"],
        me_notified=1,
        placement="edge",
    )

    report = e2e.assess(
        {"escalation": esc, "ack": ack, "offline": offline}, mode="stub", deterministic=True
    )

    assert report.passed is False
    assert "有紅燈" in e2e.render(report) and "[ 紅燈 ]" in e2e.render(report)


# ── model-backend selection (offline-safe) ───────────────────────────────────


def test_build_detector_uses_stub_without_a_url():
    detector, mode = e2e.build_detector({})

    assert detector is e2e._stub_detector and "示範模型" in mode


def test_build_detector_falls_back_to_stub_when_unreachable():
    # a dead loopback port → health() fails fast → stub fallback (offline-safe, no external net)
    detector, mode = e2e.build_detector({"CONVILYN_EDGE_MODEL_URL": "http://127.0.0.1:1"})

    assert detector is e2e._stub_detector and "連不上" in mode


def test_build_detector_rejects_a_non_http_scheme():
    # F3: a file:// (or any non-http) URL must not reach urllib — fall back to the stub.
    detector, mode = e2e.build_detector({"CONVILYN_EDGE_MODEL_URL": "file:///etc/passwd"})

    assert detector is e2e._stub_detector and "協定" in mode


def test_build_detector_redacts_credentials_in_the_url_label():
    # F1: a URL with embedded credentials must never appear verbatim in the printed mode label.
    _detector, mode = e2e.build_detector({"CONVILYN_EDGE_MODEL_URL": "http://user:s3cret@127.0.0.1:1"})

    assert "s3cret" not in mode and "user" not in mode
