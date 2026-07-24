"""Unit tests for ClientComputeBridge — the end-to-end round-trip (fakes only)."""

from __future__ import annotations

from typing import Any

import pytest

from convilyn_edge.clientcompute.bridge import (
    ClientComputeBridge,
    MissingLocalSourceError,
    find_client_compute_interrupt,
)
from convilyn_edge.clientcompute.operator import EdgeModelOperator

_PAYLOAD = {
    "nonce": "n-1",
    "required_anchors": ["title"],
    "extractor_prompt": "Extract.",
    "extractor_prompt_id": None,
    "file_ids": ["file-a"],
    "anchors_contract": {"missing_sentinel": "Not specified"},
    "strategy": "client_delegated",
}


class _Job:
    def __init__(self, interrupts: list[dict[str, Any]], *, version: int | None = 7):
        self.job_spec_id = "job-1"
        self.item_version = version
        self.pending_interrupts = interrupts


class _FakeGoalClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], int | None]] = []

    async def fill_slots(self, job_spec_id, answers, *, expected_version=None):
        self.calls.append((job_spec_id, answers, expected_version))
        return _Job([])


class _FakeExtractor:
    def __init__(self, result: dict[str, Any]):
        self._result = result

    def extract(self, *, prompt, sources, required_anchors):
        return self._result


class _MapResolver:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def resolve(self, file_id: str) -> str | None:
        return self._mapping.get(file_id)


def _interrupt(status: str = "pending", itype: str = "client_compute") -> dict[str, Any]:
    return {
        "interruptType": itype,
        "interruptId": "int-1",
        "status": status,
        "payload": _PAYLOAD,
    }


def _bridge(extract_result: dict[str, Any], sources: dict[str, str]) -> ClientComputeBridge:
    op = EdgeModelOperator(_FakeExtractor(extract_result))
    return ClientComputeBridge(op, _MapResolver(sources))


# ── find_client_compute_interrupt ────────────────────────────────────────────


def test_find_returns_pending_client_compute():
    job = _Job([_interrupt()])

    assert find_client_compute_interrupt(job)["interruptId"] == "int-1"


def test_find_ignores_non_pending():
    job = _Job([_interrupt(status="resolved")])

    assert find_client_compute_interrupt(job) is None


def test_find_ignores_other_interrupt_types():
    job = _Job([_interrupt(itype="slot_fill")])

    assert find_client_compute_interrupt(job) is None


# ── round-trip (handle_if_present) ───────────────────────────────────────────


async def test_round_trip_submits_grounded_anchor():
    bridge = _bridge({"title": "Staff Engineer"}, {"file-a": "Role: Staff Engineer."})
    client = _FakeGoalClient()

    await bridge.handle_if_present(client, _Job([_interrupt()]))

    assert client.calls[0][1] == {"int-1": {"anchors": {"title": "Staff Engineer"}, "nonce": "n-1"}}


async def test_round_trip_forwards_expected_version():
    bridge = _bridge({"title": "Staff Engineer"}, {"file-a": "Role: Staff Engineer."})
    client = _FakeGoalClient()

    await bridge.handle_if_present(client, _Job([_interrupt()], version=42))

    assert client.calls[0][2] == 42


async def test_ungrounded_round_trip_submits_sentinel():
    bridge = _bridge({"title": "Hallucinated"}, {"file-a": "unrelated text"})
    client = _FakeGoalClient()

    await bridge.handle_if_present(client, _Job([_interrupt()]))

    assert client.calls[0][1]["int-1"]["anchors"] == {"title": "Not specified"}


async def test_no_interrupt_returns_none_and_no_call():
    bridge = _bridge({}, {})
    client = _FakeGoalClient()

    result = await bridge.handle_if_present(client, _Job([]))

    assert result is None and client.calls == []


# ── error: missing local source ──────────────────────────────────────────────


async def test_referenced_file_with_no_local_copy_raises():
    bridge = _bridge({"title": "X"}, {})  # resolver has nothing for file-a
    client = _FakeGoalClient()

    with pytest.raises(MissingLocalSourceError):
        await bridge.handle_if_present(client, _Job([_interrupt()]))
