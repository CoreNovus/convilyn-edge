"""Unit tests for EdgeModelOperator (async; fake extractor, no network)."""

from __future__ import annotations

import time
from typing import Any

from convilyn_edge.clientcompute.contract import AnchorsContract
from convilyn_edge.clientcompute.operator import EdgeModelOperator, ExtractInput

_SOURCES = {"f": "The role is Staff Engineer at Acme."}
_SCHEMA: dict[str, Any] = {"type": "object"}


class _FakeExtractor:
    def __init__(self, result=None, *, raises: Exception | None = None, sleep: float = 0.0):
        self._result = result or {}
        self._raises = raises
        self._sleep = sleep

    def extract(self, *, prompt, sources, required_anchors):
        if self._sleep:
            time.sleep(self._sleep)
        if self._raises is not None:
            raise self._raises
        return self._result


def _input(required=("title",)) -> ExtractInput:
    return ExtractInput(prompt="p", sources=_SOURCES, required_anchors=list(required))


# ── logic / object-state ─────────────────────────────────────────────────────


async def test_grounded_extraction_is_success():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.status == "success"


async def test_success_output_carries_grounded_anchor():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.output == {"title": "Staff Engineer"}


async def test_success_emits_evidence():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.evidence[0].snippet == "Staff Engineer"


async def test_confidence_is_grounded_fraction():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer", "x": "nope"}))

    result = await op.infer(_input(("title", "x")), schema=_SCHEMA)

    assert result.confidence == 0.5


# ── boundary: ungrounded → uncertain ─────────────────────────────────────────


async def test_ungrounded_extraction_is_uncertain():
    op = EdgeModelOperator(_FakeExtractor({"title": "Not in source"}))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.status == "uncertain"


async def test_uncertain_still_returns_total_output():
    op = EdgeModelOperator(_FakeExtractor({"title": "Not in source"}))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.output == {"title": "Not specified"}


# ── error: extractor failure / deadline → unavailable ────────────────────────


async def test_extractor_exception_is_unavailable():
    op = EdgeModelOperator(_FakeExtractor(raises=RuntimeError("model down")))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.status == "unavailable"


async def test_unavailable_has_no_output():
    op = EdgeModelOperator(_FakeExtractor(raises=RuntimeError("model down")))

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.output is None


async def test_deadline_exceeded_is_unavailable():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}, sleep=0.2))

    result = await op.infer(_input(), schema=_SCHEMA, deadline_ms=1)

    assert result.status == "unavailable"


async def test_custom_sentinel_is_not_counted_as_grounded():
    # A server-advertised custom sentinel must be classified against itself: an
    # ungrounded value degraded to "N/A" is NOT grounded (→ uncertain), not
    # mistaken for a real value because it differs from the module default.
    contract = AnchorsContract(missing_sentinel="N/A")
    op = EdgeModelOperator(_FakeExtractor({"title": "not in the source"}))
    inp = ExtractInput(prompt="p", sources=_SOURCES, required_anchors=["title"], contract=contract)

    result = await op.infer(inp, schema=_SCHEMA)

    assert result.status == "uncertain"


async def test_model_id_is_reported():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}), model_id="qwen3:4b")

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.model_id == "qwen3:4b"


# ── object-state: bounded executor lifecycle ─────────────────────────────────


async def test_infer_after_close_recreates_the_bounded_executor():
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}))
    await op.infer(_input(), schema=_SCHEMA)
    op.close()

    result = await op.infer(_input(), schema=_SCHEMA)

    assert result.status == "success"


async def test_close_leaves_a_shared_executor_running():
    from concurrent.futures import ThreadPoolExecutor

    shared = ThreadPoolExecutor(max_workers=1)
    op = EdgeModelOperator(_FakeExtractor({"title": "Staff Engineer"}), executor=shared)
    await op.infer(_input(), schema=_SCHEMA)
    op.close()

    assert shared.submit(lambda: 1).result() == 1
