"""Unit tests for ContractModelOperator (async; fake extractor, no network)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from convilyn_edge.authored.contract import GroundedContract, GroundedField
from convilyn_edge.authored.operator import ContractModelOperator

_SOURCES = {"scene": "Doudou is sleeping on the sofa near the window."}


class _FakeExtractor:
    def __init__(self, result=None, *, raises: Exception | None = None, sleep: float = 0.0):
        self._result = result or {}
        self._raises = raises
        self._sleep = sleep
        self.seen_prompt: str | None = None

    def extract(self, *, prompt, sources, required_anchors):
        self.seen_prompt = prompt
        if self._sleep:
            time.sleep(self._sleep)
        if self._raises is not None:
            raise self._raises
        return self._result


def _contract() -> GroundedContract:
    return GroundedContract(
        contract_id="pet.cat_locate",
        prompt_template="Locate the cat.",
        fields=(
            GroundedField("present", mode="closed_set", allowed_values=("true", "false")),
            GroundedField("zone", mode="closed_set", allowed_values=("sofa", "unknown")),
        ),
        model_binding="local-qwen3-4b",
    )


# ── logic ────────────────────────────────────────────────────────────────────


async def test_grounded_classification_is_success():
    op = ContractModelOperator(_FakeExtractor({"present": "true", "zone": "sofa"}), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.status == "success"


async def test_output_carries_authored_canonical_labels():
    op = ContractModelOperator(_FakeExtractor({"present": True, "zone": " SOFA "}), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.output == {"present": "true", "zone": "sofa"}


async def test_confidence_is_grounded_fraction():
    op = ContractModelOperator(_FakeExtractor({"present": "true", "zone": "garage"}), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.confidence == 0.5


async def test_model_id_defaults_to_contract_model_binding():
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.model_id == "local-qwen3-4b"


async def test_manufactured_prompt_drives_the_runner():
    extractor = _FakeExtractor({"present": "true"})
    op = ContractModelOperator(extractor, _contract())

    await op.infer(_SOURCES, schema={})

    assert extractor.seen_prompt == "Locate the cat."


async def test_contract_derived_schema_is_accepted():
    contract = _contract()
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), contract)

    result = await op.infer(_SOURCES, schema=contract.schema())

    assert result.status == "success"


# ── boundary: nothing grounded → uncertain ───────────────────────────────────


async def test_all_fields_ungrounded_is_uncertain():
    op = ContractModelOperator(_FakeExtractor({"present": "maybe", "zone": "garage"}), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.status == "uncertain"


# ── error: schema enforcement / runner failure / deadline ────────────────────


async def test_schema_with_undeclared_field_raises():
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), _contract())

    with pytest.raises(ValueError, match="does not declare"):
        await op.infer(_SOURCES, schema={"type": "object", "properties": {"colour": {}}})


async def test_schema_with_drifted_enum_raises():
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), _contract())
    drifted = {"type": "object", "properties": {"zone": {"enum": ["garage"]}}}

    with pytest.raises(ValueError, match="source of truth"):
        await op.infer(_SOURCES, schema=drifted)


async def test_runner_exception_is_unavailable():
    op = ContractModelOperator(_FakeExtractor(raises=RuntimeError("model down")), _contract())

    result = await op.infer(_SOURCES, schema={})

    assert result.status == "unavailable"


async def test_deadline_exceeded_is_unavailable():
    op = ContractModelOperator(_FakeExtractor({"present": "true"}, sleep=0.2), _contract())

    result = await op.infer(_SOURCES, schema={}, deadline_ms=1)

    assert result.status == "unavailable"


# ── object-state: executor lifecycle ─────────────────────────────────────────


async def test_infer_after_close_recreates_the_bounded_executor():
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), _contract())
    await op.infer(_SOURCES, schema={})
    op.close()

    result = await op.infer(_SOURCES, schema={})

    assert result.status == "success"


async def test_close_leaves_a_shared_executor_running():
    shared = ThreadPoolExecutor(max_workers=1)
    op = ContractModelOperator(_FakeExtractor({"present": "true"}), _contract(), executor=shared)
    await op.infer(_SOURCES, schema={})
    op.close()

    assert shared.submit(lambda: 1).result() == 1


async def test_async_context_manager_closes_on_exit():
    async with ContractModelOperator(_FakeExtractor({"present": "true"}), _contract()) as op:
        await op.infer(_SOURCES, schema={})

    assert op._owned_executor is None
