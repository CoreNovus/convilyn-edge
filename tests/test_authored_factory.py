"""Unit tests for ContractModelOperator.for_contract — the pit-of-success factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from convilyn_edge.authored import ContractModelOperator, load_contract
from convilyn_edge.clientcompute.engine import HttpLocalExtractor

_PET_CONTRACT = Path(__file__).parent / "fixtures" / "pet_cat_locate_contract_golden.json"


class _ExplodingExtractor:
    def extract(self, *, prompt, sources, required_anchors):
        raise OSError("no local model")


# ── logic: one line from artifact to operator ────────────────────────────────


def test_factory_loads_contract_from_path():
    operator = ContractModelOperator.for_contract(_PET_CONTRACT, env={})

    assert operator.contract.contract_id == "uw_petcatlocatedemo00000001"


def test_factory_accepts_a_parsed_contract():
    contract = load_contract(_PET_CONTRACT)

    operator = ContractModelOperator.for_contract(contract, env={})

    assert operator.contract is contract


def test_default_extractor_resolves_model_tag_from_binding():
    operator = ContractModelOperator.for_contract(_PET_CONTRACT, env={})

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor) and extractor.model == "qwen3:4b"


def test_default_extractor_gets_closed_set_guidance_automatically():
    operator = ContractModelOperator.for_contract(_PET_CONTRACT, env={})

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor)
    assert extractor.field_guidance is not None
    assert set(extractor.field_guidance) == {"present", "zone"}


# ── boundary: overrides win over the environment defaults ────────────────────


def test_env_url_selects_openai_compat_backend():
    operator = ContractModelOperator.for_contract(
        _PET_CONTRACT, env={"EDGE_LLM_URL": "http://localhost:8080/v1"}
    )

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor) and extractor.kind == "openai-compat"


def test_model_override_beats_contract_binding():
    operator = ContractModelOperator.for_contract(_PET_CONTRACT, env={}, model="qwen3:8b")

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor) and extractor.model == "qwen3:8b"


def test_custom_extractor_is_used_as_is():
    sentinel = _ExplodingExtractor()

    operator = ContractModelOperator.for_contract(_PET_CONTRACT, extractor=sentinel)

    assert operator._extractor is sentinel


# ── error: a bad artifact fails loud, never a silently degraded operator ─────


def test_missing_contract_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ContractModelOperator.for_contract(tmp_path / "absent.uw.json", env={})


def test_malformed_contract_raises(tmp_path: Path):
    bad = tmp_path / "bad.uw.json"
    bad.write_text(json.dumps({"contract_id": "x"}), encoding="utf-8")

    with pytest.raises(KeyError):
        ContractModelOperator.for_contract(bad, env={})


# ── object-state: overrides surface in the operator's results ────────────────


async def test_model_id_override_flows_into_results():
    operator = ContractModelOperator.for_contract(
        _PET_CONTRACT, extractor=_ExplodingExtractor(), model_id="custom-id"
    )

    result = await operator.infer({"scene": "text"}, schema={})
    operator.close()

    assert result.model_id == "custom-id"
