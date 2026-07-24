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


def test_custom_extractor_with_caller_steering_is_used_as_is():
    sentinel = _ExplodingExtractor()

    operator = ContractModelOperator.for_contract(
        _PET_CONTRACT, extractor=sentinel, steering="caller"
    )

    assert operator._extractor is sentinel


# ── steering composition: the escape hatch no longer drops closed_set wiring ─


def test_auto_steering_injects_guidance_into_guidance_empty_extractor():
    from convilyn_edge.authored import guidance_from_contract, load_contract

    supplied = HttpLocalExtractor(model="qwen3:4b")

    operator = ContractModelOperator.for_contract(_PET_CONTRACT, extractor=supplied)

    assert operator._extractor.field_guidance == guidance_from_contract(
        load_contract(_PET_CONTRACT)
    )


def test_auto_steering_never_overwrites_caller_guidance():
    supplied = HttpLocalExtractor(model="qwen3:4b", field_guidance={"zone": "my own rule"})

    operator = ContractModelOperator.for_contract(_PET_CONTRACT, extractor=supplied)

    assert operator._extractor.field_guidance == {"zone": "my own rule"}


def test_auto_steering_warns_when_extractor_cannot_carry_guidance():
    with pytest.warns(UserWarning, match="closed_set steering is NOT wired"):
        ContractModelOperator.for_contract(_PET_CONTRACT, extractor=_ExplodingExtractor())


def test_caller_steering_silences_the_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ContractModelOperator.for_contract(
            _PET_CONTRACT, extractor=_ExplodingExtractor(), steering="caller"
        )


def test_caller_steering_without_extractor_raises():
    with pytest.raises(ValueError, match="requires a supplied extractor"):
        ContractModelOperator.for_contract(_PET_CONTRACT, env={}, steering="caller")


def test_gen_params_apply_to_env_built_extractor():
    operator = ContractModelOperator.for_contract(
        _PET_CONTRACT, env={}, max_tokens=64, temperature=0.2, reasoning=False
    )

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor)
    assert (extractor.max_tokens, extractor.temperature, extractor.reasoning) == (64, 0.2, False)


def test_unset_gen_params_keep_extractor_defaults():
    operator = ContractModelOperator.for_contract(_PET_CONTRACT, env={})

    extractor = operator._extractor
    assert isinstance(extractor, HttpLocalExtractor)
    assert (
        extractor.max_tokens,
        extractor.temperature,
        extractor.num_ctx,
        extractor.reasoning,
        extractor.extra_body,
    ) == (1024, 0.0, 8192, None, None)


# ── error: a bad artifact fails loud, never a silently degraded operator ─────


def test_gen_params_with_supplied_extractor_raise():
    with pytest.raises(ValueError, match="supplied extractor"):
        ContractModelOperator.for_contract(
            _PET_CONTRACT, extractor=_ExplodingExtractor(), max_tokens=64
        )


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
        _PET_CONTRACT, extractor=_ExplodingExtractor(), model_id="custom-id", steering="caller"
    )

    result = await operator.infer({"scene": "text"}, schema={})
    operator.close()

    assert result.model_id == "custom-id"
