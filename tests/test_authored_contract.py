"""Unit tests for the manufactured grounded contract (parse + ground; no I/O
except the tmp_path loader case)."""

from __future__ import annotations

import json

import pytest

from convilyn_edge.authored.contract import (
    GroundedContract,
    GroundedField,
    ground_fields,
    load_contract,
)
from convilyn_edge.clientcompute.contract import AnchorsContract

_SOURCES = {"scene": "Doudou is sleeping on the sofa near the window."}


def _contract(**overrides) -> GroundedContract:
    base = dict(
        contract_id="pet.cat_locate",
        prompt_template="Locate the cat.",
        fields=(
            GroundedField("present", mode="closed_set", allowed_values=("true", "false")),
            GroundedField("zone", mode="closed_set", allowed_values=("sofa", "kitchen", "unknown")),
            GroundedField("evidence_snippet", mode="verbatim"),
        ),
    )
    base.update(overrides)
    return GroundedContract(**base)


# ── logic: grounding per mode ────────────────────────────────────────────────


def test_closed_set_grounds_to_authored_canonical_label():
    grounded = ground_fields({"zone": "  SOFA "}, _contract(), _SOURCES)

    assert grounded["zone"] == "sofa"


def test_closed_set_coerces_json_bool_to_canonical_text():
    grounded = ground_fields({"present": True}, _contract(), _SOURCES)

    assert grounded["present"] == "true"


def test_verbatim_field_grounds_source_substring():
    grounded = ground_fields({"evidence_snippet": "sleeping on the sofa"}, _contract(), _SOURCES)

    assert grounded["evidence_snippet"] == "sleeping on the sofa"


def test_ground_fields_is_total_over_all_fields():
    grounded = ground_fields({}, _contract(), _SOURCES)

    assert set(grounded) == {"present", "zone", "evidence_snippet"}


def test_schema_derives_enum_for_closed_set():
    schema = _contract().schema()

    assert schema["properties"]["zone"]["enum"] == ["sofa", "kitchen", "unknown"]


def test_from_wire_descends_into_grounded_contract_wrapper():
    wire = {
        "grounded_contract": {
            "contract_id": "c",
            "prompt_template": "p",
            "fields": [{"name": "f"}],
        }
    }

    assert GroundedContract.from_wire(wire).contract_id == "c"


def test_load_contract_roundtrips_a_json_file(tmp_path):
    path = tmp_path / "pet_locate.uw.json"
    path.write_text(
        json.dumps(
            {
                "contract_id": "pet.cat_locate",
                "prompt_template": "p",
                "fields": [{"name": "zone", "mode": "closed_set", "allowed_values": ["sofa"]}],
                "model_binding": "local-qwen3-4b",
            }
        ),
        encoding="utf-8",
    )

    assert load_contract(path).model_binding == "local-qwen3-4b"


# ── boundary: degradation to sentinel ────────────────────────────────────────


def test_closed_set_out_of_set_degrades_to_sentinel():
    grounded = ground_fields({"zone": "garage"}, _contract(), _SOURCES)

    assert grounded["zone"] == "Not specified"


def test_closed_set_non_scalar_degrades_to_sentinel():
    grounded = ground_fields({"zone": ["sofa"]}, _contract(), _SOURCES)

    assert grounded["zone"] == "Not specified"


def test_custom_sentinel_is_used_for_degradation():
    contract = _contract(anchors_contract=AnchorsContract(missing_sentinel="N/A"))

    grounded = ground_fields({"zone": "garage"}, contract, _SOURCES)

    assert grounded["zone"] == "N/A"


def test_closed_set_integral_float_matches_integer_label():
    contract = _contract(
        fields=(GroundedField("count", mode="closed_set", allowed_values=("1", "2", "3")),)
    )

    grounded = ground_fields({"count": 3.0}, contract, _SOURCES)

    assert grounded["count"] == "3"


# ── error: malformed contracts fail loud ─────────────────────────────────────


def test_unknown_grounding_mode_is_rejected_at_parse():
    with pytest.raises(ValueError, match="unknown grounding mode"):
        GroundedField.from_wire({"name": "f", "mode": "llm_judged"})


def test_closed_set_without_allowed_values_is_rejected():
    with pytest.raises(ValueError, match="requires 'allowed_values'"):
        GroundedField("f", mode="closed_set")


def test_verbatim_with_allowed_values_is_rejected():
    with pytest.raises(ValueError, match="contradictory"):
        GroundedField("f", mode="verbatim", allowed_values=("x",))


def test_ambiguous_allowed_values_are_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        GroundedField("f", mode="closed_set", allowed_values=("Sofa", "sofa"))


def test_duplicate_field_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _contract(fields=(GroundedField("f"), GroundedField("f")))


def test_missing_contract_id_fails_loud():
    with pytest.raises(KeyError):
        GroundedContract.from_wire({"prompt_template": "p", "fields": [{"name": "f"}]})


def test_null_contract_id_fails_loud():
    with pytest.raises(ValueError, match="must be strings"):
        GroundedContract.from_wire(
            {"contract_id": None, "prompt_template": "p", "fields": [{"name": "f"}]}
        )


def test_null_field_name_fails_loud():
    with pytest.raises(ValueError, match="must be a string"):
        GroundedField.from_wire({"name": None})


def test_bare_string_allowed_values_is_rejected():
    with pytest.raises(ValueError, match="must be a list"):
        GroundedField.from_wire({"name": "f", "mode": "closed_set", "allowed_values": "sofa"})


def test_allowed_value_colliding_with_sentinel_is_rejected():
    with pytest.raises(ValueError, match="collides with the missing sentinel"):
        _contract(
            fields=(
                GroundedField("zone", mode="closed_set", allowed_values=("Not specified", "sofa")),
            )
        )


def test_contract_file_with_non_object_json_is_rejected(tmp_path):
    path = tmp_path / "bad.uw.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_contract(path)


# ── object-state: forward-compat + defaults ──────────────────────────────────


def test_unknown_wire_keys_are_preserved_in_extra():
    contract = GroundedContract.from_wire(
        {
            "contract_id": "c",
            "prompt_template": "p",
            "fields": [{"name": "f"}],
            "future_key": 1,
        }
    )

    assert contract.extra == {"future_key": 1}


def test_version_defaults_to_one():
    assert _contract().version == "1"


# ── logic: extractor steering guidance ───────────────────────────────────────


def test_guidance_renders_closed_set_labels():
    from convilyn_edge.authored.contract import guidance_from_contract

    guidance = guidance_from_contract(_contract())

    assert guidance["zone"] == "answer with EXACTLY one of: sofa | kitchen | unknown"


def test_guidance_omits_verbatim_fields():
    from convilyn_edge.authored.contract import guidance_from_contract

    guidance = guidance_from_contract(_contract())

    assert "evidence_snippet" not in guidance
