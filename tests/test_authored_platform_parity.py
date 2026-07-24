"""Cross-side parity pin: the platform producer's golden artifact parses here.

``fixtures/grounded_contract_golden.json`` is byte-identical to the backend
compiler's golden fixture (``backend-api/tests/unit/services/edge/fixtures/``,
epic #2838 P1 — the platform producer). Neither side imports the other (the
zero-cross-import rule); this file is the device half of the parity pin — if
the producer's wire drifts from what ``load_contract`` accepts, one of the two
golden tests breaks loudly.
"""

from __future__ import annotations

from pathlib import Path

from convilyn_edge.authored.contract import ground_fields, load_contract

_GOLDEN = Path(__file__).parent / "fixtures" / "grounded_contract_golden.json"


def test_platform_golden_artifact_loads():
    contract = load_contract(_GOLDEN)

    assert contract.contract_id == "uw_" + "a" * 24


def test_platform_golden_field_modes_survive_parse():
    contract = load_contract(_GOLDEN)

    assert [f.mode for f in contract.fields] == ["closed_set", "closed_set", "verbatim"]


def test_platform_golden_model_binding_survives_parse():
    contract = load_contract(_GOLDEN)

    assert contract.model_binding == "local-qwen3-4b"


def test_platform_golden_contract_grounds_a_closed_set_answer():
    contract = load_contract(_GOLDEN)
    sources = {"scene": "Doudou is sleeping by the window."}

    grounded = ground_fields({"present": True, "zone": " WINDOW "}, contract, sources)

    assert (grounded["present"], grounded["zone"]) == ("true", "window")


# ── pet cat-locate: the MANUFACTURED artifact (epic #2838 P4) ────────────────
# Same wire as backend-api/tests/unit/services/edge/fixtures/ EXCEPT
# prompt_template, which is REDACTED here (the sdist ships tests/, and the
# platform's real rendered prompt must never publish; parse/ground behaviour is
# prompt-independent). Regenerate from the backend golden via the documented
# redaction step whenever the fields/binding change.

_PET_GOLDEN = Path(__file__).parent / "fixtures" / "pet_cat_locate_contract_golden.json"


def test_pet_manufactured_artifact_loads():
    contract = load_contract(_PET_GOLDEN)

    assert contract.contract_id == "uw_petcatlocatedemo00000001"


def test_pet_field_modes_are_the_authored_contract():
    contract = load_contract(_PET_GOLDEN)

    assert [(f.name, f.mode) for f in contract.fields] == [
        ("present", "closed_set"),
        ("zone", "closed_set"),
        ("evidence_snippet", "verbatim"),
    ]


def test_pet_contract_grounds_a_scene_end_to_end():
    contract = load_contract(_PET_GOLDEN)
    sources = {"scene": "Doudou is curled up on the sofa near the feeder."}

    grounded = ground_fields(
        {"present": True, "zone": "SOFA", "evidence_snippet": "curled up on the sofa"},
        contract,
        sources,
    )

    assert grounded == {
        "present": "true",
        "zone": "sofa",
        "evidence_snippet": "curled up on the sofa",
    }


def test_pet_contract_out_of_set_zone_degrades_to_sentinel():
    contract = load_contract(_PET_GOLDEN)
    sources = {"scene": "Doudou is in the garage."}

    grounded = ground_fields({"present": True, "zone": "garage"}, contract, sources)

    assert grounded["zone"] == "Not specified"
