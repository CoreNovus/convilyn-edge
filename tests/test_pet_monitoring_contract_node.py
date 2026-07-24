"""Tests for the manufactured-contract cat-locate node + the dual-path factory."""

from __future__ import annotations

from pathlib import Path

from examples.pet_monitoring import run_e2e as e2e
from examples.pet_monitoring.contract_node import (
    PET_CONTRACT_PATH,
    ContractCatLocator,
    ScriptedContractExtractor,
    build_cat_locator,
)
from examples.pet_monitoring.model_node import CameraFrame, CatLocatorModel
from examples.pet_monitoring.pack import LOCATION_SCHEMA

_FRAME = CameraFrame(frame_id="frame-001", motion=True)


# ── logic: the committed asset loads and grounds through the real contract ───


def test_committed_contract_asset_builds_contract_node():
    locator = build_cat_locator(placement="edge")

    assert isinstance(locator, ContractCatLocator)


async def test_contract_node_grounds_scripted_answers_to_typed_location():
    locator = build_cat_locator(placement="edge")
    assert isinstance(locator, ContractCatLocator)

    result = await locator.infer(_FRAME, schema=LOCATION_SCHEMA)
    locator.close()

    assert result.output is not None and (result.output.present, result.output.zone) == (
        True,
        "litter",
    )


async def test_contract_node_reports_success_when_all_fields_ground():
    locator = build_cat_locator(placement="edge")
    assert isinstance(locator, ContractCatLocator)

    result = await locator.infer(_FRAME, schema=LOCATION_SCHEMA)
    locator.close()

    assert result.status == "success"


async def test_off_contract_answer_degrades_zone_to_none():
    locator = build_cat_locator(
        placement="edge",
        extractor=ScriptedContractExtractor({"present": "true", "zone": "garage"}),
    )
    assert isinstance(locator, ContractCatLocator)

    result = await locator.infer(_FRAME, schema=LOCATION_SCHEMA)
    locator.close()

    assert result.output is not None and result.output.zone is None


# ── error: an unavailable local model degrades, never raises ─────────────────


class _ExplodingExtractor:
    def extract(self, *, prompt, sources, required_anchors):
        raise OSError("local model gone")


async def test_extractor_failure_degrades_to_unavailable():
    locator = build_cat_locator(placement="edge", extractor=_ExplodingExtractor())
    assert isinstance(locator, ContractCatLocator)

    result = await locator.infer(_FRAME, schema=LOCATION_SCHEMA)
    locator.close()

    assert result.status == "unavailable"


# ── boundary: the zero-asset fallback path ───────────────────────────────────


def test_missing_contract_file_falls_back_to_scripted_sim(tmp_path: Path):
    locator = build_cat_locator(placement="edge", contract_path=tmp_path / "absent.uw.json")

    assert isinstance(locator, CatLocatorModel)


# ── the example smoke runs BOTH paths end to end (the DoD chimney) ───────────


async def test_e2e_contract_path_all_green(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path)  # default = committed contract asset

    assert report.passed


async def test_e2e_contract_path_mode_names_the_contract(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path)

    assert "contract" in report.mode


async def test_e2e_fallback_path_all_green_with_no_contract(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path, contract_path=tmp_path / "absent.uw.json")

    assert report.passed


async def test_e2e_fallback_path_mode_names_the_fallback(tmp_path: Path):
    report = await e2e.run(env={}, tmp=tmp_path, contract_path=tmp_path / "absent.uw.json")

    assert "scripted-sim" in report.mode


# ── object-state: the asset path is the committed example artifact ───────────


def test_pet_contract_path_points_at_committed_asset():
    assert PET_CONTRACT_PATH.name == "pet_cat_locate.uw.json" and PET_CONTRACT_PATH.exists()
