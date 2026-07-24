"""Tests for the pet-monitoring grounded model node (#2793 PR-2).

Charter rule 2 (capability-negotiated placement, never an if-silicon/if-provider) + rule 3
(the decision path consumes a GROUNDED ModelOperator — typed output + evidence, never NL).
"""

from __future__ import annotations

from convilyn_edge.probe import DeviceCapabilityManifest, InstalledAsset
from examples.pet_monitoring import (
    CameraFrame,
    CatLocation,
    CatLocatorModel,
    resolve_placement,
    scripted_detector,
)


def _manifest(*, silicon: str = "generic_x86_64", assets: tuple[InstalledAsset, ...] = ()):
    return DeviceCapabilityManifest(
        device_id="d", silicon=silicon, ram_mb=8192, installed_assets=assets
    )


_OD_ASSET = InstalledAsset(capability="object_detection", model="yolo-nano")
_OCR_ASSET = InstalledAsset(capability="ocr", model="tess")


# ── capability-negotiated placement (rule 2, no if-silicon) ──────────────────


def test_placement_is_edge_when_object_detection_is_installed():
    assert resolve_placement(_manifest(assets=(_OD_ASSET,))) == "edge"


def test_placement_is_cloud_without_a_local_detector():
    assert resolve_placement(_manifest()) == "cloud"


def test_placement_ignores_an_unrelated_installed_capability():
    assert resolve_placement(_manifest(assets=(_OCR_ASSET,))) == "cloud"


def test_placement_is_capability_not_silicon():
    # Same silicon, different installed assets → different placement (proves it's negotiated on
    # the reported capability, NOT branched on the silicon string).
    jetson_bare = _manifest(silicon="jetson_orin")
    jetson_with_detector = _manifest(silicon="jetson_orin", assets=(_OD_ASSET,))
    generic_with_detector = _manifest(silicon="generic_x86_64", assets=(_OD_ASSET,))

    assert resolve_placement(jetson_bare) == "cloud"
    assert resolve_placement(jetson_with_detector) == "edge"
    assert resolve_placement(generic_with_detector) == "edge"  # generic silicon, but edge-capable


def test_model_is_wired_for_the_resolved_placement():
    placement = resolve_placement(_manifest(assets=(_OD_ASSET,)))
    model = CatLocatorModel(placement=placement)

    assert model.placement == "edge"


# ── grounded inference (rule 3) ──────────────────────────────────────────────


async def test_infer_returns_a_typed_grounded_result():
    model = CatLocatorModel(
        placement="edge",
        detector=scripted_detector(
            {"f1": CatLocation(present=True, zone="feeder", confidence=0.95)}
        ),
    )

    result = await model.infer(CameraFrame("f1", motion=True), schema={})

    assert result.status == "success"
    assert isinstance(result.output, CatLocation) and result.output.zone == "feeder"
    # grounded: evidence ties the output back to the source frame (server re-grounds it)
    assert result.evidence and result.evidence[0].source == "f1"


async def test_low_confidence_detection_is_uncertain():
    model = CatLocatorModel(
        placement="cloud",
        detector=scripted_detector(
            {"f1": CatLocation(present=True, zone="window", confidence=0.2)}
        ),
        min_confidence=0.5,
    )

    result = await model.infer(CameraFrame("f1", motion=True), schema={})

    assert result.status == "uncertain"  # below the bar → the workflow falls back


async def test_unavailable_model_falls_back_not_raises():
    model = CatLocatorModel(placement="edge", available=False)

    result = await model.infer(CameraFrame("f1", motion=True), schema={})

    assert result.status == "unavailable" and result.output is None


async def test_default_detector_uses_motion():
    model = CatLocatorModel(placement="cloud")

    present = await model.infer(CameraFrame("f1", motion=True), schema={})
    absent = await model.infer(CameraFrame("f2", motion=False), schema={})

    assert present.output == CatLocation(present=True, zone="unknown", confidence=0.9)
    assert absent.output is not None and absent.output.present is False


async def test_placement_param_overrides_wired_placement():
    # The SPI's placement param overrides the wired default when not "auto"; the sim's grounded
    # result is placement-independent (a real impl differs behind the Protocol).
    model = CatLocatorModel(placement="edge")

    result = await model.infer(CameraFrame("f1", motion=True), schema={}, placement="cloud")

    assert result.status == "success"  # still a valid typed result under the override


async def test_output_is_never_natural_language():
    model = CatLocatorModel(placement="edge")

    result = await model.infer(CameraFrame("f1", motion=True), schema={})

    # the keystone: a typed value on the success path, never a bare string / natural language
    assert isinstance(result.output, CatLocation)
