"""The grounded model node — "is the cat present, and where?" (charter rules 2 + 3).

The pack's decision-critical AI step: a :class:`~convilyn_edge.spi.model.ModelOperator` that takes
a camera frame and returns a **typed, grounded** :class:`~convilyn_edge.spi.model.ModelResult`
(never a bare string of natural language). This is what makes the scenario *lock in* to Convilyn
(charter rule 3): a grounded model output sits on the alert-branch path, so the cloud's
re-grounding IP is load-bearing — the device is never a second source of truth (every returned
value carries :class:`~convilyn_edge.spi.model.Evidence` tying it back to the source frame, which
the server re-grounds before trusting).

**Placement is capability-negotiated, never an ``if provider`` / ``if silicon``** (charter rule 2).
:func:`resolve_placement` reads the device's :class:`~convilyn_edge.probe.DeviceCapabilityManifest`
— the capability registry the device reports — and picks ``edge`` when the device advertises a local
``object_detection`` asset, else ``cloud``. It is a pure descriptor lookup: two devices with the
same silicon but different installed assets resolve differently, and a never-enumerated device
plugs in as the assets it reports. The same :class:`ModelOperator` Protocol serves both placements
(LSP); which concrete runs is *data*.

This is the reference **simulator** model (a scripted detector) so the pack is testable with no
accelerator; a real deployment binds an on-device ``object_detection`` runner (edge) or the
consumer SDK's ``goals.run`` (cloud) behind this same Protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from convilyn_edge.probe import Capability, DeviceCapabilityManifest
from convilyn_edge.spi.model import Evidence, ModelResult, Placement


@dataclass(frozen=True)
class CameraFrame:
    """The model's input: one camera frame reference (never raw pixels on the wire)."""

    frame_id: str
    motion: bool = False
    image_ref: str | None = None


@dataclass(frozen=True)
class CatLocation:
    """The model's typed output: is the cat present, and in which monitored zone."""

    present: bool
    zone: str | None = None
    confidence: float | None = None


#: The capability a device must report installed to run this model on-device.
_OBJECT_DETECTION: Capability = "object_detection"


def resolve_placement(
    manifest: DeviceCapabilityManifest, *, capability: Capability = _OBJECT_DETECTION
) -> Placement:
    """Pick ``edge`` vs ``cloud`` from the device's reported capabilities — deterministic.

    ``edge`` iff the manifest advertises an installed asset for ``capability`` (the device can run
    inference locally), else ``cloud``. A pure capability-registry lookup over
    ``manifest.installed_assets`` — no silicon-equality, no provider branch. Extending to another
    modality is a different ``capability`` argument, never a new ``if``.
    """
    has_local = any(asset.capability == capability for asset in manifest.installed_assets)
    return "edge" if has_local else "cloud"


#: A detector maps a frame to a location — the sim's stand-in for real inference (edge or cloud).
CatDetector = Callable[[CameraFrame], CatLocation]


def _default_detector(frame: CameraFrame) -> CatLocation:
    """A trivial reference detector: motion ⇒ cat present (zone unknown), else absent."""
    if frame.motion:
        return CatLocation(present=True, zone="unknown", confidence=0.9)
    return CatLocation(present=False, confidence=0.8)


def scripted_detector(
    locations: Mapping[str, CatLocation], *, fallback: CatDetector = _default_detector
) -> CatDetector:
    """A detector that returns a scripted location per ``frame_id`` (else ``fallback``).

    Lets a test pin exactly what the model "sees" for each frame — deterministic, replay-safe.
    """

    def _detect(frame: CameraFrame) -> CatLocation:
        return locations.get(frame.frame_id) or fallback(frame)

    return _detect


class CatLocatorModel:
    """``ModelOperator[CameraFrame, CatLocation]`` — the grounded cat-locate node (reference sim).

    Construct with the placement :func:`resolve_placement` chose; the same object serves as the
    node behind the pipeline's ``.model(...)`` stage. ``infer`` returns a typed, grounded
    :class:`ModelResult`: ``success`` when confidence clears ``min_confidence``, else ``uncertain``
    (the workflow falls back); ``unavailable`` when the model is marked offline (offline-first).
    ``output`` is always a typed :class:`CatLocation`, never natural language.
    """

    def __init__(
        self,
        *,
        placement: Placement,
        detector: CatDetector | None = None,
        available: bool = True,
        min_confidence: float = 0.5,
        model_id: str = "pet.cat_locator.sim",
        model_version: str = "sim-1",
    ) -> None:
        self._placement: Placement = placement
        self._detector = detector if detector is not None else _default_detector
        self._available = available
        self._min_confidence = min_confidence
        self._model_id = model_id
        self._model_version = model_version

    @property
    def placement(self) -> Placement:
        """The placement this node was wired for (the resolver's choice)."""
        return self._placement

    async def infer(
        self,
        input: CameraFrame,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "auto",
    ) -> ModelResult[CatLocation]:
        """Locate the cat in ``input``; return a typed, grounded :class:`ModelResult`.

        ``placement`` overrides the wired placement when not ``"auto"`` (the SPI contract); the sim
        produces the same grounded result either way — a real edge/cloud impl differs behind the
        Protocol. ``schema`` is accepted per the SPI (a real impl validates against it); a real impl
        also maps a breached ``deadline_ms`` to ``status="unavailable"`` (the sim is instant, so it
        does not need to).
        """
        if not self._available:
            # Offline / unreachable → the workflow takes its fixed fallback path (never raise).
            return ModelResult(
                status="unavailable",
                model_id=self._model_id,
                model_version=self._model_version,
                latency_ms=0.0,
            )
        location = self._detector(input)
        confidence = location.confidence if location.confidence is not None else 0.8
        # Evidence ties the typed output back to the SOURCE frame so the server can re-ground it —
        # the device is never a second source of truth. For a vision source, "re-grounding" means
        # the server re-runs detection keyed by ``source=frame_id`` — NOT substring-anchoring the
        # text ``snippet`` (do not apply SubstringAnchorVerifier to an image ref).
        evidence = (
            Evidence(
                source=input.frame_id,
                snippet=location.zone or ("motion" if input.motion else "still"),
            ),
        )
        status = "success" if confidence >= self._min_confidence else "uncertain"
        return ModelResult(
            status=status,
            model_id=self._model_id,
            model_version=self._model_version,
            latency_ms=1.0,
            output=location,
            confidence=confidence,
            evidence=evidence,
        )


__all__ = [
    "CameraFrame",
    "CatLocation",
    "CatDetector",
    "scripted_detector",
    "resolve_placement",
    "CatLocatorModel",
]
