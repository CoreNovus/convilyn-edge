"""The manufactured-contract model node — the cloud makes the brain, the pack wires the body.

The pet pack's cat-locate node is no longer a hand-written simulator: the platform
**manufactures** a grounded contract (prompt + typed fields + per-field grounding
rules) when the workflow is authored, the artifact ships to the device inside a
``uw_*`` bundle, and this module mounts it straight into the DAG via
``load_contract(...)`` + :class:`~convilyn_edge.authored.operator.ContractModelOperator`.
The committed ``authored/pet_cat_locate.uw.json`` is that artifact, checked in as
the example's asset (a real device receives it through the digest-verified
:mod:`convilyn_edge.bundle` chain instead).

Every answer is grounded by the contract's authored rules — ``present`` / ``zone``
by closed-set membership (the answer is always an authored canonical label),
``evidence_snippet`` verbatim against the device's own scene text — regardless of
which local model produced it. The zero-asset fallback (no contract file on disk)
remains the scripted simulator, so the example still runs offline with nothing
installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from convilyn_edge.authored import ContractModelOperator, GroundedContract, load_contract
from convilyn_edge.clientcompute.engine import LocalExtractor
from convilyn_edge.spi.model import Evidence, ModelOperator, ModelResult, Placement
from examples.pet_monitoring.model_node import (
    CameraFrame,
    CatDetector,
    CatLocation,
    CatLocatorModel,
)

#: The manufactured contract artifact committed as the example's asset.
PET_CONTRACT_PATH = Path(__file__).parent / "authored" / "pet_cat_locate.uw.json"


def describe_frame(frame: CameraFrame) -> str:
    """The device-side scene text the contract's extractor reads (a real pack
    would attach a vision caption / detector transcript here)."""
    return (
        f"Pet camera frame {frame.frame_id}. Motion detected: {frame.motion}. "
        "Monitored zones: sofa, kitchen, window, litter."
    )


class ScriptedContractExtractor:
    """A deterministic ``LocalExtractor`` standing in for the local LLM runner.

    Lets the demo exercise the REAL manufactured-contract path — load, prompt,
    per-field grounding — with no model installed: the answers are scripted, but
    everything downstream of the extractor is exactly what a real local model's
    output goes through. ``evidence_snippet`` is taken from the actual source
    text so the contract's verbatim rule grounds it like a real quote.
    """

    def __init__(self, answers: Mapping[str, Any] | None = None) -> None:
        default = {"present": "true", "zone": "litter"}
        self._answers = dict(answers) if answers is not None else default

    def extract(
        self,
        *,
        prompt: str,
        sources: Mapping[str, str],
        required_anchors: Sequence[str],
    ) -> Mapping[str, Any]:
        first_source = next(iter(sources.values()), "")
        snippet = first_source.split(". ")[0] if first_source else ""
        return {**self._answers, "evidence_snippet": snippet}


class ContractCatLocator:
    """``ModelOperator[CameraFrame, CatLocation]`` executing the manufactured contract.

    The adapter between the pack's typed camera world and the generic
    :class:`ContractModelOperator`: a frame becomes the device's scene text, the
    contract's grounded field dict becomes a typed :class:`CatLocation`. The
    inner operator always enforces the **contract's own** schema (the contract is
    the single source of truth for the raw fields); the pipeline-level schema
    describes this node's typed output and is accepted as-is.
    """

    def __init__(
        self,
        contract: GroundedContract,
        extractor: LocalExtractor,
        *,
        placement: Placement = "edge",
    ) -> None:
        self._contract = contract
        self._operator = ContractModelOperator(extractor, contract)
        self._placement: Placement = placement

    @property
    def placement(self) -> Placement:
        """The placement this node was wired for (the resolver's choice)."""
        return self._placement

    @property
    def contract(self) -> GroundedContract:
        """The manufactured contract this node executes."""
        return self._contract

    async def infer(
        self,
        input: CameraFrame,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "auto",
    ) -> ModelResult[CatLocation]:
        """Locate the cat by executing the manufactured contract over the scene text."""
        inner = await self._operator.infer(
            {"scene": describe_frame(input)},
            schema=self._contract.schema(),
            deadline_ms=deadline_ms,
        )
        if inner.status == "unavailable" or inner.output is None:
            return ModelResult(
                status="unavailable",
                model_id=inner.model_id,
                model_version=inner.model_version,
                latency_ms=inner.latency_ms,
            )
        sentinel = self._contract.anchors_contract.missing_sentinel
        grounded = inner.output
        zone = grounded.get("zone", sentinel)
        snippet = grounded.get("evidence_snippet", sentinel)
        location = CatLocation(
            present=grounded.get("present") == "true",
            zone=zone if zone != sentinel else None,
            confidence=inner.confidence,
        )
        evidence = (
            Evidence(source=input.frame_id, snippet=snippet if snippet != sentinel else None),
        )
        return ModelResult(
            status=inner.status,
            model_id=inner.model_id,
            model_version=inner.model_version,
            latency_ms=inner.latency_ms,
            output=location,
            confidence=inner.confidence,
            evidence=evidence,
        )

    def close(self) -> None:
        """Release the inner operator's bounded executor."""
        self._operator.close()


def build_cat_locator(
    *,
    placement: Placement,
    contract_path: str | Path = PET_CONTRACT_PATH,
    extractor: LocalExtractor | None = None,
    detector: CatDetector | None = None,
) -> ModelOperator[CameraFrame, CatLocation]:
    """The pack's dual-path model-node factory.

    A contract file at ``contract_path`` mounts the manufactured-contract node
    (``load_contract`` + :class:`ContractCatLocator`; ``extractor`` defaults to
    the scripted stand-in so no model install is needed). No contract file →
    the scripted-sim :class:`CatLocatorModel` fallback, so the example runs
    offline with zero assets.
    """
    path = Path(contract_path)
    if not path.exists():
        return CatLocatorModel(placement=placement, detector=detector)
    contract = load_contract(path)
    backend = extractor if extractor is not None else ScriptedContractExtractor()
    return ContractCatLocator(contract, backend, placement=placement)


__all__ = [
    "PET_CONTRACT_PATH",
    "describe_frame",
    "ScriptedContractExtractor",
    "ContractCatLocator",
    "build_cat_locator",
]
