"""``ClientComputeBridge`` — drive one ``client_compute`` interrupt to resume.

The round-trip the keystone proves, end to end:

    cloud pauses with a client_compute interrupt
      → bridge parses the frozen payload (contract.py)
      → resolves each file_id to the device's OWN local text (FileTextResolver)
      → EdgeModelOperator.infer runs the local model + grounds (operator.py)
      → bridge submits {anchors, nonce} via the consumer SDK's fill_slots
      → server re-grounds and resumes

The bridge depends only on **narrow Protocols** (DIP) — never on the consumer SDK
package — so ``convilyn-edge`` stays zero-dependency: the caller injects a live
``AsyncConvilyn().goals`` (which satisfies :class:`GoalClientPort` structurally)
and a resolver that maps a ``file_id`` to the local file's text. Privacy invariant:
the interrupt carries file *references* only; the device reads its own copy — file
bytes never leave the device.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from convilyn_edge.clientcompute.contract import (
    INTERRUPT_TYPE,
    ClientComputeRequest,
    build_resume_answer,
)
from convilyn_edge.clientcompute.operator import EdgeModelOperator, ExtractInput


class ClientComputeError(Exception):
    """The bridge could not fulfil a client_compute interrupt."""


class MissingLocalSourceError(ClientComputeError):
    """The interrupt referenced files but none resolved to local text on this
    device — a fail-loud misconfiguration (the extraction cannot be meaningful),
    never a silent all-sentinel submission."""


class FileTextResolver(Protocol):
    """Maps a delegation ``file_id`` to the device's own local copy of its text.

    The device reads its OWN file (the point of on-device compute); returning
    ``None`` means "this device has no local copy of that file"."""

    def resolve(self, file_id: str) -> str | None:
        """Return the local text for ``file_id``, or ``None`` if unavailable."""
        ...


class GoalJobLike(Protocol):
    """The subset of a consumer-SDK ``GoalJob`` the bridge reads."""

    job_spec_id: str
    item_version: int | None
    pending_interrupts: Sequence[Mapping[str, Any]]


class GoalClientPort(Protocol):
    """The subset of the consumer SDK's ``goals`` surface the bridge drives.

    A live ``AsyncConvilyn().goals`` satisfies this structurally."""

    async def fill_slots(
        self,
        job_spec_id: str,
        answers: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> GoalJobLike:
        """Submit slot/interrupt answers, returning the updated job."""
        ...


def find_client_compute_interrupt(job: GoalJobLike) -> Mapping[str, Any] | None:
    """Return the first pending ``client_compute`` interrupt on ``job``, or None."""
    for entry in job.pending_interrupts:
        if entry.get("interruptType") == INTERRUPT_TYPE and entry.get("status") == "pending":
            return entry
    return None


def _anchor_schema(required_anchors: Sequence[str]) -> dict[str, Any]:
    """A minimal JSON Schema for the anchor object (SPI conformance)."""
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in required_anchors},
        "required": list(required_anchors),
    }


class ClientComputeBridge:
    """Fulfil ``client_compute`` interrupts with a device-local model."""

    def __init__(
        self,
        operator: EdgeModelOperator,
        file_resolver: FileTextResolver,
    ) -> None:
        self._operator = operator
        self._file_resolver = file_resolver

    def _resolve_sources(self, request: ClientComputeRequest) -> dict[str, str]:
        """Map each referenced ``file_id`` to its local text. Raises
        :class:`MissingLocalSourceError` if files were referenced but none
        resolved."""
        sources: dict[str, str] = {}
        for file_id in request.file_ids:
            text = self._file_resolver.resolve(file_id)
            if text is not None:
                sources[file_id] = text
        if request.file_ids and not sources:
            raise MissingLocalSourceError(
                f"client_compute referenced {len(request.file_ids)} file(s) but none "
                "resolved to local text on this device"
            )
        return sources

    async def handle_interrupt(
        self,
        client: GoalClientPort,
        job: GoalJobLike,
        interrupt: Mapping[str, Any],
        *,
        deadline_ms: int | None = None,
    ) -> GoalJobLike:
        """Run the local extractor for one ``client_compute`` ``interrupt`` and
        submit the grounded anchors via ``client.fill_slots``. Returns the updated
        job."""
        request = ClientComputeRequest.from_payload(interrupt.get("payload") or {})
        sources = self._resolve_sources(request)

        result = await self._operator.infer(
            ExtractInput(
                prompt=request.extractor_prompt,
                sources=sources,
                required_anchors=request.required_anchors,
                contract=request.anchors_contract,
            ),
            schema=_anchor_schema(request.required_anchors),
            deadline_ms=deadline_ms,
            placement="edge",
        )
        # Even an "unavailable" result yields a total all-sentinel answer, so the
        # server can re-ground (→ all indeterminate) and resume rather than hang.
        anchors = result.output or {
            key: request.anchors_contract.missing_sentinel for key in request.required_anchors
        }
        answer = build_resume_answer(anchors, request.nonce)
        interrupt_id = interrupt["interruptId"]
        return await client.fill_slots(
            job.job_spec_id,
            {interrupt_id: answer},
            expected_version=job.item_version,
        )

    async def handle_if_present(
        self,
        client: GoalClientPort,
        job: GoalJobLike,
        *,
        deadline_ms: int | None = None,
    ) -> GoalJobLike | None:
        """If ``job`` has a pending client_compute interrupt, fulfil it and return
        the updated job; otherwise return ``None`` (nothing to do — e.g. the
        server-side delegation feature flag is OFF)."""
        interrupt = find_client_compute_interrupt(job)
        if interrupt is None:
            return None
        return await self.handle_interrupt(client, job, interrupt, deadline_ms=deadline_ms)


__all__ = [
    "ClientComputeError",
    "MissingLocalSourceError",
    "FileTextResolver",
    "GoalJobLike",
    "GoalClientPort",
    "ClientComputeBridge",
    "find_client_compute_interrupt",
]
