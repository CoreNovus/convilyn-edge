"""The client-compute interrupt contract v1 — device side (parse + self-verify).

This is the SDK's typed, zero-dependency mirror of the **frozen v1** device-facing
wire contract. The cloud, when a workflow routes the extractor role to the device,
pauses the job and hands the device a *content-free* delegation request; the
device runs a local model over its OWN copy of the file and submits structured
anchors, which the server **re-grounds** before trusting.

The SDK **confirms-and-consumes** this contract — it does not co-design it. The
authoritative spec lives server-side; this module encodes only the device's half:

* :class:`ClientComputeRequest` — parse the interrupt payload (the 7 frozen keys,
  tolerant of forward-compatible additions).
* :func:`build_resume_answer` — shape the ``{anchors, nonce}`` resume body.
* :func:`ground_anchors` — the **local self-verify**: every returned value is a
  verbatim source substring or the ``"Not specified"`` sentinel, within the
  advertised size caps. This mirrors the server's re-grounding so a bad local
  extraction degrades *here* exactly as it would server-side — the device never
  ships a value the server would silently drop, and never smuggles an ungrounded
  (potentially injected) instruction past the boundary.

Pure stdlib; no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Frozen v1 constants (== the server's enforced constants).
INTERRUPT_TYPE = "client_compute"
STRATEGY_V1 = "client_delegated"
MISSING_SENTINEL = "Not specified"
DEFAULT_MAX_VALUE_CHARS = 2000
DEFAULT_MAX_TOTAL_CHARS = 20000

_WS_RE = re.compile(r"\s+")
# C0/C1 control chars except tab/newline/carriage-return (stripped from values,
# matching the server's structural screen).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _collapse_ws(text: str) -> str:
    """Whitespace-collapsed, stripped form used for verbatim-substring matching."""
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class AnchorsContract:
    """Limits the device must respect. Advertised == enforced."""

    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS
    value_type: str = "string"
    missing_sentinel: str = MISSING_SENTINEL

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any] | None) -> AnchorsContract:
        # ``x or DEFAULT`` (not ``.get(k, DEFAULT)``) so an explicit null / empty /
        # zero in the wire falls back to the safe default rather than becoming
        # ``int(None)`` (raises) or the literal ``"None"`` sentinel (silently wrong).
        wire = wire or {}
        return cls(
            max_value_chars=int(wire.get("max_value_chars") or DEFAULT_MAX_VALUE_CHARS),
            max_total_chars=int(wire.get("max_total_chars") or DEFAULT_MAX_TOTAL_CHARS),
            value_type=str(wire.get("value_type") or "string"),
            missing_sentinel=str(wire.get("missing_sentinel") or MISSING_SENTINEL),
        )


@dataclass(frozen=True)
class ClientComputeRequest:
    """A parsed ``client_compute`` interrupt payload (device side).

    The frozen v1 payload pins 7 always-present keys; a device MUST tolerate the
    payload growing *additional* keys in a future minor (forward-compatible), so
    unknown keys are preserved in :attr:`extra` rather than rejected.
    ``role_model_map`` is deliberately NOT device-facing in v1 (the server's
    allowlist projection drops it), so it never appears here.
    """

    nonce: str
    required_anchors: tuple[str, ...]
    extractor_prompt: str
    extractor_prompt_id: str | None
    file_ids: tuple[str, ...]
    anchors_contract: AnchorsContract
    strategy: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ClientComputeRequest:
        """Parse a raw interrupt ``payload`` dict.

        The three keys extraction cannot proceed without — ``nonce``,
        ``required_anchors``, ``extractor_prompt`` — are hard-required and raise
        ``KeyError`` if absent (a malformed payload fails loud). The remaining
        frozen keys default to their safe v1 values when absent
        (``extractor_prompt_id→None``, ``file_ids→()``,
        ``anchors_contract→v1 defaults``, ``strategy→"client_delegated"``), which
        keeps the SDK tolerant of a forward/backward-compatible payload."""
        known = {
            "nonce",
            "required_anchors",
            "extractor_prompt",
            "extractor_prompt_id",
            "file_ids",
            "anchors_contract",
            "strategy",
        }
        return cls(
            nonce=payload["nonce"],
            required_anchors=tuple(payload["required_anchors"]),
            extractor_prompt=payload["extractor_prompt"],
            extractor_prompt_id=payload.get("extractor_prompt_id"),
            file_ids=tuple(payload.get("file_ids", ())),
            anchors_contract=AnchorsContract.from_wire(payload.get("anchors_contract")),
            strategy=payload.get("strategy", STRATEGY_V1),
            extra={k: v for k, v in payload.items() if k not in known},
        )


def _clean_value(value: Any, max_value_chars: int) -> str | None:
    """Return a cleaned string value, or ``None`` if it fails structural screening
    (wrong type / empty / over per-value cap). Control chars are stripped."""
    if not isinstance(value, str):
        return None
    cleaned = _CONTROL_RE.sub("", value).strip()
    if not cleaned or len(cleaned) > max_value_chars:
        return None
    return cleaned


def ground_anchors(
    raw: Mapping[str, Any],
    required_anchors: Sequence[str],
    sources: Mapping[str, str],
    contract: AnchorsContract,
) -> dict[str, str]:
    """The device-side self-verify — the local mirror of server re-grounding.

    Returns a **total** dict (every ``required_anchors`` key present). Each value
    is either a verbatim, whitespace-collapsed substring of some ``sources`` block
    (within the per-value cap) or the ``missing_sentinel``. Anything that would
    fail the server's substring grounding is degraded to the sentinel *here*, so
    the submission carries only values the server will accept — and an ungrounded
    (possibly injected) string never crosses the boundary.

    Finally enforces the total-chars cap: if the summed value length exceeds
    ``max_total_chars``, the longest values are degraded to the sentinel (longest
    first, then by key for determinism) until the sum fits — the server rejects an
    over-total answer outright, so degrading locally keeps the round-trip alive.
    """
    collapsed_sources = [_collapse_ws(text) for text in sources.values()]
    sentinel = contract.missing_sentinel

    anchors: dict[str, str] = {}
    for key in required_anchors:
        cleaned = _clean_value(raw.get(key), contract.max_value_chars)
        if cleaned is None:
            anchors[key] = sentinel
            continue
        if cleaned == sentinel:
            anchors[key] = sentinel
            continue
        needle = _collapse_ws(cleaned)
        if any(needle in src for src in collapsed_sources):
            anchors[key] = cleaned
        else:
            anchors[key] = sentinel

    _enforce_total_cap(anchors, sentinel, contract.max_total_chars)
    return anchors


def _enforce_total_cap(anchors: dict[str, str], sentinel: str, max_total_chars: int) -> None:
    """Degrade longest grounded values to ``sentinel`` until the total fits.

    Mutates ``anchors`` in place (a local, freshly-built dict — never a caller's).
    Order is deterministic: longest value first, ties broken by key.
    """

    def total() -> int:
        return sum(len(v) for v in anchors.values())

    if total() <= max_total_chars:
        return
    # Degrading a value to the sentinel only REDUCES the total when the value is
    # longer than the sentinel itself — so only those are candidates, longest
    # first (ties by key). If the cap is smaller than the sentinels alone, no
    # local degradation can satisfy it (a pathological server config); we do the
    # best reduction possible and let the server make the final call.
    sentinel_len = len(sentinel)
    candidates = sorted(
        (k for k, v in anchors.items() if v != sentinel and len(v) > sentinel_len),
        key=lambda k: (-len(anchors[k]), k),
    )
    for key in candidates:
        if total() <= max_total_chars:
            break
        anchors[key] = sentinel


def build_resume_answer(anchors: Mapping[str, str], nonce: str | None) -> dict[str, Any]:
    """Shape the device→server resume body: ``{"anchors": ..., "nonce": ...}``.

    Submitted via the standard slot-resume corridor
    (``goals.fill_slots({interrupt_id: build_resume_answer(...)})``). The device
    echoes the ``nonce`` back for anti-replay.
    """
    return {"anchors": dict(anchors), "nonce": nonce}


__all__ = [
    "INTERRUPT_TYPE",
    "STRATEGY_V1",
    "MISSING_SENTINEL",
    "DEFAULT_MAX_VALUE_CHARS",
    "DEFAULT_MAX_TOTAL_CHARS",
    "AnchorsContract",
    "ClientComputeRequest",
    "ground_anchors",
    "build_resume_answer",
]
