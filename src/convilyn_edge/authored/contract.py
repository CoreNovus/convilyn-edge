"""The manufactured grounded contract — device-side format, parse, and grounding.

A workflow authored on the Convilyn platform compiles its decision-critical AI
node into a **grounded contract**: the prompt, the typed output fields, and the
deterministic grounding rule for each field. The contract travels inside a
``uw_*`` bundle (fetched/verified/installed via :mod:`convilyn_edge.bundle`) and
is what a device :class:`~convilyn_edge.authored.operator.ContractModelOperator`
executes — the platform makes the brain, the pack wires the body.

This module is the SDK's zero-dependency mirror of that artifact's wire shape.
It **confirms-and-consumes** the contract — the authoritative compiler lives
server-side; the device only parses, derives the effective schema, and grounds.

Two grounding modes, both deterministic (never an LLM judging its own output):

* ``verbatim`` — the value must be a whitespace-collapsed substring of the
  device's own source text (the existing anchor-grounding rule, shared with
  :func:`~convilyn_edge.clientcompute.contract.ground_anchors`). For extraction.
* ``closed_set`` — the value must normalise to one of the field's **authored**
  ``allowed_values``; the grounded output is always the authored canonical
  label, never the model's raw string. For classification / derived outputs
  ("is the cat present" → ``{"true","false"}``, "which zone" → the authored
  zone labels) — values that legitimately never appear verbatim in the source.

Anything that fails its field's rule degrades to the missing sentinel — blank
over fabrication, exactly as the server-side re-grounding would degrade it.
Pure stdlib; no I/O except :func:`load_contract`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from convilyn_edge.clientcompute.contract import AnchorsContract, ground_anchors

#: The grounding modes a device can enforce. An unknown mode is REJECTED at
#: parse time (never run ungrounded): the device cannot deterministically
#: enforce a rule it does not understand, and running the field without its
#: rule would silently drop the grounding guarantee.
GroundingMode = Literal["verbatim", "closed_set"]

_KNOWN_MODES: frozenset[str] = frozenset(("verbatim", "closed_set"))

#: Wire key a ``uw_*`` workflow spec uses to embed its grounded contract.
_WRAPPER_KEY = "grounded_contract"


def _normalise(value: str) -> str:
    """Whitespace-collapsed, case-folded form used for closed-set matching."""
    return " ".join(value.split()).casefold()


def _coerce_scalar(value: Any) -> str | None:
    """Coerce a raw model value to a comparable string, or ``None`` if untyped.

    A JSON-mode model legitimately returns ``true`` / ``3`` for a closed-set
    field authored as ``"true"`` / ``"3"`` — coerce bool/int/float to their
    canonical JSON text so the membership check sees them. JSON has no
    int/float distinction, so an integral float (``3.0``) coerces to the
    integer text (``"3"``), matching how the label was authored. Anything else
    (dict, list, None) is not a scalar answer and fails grounding.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return json.dumps(int(value))
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return None


@dataclass(frozen=True)
class GroundedField:
    """One typed output field and the deterministic rule that grounds it."""

    name: str
    mode: GroundingMode = "verbatim"
    #: The authored canonical labels (``closed_set`` only; empty for ``verbatim``).
    allowed_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field requires a non-empty 'name'")
        if self.mode not in _KNOWN_MODES:
            raise ValueError(
                f"field {self.name!r}: unknown grounding mode {self.mode!r} — "
                "the device cannot enforce a rule it does not understand"
            )
        if self.mode == "closed_set":
            if not self.allowed_values:
                raise ValueError(f"closed_set field {self.name!r} requires 'allowed_values'")
            if any(not v for v in self.allowed_values):
                raise ValueError(f"closed_set field {self.name!r} has an empty allowed value")
            normalised = [_normalise(v) for v in self.allowed_values]
            if len(set(normalised)) != len(normalised):
                raise ValueError(
                    f"closed_set field {self.name!r}: allowed_values collide after "
                    "normalisation — the authored set is ambiguous"
                )
        elif self.allowed_values:
            raise ValueError(
                f"verbatim field {self.name!r} carries 'allowed_values' — a "
                "contradictory contract fails loud rather than half-applying"
            )

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> GroundedField:
        """Parse one wire field object. ``name`` is hard-required and must be a
        non-null string (guarded before ``str()`` coercion — ``str(None)`` is
        the truthy ``"None"`` and would slip a null through); ``mode`` defaults
        to ``verbatim``; ``allowed_values`` must be a real list (a bare string
        would iterate per-character into nonsense labels); unknown modes fail
        loud (see module doc)."""
        name = wire["name"]  # KeyError on absence
        if not isinstance(name, str):
            raise ValueError("field 'name' must be a string")
        raw_allowed = wire.get("allowed_values") or ()
        if isinstance(raw_allowed, (str, bytes)) or not isinstance(raw_allowed, Sequence):
            raise ValueError(f"field {name!r}: 'allowed_values' must be a list")
        return cls(
            name=name,
            mode=wire.get("mode", "verbatim"),
            allowed_values=tuple(str(v) for v in raw_allowed),
        )

    def ground(self, raw_value: Any, *, sentinel: str) -> str | None:
        """Ground a raw ``closed_set`` value → the authored canonical label,
        or ``None`` when it fails membership. (``verbatim`` fields ground via
        :func:`ground_anchors` — see :func:`ground_fields`.)"""
        if self.mode != "closed_set":
            raise ValueError(f"field {self.name!r} is not closed_set")
        coerced = _coerce_scalar(raw_value)
        if coerced is None or coerced == sentinel:
            return None
        needle = _normalise(coerced)
        for canonical in self.allowed_values:
            if _normalise(canonical) == needle:
                return canonical
        return None


@dataclass(frozen=True)
class GroundedContract:
    """A parsed manufactured contract (device side).

    ``contract_id`` / ``prompt_template`` / ``fields`` are hard-required (a
    malformed contract fails loud); ``model_binding`` is the server-resolved
    model id (e.g. ``local-qwen3-4b`` — translate with
    :func:`~convilyn_edge.clientcompute.engine.resolve_local_model_tag`);
    unknown wire keys are preserved in :attr:`extra` (forward-compatible).
    """

    contract_id: str
    prompt_template: str
    fields: tuple[GroundedField, ...]
    version: str = "1"
    model_binding: str | None = None
    anchors_contract: AnchorsContract = field(default_factory=AnchorsContract)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract requires a non-empty 'contract_id'")
        if not self.prompt_template:
            raise ValueError("contract requires a non-empty 'prompt_template'")
        if not self.fields:
            raise ValueError("contract requires at least one field")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("contract field names must be unique")
        sentinel = _normalise(self.anchors_contract.missing_sentinel)
        for f in self.fields:
            if f.mode == "closed_set" and any(_normalise(v) == sentinel for v in f.allowed_values):
                raise ValueError(
                    f"closed_set field {f.name!r}: an allowed value collides with the "
                    "missing sentinel — a degraded answer would be indistinguishable "
                    "from a grounded one"
                )

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> GroundedContract:
        """Parse a wire contract object, or a ``uw_*`` spec that embeds one
        under ``"grounded_contract"`` (the wrapper is descended into)."""
        if _WRAPPER_KEY in wire:
            inner = wire[_WRAPPER_KEY]
            if not isinstance(inner, Mapping):
                raise ValueError(f"'{_WRAPPER_KEY}' must be an object")
            return cls.from_wire(inner)
        known = {
            "contract_id",
            "prompt_template",
            "fields",
            "version",
            "model_binding",
            "anchors_contract",
        }
        raw_id, raw_prompt = wire["contract_id"], wire["prompt_template"]  # KeyError on absence
        # Guard nulls BEFORE str() coercion — ``str(None) == "None"`` is truthy
        # and would silently pass the non-empty check in ``__post_init__``.
        if not isinstance(raw_id, str) or not isinstance(raw_prompt, str):
            raise ValueError("'contract_id' and 'prompt_template' must be strings")
        raw_fields = wire["fields"]
        if isinstance(raw_fields, (str, bytes)) or not isinstance(raw_fields, Sequence):
            raise ValueError("'fields' must be a list of field objects")
        model_binding = wire.get("model_binding")
        return cls(
            contract_id=raw_id,
            prompt_template=raw_prompt,
            fields=tuple(GroundedField.from_wire(f) for f in raw_fields),
            version=str(wire.get("version") or "1"),
            model_binding=(str(model_binding) if model_binding is not None else None),
            anchors_contract=AnchorsContract.from_wire(wire.get("anchors_contract")),
            extra={k: v for k, v in wire.items() if k not in known},
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every output field name, in authored order."""
        return tuple(f.name for f in self.fields)

    def schema(self) -> dict[str, Any]:
        """The **effective** output JSON Schema, derived from the fields.

        This is the schema a :class:`ContractModelOperator` enforces — the
        contract is the single source of truth, so the schema can never drift
        from the grounding rules. ``closed_set`` fields carry their ``enum``.
        """
        properties: dict[str, Any] = {}
        for f in self.fields:
            prop: dict[str, Any] = {"type": "string"}
            if f.mode == "closed_set":
                prop["enum"] = list(f.allowed_values)
            properties[f.name] = prop
        return {
            "type": "object",
            "properties": properties,
            "required": list(self.field_names),
            "additionalProperties": False,
        }


def ground_fields(
    raw: Mapping[str, Any],
    contract: GroundedContract,
    sources: Mapping[str, str],
) -> dict[str, str]:
    """Ground a raw model output against the contract — the device self-verify.

    Returns a **total** dict (every contract field present, authored order).
    ``verbatim`` fields ground by the shared anchor rule
    (:func:`~convilyn_edge.clientcompute.contract.ground_anchors`, including its
    size caps); ``closed_set`` fields ground by authored-set membership and
    always yield the authored canonical label. Any failure degrades that field
    to the contract's missing sentinel — blank over fabrication.
    """
    sentinel = contract.anchors_contract.missing_sentinel
    verbatim_names = [f.name for f in contract.fields if f.mode == "verbatim"]
    verbatim = ground_anchors(raw, verbatim_names, sources, contract.anchors_contract)

    grounded: dict[str, str] = {}
    for f in contract.fields:
        if f.mode == "verbatim":
            grounded[f.name] = verbatim[f.name]
        else:
            grounded[f.name] = f.ground(raw.get(f.name), sentinel=sentinel) or sentinel
    return grounded


def guidance_from_contract(contract: GroundedContract) -> dict[str, str]:
    """Render per-field answer rules for the reference extractor's message.

    The reference :class:`~convilyn_edge.clientcompute.engine.HttpLocalExtractor`
    steers every value toward "verbatim in a source" by default — correct for
    ``verbatim`` fields, but a ``closed_set`` answer legitimately never appears
    in the source, so an unguided model is steered to the sentinel. Pass this
    mapping as the extractor's ``field_guidance`` so each ``closed_set`` field
    is steered toward its **authored** labels instead::

        extractor = HttpLocalExtractor(
            model="qwen3:4b", field_guidance=guidance_from_contract(contract)
        )

    Steering only — grounding (:func:`ground_fields`) is enforced regardless.
    ``verbatim`` fields get no entry (the blanket rule already fits them).
    """
    return {
        f.name: "answer with EXACTLY one of: " + " | ".join(f.allowed_values)
        for f in contract.fields
        if f.mode == "closed_set"
    }


def load_contract(path: str | Path) -> GroundedContract:
    """Load a manufactured contract from an installed bundle artifact (JSON file).

    The path is typically an
    :class:`~convilyn_edge.bundle.InstalledArtifact` destination (the bundle
    chain has already digest-verified the bytes) or the pack's committed
    ``authored/<name>.uw.json``. Malformed JSON / a malformed contract fails
    loud — never a silently degraded contract.
    """
    wire = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(wire, Mapping):
        raise ValueError(f"contract file {Path(path).name!r} must contain a JSON object")
    return GroundedContract.from_wire(wire)


__all__ = [
    "GroundingMode",
    "GroundedField",
    "GroundedContract",
    "ground_fields",
    "guidance_from_contract",
    "load_contract",
]
