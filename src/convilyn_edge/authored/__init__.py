"""Manufactured grounded contracts — load, ground, and execute on-device.

The device-side half of "author once, grounded everywhere": a workflow authored
on the platform compiles its AI node into a **grounded contract** (prompt +
typed fields + a deterministic grounding rule per field); this package parses
that artifact from an installed ``uw_*`` bundle and executes it via any local
runner, grounding every returned value before anything downstream sees it.

    from convilyn_edge.authored import (
        ContractModelOperator, guidance_from_contract, load_contract,
    )
    from convilyn_edge.clientcompute.engine import HttpLocalExtractor

    contract = load_contract("authored/pet_locate.uw.json")   # bundle-verified
    extractor = HttpLocalExtractor(
        model="qwen3:4b", field_guidance=guidance_from_contract(contract)
    )
    async with ContractModelOperator(extractor, contract) as operator:
        result = await operator.infer(sources, schema=contract.schema())

Grounding modes (both deterministic, never LLM-judged):

* ``verbatim`` — value must be a substring of the device's own source text.
* ``closed_set`` — value must normalise to one of the **authored** labels; the
  output is always the canonical label (classification / derived outputs).
"""

from __future__ import annotations

from convilyn_edge.authored.contract import (
    GroundedContract,
    GroundedField,
    GroundingMode,
    ground_fields,
    guidance_from_contract,
    load_contract,
)
from convilyn_edge.authored.operator import ContractModelOperator

__all__ = [
    "GroundingMode",
    "GroundedField",
    "GroundedContract",
    "ground_fields",
    "guidance_from_contract",
    "load_contract",
    "ContractModelOperator",
]
