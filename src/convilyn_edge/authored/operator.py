"""``ContractModelOperator`` — run a manufactured contract on-device, grounded.

The generic grounded ``ModelOperator`` the integrator does NOT write: construct
it from a :class:`~convilyn_edge.authored.contract.GroundedContract` (loaded
from the installed ``uw_*`` bundle) plus any
:class:`~convilyn_edge.clientcompute.engine.LocalExtractor`-shaped runner, and
every ``infer`` executes the **manufactured** prompt, then grounds every field
by its authored rule (``verbatim`` substring / ``closed_set`` membership).

Unlike the extractor-only ``EdgeModelOperator``, the ``schema`` argument here is
**enforced, not decorative**: the effective schema is derived from the contract
(:meth:`GroundedContract.schema`), and a caller-passed schema that names fields
the contract does not declare fails loud — the contract is the single source of
truth, so schema and grounding rules cannot drift apart.

Implements ``ModelOperator[Mapping[str, str], dict[str, str]]`` — the input is
the device's own source texts (keyed by namespace), the output a total,
grounded field dict. The blocking runner call is offloaded to a per-operator
**bounded** executor (one thread), so a deadline-abandoned call cannot pile up
threads on a slow local model (Jetson cold-load); overlapping ``infer`` calls
serialise behind it. Pass a shared ``executor`` to pool across operators.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from convilyn_edge.authored.contract import GroundedContract, ground_fields
from convilyn_edge.clientcompute.engine import LocalExtractor
from convilyn_edge.spi.model import Evidence, ModelResult, Placement


class ContractModelOperator:
    """Run a manufactured :class:`GroundedContract` over a local runner.

    ``model_id`` defaults to the contract's ``model_binding`` (the
    server-resolved id the contract was authored against); ``model_version``
    defaults to the contract version, so provenance in every ``ModelResult``
    traces back to the manufactured artifact.
    """

    def __init__(
        self,
        extractor: LocalExtractor,
        contract: GroundedContract,
        *,
        model_id: str | None = None,
        model_version: str | None = None,
        executor: Executor | None = None,
    ) -> None:
        self._extractor = extractor
        self._contract = contract
        self._model_id = model_id or contract.model_binding or "edge-local"
        self._model_version = model_version or contract.version
        self._executor = executor
        self._owned_executor: ThreadPoolExecutor | None = None

    @property
    def contract(self) -> GroundedContract:
        """The manufactured contract this operator executes."""
        return self._contract

    async def infer(
        self,
        input: Mapping[str, str],
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "edge",
    ) -> ModelResult[dict[str, str]]:
        """Execute the manufactured prompt over ``input`` sources; ground per field.

        ``schema`` is validated against the contract (pass ``{}`` or
        ``contract.schema()`` — a schema naming undeclared fields raises
        ``ValueError``). This operator is edge-bound; ``placement`` is
        informational.
        """
        self._check_schema(schema)
        started = time.perf_counter()
        try:
            raw = await self._run(input, deadline_ms)
        except Exception:  # noqa: BLE001 — degrade to "unavailable", never raise
            return ModelResult(
                status="unavailable",
                model_id=self._model_id,
                model_version=self._model_version,
                latency_ms=self._elapsed_ms(started),
            )

        grounded = ground_fields(raw, self._contract, input)
        return self._result(grounded, started)

    def close(self) -> None:
        """Shut down the owned bounded executor (no-op for a shared one)."""
        if self._owned_executor is not None:
            self._owned_executor.shutdown(wait=False, cancel_futures=True)
            self._owned_executor = None

    def __enter__(self) -> ContractModelOperator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> ContractModelOperator:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.close()

    def _check_schema(self, schema: Mapping[str, Any]) -> None:
        """Fail loud when the caller's schema contradicts the contract.

        The contract is the single source of truth (grounding is always
        contract-driven regardless); this guard catches a *drifted* caller
        schema early: a ``properties`` key the contract does not declare, or an
        ``enum`` that differs from the field's authored ``allowed_values``,
        raises. An empty mapping means "use the contract's own schema"; a
        schema without ``properties`` makes no field claims and passes.
        """
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return
        declared = {f.name: f for f in self._contract.fields}
        undeclared = sorted(set(properties) - set(declared))
        if undeclared:
            raise ValueError(
                f"schema names fields the contract {self._contract.contract_id!r} "
                f"does not declare: {undeclared}"
            )
        for name, prop in properties.items():
            enum = prop.get("enum") if isinstance(prop, Mapping) else None
            if enum is not None and tuple(enum) != declared[name].allowed_values:
                raise ValueError(
                    f"schema enum for field {name!r} differs from the contract's "
                    f"authored allowed_values — the contract is the source of truth"
                )

    async def _run(self, sources: Mapping[str, str], deadline_ms: int | None) -> Mapping[str, Any]:
        """Run the blocking runner on the bounded executor, honouring a deadline.

        A synchronous runner call cannot be cancelled: on ``deadline_ms`` the
        coroutine returns (→ ``unavailable``) while the single worker thread
        runs to completion — bounding the leak to one thread; a subsequent
        ``infer`` queues behind it (honest backpressure) instead of stacking a
        new thread per timeout.
        """
        loop = asyncio.get_running_loop()
        call = loop.run_in_executor(
            self._bound_executor(),
            functools.partial(
                self._extractor.extract,
                prompt=self._contract.prompt_template,
                sources=sources,
                required_anchors=self._contract.field_names,
            ),
        )
        if deadline_ms is not None:
            return await asyncio.wait_for(call, timeout=deadline_ms / 1000)
        return await call

    def _bound_executor(self) -> Executor:
        if self._executor is not None:
            return self._executor
        if self._owned_executor is None:
            self._owned_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"contract-model-{self._contract.contract_id}"
            )
        return self._owned_executor

    def _result(self, grounded: dict[str, str], started: float) -> ModelResult[dict[str, str]]:
        sentinel = self._contract.anchors_contract.missing_sentinel
        required = self._contract.field_names
        grounded_keys = [k for k in required if grounded.get(k, sentinel) != sentinel]
        evidence = tuple(Evidence(source=k, snippet=grounded[k]) for k in grounded_keys)
        return ModelResult(
            status="success" if grounded_keys else "uncertain",
            model_id=self._model_id,
            model_version=self._model_version,
            latency_ms=self._elapsed_ms(started),
            output=grounded,
            confidence=(len(grounded_keys) / len(required)) if required else None,
            evidence=evidence,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000


__all__ = ["ContractModelOperator"]
