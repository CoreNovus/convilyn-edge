"""``EdgeModelOperator`` — the ``ModelOperator`` SPI, run on-device (the keystone).

Wraps a :class:`~convilyn_edge.clientcompute.engine.LocalExtractor` as an
``edge``-placement :class:`~convilyn_edge.spi.model.ModelOperator`. ``infer`` runs
the local model, then applies :func:`~convilyn_edge.clientcompute.contract.ground_anchors`
so what it returns is already what the server will accept — the device produces,
the local verifier grounds, and (surface 3) the server re-grounds again before
trusting it. The device is never a second source of truth.

The result is always a typed :class:`ModelResult`, never raw natural language:

* extractor raises (model unreachable) / deadline exceeded → ``unavailable`` (the
  workflow takes its fixed fallback path — offline-first).
* ran, but zero anchors grounded → ``uncertain`` (a valid, submittable
  all-sentinel answer; the caller may prefer a fixed message).
* ran, ≥1 anchor grounded → ``success``.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from convilyn_edge.clientcompute.contract import AnchorsContract, ground_anchors
from convilyn_edge.clientcompute.engine import LocalExtractor
from convilyn_edge.spi.model import Evidence, ModelResult, Placement


@dataclass(frozen=True)
class ExtractInput:
    """The typed input to the edge extractor: the prompt, the device's own source
    texts (keyed by an anchor namespace), the anchor keys to return, and the
    advertised limits."""

    prompt: str
    sources: Mapping[str, str]
    required_anchors: Sequence[str]
    contract: AnchorsContract = field(default_factory=AnchorsContract)


class EdgeModelOperator:
    """Run the extractor role on-device, returning grounded anchors as a
    ``ModelResult``. Implements ``ModelOperator[ExtractInput, dict[str, str]]``."""

    def __init__(
        self,
        extractor: LocalExtractor,
        *,
        model_id: str = "edge-local",
        model_version: str = "0",
        executor: Executor | None = None,
    ) -> None:
        self._extractor = extractor
        self._model_id = model_id
        self._model_version = model_version
        self._executor = executor
        self._owned_executor: ThreadPoolExecutor | None = None

    async def infer(
        self,
        input: ExtractInput,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "edge",
    ) -> ModelResult[dict[str, str]]:
        """Extract + ground on-device. ``schema`` is accepted for SPI conformance
        (the anchor key set is the effective schema here); this operator is
        edge-bound, so ``placement`` is informational."""
        started = time.perf_counter()
        try:
            raw = await self._run(input, deadline_ms)
        except Exception:  # noqa: BLE001 — degrade to "unavailable", never raise
            return ModelResult(
                status="unavailable",
                model_id=self._model_id,
                model_version=self._model_version,
                latency_ms=self._elapsed_ms(started),
                confidence=None,
                evidence=(),
                output=None,
            )

        grounded = ground_anchors(raw, input.required_anchors, input.sources, input.contract)
        return self._result(grounded, input, started)

    def close(self) -> None:
        """Shut down the owned bounded executor (no-op for a shared one)."""
        if self._owned_executor is not None:
            self._owned_executor.shutdown(wait=False, cancel_futures=True)
            self._owned_executor = None

    def __enter__(self) -> EdgeModelOperator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> EdgeModelOperator:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.close()

    async def _run(self, input: ExtractInput, deadline_ms: int | None) -> Mapping[str, Any]:
        """Run the (blocking) extractor on a **bounded** executor, honouring a deadline.

        A synchronous ``extract()`` cannot be cancelled: on ``deadline_ms`` the
        coroutine returns (→ ``unavailable``) while the worker thread runs to
        completion. The default per-operator single-thread executor bounds that
        leak to one thread (a slow local model — e.g. a Jetson cold-load — can
        no longer stack a new thread per timeout); a subsequent ``infer`` queues
        behind it, which is honest backpressure. Pass a shared ``executor`` to
        pool across operators instead.
        """
        loop = asyncio.get_running_loop()
        call = loop.run_in_executor(
            self._bound_executor(),
            functools.partial(
                self._extractor.extract,
                prompt=input.prompt,
                sources=input.sources,
                required_anchors=input.required_anchors,
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
                max_workers=1, thread_name_prefix=f"edge-model-{self._model_id}"
            )
        return self._owned_executor

    def _result(
        self, grounded: dict[str, str], input: ExtractInput, started: float
    ) -> ModelResult[dict[str, str]]:
        required = list(input.required_anchors)
        # Classify against the SAME sentinel ground_anchors degraded to — the
        # advertised one, which may differ from the module default.
        sentinel = input.contract.missing_sentinel
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


__all__ = ["ExtractInput", "EdgeModelOperator"]
