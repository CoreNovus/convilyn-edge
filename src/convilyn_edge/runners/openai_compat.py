"""``OpenAICompatRunner`` — the productized on-device HTTP-local model runner.

Wraps the reference :class:`~convilyn_edge.clientcompute.engine.HttpLocalExtractor`
behind the ``ModelOperator`` SPI (via
:class:`~convilyn_edge.clientcompute.operator.EdgeModelOperator`) as a named,
selectable **runner** — the concrete on-device execution binding the runner
registry resolves for a ``llama_cpp`` / ``llama_cpp_cuda`` / ``ollama`` /
``openai-compat`` runtime. It speaks the two common local-inference wires with
ZERO runtime dependencies (stdlib ``urllib``):

* ``ollama`` — Ollama's ``/api/chat`` (Jetson/CUDA, ARM/x86 CPU, Apple).
* ``openai-compat`` — any ``/v1/chat/completions`` server (llama.cpp
  ``llama-server``, LM Studio, vLLM, …).

Wire selection is by construction (a classmethod / config), never by branching on
device identity — the model is a capability sink; the SDK never asks "am I on a
Jetson?". Implements ``ModelOperator[ExtractInput, dict[str, str]]``; the server
re-grounds every returned value before trusting it (the device is never a second
source of truth).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from convilyn_edge.clientcompute.engine import BackendKind, HttpLocalExtractor
from convilyn_edge.clientcompute.operator import EdgeModelOperator, ExtractInput
from convilyn_edge.spi.model import ModelResult, Placement

_DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OpenAICompatRunner:
    """A ``ModelOperator`` that runs the extractor role over an HTTP local server.

    Construct via :meth:`ollama` / :meth:`openai_compat` / :meth:`from_env`, or
    directly from a configured :class:`HttpLocalExtractor`. Grounding/self-verify
    is applied by the wrapped ``EdgeModelOperator`` — the runner produces, the
    local verifier grounds, the server re-grounds.
    """

    def __init__(
        self,
        extractor: HttpLocalExtractor,
        *,
        model_id: str | None = None,
        model_version: str = "0",
    ) -> None:
        self._extractor = extractor
        self._operator = EdgeModelOperator(
            extractor,
            model_id=model_id or extractor.model,
            model_version=model_version,
        )

    @classmethod
    def ollama(
        cls,
        *,
        model: str,
        base_url: str = _DEFAULT_OLLAMA_URL,
        api_key: str | None = None,
        transport: Any = None,
        model_id: str | None = None,
    ) -> OpenAICompatRunner:
        """Build a runner over an Ollama server (``/api/chat``)."""
        return cls(
            _http_extractor(
                "ollama", model=model, base_url=base_url, api_key=api_key, transport=transport
            ),
            model_id=model_id,
        )

    @classmethod
    def openai_compat(
        cls,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        transport: Any = None,
        model_id: str | None = None,
    ) -> OpenAICompatRunner:
        """Build a runner over any OpenAI-compatible server (``/chat/completions``)."""
        return cls(
            _http_extractor(
                "openai-compat",
                model=model,
                base_url=base_url,
                api_key=api_key,
                transport=transport,
            ),
            model_id=model_id,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, model: str | None = None) -> OpenAICompatRunner:
        """Build from environment — ``EDGE_LLM_URL`` selects the openai-compat
        backend, otherwise Ollama at ``OLLAMA_BASE`` (see ``HttpLocalExtractor``)."""
        return cls(HttpLocalExtractor.from_env(env, model=model))

    async def infer(
        self,
        input: ExtractInput,
        *,
        schema: Mapping[str, Any],
        deadline_ms: int | None = None,
        placement: Placement = "edge",
    ) -> ModelResult[dict[str, str]]:
        """Run the extractor role on-device, returning a grounded ``ModelResult``."""
        return await self._operator.infer(
            input, schema=schema, deadline_ms=deadline_ms, placement=placement
        )

    def health(self) -> str | None:
        """``None`` if the local inference server is reachable, else a problem string."""
        return self._extractor.health()


def _http_extractor(
    kind: BackendKind,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    transport: Any,
) -> HttpLocalExtractor:
    """Build an ``HttpLocalExtractor``, passing ``transport`` only when supplied so
    the extractor keeps its stdlib default (tests inject a fake to avoid I/O)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "kind": kind,
        "base_url": base_url,
        "api_key": api_key,
    }
    if transport is not None:
        kwargs["transport"] = transport
    return HttpLocalExtractor(**kwargs)


__all__ = ["OpenAICompatRunner"]
