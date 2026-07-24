"""Unit tests for OpenAICompatRunner (productized HTTP-local model runner)."""

from __future__ import annotations

import json
from typing import Any

from convilyn_edge.clientcompute.operator import ExtractInput
from convilyn_edge.runners import OpenAICompatRunner, failing_fields_count

_SOURCES = {"label": "Milk 1L costs 49 dollars at aisle 3"}
_ANCHORS = ["product_name", "price"]


def _extract_input() -> ExtractInput:
    return ExtractInput(prompt="extract", sources=_SOURCES, required_anchors=_ANCHORS)


def _openai_transport(content: str):
    """A fake OpenAI-compatible transport returning ``content`` as the message."""

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
        return {"choices": [{"message": {"content": content}}]}

    return transport


def _ollama_transport(content: str):
    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
        return {"message": {"content": content}}

    return transport


_GOOD = json.dumps({"product_name": "Milk 1L", "price": "49"})


# ── logic: successful grounded extraction ────────────────────────────────────


async def test_openai_compat_returns_success():
    runner = OpenAICompatRunner.openai_compat(
        model="qwen3:4b", base_url="http://sim/v1", transport=_openai_transport(_GOOD)
    )

    result = await runner.infer(_extract_input(), schema={})

    assert result.status == "success"


async def test_openai_compat_grounds_every_field():
    runner = OpenAICompatRunner.openai_compat(
        model="qwen3:4b", base_url="http://sim/v1", transport=_openai_transport(_GOOD)
    )

    result = await runner.infer(_extract_input(), schema={})

    assert failing_fields_count(result, _ANCHORS) == 0


async def test_ollama_wire_also_succeeds():
    runner = OpenAICompatRunner.ollama(
        model="qwen3:4b", base_url="http://sim", transport=_ollama_transport(_GOOD)
    )

    result = await runner.infer(_extract_input(), schema={})

    assert result.status == "success"


# ── boundary: ungrounded value falls to the sentinel ─────────────────────────


async def test_hallucinated_value_is_not_grounded():
    # "Bread" never appears in the source → grounding rejects it (→ sentinel).
    payload = json.dumps({"product_name": "Bread", "price": "49"})
    runner = OpenAICompatRunner.openai_compat(
        model="m", base_url="http://sim/v1", transport=_openai_transport(payload)
    )

    result = await runner.infer(_extract_input(), schema={})

    assert "product_name" in _failing(result)


def _failing(result):
    from convilyn_edge.runners import failing_fields

    return failing_fields(result, _ANCHORS)


# ── error: transport failure degrades to unavailable (offline-first) ─────────


async def test_transport_error_yields_unavailable():
    def boom(url, body, headers, timeout):
        raise ConnectionError("server down")

    runner = OpenAICompatRunner.openai_compat(model="m", base_url="http://sim/v1", transport=boom)

    result = await runner.infer(_extract_input(), schema={})

    assert result.status == "unavailable"


# ── object-state: from_env selects the backend by config ─────────────────────


def test_from_env_openai_when_url_set():
    runner = OpenAICompatRunner.from_env({"EDGE_LLM_URL": "http://x/v1", "EDGE_LLM_MODEL": "m"})

    # EDGE_LLM_URL selects the openai-compat wire (no network — inspect config).
    assert runner._extractor.kind == "openai-compat"


def test_from_env_ollama_when_no_url():
    runner = OpenAICompatRunner.from_env({"EDGE_LLM_MODEL": "m"})

    assert runner._extractor.kind == "ollama"
