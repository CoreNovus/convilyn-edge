"""Unit tests for the local extraction engine (wire-shaping, no network)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from convilyn_edge.clientcompute.engine import (
    ExtractorOutputError,
    ExtractorTransportError,
    HttpLocalExtractor,
    build_extract_messages,
    resolve_local_model_tag,
)


class _FakeTransport:
    """Records the last request and returns a canned response for a backend."""

    def __init__(self, content: str, *, kind: str) -> None:
        self.content = content
        self.kind = kind
        self.last_url: str | None = None
        self.last_body: dict[str, Any] | None = None

    def __call__(self, url, body, headers, timeout) -> dict[str, Any]:
        self.last_url = url
        self.last_body = body
        if self.kind == "openai-compat":
            return {"choices": [{"message": {"content": self.content}}]}
        return {"message": {"content": self.content}}


# ── resolve_local_model_tag ──────────────────────────────────────────────────


def test_resolve_strips_local_prefix_and_sizes():
    assert resolve_local_model_tag("local-qwen3-8b") == "qwen3:8b"


def test_resolve_passes_through_runtime_tag():
    assert resolve_local_model_tag("qwen3:4b") == "qwen3:4b"


# ── build_extract_messages ───────────────────────────────────────────────────


def test_messages_put_prompt_in_system_role():
    messages = build_extract_messages("SYS", {"f": "text"}, ["a"])

    assert messages[0] == {"role": "system", "content": "SYS"}


def test_messages_user_turn_lists_required_keys():
    messages = build_extract_messages("SYS", {"f": "text"}, ["title", "company"])

    assert "- title" in messages[1]["content"] and "- company" in messages[1]["content"]


def test_no_guidance_keeps_the_v1_blanket_verbatim_rule():
    messages = build_extract_messages("SYS", {"f": "text"}, ["a"])

    assert "Every value MUST appear verbatim" in messages[1]["content"]


def test_field_guidance_renders_the_per_key_rule():
    messages = build_extract_messages(
        "SYS", {"f": "text"}, ["zone"], field_guidance={"zone": "answer with EXACTLY one of: a | b"}
    )

    assert "- zone — answer with EXACTLY one of: a | b" in messages[1]["content"]


def test_field_guidance_scopes_the_verbatim_rule_to_unguided_keys():
    messages = build_extract_messages(
        "SYS", {"f": "text"}, ["zone", "quote"], field_guidance={"zone": "rule"}
    )

    assert "Unless a key states its own answer rule" in messages[1]["content"]


# ── HttpLocalExtractor.extract ───────────────────────────────────────────────


def test_ollama_extract_parses_content():
    transport = _FakeTransport('{"title": "X"}', kind="ollama")
    extractor = HttpLocalExtractor(model="qwen3:4b", kind="ollama", transport=transport)

    result = extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert result == {"title": "X"}


def test_ollama_hits_api_chat_endpoint():
    transport = _FakeTransport('{"title": "X"}', kind="ollama")
    extractor = HttpLocalExtractor(
        model="m", kind="ollama", base_url="http://host:11434", transport=transport
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert transport.last_url == "http://host:11434/api/chat"


def test_openai_compat_extract_parses_content():
    transport = _FakeTransport('{"title": "Y"}', kind="openai-compat")
    extractor = HttpLocalExtractor(model="m", kind="openai-compat", transport=transport)

    result = extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert result == {"title": "Y"}


def test_extract_strips_think_block():
    transport = _FakeTransport('<think>reasoning</think>{"title": "X"}', kind="ollama")
    extractor = HttpLocalExtractor(model="m", transport=transport)

    result = extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert result == {"title": "X"}


def test_extract_tolerates_json_fence():
    transport = _FakeTransport('```json\n{"title": "X"}\n```', kind="ollama")
    extractor = HttpLocalExtractor(model="m", transport=transport)

    result = extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert result == {"title": "X"}


# ── generation params: b21 defaults stay byte-identical to the b20 wire ──────


def test_openai_compat_default_body_is_byte_identical_to_b20():
    transport = _FakeTransport('{"title": "X"}', kind="openai-compat")
    extractor = HttpLocalExtractor(model="m", kind="openai-compat", transport=transport)

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert transport.last_body == {
        "model": "m",
        "messages": build_extract_messages("p", {"f": "t"}, ["title"]),
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }


def test_ollama_default_body_is_byte_identical_to_b20():
    transport = _FakeTransport('{"title": "X"}', kind="ollama")
    extractor = HttpLocalExtractor(model="m", kind="ollama", transport=transport)

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert transport.last_body == {
        "model": "m",
        "messages": build_extract_messages("p", {"f": "t"}, ["title"]),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 1024},
    }


def test_openai_compat_gen_params_land_in_body():
    transport = _FakeTransport('{"a": "x"}', kind="openai-compat")
    extractor = HttpLocalExtractor(
        model="m", kind="openai-compat", transport=transport, max_tokens=4096, temperature=0.7
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert (transport.last_body["max_tokens"], transport.last_body["temperature"]) == (4096, 0.7)


def test_ollama_gen_params_land_in_options():
    transport = _FakeTransport('{"a": "x"}', kind="ollama")
    extractor = HttpLocalExtractor(
        model="m",
        kind="ollama",
        transport=transport,
        max_tokens=4096,
        temperature=0.7,
        num_ctx=4096,
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body["options"] == {
        "temperature": 0.7,
        "num_ctx": 4096,
        "num_predict": 4096,
    }


@pytest.mark.parametrize(
    "reasoning, expected_think", [(None, False), (False, False), (True, True)]
)
def test_ollama_reasoning_tristate_maps_to_think(reasoning, expected_think):
    transport = _FakeTransport('{"a": "x"}', kind="ollama")
    extractor = HttpLocalExtractor(
        model="m", kind="ollama", transport=transport, reasoning=reasoning
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body["think"] is expected_think


@pytest.mark.parametrize(
    "reasoning, expected_kwargs",
    [(None, None), (False, {"enable_thinking": False}), (True, None)],
)
def test_openai_compat_reasoning_tristate_maps_to_chat_template_kwargs(reasoning, expected_kwargs):
    transport = _FakeTransport('{"a": "x"}', kind="openai-compat")
    extractor = HttpLocalExtractor(
        model="m", kind="openai-compat", transport=transport, reasoning=reasoning
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body.get("chat_template_kwargs") == expected_kwargs


def test_extra_body_scalar_key_is_added():
    transport = _FakeTransport('{"a": "x"}', kind="openai-compat")
    extractor = HttpLocalExtractor(
        model="m", kind="openai-compat", transport=transport, extra_body={"reasoning_effort": "low"}
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body["reasoning_effort"] == "low"


def test_extra_body_dict_value_merges_over_built_options():
    transport = _FakeTransport('{"a": "x"}', kind="ollama")
    extractor = HttpLocalExtractor(
        model="m", kind="ollama", transport=transport, extra_body={"options": {"top_k": 1}}
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body["options"] == {
        "temperature": 0.0,
        "num_ctx": 8192,
        "num_predict": 1024,
        "top_k": 1,
    }


def test_extra_body_scalar_replaces_built_key():
    transport = _FakeTransport('{"a": "x"}', kind="openai-compat")
    extractor = HttpLocalExtractor(
        model="m", kind="openai-compat", transport=transport, extra_body={"max_tokens": 8}
    )

    extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])

    assert transport.last_body["max_tokens"] == 8


# ── typed failures: transport vs model-output ────────────────────────────────


class _RawTransport:
    """Returns a caller-provided full response dict (any message shape)."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def __call__(self, url, body, headers, timeout) -> dict[str, Any]:
        return self.response


def test_transport_failure_raises_extractor_transport_error():
    def boom(url, body, headers, timeout):
        raise OSError("connection refused")

    extractor = HttpLocalExtractor(model="m", kind="ollama", transport=boom)

    with pytest.raises(ExtractorTransportError, match="connection refused"):
        extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])


def test_garbage_output_raises_extractor_output_error():
    transport = _FakeTransport("the model rambled instead of emitting JSON", kind="ollama")
    extractor = HttpLocalExtractor(model="m", kind="ollama", transport=transport)

    with pytest.raises(ExtractorOutputError, match="not parseable"):
        extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])


def test_empty_content_raises_extractor_output_error():
    transport = _RawTransport({"message": {"content": ""}})
    extractor = HttpLocalExtractor(model="m", kind="ollama", transport=transport)

    with pytest.raises(ExtractorOutputError, match="not parseable"):
        extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])


@pytest.mark.parametrize(
    "kind, message",
    [
        ("openai-compat", {"content": "", "reasoning_content": "step 1: think..."}),
        ("openai-compat", {"content": None, "reasoning": "step 1: think..."}),
        ("ollama", {"content": "", "thinking": "step 1: think..."}),
    ],
)
def test_reasoning_only_response_raises_extractor_output_error(kind, message):
    response = (
        {"choices": [{"message": message}]} if kind == "openai-compat" else {"message": message}
    )
    extractor = HttpLocalExtractor(model="m", kind=kind, transport=_RawTransport(response))

    with pytest.raises(ExtractorOutputError, match="reasoning-only"):
        extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["a"])


# ── model_available: the binding check health() can't answer ─────────────────


def _get_transport(body):
    def transport(url, headers, timeout):
        return body

    return transport


def test_model_available_ollama_finds_exact_tag():
    extractor = HttpLocalExtractor(
        model="qwen3:4b",
        kind="ollama",
        get_transport=_get_transport({"models": [{"name": "qwen3:4b"}, {"name": "llava:7b"}]}),
    )

    assert extractor.model_available().state == "available"


def test_model_available_ollama_matches_latest_suffix_for_bare_tag():
    extractor = HttpLocalExtractor(
        model="qwen3",
        kind="ollama",
        get_transport=_get_transport({"models": [{"name": "qwen3:latest"}]}),
    )

    assert extractor.model_available().state == "available"


def test_model_available_reports_missing_with_known_models():
    extractor = HttpLocalExtractor(
        model="qwen3:4b",
        kind="ollama",
        get_transport=_get_transport({"models": [{"name": "llava:7b"}]}),
    )

    report = extractor.model_available()
    assert (report.state, report.known_models) == ("missing", ("llava:7b",))


def test_model_available_openai_compat_matches_data_id():
    extractor = HttpLocalExtractor(
        model="qwen3-4b",
        kind="openai-compat",
        get_transport=_get_transport({"data": [{"id": "qwen3-4b"}]}),
    )

    assert extractor.model_available().state == "available"


def test_model_available_empty_listing_is_unknown_never_missing():
    # Single-model openai-compat servers (llama.cpp/LM Studio) may report an
    # empty or arbitrary listing — that must never read as a false "missing".
    extractor = HttpLocalExtractor(
        model="qwen3-4b", kind="openai-compat", get_transport=_get_transport({"data": []})
    )

    assert extractor.model_available().state == "unknown"


def test_model_available_odd_listing_shape_is_unknown():
    extractor = HttpLocalExtractor(
        model="qwen3:4b", kind="ollama", get_transport=_get_transport(["not", "a", "dict"])
    )

    assert extractor.model_available().state == "unknown"


def test_model_available_transport_failure_is_unreachable():
    def boom(url, headers, timeout):
        raise OSError("refused")

    extractor = HttpLocalExtractor(model="qwen3:4b", kind="ollama", get_transport=boom)

    assert extractor.model_available().state == "unreachable"


def test_model_available_truthiness_means_available():
    extractor = HttpLocalExtractor(
        model="qwen3:4b",
        kind="ollama",
        get_transport=_get_transport({"models": [{"name": "qwen3:4b"}]}),
    )

    assert bool(extractor.model_available()) is True


# ── from_env backend selection ───────────────────────────────────────────────


def test_from_env_selects_openai_compat_when_url_present():
    extractor = HttpLocalExtractor.from_env({"EDGE_LLM_URL": "http://x/v1/"})

    assert extractor.kind == "openai-compat"


def test_from_env_defaults_to_ollama():
    extractor = HttpLocalExtractor.from_env({})

    assert extractor.kind == "ollama"


def test_from_env_model_override_wins():
    extractor = HttpLocalExtractor.from_env({"EDGE_LLM_MODEL": "ignored"}, model="chosen")

    assert extractor.model == "chosen"


# ── error / default transport (network boundary) ─────────────────────────────


def test_non_object_json_raises():
    transport = _FakeTransport("[1, 2, 3]", kind="ollama")  # a JSON array, not an object
    extractor = HttpLocalExtractor(model="m", transport=transport)

    with pytest.raises(ValueError, match="not a JSON object"):
        extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def test_health_reachable_returns_none(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(b"{}"))

    assert HttpLocalExtractor(model="m").health() is None


def test_health_unreachable_returns_problem(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    assert "unreachable" in HttpLocalExtractor(model="m").health()


def test_default_transport_posts_json_via_urllib(monkeypatch):
    # Exercise the real _urllib_post_json path (default transport) without a
    # network — assert it issues a POST and parses the response body.
    body = json.dumps({"message": {"content": json.dumps({"title": "X"})}}).encode("utf-8")
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["method"] = request.method
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    extractor = HttpLocalExtractor(model="m", kind="ollama")  # default urllib transport

    result = extractor.extract(prompt="p", sources={"f": "t"}, required_anchors=["title"])

    assert result == {"title": "X"} and seen["method"] == "POST"
