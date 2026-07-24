"""Unit tests for the local extraction engine (wire-shaping, no network)."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from convilyn_edge.clientcompute.engine import (
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
