"""Local extraction engine — the device-side model call for the extractor role.

``LocalExtractor`` is the narrow extension point the client-compute flow depends on (DIP):
"given the extractor prompt + the device's own source texts + the required anchor
keys, return the model's raw anchor dict." Grounding/self-verify is NOT the
extractor's job — it is applied by :class:`~convilyn_edge.clientcompute.operator`
via :func:`~convilyn_edge.clientcompute.contract.ground_anchors`, mirroring the
server split (model produces, verifier grounds).

``HttpLocalExtractor`` is the reference implementation — it speaks the two common
local-inference wires with **zero runtime dependencies** (stdlib ``urllib``):

* ``ollama``          — Ollama's ``/api/chat`` (Jetson/CUDA, ARM/x86 CPU, Apple).
* ``openai-compat``   — any ``/v1/chat/completions`` server: llama.cpp
  ``llama-server``, LM Studio, vLLM, or an ONNX/QNN NPU wrapper.

Backend selection is by configuration, never by branching on device identity (the
model is a capability sink; the SDK never asks "am I on a Jetson?"). This
productizes the extract + self-verify logic proven in the (now-retired) edge
runner POC.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from convilyn_edge.clientcompute.contract import MISSING_SENTINEL

BackendKind = Literal["ollama", "openai-compat"]

#: A transport is ``(url, body, headers, timeout) -> parsed-json-response``. The
#: default uses stdlib urllib; tests inject a fake to avoid the network.
Transport = Callable[[str, "dict[str, Any]", "dict[str, str]", float], "dict[str, Any]"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalExtractor(Protocol):
    """Run the extractor role on-device; return the model's raw anchor dict."""

    def extract(
        self,
        *,
        prompt: str,
        sources: Mapping[str, str],
        required_anchors: Sequence[str],
    ) -> Mapping[str, Any]:
        """Return the raw JSON object the local model produced (ungrounded)."""
        ...


def resolve_local_model_tag(server_model_id: str) -> str:
    """Map a server-resolved model id to a local runtime tag. Deterministic, pure.

    The server may transport its own id (e.g. ``local-qwen3-8b``); a local runtime
    (Ollama / llama.cpp) wants a tag like ``qwen3:8b``. Strip a ``local-`` prefix,
    then turn a trailing ``-<size>`` into ``:<size>``. An id that is already a
    runtime tag (``qwen3:4b``) passes through unchanged.
    """
    tag = server_model_id.removeprefix("local-")
    if ":" not in tag and "-" in tag:
        head, _, size = tag.rpartition("-")
        tag = f"{head}:{size}"
    return tag


def _urllib_post_json(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    """Default transport: POST ``body`` as JSON, return the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 — operator-configured local URL
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _strip_think(content: str) -> str:
    """Remove a reasoning model's ``<think>...</think>`` block."""
    return _THINK_RE.sub("", content).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating a ```` ```json ```` fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model output was not a JSON object")
    return parsed


def build_extract_messages(
    prompt: str,
    sources: Mapping[str, str],
    required_anchors: Sequence[str],
    *,
    field_guidance: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Compose the chat messages for the extractor role.

    The extractor system ``prompt`` arrives in the interrupt payload; the user
    turn presents each source block and asks for a strict JSON object whose values
    each appear **verbatim** in a source (or the sentinel). The device grounds the
    result afterwards regardless — this instruction just steers the model toward
    values that will survive grounding.

    ``field_guidance`` optionally overrides the answer rule **per key** (generic
    data — any caller may pass it): a key with guidance is steered by its own
    rule line instead of the blanket verbatim instruction. This is how a
    manufactured contract's ``closed_set`` fields (answers that legitimately
    never appear verbatim in the source — see
    :func:`convilyn_edge.authored.contract.guidance_from_contract`) are steered
    toward their authored labels. With no guidance the message is byte-identical
    to the v1 interrupt-flow shape.
    """
    guidance = field_guidance or {}
    source_xml = "\n\n".join(
        f'<source name="{name}">\n{text}\n</source>' for name, text in sources.items()
    )
    required_lines = "\n".join(
        f"  - {key} — {guidance[key]}" if key in guidance else f"  - {key}"
        for key in required_anchors
    )
    verbatim_rule = (
        "Unless a key states its own answer rule above, every value MUST appear "
        "verbatim in at least one <source> block above.\n"
        if guidance
        else "Every value MUST appear verbatim in at least one <source> block above.\n"
    )
    user_text = (
        f"{source_xml}\n\n"
        "<extract_request>\n"
        "Emit ONE strict JSON object with EXACTLY these keys (no extras):\n"
        f"{required_lines}\n\n"
        f"{verbatim_rule}"
        f'If a value is genuinely absent from all sources, set it to "{MISSING_SENTINEL}" '
        "— do NOT invent.\n"
        "Output: raw JSON only. No prose. No markdown fence.\n"
        "</extract_request>"
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]


@dataclass
class HttpLocalExtractor:
    """Reference ``LocalExtractor`` over an HTTP local-inference server (zero-dep).

    Construct directly (tests / explicit config) or via :meth:`from_env`. The
    ``transport`` is injectable so the wire-shaping is unit-testable without a
    running model.
    """

    model: str
    kind: BackendKind = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    timeout: float = 600.0
    transport: Transport = _urllib_post_json
    #: Optional per-key answer-rule overrides passed through to
    #: :func:`build_extract_messages` (e.g. a manufactured contract's
    #: ``closed_set`` label sets via ``guidance_from_contract``). ``None`` keeps
    #: the v1 verbatim-only message byte-identical.
    field_guidance: Mapping[str, str] | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, model: str | None = None) -> HttpLocalExtractor:
        """Build from environment. ``EDGE_LLM_URL`` (an OpenAI-compatible base URL)
        selects the ``openai-compat`` backend; otherwise Ollama at ``OLLAMA_BASE``.
        ``model`` overrides ``EDGE_LLM_MODEL`` (e.g. a server-resolved tag)."""
        resolved_model = model or env.get("EDGE_LLM_MODEL", "qwen3:4b")
        openai_url = env.get("EDGE_LLM_URL")
        if openai_url:
            return cls(
                model=resolved_model,
                kind="openai-compat",
                base_url=openai_url.rstrip("/"),
                api_key=env.get("EDGE_LLM_API_KEY"),
            )
        return cls(
            model=resolved_model,
            kind="ollama",
            base_url=env.get("OLLAMA_BASE", "http://localhost:11434"),
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> str | None:
        """Return ``None`` if the inference server is reachable, else a
        human-actionable problem string (for a device ``doctor`` check)."""
        url = f"{self.base_url}/api/tags" if self.kind == "ollama" else f"{self.base_url}/models"
        request = urllib.request.Request(  # noqa: S310 — operator-configured local URL
            url, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                response.read()
        except Exception as exc:  # noqa: BLE001 — health reports, never raises
            return f"local inference server unreachable at {self.base_url} ({type(exc).__name__})"
        return None

    def _request(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        """Return (url, body) for this backend's chat endpoint."""
        if self.kind == "openai-compat":
            return (
                f"{self.base_url}/chat/completions",
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "response_format": {"type": "json_object"},
                },
            )
        return (
            f"{self.base_url}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 1024},
            },
        )

    def _content(self, response: Mapping[str, Any]) -> str:
        """Pull the assistant content out of either backend's response shape."""
        if self.kind == "openai-compat":
            return response["choices"][0]["message"]["content"]
        return response["message"]["content"]

    def extract(
        self,
        *,
        prompt: str,
        sources: Mapping[str, str],
        required_anchors: Sequence[str],
    ) -> Mapping[str, Any]:
        messages = build_extract_messages(
            prompt, sources, required_anchors, field_guidance=self.field_guidance
        )
        url, body = self._request(messages)
        response = self.transport(url, body, self._headers(), self.timeout)
        return _parse_json_object(_strip_think(self._content(response)))


__all__ = [
    "BackendKind",
    "Transport",
    "LocalExtractor",
    "HttpLocalExtractor",
    "resolve_local_model_tag",
    "build_extract_messages",
]
