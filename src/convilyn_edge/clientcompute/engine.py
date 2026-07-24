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
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from convilyn_edge.clientcompute.contract import MISSING_SENTINEL
from convilyn_edge.warmup import DEFAULT_WARM_THRESHOLD_MS, WarmupResult

BackendKind = Literal["ollama", "openai-compat"]

#: A transport is ``(url, body, headers, timeout) -> parsed-json-response``. The
#: default uses stdlib urllib; tests inject a fake to avoid the network.
Transport = Callable[[str, "dict[str, Any]", "dict[str, str]", float], "dict[str, Any]"]

#: A GET transport is ``(url, headers, timeout) -> parsed-json-response`` — the
#: read-only sibling of :data:`Transport`, used by :meth:`HttpLocalExtractor.model_available`
#: to fetch the server's model listing. Injectable for tests.
GetTransport = Callable[[str, "dict[str, str]", float], Any]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ExtractorTransportError(RuntimeError):
    """The local inference server could not be reached (the transport call failed).

    Subclasses ``RuntimeError`` so pre-b21 blanket catch-sites keep working;
    operators map it to ``degrade_reason="server_unreachable"``.
    """


class ExtractorOutputError(ValueError):
    """The server responded, but the model produced nothing parseable.

    Covers empty content, non-JSON output, and reasoning-only responses (the
    model spent its whole token budget thinking — ``reasoning_content`` present,
    ``content`` empty). Subclasses ``ValueError`` so pre-b21 blanket catch-sites
    keep working; operators map it to ``degrade_reason="output_unparseable"`` — so
    a model that ran but produced nothing is never conflated with an unreachable
    server.
    """


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


def _urllib_get_json(url: str, headers: dict[str, str], timeout: float) -> Any:
    """Default GET transport: fetch ``url``, return the parsed JSON body."""
    request = urllib.request.Request(  # noqa: S310 — operator-configured local URL
        url, headers=headers, method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class ModelAvailability:
    """Whether the extractor's bound model is actually servable on the server.

    ``health()`` proves reachability; this proves the *binding* resolves — the
    difference between "the server is up" and "the server can run THIS model"
    (an honest banner needs the latter). Four states:

    * ``available``   — the model tag is in the server's listing.
    * ``missing``     — the server listed its models and the tag is not there.
    * ``unreachable`` — the listing endpoint could not be fetched.
    * ``unknown``     — the server responded but membership cannot be decided
      (empty/odd listing — e.g. single-model OpenAI-compat servers report
      arbitrary ids). Never a false ``missing``.

    Truthiness is ``state == "available"`` — but this is doctor-surface data;
    never gate execution on it (the operator's degrade path stays authoritative).
    """

    state: Literal["available", "missing", "unreachable", "unknown"]
    model: str
    detail: str | None = None
    known_models: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.state == "available"


def _strip_think(content: str) -> str:
    """Remove a reasoning model's ``<think>...</think>`` block."""
    return _THINK_RE.sub("", content).strip()


def _merge_extra_body(body: dict[str, Any], extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge caller passthrough keys over a built request body — applied LAST.

    One-level-deep and deterministic: a dict-valued key (``options``,
    ``chat_template_kwargs``, ``response_format``) merges key-wise over the
    built dict; any other value replaces. The escape hatch for vendor
    generation knobs (``reasoning_effort``, sampler options, …) the typed
    fields don't model.
    """
    if not extra:
        return body
    merged = dict(body)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


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
    #: Generation caps/params. Defaults equal the pre-b21 hard-coded wire, so an
    #: unconfigured extractor sends byte-identical request bodies. ``max_tokens``
    #: maps to openai-compat ``max_tokens`` / ollama ``options.num_predict``;
    #: ``num_ctx`` applies to ollama only.
    max_tokens: int = 1024
    temperature: float = 0.0
    num_ctx: int = 8192
    #: Reasoning switch (tri-state). ``None`` (default) keeps today's wire:
    #: ollama ``think: false``, openai-compat nothing extra. ``False`` opts out
    #: explicitly everywhere: ollama ``think: false``; openai-compat adds
    #: ``chat_template_kwargs: {"enable_thinking": false}`` — the Qwen3
    #: convention honoured by vLLM / SGLang / recent llama.cpp (a strict server
    #: that rejects unknown keys wants ``None`` + a vendor knob via
    #: ``extra_body`` instead). ``True`` lets the model reason: ollama
    #: ``think: true``; openai-compat sends nothing (server default is on).
    reasoning: bool | None = None
    #: Vendor passthrough merged over the built request body LAST (see
    #: :func:`_merge_extra_body`) — e.g. ``{"reasoning_effort": "low"}`` or
    #: ``{"options": {"top_k": 1}}``. ``None`` sends the typed body untouched.
    extra_body: Mapping[str, Any] | None = None
    #: Injectable GET transport for :meth:`model_available` (tests inject a
    #: fake; the default is stdlib urllib).
    get_transport: GetTransport = _urllib_get_json

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

    def model_available(self) -> ModelAvailability:
        """Check the bound model against the server's listing — never raises.

        Ollama: ``/api/tags`` entry names (a bare tag also matches its
        ``:latest`` form). OpenAI-compat: ``/models`` ``data[*].id`` exact
        match; an empty or unrecognizable listing is ``unknown``, never a false
        ``missing``. Doctor-surface data — never gate execution on it.
        """
        url = f"{self.base_url}/api/tags" if self.kind == "ollama" else f"{self.base_url}/models"
        try:
            body = self.get_transport(url, self._headers(), 5.0)
        except Exception as exc:  # noqa: BLE001 — availability reports, never raises
            return ModelAvailability(
                state="unreachable",
                model=self.model,
                detail=(
                    f"local inference server unreachable at {self.base_url} "
                    f"({type(exc).__name__})"
                ),
            )
        try:
            if self.kind == "ollama":
                entries = body.get("models") or []
                names = tuple(str(e.get("name") or e.get("model") or "") for e in entries)
            else:
                entries = body.get("data") or []
                names = tuple(str(e.get("id") or "") for e in entries)
        except Exception as exc:  # noqa: BLE001 — odd listing shape is "unknown", not a crash
            return ModelAvailability(
                state="unknown",
                model=self.model,
                detail=f"unrecognized model-listing shape ({type(exc).__name__})",
            )
        names = tuple(name for name in names if name)
        if not names:
            return ModelAvailability(
                state="unknown",
                model=self.model,
                detail="server reported no model listing; membership cannot be checked",
            )
        candidates = {self.model}
        if ":" not in self.model:
            candidates.add(f"{self.model}:latest")
        if candidates & set(names):
            return ModelAvailability(state="available", model=self.model, known_models=names)
        return ModelAvailability(
            state="missing",
            model=self.model,
            known_models=names,
            detail=f"model {self.model!r} is not in the server's listing",
        )

    def warmup(
        self,
        deadline_ms: int | None = None,
        *,
        warm_threshold_ms: float = DEFAULT_WARM_THRESHOLD_MS,
    ) -> WarmupResult:
        """Pay the model's cold-start now and report which state the server was in.

        Runs :meth:`health` first — an unreachable server is reported as
        ``unreachable``, never mislabelled as a slow cold start. Then times one
        minimal probe inference: within ``warm_threshold_ms`` → ``warm``; slower →
        ``cold_started`` (the warmup itself absorbed the load cost). A probe that
        fails while the server is still up (e.g. ``deadline_ms`` elapsed during
        weight loading) is ``cold_started`` with the observed time; a server that
        dropped mid-probe is ``unreachable``.
        """
        problem = self.health()
        if problem is not None:
            return WarmupResult(state="unreachable", detail=problem)
        timeout = (deadline_ms / 1000.0) if deadline_ms is not None else self.timeout
        url, body = self._request(
            [
                {"role": "system", "content": "You are a warmup probe."},
                {"role": "user", "content": 'Reply with the JSON object {"ok": true}.'},
            ]
        )
        start = time.monotonic()
        try:
            self.transport(url, body, self._headers(), timeout)
        except Exception as exc:  # noqa: BLE001 — warmup reports, never raises
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if self.health() is not None:
                return WarmupResult(
                    state="unreachable",
                    detail=f"server dropped during warmup probe ({type(exc).__name__})",
                )
            return WarmupResult(
                state="cold_started",
                latency_ms=elapsed_ms,
                detail=(
                    f"warmup probe did not finish within its deadline while the server "
                    f"stayed reachable — model still loading ({type(exc).__name__})"
                ),
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms <= warm_threshold_ms:
            return WarmupResult(state="warm", latency_ms=elapsed_ms)
        return WarmupResult(state="cold_started", latency_ms=elapsed_ms)

    def _request(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        """Return (url, body) for this backend's chat endpoint."""
        if self.kind == "openai-compat":
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if self.reasoning is False:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            return f"{self.base_url}/chat/completions", _merge_extra_body(body, self.extra_body)
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.reasoning if self.reasoning is not None else False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        return f"{self.base_url}/api/chat", _merge_extra_body(body, self.extra_body)

    def _content(self, response: Mapping[str, Any]) -> str:
        """Pull the assistant content out of either backend's response shape.

        A reasoning-only response — reasoning fields populated while ``content``
        is empty (the model spent its whole token budget thinking) — raises
        :class:`ExtractorOutputError` so it surfaces as
        ``output_unparseable``, never conflated with an unreachable server.
        """
        if self.kind == "openai-compat":
            message = response["choices"][0]["message"]
            reasoning_keys = ("reasoning_content", "reasoning")
        else:
            message = response["message"]
            reasoning_keys = ("thinking",)
        content = message.get("content")
        if not content and any(message.get(key) for key in reasoning_keys):
            raise ExtractorOutputError(
                "reasoning-only response: the model produced reasoning but no content "
                "(token budget likely consumed by thinking — raise max_tokens or set "
                "reasoning=False)"
            )
        return content or ""

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
        try:
            response = self.transport(url, body, self._headers(), self.timeout)
        except Exception as exc:
            raise ExtractorTransportError(
                f"local inference request to {url} failed ({type(exc).__name__}: {exc})"
            ) from exc
        try:
            return _parse_json_object(_strip_think(self._content(response)))
        except ExtractorOutputError:
            raise
        except Exception as exc:
            raise ExtractorOutputError(
                f"model output was not parseable ({type(exc).__name__}: {exc})"
            ) from exc


__all__ = [
    "BackendKind",
    "Transport",
    "GetTransport",
    "LocalExtractor",
    "HttpLocalExtractor",
    "ExtractorTransportError",
    "ExtractorOutputError",
    "ModelAvailability",
    "resolve_local_model_tag",
    "build_extract_messages",
]
