"""Per-artifact **fetch** over a short-TTL presigned URL — the device pull edge.

The push payload (:mod:`convilyn_edge.bundle.payload`) hands the device a fetch
location per artifact; the server never proxies the bytes. This module is the
seam that turns one such location into bytes on the device, and it is the SDK's
network **egress boundary** — so it carries the device-side fetch guarantees the
push contract requires (``backend-api/docs/engineering/edge_push_device_fetch_contract.md``
§4):

* **https only.** A non-``https`` URL (``file:`` / ``http:`` / ``ftp:`` / ``data:``)
  is refused before a socket opens — the load-bearing SSRF guard: a tampered
  payload cannot make the device read a local file or reach an internal metadata
  endpoint. Defence-in-depth: the opener does not even *load* the file/ftp/data
  handlers, so a future code path that reached ``open()`` on another scheme still
  cannot resolve one.
* **TLS-verified.** The default :func:`ssl.create_default_context` (cert chain +
  hostname checked) protects against a MITM even through a CONNECT proxy, since
  verification is end-to-end to the origin.
* **No redirects.** A presigned single-object URL never redirects; any 3xx is
  refused, closing an https→http downgrade / redirect-to-internal rebind.
* **Bounded.** A per-read socket timeout (idle guard) *and* a total-transfer
  wall-clock deadline (slow-drip guard) always apply; an optional byte ceiling
  (``max_bytes``) additionally caps size. **Set ``max_bytes`` for any untrusted
  egress** — it is the disk-fill / OOM bound and is OFF by default because only
  the integrator knows their device's storage budget (see ``max_bytes`` below).
* **URL never logged.** The URL is the only credential in the flow; no error
  message here ever embeds it (only the exception *type*), and the wrapping
  ``FetchError`` breaks the ``__cause__`` chain (``from None``) so a leaked URL
  cannot resurface through a printed traceback.

:class:`ArtifactFetcher` is the injectable seam (dependency inversion): the verify
and install layers, and every test, depend on this Protocol, never on the concrete
:class:`UrllibArtifactFetcher`. Stdlib only — zero runtime dependencies.
"""

from __future__ import annotations

import http.client
import io
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import BinaryIO, Protocol, runtime_checkable

#: Default per-read socket timeout (seconds) — the *idle* guard: a stalled socket
#: that delivers no bytes fails rather than hanging.
DEFAULT_TIMEOUT_SECONDS = 30.0
#: Default total-transfer wall-clock deadline (seconds) — the *slow-drip* guard: a
#: byte-per-``timeout`` endpoint keeps resetting the socket timer, so within a size
#: ceiling a trickle can still tie up a thread for years; a whole-transfer deadline
#: caps that. 2 h is generous enough for a multi-GB model asset on a modest edge
#: link (≈9 Mbps clears 8 GB) yet bounds a pathological drip. Raise it for a very
#: large asset on a very slow link; set ``max_total_seconds=None`` to disable.
DEFAULT_MAX_TOTAL_SECONDS = 7200.0
#: Default streaming chunk (64 KiB) — bounds peak memory regardless of asset size.
DEFAULT_CHUNK_SIZE = 65536


class FetchError(Exception):
    """A device-side artifact fetch was refused or failed.

    Raised for a rejected scheme, an unexpected redirect, a transport failure, a
    transfer exceeding the byte ceiling, or one exceeding the total deadline. The
    message never contains the URL (the only credential in the flow) — only the
    failure kind and, where useful, the originating exception *type* — and the
    ``__cause__`` chain is deliberately broken so the URL cannot leak via a
    formatted traceback either.
    """


@runtime_checkable
class ArtifactFetcher(Protocol):
    """Fetch one artifact's bytes from its location into a caller-owned sink.

    The single seam between the bundle-consumption layer and network egress.
    ``fetch`` streams the object at ``url`` into ``sink`` (a writable binary
    stream) and returns the number of bytes written. Implementations MUST NOT
    buffer the whole object in memory (stream it) and MUST NOT log the URL.
    """

    def fetch(self, url: str, sink: BinaryIO) -> int:
        """Stream ``url`` into ``sink``; return the byte count written."""
        ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect — a presigned single-object URL must resolve directly."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise FetchError(
            f"artifact url returned an unexpected redirect (HTTP {code}); "
            "presigned single-object urls must not redirect"
        )


def _build_secure_opener(context: ssl.SSLContext) -> urllib.request.OpenerDirector:
    """An opener that can resolve ONLY https, and never redirects.

    Built by hand rather than via :func:`urllib.request.build_opener` so the
    dangerous default handlers — ``FileHandler`` / ``FTPHandler`` / ``DataHandler``
    / ``HTTPHandler`` / ``UnknownHandler`` — are never even loaded. The scheme
    guard in :meth:`UrllibArtifactFetcher.fetch` is then not the *only* thing
    standing between a tampered ``file:``/``ftp:``/``data:`` URL and a live
    handler: no such handler exists to resolve it. ``ProxyHandler`` is kept so an
    integrator behind a corporate proxy still works (TLS stays end-to-end to the
    origin through the CONNECT tunnel); the error handlers turn a 4xx/5xx — e.g. an
    expired-URL 403 — into an ``HTTPError`` the caller sees as a ``FetchError``.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirectHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


class UrllibArtifactFetcher:
    """Stdlib :class:`ArtifactFetcher` — https-only, TLS-verified, streaming.

    Builds one hardened :class:`urllib.request.OpenerDirector` at construction
    (:func:`_build_secure_opener`) — https + proxy only, no redirects, no
    file/ftp/data handlers — and pins the TLS context. A general capability: it
    fetches every artifact kind identically and branches on no scenario.

    Bounds: ``timeout_seconds`` is the per-read idle guard; ``max_total_seconds``
    is the whole-transfer wall-clock deadline (both always on). ``max_bytes`` is
    the OPTIONAL size ceiling — **pass it for untrusted egress**; it is ``None``
    (uncapped) by default because the right disk/RAM budget is the integrator's to
    choose, not a value the SDK can guess (a Jetson and a workstation differ by
    orders of magnitude). The uncapped default is safe only for a trusted payload
    whose sizes are known; :func:`~convilyn_edge.bundle.staging.stage_bundle`
    additionally bounds each artifact by its server-advertised ``size_bytes``.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_total_seconds: float | None = DEFAULT_MAX_TOTAL_SECONDS,
        max_bytes: int | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_total_seconds is not None and max_total_seconds <= 0:
            raise ValueError("max_total_seconds must be positive when set")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive when set")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._timeout = timeout_seconds
        self._max_total_seconds = max_total_seconds
        self._max_bytes = max_bytes
        self._chunk_size = chunk_size
        context = ssl_context or ssl.create_default_context()
        self._opener = _build_secure_opener(context)

    @property
    def max_bytes(self) -> int | None:
        """The single-artifact byte ceiling, or ``None`` when uncapped."""
        return self._max_bytes

    def fetch(self, url: str, sink: BinaryIO) -> int:
        try:
            # Parse inside the try: a malformed URL (e.g. a bad IPv6 literal) makes
            # urlsplit raise ValueError, which must surface as a URL-free FetchError
            # like any other failure, not escape the contract raw.
            scheme = urllib.parse.urlsplit(url).scheme.lower()
            if scheme != "https":
                raise FetchError(f"refusing to fetch non-https artifact url (scheme={scheme!r})")
            response = self._opener.open(url, timeout=self._timeout)
        except FetchError:
            raise  # scheme rejection / refused redirect — already URL-free
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            # `from None`: never chain the cause. http.client.InvalidURL (a control
            # char in the URL) embeds the URL in its message, so a chained traceback
            # would leak the only credential in the flow. Keep just the type name.
            raise FetchError(f"artifact fetch failed ({type(exc).__name__})") from None
        return self._stream(response, sink)

    def _stream(self, response: BinaryIO, sink: BinaryIO) -> int:
        """Copy ``response`` into ``sink`` in bounded chunks; enforce every ceiling."""
        deadline = (
            time.monotonic() + self._max_total_seconds
            if self._max_total_seconds is not None
            else None
        )
        total = 0
        try:
            with response:
                while True:
                    if deadline is not None and time.monotonic() > deadline:
                        raise FetchError("artifact transfer exceeded the total deadline")
                    chunk = response.read(self._chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if self._max_bytes is not None and total > self._max_bytes:
                        raise FetchError(f"artifact exceeds max_bytes ({self._max_bytes})")
                    sink.write(chunk)
        except FetchError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            # OSError = socket drop; HTTPException = e.g. IncompleteRead on a
            # truncated response. `from None` for the same no-URL-in-traceback rule.
            raise FetchError(f"artifact stream failed ({type(exc).__name__})") from None
        return total


def fetch_to_bytes(fetcher: ArtifactFetcher, url: str) -> bytes:
    """Fetch ``url`` fully into memory via ``fetcher`` and return the bytes.

    A convenience for **small** artifacts (a workflow spec, a trusted core) where an
    in-memory value is more ergonomic than a file sink. This buffers the WHOLE body
    in memory, so it is bounded only by the ``fetcher``'s ``max_bytes`` — pass a
    capped :class:`UrllibArtifactFetcher` for untrusted sizes or it can OOM the
    device. Large model assets must stream to a file sink via
    :meth:`ArtifactFetcher.fetch` (or :func:`~convilyn_edge.bundle.staging.stage_bundle`).
    """
    buffer = io.BytesIO()
    fetcher.fetch(url, buffer)
    return buffer.getvalue()


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_TOTAL_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "FetchError",
    "ArtifactFetcher",
    "UrllibArtifactFetcher",
    "fetch_to_bytes",
]
