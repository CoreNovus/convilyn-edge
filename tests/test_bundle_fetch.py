"""Unit tests for the device-side artifact fetcher (network egress boundary).

Network-free: the concrete :class:`UrllibArtifactFetcher` is exercised through a
fake opener injected over its private ``_opener`` seam, so no socket is opened and
the scheme / redirect / ceiling / streaming guards are each pinned in isolation.
"""

from __future__ import annotations

import http.client
import io
import traceback
import urllib.error
import urllib.request

import pytest

from convilyn_edge.bundle import fetch as fetch_module
from convilyn_edge.bundle.fetch import (
    ArtifactFetcher,
    FetchError,
    UrllibArtifactFetcher,
    _NoRedirectHandler,
    fetch_to_bytes,
)

_URL = "https://bucket.example/obj?sig=abc"


class _FakeOpener:
    """Stands in for ``urllib`` — returns a canned response or raises on open."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def open(self, url, timeout=None):
        if self._error is not None:
            raise self._error
        return self._response


class _RaisingRead(io.BytesIO):
    """A response whose ``read`` fails mid-stream (transport drop)."""

    def read(self, size=-1):
        raise OSError("connection reset")


def _fetcher_with(response=None, error=None, **kwargs) -> UrllibArtifactFetcher:
    fetcher = UrllibArtifactFetcher(**kwargs)
    fetcher._opener = _FakeOpener(response=response, error=error)
    return fetcher


# ── construction validation ──────────────────────────────────────────────────


def test_init_rejects_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout_seconds"):
        UrllibArtifactFetcher(timeout_seconds=0)


def test_init_rejects_non_positive_max_bytes():
    with pytest.raises(ValueError, match="max_bytes"):
        UrllibArtifactFetcher(max_bytes=0)


def test_init_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        UrllibArtifactFetcher(chunk_size=0)


def test_max_bytes_property_exposes_the_ceiling():
    assert UrllibArtifactFetcher(max_bytes=42).max_bytes == 42


# ── scheme guard (the load-bearing SSRF guard) ───────────────────────────────


@pytest.mark.parametrize("url", ["http://h/o", "file:///etc/passwd", "ftp://h/o", "data:,x"])
def test_fetch_refuses_non_https_scheme(url):
    fetcher = _fetcher_with(response=io.BytesIO(b"x"))

    with pytest.raises(FetchError, match="non-https"):
        fetcher.fetch(url, io.BytesIO())


# ── redirect guard ───────────────────────────────────────────────────────────


def test_no_redirect_handler_refuses_any_redirect():
    handler = _NoRedirectHandler()

    with pytest.raises(FetchError, match="redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil/x")


# ── transport failures map to FetchError (never leak the URL) ─────────────────


def test_fetch_wraps_transport_error():
    fetcher = _fetcher_with(error=urllib.error.URLError("dns"))

    with pytest.raises(FetchError, match="fetch failed"):
        fetcher.fetch(_URL, io.BytesIO())


def test_fetch_error_message_never_contains_the_url():
    fetcher = _fetcher_with(error=urllib.error.URLError("dns"))

    with pytest.raises(FetchError) as excinfo:
        fetcher.fetch(_URL, io.BytesIO())

    assert "bucket.example" not in str(excinfo.value)


def test_fetch_propagates_refused_redirect_verbatim():
    # A refused redirect surfaces as FetchError from open(); it must pass through
    # the non-FetchError except arm untouched.
    fetcher = _fetcher_with(error=FetchError("unexpected redirect"))

    with pytest.raises(FetchError, match="redirect"):
        fetcher.fetch(_URL, io.BytesIO())


def test_fetch_wraps_malformed_url_as_fetcherror():
    # urlsplit on a bad IPv6 literal raises ValueError; parsed inside the try it must
    # surface as a URL-free FetchError, not escape the fetch contract raw.
    fetcher = _fetcher_with(response=io.BytesIO(b"x"))

    with pytest.raises(FetchError, match="fetch failed"):
        fetcher.fetch("https://[oops", io.BytesIO())


def test_fetch_wraps_invalid_url_httpexception_without_leaking_url():
    # http.client.InvalidURL (a control char in the URL) is NOT a URLError/OSError
    # and its own message embeds the URL — it must be caught and re-wrapped URL-free.
    fetcher = _fetcher_with(error=http.client.InvalidURL("URL can't contain 'bucket.example'"))

    with pytest.raises(FetchError) as excinfo:
        fetcher.fetch(_URL, io.BytesIO())

    assert "bucket.example" not in str(excinfo.value)


def test_fetch_error_breaks_cause_chain_so_url_cannot_leak_via_traceback():
    # The wrapping FetchError must NOT chain the InvalidURL cause (whose message
    # embeds the URL) — else a formatted traceback would print the credential.
    leaking = http.client.InvalidURL(
        "URL can't contain ctrl chars. 'https://bucket.example/x?sig=SECRET'"
    )
    fetcher = _fetcher_with(error=leaking)

    with pytest.raises(FetchError) as excinfo:
        fetcher.fetch(_URL, io.BytesIO())

    assert "SECRET" not in "".join(traceback.format_exception(excinfo.value))


def test_fetch_wraps_mid_stream_read_error():
    fetcher = _fetcher_with(response=_RaisingRead(b"partial"))

    with pytest.raises(FetchError, match="stream failed"):
        fetcher.fetch(_URL, io.BytesIO())


def test_fetch_wraps_incomplete_read_httpexception():
    class _IncompleteRead(io.BytesIO):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"", 100)

    fetcher = _fetcher_with(response=_IncompleteRead(b"x"))

    with pytest.raises(FetchError, match="stream failed"):
        fetcher.fetch(_URL, io.BytesIO())


# ── happy path + ceiling ─────────────────────────────────────────────────────


def test_fetch_streams_bytes_into_sink():
    fetcher = _fetcher_with(response=io.BytesIO(b"hello world"))
    sink = io.BytesIO()

    fetcher.fetch(_URL, sink)

    assert sink.getvalue() == b"hello world"


def test_fetch_returns_byte_count():
    fetcher = _fetcher_with(response=io.BytesIO(b"hello world"))

    assert fetcher.fetch(_URL, io.BytesIO()) == 11


def test_fetch_within_ceiling_succeeds():
    fetcher = _fetcher_with(response=io.BytesIO(b"1234"), max_bytes=10, chunk_size=2)

    assert fetcher.fetch(_URL, io.BytesIO()) == 4


def test_fetch_at_exact_ceiling_succeeds():
    # boundary: total == max_bytes is allowed (the guard is a strict `>`).
    fetcher = _fetcher_with(response=io.BytesIO(b"1234"), max_bytes=4, chunk_size=2)

    assert fetcher.fetch(_URL, io.BytesIO()) == 4


def test_fetch_exceeding_ceiling_raises():
    fetcher = _fetcher_with(response=io.BytesIO(b"way too big"), max_bytes=4, chunk_size=2)

    with pytest.raises(FetchError, match="max_bytes"):
        fetcher.fetch(_URL, io.BytesIO())


def test_fetch_enforces_total_transfer_deadline(monkeypatch):
    # A slow-drip endpoint resets the per-read socket timeout forever; the whole-
    # transfer deadline must still cut it. Drive time.monotonic deterministically.
    class _InfiniteResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            return b"x" * 8

    ticks = iter([0.0, 0.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(fetch_module.time, "monotonic", lambda: next(ticks))
    fetcher = _fetcher_with(response=_InfiniteResponse(), max_total_seconds=1.0)

    with pytest.raises(FetchError, match="deadline"):
        fetcher.fetch(_URL, io.BytesIO())


def test_init_rejects_non_positive_max_total_seconds():
    with pytest.raises(ValueError, match="max_total_seconds"):
        UrllibArtifactFetcher(max_total_seconds=0)


# ── fetch_to_bytes + Protocol conformance ────────────────────────────────────


def test_fetch_to_bytes_returns_full_body():
    fetcher = _fetcher_with(response=io.BytesIO(b"in memory"))

    assert fetch_to_bytes(fetcher, _URL) == b"in memory"


def test_urllib_fetcher_satisfies_the_protocol():
    assert isinstance(UrllibArtifactFetcher(), ArtifactFetcher)


def test_opener_is_wired_with_the_no_redirect_handler():
    # The real (non-fake) opener must carry _NoRedirectHandler — otherwise the
    # redirect guard is dead code that never runs on a live fetch.
    fetcher = UrllibArtifactFetcher()

    assert any(isinstance(h, _NoRedirectHandler) for h in fetcher._opener.handlers)


def test_opener_does_not_load_dangerous_scheme_handlers():
    # Defence-in-depth: the file/ftp/data/http handlers must be absent, so the
    # https-only scheme guard is not the single thing between a tampered URL and a
    # live handler for a dangerous scheme.
    fetcher = UrllibArtifactFetcher()

    loaded = {type(h).__name__ for h in fetcher._opener.handlers}
    assert not (loaded & {"FileHandler", "FTPHandler", "DataHandler", "HTTPHandler"})


def test_opener_pins_a_custom_tls_context():
    import ssl

    context = ssl.create_default_context()
    fetcher = UrllibArtifactFetcher(ssl_context=context)

    https = next(h for h in fetcher._opener.handlers if isinstance(h, urllib.request.HTTPSHandler))
    assert https._context is context


def test_object_without_fetch_is_not_an_artifact_fetcher():
    assert not isinstance(object(), ArtifactFetcher)
