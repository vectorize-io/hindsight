"""
Tests for the LlamaParse file parser.

Unit tests run always (stub HTTP upstream). Integration tests require
HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY in the environment.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest
from aiohttp import web

from hindsight_api.config import ENV_FILE_PARSER_LLAMA_PARSE_API_KEY
from hindsight_api.engine.parsers import llama_parse
from hindsight_api.engine.parsers.base import UnsupportedFileTypeError
from hindsight_api.engine.parsers.llama_parse import LlamaParseParser
from tests.aiohttp_stub import stub_server

_api_key = os.getenv(ENV_FILE_PARSER_LLAMA_PARSE_API_KEY)

# Minimal valid PDF with the text "Hello from Hindsight"
_SAMPLE_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello from Hindsight) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
trailer << /Size 5 /Root 1 0 R >>
startxref
369
%%EOF"""


@dataclass
class _Reply:
    status: int = 200
    json: dict | None = None
    text: str = ""


@dataclass
class _Upstream:
    """Scripted LlamaParse API: the upload reply, then status/result replies in order."""

    upload: _Reply
    gets: list[_Reply] = field(default_factory=list)
    get_count: int = 0
    upload_parts: list[tuple[str | None, str | None, bytes]] = field(default_factory=list)

    async def handler(self, request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"].startswith("Bearer ")
        if request.method == "POST":
            assert request.path.endswith("/upload")
            reader = await request.multipart()
            part = await reader.next()
            assert part is not None and part.name == "file"
            self.upload_parts.append((part.filename, part.headers.get("Content-Type"), bytes(await part.read())))
            reply = self.upload
        else:
            self.get_count += 1
            reply = self.gets.pop(0) if len(self.gets) > 1 else self.gets[0]
        if reply.json is not None:
            return web.json_response(reply.json, status=reply.status)
        return web.Response(status=reply.status, text=reply.text)


@asynccontextmanager
async def _serve(monkeypatch: pytest.MonkeyPatch, upstream: _Upstream) -> AsyncIterator[None]:
    async with stub_server(upstream.handler) as base_url:
        monkeypatch.setattr(llama_parse, "_LLAMA_PARSE_BASE_URL", base_url)
        yield


# ---------------------------------------------------------------------------
# Unit tests (always run — stub HTTP upstream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_success(monkeypatch):
    """Happy path: upload → poll SUCCESS → fetch markdown."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)
    upstream = _Upstream(
        upload=_Reply(json={"id": "job-123"}),
        gets=[_Reply(json={"status": "SUCCESS"}), _Reply(json={"markdown": "# Hello"})],
    )

    async with _serve(monkeypatch, upstream):
        result = await parser.convert(b"fake-pdf", "test.pdf")
    assert result == "# Hello"
    # The file goes up as a multipart part carrying its name and guessed type.
    assert upstream.upload_parts == [("test.pdf", "application/pdf", b"fake-pdf")]


@pytest.mark.asyncio
async def test_convert_polls_until_success(monkeypatch):
    """Parser should poll multiple times before SUCCESS."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)
    pending = _Reply(json={"status": "PENDING"})
    upstream = _Upstream(
        upload=_Reply(json={"id": "job-456"}),
        gets=[pending, pending, _Reply(json={"status": "SUCCESS"}), _Reply(json={"markdown": "parsed content"})],
    )

    async with _serve(monkeypatch, upstream):
        result = await parser.convert(b"fake", "doc.pdf")
    assert result == "parsed content"
    assert upstream.get_count == 4  # 2 pending + 1 success + 1 result


@pytest.mark.asyncio
async def test_convert_job_error(monkeypatch):
    """Parser should raise RuntimeError when job status is ERROR."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=10.0)
    upstream = _Upstream(
        upload=_Reply(json={"id": "job-err"}),
        gets=[_Reply(json={"status": "ERROR", "error_code": "PARSE_FAILED"})],
    )

    async with _serve(monkeypatch, upstream):
        with pytest.raises(RuntimeError, match="PARSE_FAILED"):
            await parser.convert(b"bad", "bad.pdf")


@pytest.mark.asyncio
async def test_convert_timeout(monkeypatch):
    """Parser should raise RuntimeError on timeout."""
    parser = LlamaParseParser(api_key="llx-test", poll_interval=0.0, timeout=0.0)
    upstream = _Upstream(upload=_Reply(json={"id": "job-slow"}), gets=[_Reply(json={"status": "PENDING"})])

    async with _serve(monkeypatch, upstream):
        with pytest.raises(RuntimeError, match="timed out"):
            await parser.convert(b"data", "slow.pdf")


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(monkeypatch):
    """400/415/422 on upload should raise UnsupportedFileTypeError."""
    for status_code in (400, 415, 422):
        parser = LlamaParseParser(api_key="llx-test")
        upstream = _Upstream(upload=_Reply(status=status_code, text="unsupported format"))

        async with _serve(monkeypatch, upstream):
            with pytest.raises(UnsupportedFileTypeError, match="unsupported format"):
                await parser.convert(b"data", "file.xyz")


@pytest.mark.asyncio
async def test_auth_error_raises_runtime_error(monkeypatch):
    """401/403 should raise RuntimeError, not UnsupportedFileTypeError."""
    for status_code in (401, 403):
        parser = LlamaParseParser(api_key="bad-key")
        upstream = _Upstream(upload=_Reply(status=status_code, text="unauthorized"))

        async with _serve(monkeypatch, upstream):
            with pytest.raises(RuntimeError, match="unauthorized") as exc:
                await parser.convert(b"data", "file.pdf")
        assert not isinstance(exc.value, UnsupportedFileTypeError)


@pytest.mark.asyncio
async def test_rate_limit_raises_runtime_error(monkeypatch):
    """429 should raise RuntimeError, not UnsupportedFileTypeError."""
    parser = LlamaParseParser(api_key="llx-test")
    upstream = _Upstream(upload=_Reply(status=429, text="rate limited"))

    async with _serve(monkeypatch, upstream):
        with pytest.raises(RuntimeError, match="rate limited") as exc:
            await parser.convert(b"data", "file.pdf")
    assert not isinstance(exc.value, UnsupportedFileTypeError)


def test_parser_name():
    """LlamaParseParser.name() should return 'llama_parse'."""
    parser = LlamaParseParser(api_key="llx-test")
    assert parser.name() == "llama_parse"


# ---------------------------------------------------------------------------
# Integration tests (require API key)
# ---------------------------------------------------------------------------

_integration = pytest.mark.skipif(
    not _api_key,
    reason="HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY not set",
)


@pytest.fixture
def llama_parse_parser() -> LlamaParseParser:
    assert _api_key is not None
    return LlamaParseParser(api_key=_api_key)


@_integration
@pytest.mark.asyncio
async def test_llama_parse_parser_converts_pdf(llama_parse_parser: LlamaParseParser):
    """LlamaParseParser should extract text from a valid PDF."""
    result = await llama_parse_parser.convert(_SAMPLE_PDF, "sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0
