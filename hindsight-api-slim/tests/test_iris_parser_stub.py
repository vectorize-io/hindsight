"""Unit tests for the Iris parser against a stub Vectorize API (no credentials needed).

The live-API tests are in ``test_iris_parser.py`` and skip without a token.
"""

from dataclasses import dataclass, field

import pytest
from aiohttp import web

from hindsight_api.engine.parsers import iris
from hindsight_api.engine.parsers.base import UnsupportedFileTypeError
from hindsight_api.engine.parsers.iris import IrisParser
from tests.aiohttp_stub import stub_server

# A pre-signed query string: its percent-escapes must reach the server untouched.
_SIGNATURE = "X-Amz-Signature=ab%2Fcd%3D%3D"


@dataclass
class _Upstream:
    extraction_statuses: list[dict]
    fail: dict[str, int] = field(default_factory=dict)  # step path suffix -> status
    uploaded: list[tuple[bytes, str | None, str | None]] = field(default_factory=list)
    polls: int = 0

    async def handler(self, request: web.Request) -> web.StreamResponse:
        path = request.path
        for suffix, status in self.fail.items():
            if path.endswith(suffix):
                return web.Response(status=status, text=f"refused at {suffix}")
        if request.method == "POST" and path.endswith("/org/org-1/files"):
            assert request.headers["Authorization"] == "Bearer tok"
            body = await request.json()
            assert body == {"name": "doc.pdf", "contentType": "application/pdf"}
            upload_url = f"{request.scheme}://{request.host}/presigned/upload?{_SIGNATURE}"
            return web.json_response({"fileId": "file-1", "uploadUrl": upload_url})
        if request.method == "PUT" and path == "/presigned/upload":
            assert request.raw_path.partition("?")[2] == _SIGNATURE
            self.uploaded.append(
                (await request.read(), request.headers.get("Content-Type"), request.headers.get("Authorization"))
            )
            return web.Response(status=200)
        if request.method == "POST" and path.endswith("/org/org-1/extraction"):
            assert await request.json() == {"fileId": "file-1"}
            return web.json_response({"extractionId": "ext-1"})
        if request.method == "GET" and path.endswith("/org/org-1/extraction/ext-1"):
            self.polls += 1
            statuses = self.extraction_statuses
            return web.json_response(statuses.pop(0) if len(statuses) > 1 else statuses[0])
        raise AssertionError(f"unexpected request: {request.method} {path}")


async def _convert(monkeypatch: pytest.MonkeyPatch, upstream: _Upstream, *, timeout: float = 10.0) -> str:
    async with stub_server(upstream.handler) as base_url:
        monkeypatch.setattr(iris, "_IRIS_BASE_URL", base_url)
        parser = IrisParser(token="tok", org_id="org-1", poll_interval=0.0, timeout=timeout)
        return await parser.convert(b"%PDF-bytes", "doc.pdf")


@pytest.mark.asyncio
async def test_convert_uploads_to_presigned_url_and_polls_until_ready(monkeypatch):
    upstream = _Upstream(
        extraction_statuses=[
            {"ready": False},
            {"ready": True, "data": {"success": True, "text": "Hello from Hindsight"}},
        ]
    )

    assert await _convert(monkeypatch, upstream) == "Hello from Hindsight"
    assert upstream.polls == 2
    # Raw bytes with the guessed type, and no bearer token sent to the presigned URL.
    assert upstream.uploaded == [(b"%PDF-bytes", "application/pdf", None)]


@pytest.mark.asyncio
async def test_extraction_failure_raises_runtime_error(monkeypatch):
    upstream = _Upstream(extraction_statuses=[{"ready": True, "data": {"success": False, "error": "corrupt"}}])

    with pytest.raises(RuntimeError, match="corrupt"):
        await _convert(monkeypatch, upstream)


@pytest.mark.asyncio
async def test_timeout_raises_runtime_error(monkeypatch):
    upstream = _Upstream(extraction_statuses=[{"ready": False}])

    with pytest.raises(RuntimeError, match="timed out"):
        await _convert(monkeypatch, upstream, timeout=0.0)


@pytest.mark.asyncio
async def test_client_error_is_an_unsupported_file_type(monkeypatch):
    upstream = _Upstream(extraction_statuses=[], fail={"/org/org-1/files": 415})

    with pytest.raises(UnsupportedFileTypeError, match="file upload init.*415.*refused at"):
        await _convert(monkeypatch, upstream)


@pytest.mark.asyncio
async def test_server_error_is_a_runtime_error(monkeypatch):
    upstream = _Upstream(extraction_statuses=[], fail={"/org/org-1/extraction": 503})

    with pytest.raises(RuntimeError, match="start extraction.*503") as exc:
        await _convert(monkeypatch, upstream)
    assert not isinstance(exc.value, UnsupportedFileTypeError)
