"""Run the preflight against a fake Hindsight MCP server that behaves like Hindsight Cloud.

The fake mirrors what production returns (checked 2026-09-19): a 401 with a
``resource_metadata`` challenge, RFC 9728 / RFC 8414 metadata with a registration endpoint
and S256, and FastMCP-style SSE replies to MCP requests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hindsight_meta_muse.preflight import PreflightReport, run_preflight

GOOD_TOKEN = "good-token"


@dataclass
class FakeHindsight:
    tools: list[str] = field(
        default_factory=lambda: [
            "recall",
            "retain",
            "reflect",
            "create_mental_model",
            "get_mental_model",
            "list_banks",
            "create_bank",
        ]
    )
    tools_list_error: bool = False
    challenge: bool = True
    registration: bool = True
    sse: bool = True
    seen_methods: list[str] = field(default_factory=list)
    seen_session_ids: list[str | None] = field(default_factory=list)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/mcp", self.mcp)
        app.router.add_post("/mcp/{bank}/", self.mcp)
        app.router.add_get("/.well-known/oauth-protected-resource", self.resource_metadata)
        app.router.add_get("/.well-known/oauth-authorization-server", self.server_metadata)
        return app

    async def resource_metadata(self, request: web.Request) -> web.Response:
        base = f"{request.scheme}://{request.host}"
        return web.json_response({"resource": f"{base}/mcp", "authorization_servers": [base]})

    async def server_metadata(self, request: web.Request) -> web.Response:
        base = f"{request.scheme}://{request.host}"
        body = {
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "code_challenge_methods_supported": ["S256"],
        }
        if self.registration:
            body["registration_endpoint"] = f"{base}/oauth/register"
        return web.json_response(body)

    async def mcp(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {GOOD_TOKEN}":
            headers = {}
            if self.challenge:
                base = f"{request.scheme}://{request.host}"
                headers["WWW-Authenticate"] = f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'
            return web.json_response({"detail": "unauthorized"}, status=401, headers=headers)

        body = await request.json()
        self.seen_methods.append(body["method"])
        self.seen_session_ids.append(request.headers.get("Mcp-Session-Id"))
        if "id" not in body:
            return web.Response(status=202)
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}}
        elif body["method"] == "tools/list" and not self.tools_list_error:
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}} for name in self.tools]}
        else:
            error = {"code": -32601, "message": f"method not allowed: {body['method']}"}
            return web.json_response({"jsonrpc": "2.0", "id": body["id"], "error": error})
        message = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        headers = {"Mcp-Session-Id": "session-1"}
        if self.sse:
            return web.Response(
                text=f"event: message\ndata: {json.dumps(message)}\n\n",
                content_type="text/event-stream",
                headers=headers,
            )
        return web.json_response(message, headers=headers)


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


async def _run(
    fake: FakeHindsight, http: aiohttp.ClientSession, token: str | None, path: str = "/mcp/my-bank/"
) -> PreflightReport:
    server = TestServer(fake.app())
    await server.start_server()
    try:
        return await run_preflight(str(server.make_url(path)), token, http)
    finally:
        await server.close()


def _failed(report: PreflightReport) -> list[str]:
    return [check.name for check in report.checks if not check.ok]


async def test_passes_against_cloud_like_server(http: aiohttp.ClientSession) -> None:
    fake = FakeHindsight()
    report = await _run(fake, http, GOOD_TOKEN)

    assert report.ok, _failed(report)
    assert fake.seen_methods == ["initialize", "notifications/initialized", "tools/list"]
    # The session id from initialize is sent back on later requests, as stateful servers require.
    assert fake.seen_session_ids == [None, "session-1", "session-1"]


async def test_plain_json_replies_are_accepted(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(sse=False), http, GOOD_TOKEN)

    assert report.ok, _failed(report)


async def test_without_token_only_oauth_is_checked(http: aiohttp.ClientSession) -> None:
    fake = FakeHindsight()
    report = await _run(fake, http, None)

    assert report.ok, _failed(report)
    assert fake.seen_methods == []
    assert "PKCE S256 is supported" in [check.name for check in report.checks]


async def test_missing_memory_tool_fails(http: aiohttp.ClientSession) -> None:
    tools = ["recall", "retain", "create_mental_model", "get_mental_model"]
    report = await _run(FakeHindsight(tools=tools), http, GOOD_TOKEN)

    assert _failed(report) == ["the tools the connect prompt uses are exposed"]
    assert report.checks[-1].detail == "missing: reflect"


async def test_bank_url_does_not_need_bank_management_tools(http: aiohttp.ClientSession) -> None:
    tools = ["recall", "retain", "reflect", "create_mental_model", "get_mental_model"]
    report = await _run(FakeHindsight(tools=tools), http, GOOD_TOKEN)

    assert report.ok, _failed(report)


async def test_root_url_passes_with_bank_management_tools(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(), http, GOOD_TOKEN, path="/mcp")

    assert report.ok, _failed(report)


async def test_root_url_needs_bank_management_tools(http: aiohttp.ClientSession) -> None:
    # The connect prompt points Muse at the root URL and has it call list_banks and create_bank there.
    tools = ["recall", "retain", "reflect", "create_mental_model", "get_mental_model"]
    report = await _run(FakeHindsight(tools=tools), http, GOOD_TOKEN, path="/mcp")

    assert _failed(report) == ["the tools the connect prompt uses are exposed"]
    assert report.checks[-1].detail == "missing: list_banks, create_bank"


async def test_tools_list_error_is_reported(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(tools_list_error=True), http, GOOD_TOKEN)

    assert _failed(report) == ["tools/list returns tools"]
    assert report.checks[-1].detail == "method not allowed: tools/list"


async def test_missing_oauth_challenge_fails(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(challenge=False), http, None)

    assert _failed(report) == ["401 points to OAuth resource metadata"]


async def test_missing_dynamic_client_registration_fails(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(registration=False), http, None)

    assert _failed(report) == ["dynamic client registration is offered"]


async def test_rejected_token_fails(http: aiohttp.ClientSession) -> None:
    report = await _run(FakeHindsight(), http, "wrong-token")

    assert _failed(report) == ["MCP initialize succeeds with the token"]
