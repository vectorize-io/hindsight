"""Check that a Hindsight MCP endpoint can be connected the way Meta Muse connects to it.

Muse's custom-connector flow, observed against Hindsight Cloud:
  1. The user gives Muse the MCP URL.
  2. Muse calls it unauthenticated, gets a 401 whose ``WWW-Authenticate`` header points at
     the OAuth protected-resource metadata (RFC 9728).
  3. Muse reads the authorization-server metadata (RFC 8414), registers itself as a client
     (dynamic client registration, RFC 7591) and runs an authorization-code flow with PKCE.
  4. With the token it speaks MCP over Streamable HTTP: ``initialize`` then ``tools/list``.

This module replays steps 2-4 so a self-hosted deployment (or an OAuth proxy in front of
one) can be checked before a user tries it in Muse. Step 4 needs a token; without one the
MCP checks are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from pydantic import BaseModel, ValidationError

# The memory tools the connect prompt tells Muse to call. The root URL (multi-bank) and a
# bank-scoped URL (/mcp/<bank>/) both expose these; only the root URL adds list_banks and
# create_bank, which the prompt uses to give Muse its own bank.
REQUIRED_TOOLS: tuple[str, ...] = ("recall", "retain", "reflect")

# Protocol revision sent in `initialize`. Servers negotiate down if they speak an older one.
MCP_PROTOCOL_VERSION = "2025-06-18"

_RESOURCE_METADATA_RE = re.compile(r'resource_metadata="([^"]+)"')


class ProtectedResourceMetadata(BaseModel):
    resource: str
    authorization_servers: list[str]


class AuthorizationServerMetadata(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    code_challenge_methods_supported: list[str] = []


class ToolInfo(BaseModel):
    name: str


class ToolsListResult(BaseModel):
    tools: list[ToolInfo]


class JsonRpcError(BaseModel):
    code: int
    message: str


class JsonRpcResponse(BaseModel):
    id: int | str | None = None
    # `result` differs per method; it is validated into a method-specific model by the caller.
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append(CheckResult(name=name, ok=ok, detail=detail))
        return ok


@dataclass
class McpReply:
    status: int
    session_id: str | None
    message: JsonRpcResponse | None


# JSON-RPC envelopes stay as dicts rather than models: `params` and `result` are defined per method
# by the MCP spec, so there is no fixed schema to model here. Each `result` this module actually reads
# is parsed into a model at the boundary (see `ToolsListResult` in `_check_tools`).
def _rpc(method: str, request_id: int | None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        body["id"] = request_id
    if params is not None:
        body["params"] = params
    return body


def _parse_body(content_type: str, text: str, request_id: int) -> JsonRpcResponse | None:
    """Pull the response for `request_id` out of a JSON or SSE (text/event-stream) body.

    Streamable HTTP servers may answer a POST with either; FastMCP answers with SSE by default.
    """
    if "text/event-stream" in content_type:
        payloads = [line[len("data:") :].strip() for line in text.splitlines() if line.startswith("data:")]
    else:
        payloads = [text]
    for payload in payloads:
        if not payload:
            continue
        try:
            message = JsonRpcResponse.model_validate(json.loads(payload))
        except (json.JSONDecodeError, ValidationError):
            continue
        if message.id == request_id:
            return message
    return None


async def _post_mcp(
    session: aiohttp.ClientSession,
    url: str,
    body: dict[str, Any],
    token: str | None,
    session_id: str | None = None,
) -> McpReply:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    async with session.post(url, json=body, headers=headers) as resp:
        text = await resp.text()
        request_id = body.get("id")
        message = _parse_body(resp.content_type, text, request_id) if isinstance(request_id, int) else None
        return McpReply(status=resp.status, session_id=resp.headers.get("Mcp-Session-Id"), message=message)


async def _check_oauth(session: aiohttp.ClientSession, url: str, report: PreflightReport) -> None:
    init = _rpc("initialize", 1, {"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {}})
    async with session.post(url, json=init, headers={"Accept": "application/json, text/event-stream"}) as resp:
        status = resp.status
        challenge = resp.headers.get("WWW-Authenticate", "")
    if not report.add("unauthenticated request is refused", status == 401, f"HTTP {status}"):
        return

    match = _RESOURCE_METADATA_RE.search(challenge)
    if match is None:
        report.add("401 points to OAuth resource metadata", False, challenge or "no WWW-Authenticate header")
        return
    report.add("401 points to OAuth resource metadata", True, match.group(1))

    try:
        async with session.get(match.group(1)) as resp:
            resource = ProtectedResourceMetadata.model_validate(await resp.json(content_type=None))
    except (aiohttp.ClientError, json.JSONDecodeError, ValidationError) as exc:
        report.add("resource metadata is readable", False, str(exc))
        return
    if not report.add(
        "resource metadata names an authorization server",
        len(resource.authorization_servers) > 0,
        resource.resource,
    ):
        return

    issuer = resource.authorization_servers[0].rstrip("/")
    try:
        async with session.get(f"{issuer}/.well-known/oauth-authorization-server") as resp:
            server = AuthorizationServerMetadata.model_validate(await resp.json(content_type=None))
    except (aiohttp.ClientError, json.JSONDecodeError, ValidationError) as exc:
        report.add("authorization server metadata is readable", False, str(exc))
        return
    report.add("authorization server metadata is readable", True, server.issuer)
    # Muse registers itself; there is no place in its flow to paste a pre-issued client id.
    report.add(
        "dynamic client registration is offered",
        server.registration_endpoint is not None,
        server.registration_endpoint or "no registration_endpoint",
    )
    report.add(
        "PKCE S256 is supported",
        "S256" in server.code_challenge_methods_supported,
        ", ".join(server.code_challenge_methods_supported) or "none listed",
    )


async def _check_tools(session: aiohttp.ClientSession, url: str, token: str, report: PreflightReport) -> None:
    init = await _post_mcp(
        session,
        url,
        _rpc(
            "initialize",
            1,
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "hindsight-meta-muse-preflight", "version": "0.1.0"},
            },
        ),
        token,
    )
    ok = init.status == 200 and init.message is not None and init.message.error is None
    if not report.add("MCP initialize succeeds with the token", ok, f"HTTP {init.status}"):
        return

    # Stateless servers return no session id; stateful ones require it on every later request.
    await _post_mcp(session, url, _rpc("notifications/initialized", None), token, init.session_id)
    listed = await _post_mcp(session, url, _rpc("tools/list", 2), token, init.session_id)
    if listed.message is None or listed.message.result is None:
        report.add("tools/list returns tools", False, f"HTTP {listed.status}")
        return
    try:
        tools = ToolsListResult.model_validate(listed.message.result)
    except ValidationError as exc:
        report.add("tools/list returns tools", False, str(exc))
        return
    names = {tool.name for tool in tools.tools}
    missing = [name for name in REQUIRED_TOOLS if name not in names]
    report.add(
        "recall, retain and reflect are exposed",
        not missing,
        f"missing: {', '.join(missing)}" if missing else f"{len(names)} tools",
    )


async def run_preflight(url: str, token: str | None, session: aiohttp.ClientSession) -> PreflightReport:
    report = PreflightReport()
    await _check_oauth(session, url, report)
    if token:
        await _check_tools(session, url, token, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a Hindsight MCP endpoint is ready for Meta Muse.")
    parser.add_argument("url", help="MCP URL, e.g. https://api.hindsight.vectorize.io/mcp")
    parser.add_argument(
        "--token-env",
        default="HINDSIGHT_API_KEY",
        help="Environment variable holding a token for the MCP checks (default: HINDSIGHT_API_KEY). "
        "Read from the environment so the token stays out of shell history.",
    )
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env) or None

    async def _run() -> PreflightReport:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await run_preflight(args.url, token, session)

    report = asyncio.run(_run())
    for check in report.checks:
        print(f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}")
    if not token:
        print(f"SKIP  MCP tool checks: set {args.token_env} to run them")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
