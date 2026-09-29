"""Smoke tests for the Roo Code Hindsight integration.

Requires a live Hindsight instance. Set HINDSIGHT_SMOKE_URL to run:

    HINDSIGHT_SMOKE_URL=http://localhost:8888 python -m pytest tests/test_smoke.py -v

Skipped automatically if HINDSIGHT_SMOKE_URL is not set.

Covers:
  1. Health check — Hindsight is reachable
  2. MCP endpoint — /mcp responds to initialize + tools/list, recall/retain present
  3. REST retain — memories can be stored
  4. REST recall — stored memories are returned
  5. Round-trip — retain then recall returns the retained content
  6. MCP recall — recall tool works via MCP protocol (what Roo Code actually calls)
  7. MCP retain — retain tool works via MCP protocol
"""

import json
import os
import urllib.request
import urllib.error
from urllib.parse import urljoin

import pytest

SMOKE_URL = os.environ.get("HINDSIGHT_SMOKE_URL", "")
SMOKE_API_KEY = os.environ.get("HINDSIGHT_SMOKE_API_KEY", "")
SMOKE_BANK = "roo-smoke-test"

skip_if_no_server = pytest.mark.skipif(
    not SMOKE_URL,
    reason="HINDSIGHT_SMOKE_URL not set — skipping live smoke tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    h: dict = {}
    if SMOKE_API_KEY:
        h["Authorization"] = f"Bearer {SMOKE_API_KEY}"
    return h


def _post(path: str, body: dict) -> dict:
    url = urljoin(SMOKE_URL.rstrip("/") + "/", path.lstrip("/"))
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    url = urljoin(SMOKE_URL.rstrip("/") + "/", path.lstrip("/"))
    req = urllib.request.Request(url, headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _mcp_call(method: str, params: dict, session_id: str | None = None) -> tuple[dict, str | None]:
    """Send a single JSON-RPC MCP request. Returns (response_body, session_id)."""
    url = SMOKE_URL.rstrip("/") + "/mcp"
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **_auth_headers(),
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        new_session_id = resp.headers.get("Mcp-Session-Id")
        raw = resp.read().decode()
        # Streamable HTTP returns SSE: "event: message\r\ndata: {...}\r\n\r\n"
        # Extract the last data: line
        data_line = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                data_line = line[5:].strip()
        if data_line:
            raw = data_line
        return json.loads(raw), new_session_id


def _mcp_initialize() -> str | None:
    """Run MCP initialize handshake and return session ID."""
    _, session_id = _mcp_call(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "roo-smoke-test", "version": "1.0"},
        },
    )
    return session_id


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_health_check() -> None:
    result = _get("/health")
    assert result.get("status") == "ok" or "status" in result


# ---------------------------------------------------------------------------
# 2. MCP endpoint — tools listed
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_mcp_initialize() -> None:
    session_id = _mcp_initialize()
    # session_id may be None for stateless servers — that's fine
    _ = session_id


@skip_if_no_server
def test_mcp_tools_list_contains_recall_and_retain() -> None:
    session_id = _mcp_initialize()
    result, _ = _mcp_call("tools/list", {}, session_id)
    tool_names = {t["name"] for t in result.get("result", {}).get("tools", [])}
    assert "recall" in tool_names, f"recall not in MCP tools: {tool_names}"
    assert "retain" in tool_names, f"retain not in MCP tools: {tool_names}"


# ---------------------------------------------------------------------------
# 3 & 4. REST retain + recall
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_rest_retain_and_recall() -> None:
    marker = "roo-smoke-test-marker-abc123"

    # Retain via REST
    _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories",
        {"items": [{"content": f"Smoke test memory: {marker}"}]},
    )

    # Recall via REST
    result = _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories/recall",
        {"query": marker, "budget": "low"},
    )
    texts = [r.get("text", "") for r in result.get("results", [])]
    assert any(marker in t for t in texts), f"Marker not recalled. Got: {texts}"


# ---------------------------------------------------------------------------
# 5. Round-trip — retain then recall returns content
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_round_trip_unique_content() -> None:
    unique = "roo-smoke-unique-decision-xyz987"

    _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories",
        {"items": [{"content": f"Decision: always use {unique} for routing"}]},
    )

    result = _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories/recall",
        {"query": unique, "budget": "low"},
    )
    texts = " ".join(r.get("text", "") for r in result.get("results", []))
    assert unique in texts


# ---------------------------------------------------------------------------
# 6. MCP recall tool
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_mcp_recall_tool() -> None:
    marker = "roo-mcp-recall-test-marker"

    # Seed via REST first
    _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories",
        {"items": [{"content": f"MCP recall test: {marker}"}]},
    )

    session_id = _mcp_initialize()
    result, _ = _mcp_call(
        "tools/call",
        {
            "name": "recall",
            "arguments": {"query": marker, "bank_id": SMOKE_BANK},
        },
        session_id,
    )

    content = result.get("result", {})
    assert content, f"MCP recall returned empty result: {result}"


# ---------------------------------------------------------------------------
# 7. MCP retain tool
# ---------------------------------------------------------------------------


@skip_if_no_server
def test_mcp_retain_tool() -> None:
    marker = "roo-mcp-retain-test-marker"

    session_id = _mcp_initialize()
    # Use sync_retain so processing completes before we verify recall
    result, _ = _mcp_call(
        "tools/call",
        {
            "name": "sync_retain",
            "arguments": {
                "content": f"MCP retain test: {marker}",
                "bank_id": SMOKE_BANK,
            },
        },
        session_id,
    )

    assert "error" not in result, f"MCP sync_retain returned error: {result}"

    # Verify something is recallable for this topic (LLM extracts semantics, not verbatim strings)
    recall_result = _post(
        f"/v1/default/banks/{SMOKE_BANK}/memories/recall",
        {"query": "MCP retain test", "budget": "low"},
    )
    assert len(recall_result.get("results", [])) > 0, "MCP retain: nothing recalled after sync_retain"
