"""Gated MCP-endpoint E2E (requires_real_llm)."""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from hindsight_devin_desktop.mcp_config import mcp_endpoint_url

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
HINDSIGHT_API_TOKEN = os.getenv("HINDSIGHT_API_TOKEN")


def _reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{HINDSIGHT_API_URL}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_real_llm,
    pytest.mark.skipif(not _reachable(), reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}"),
]


def _rpc(url, payload, session=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if session:
        req.add_header("Mcp-Session-Id", session)
    if HINDSIGHT_API_TOKEN:
        req.add_header("Authorization", f"Bearer {HINDSIGHT_API_TOKEN}")
    return urllib.request.urlopen(req, timeout=15)


def test_mcp_endpoint_lists_memory_tools():
    # Multi-bank endpoint (no bank pinned in the path).
    url = mcp_endpoint_url(HINDSIGHT_API_URL)
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "devin-desktop-e2e", "version": "0"},
        },
    }
    resp = _rpc(url, init)
    session = resp.headers.get("Mcp-Session-Id")
    resp.read()
    _rpc(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session=session).read()
    resp = _rpc(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session=session)
    text = resp.read().decode("utf-8", "replace")
    assert "recall" in text and "retain" in text, f"tools/list missing memory tools: {text[:300]}"
