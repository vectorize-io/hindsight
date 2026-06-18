"""End-to-end: the MCP endpoint our Zed config points at actually serves the tools.

This is the real-LLM bucket (``requires_real_llm``) and is skipped unless a
Hindsight server is reachable. It builds the same MCP URL the integration writes
into Zed's settings and confirms a JSON-RPC ``tools/list`` returns the Hindsight
memory tools — i.e. that the config we generate points at a working server.

    HINDSIGHT_API_URL=http://localhost:8888 \
        uv run pytest tests -v -m requires_real_llm
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from hindsight_zed.zed_settings import mcp_endpoint_url

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


def test_mcp_endpoint_lists_memory_tools():
    url = mcp_endpoint_url(HINDSIGHT_API_URL, "zed-e2e")
    body = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if HINDSIGHT_API_TOKEN:
        req.add_header("Authorization", f"Bearer {HINDSIGHT_API_TOKEN}")

    with urllib.request.urlopen(req, timeout=15) as r:
        text = r.read().decode("utf-8", "replace")

    # Streamable-HTTP may answer as SSE; tolerate either by scanning the text.
    assert "recall" in text and "retain" in text, f"tools/list did not surface memory tools: {text[:300]}"
