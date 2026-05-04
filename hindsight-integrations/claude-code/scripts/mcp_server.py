#!/usr/bin/env python3
"""Hindsight MCP server for Claude Code plugin.

Runs as a stdio subprocess managed by the plugin system.
Exposes knowledge tools (list/get/create/update/delete pages, recall, ingest).
Reuses the existing plugin config chain and client.

Each tool accepts an optional bank_id parameter. When omitted, falls back to the
default bank derived from config at startup. The PreToolUse hook (inject_bank_id.py)
injects bank_id from session context (cwd, agentName) before calls reach here.
"""

import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.daemon import get_api_url
from lib.bank import derive_bank_id

# ── Server setup ────────────────────────────────────────

mcp = FastMCP("hindsight")

# Resolve config at startup
_config = load_config()
_dbg = lambda *a: debug_log(_config, *a)

if not _config.get("enableKnowledgeTools"):
    _dbg("Knowledge tools disabled (enableKnowledgeTools=false), MCP server exiting")
    sys.exit(0)

try:
    _api_url = get_api_url(_config, debug_fn=_dbg, allow_daemon_start=True)
except Exception as e:
    print(f"[Hindsight MCP] Failed to resolve API URL: {e}", file=sys.stderr)
    sys.exit(1)

_hook_input = {"cwd": os.getcwd(), "session_id": ""}
_default_bank_id = derive_bank_id(_hook_input, _config)
_client = HindsightClient(_api_url, _config.get("hindsightApiToken"))

_dbg(f"MCP server starting — API: {_api_url}, default bank: {_default_bank_id}")


def _bank(bank_id: str = "") -> str:
    """Resolve bank ID: use explicit value if provided, else fall back to default."""
    return bank_id if bank_id else _default_bank_id


def _encode_bank(bank_id: str) -> str:
    return urllib.parse.quote(bank_id, safe="")


# ── Mental model defaults ───────────────────────────────

PAGE_DEFAULTS = {
    "mode": "delta",
    "refresh_after_consolidation": True,
    "fact_types": ["observation"],
    "exclude_mental_models": True,
}

# ── Tools ───────────────────────────────────────────────


@mcp.tool()
def agent_knowledge_list_pages(bank_id: str = "") -> str:
    """List all your knowledge pages (IDs and names only). Use agent_knowledge_get_page to read the full content of a specific page."""
    bid = _bank(bank_id)
    resp = _client.request("GET", f"/v1/default/banks/{_encode_bank(bid)}/mental-models", timeout=10)
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_get_page(page_id: str, bank_id: str = "") -> str:
    """Read a specific knowledge page by its ID. Returns the full synthesized content."""
    bid = _bank(bank_id)
    resp = _client.request(
        "GET", f"/v1/default/banks/{_encode_bank(bid)}/mental-models/{page_id}?detail=full", timeout=10
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_create_page(page_id: str, name: str, source_query: str, bank_id: str = "") -> str:
    """Create a new knowledge page. The source_query is a question the system re-asks after each consolidation to rebuild the page from conversation observations. Pages auto-update as you have more conversations."""
    bid = _bank(bank_id)
    resp = _client.request(
        "POST",
        f"/v1/default/banks/{_encode_bank(bid)}/mental-models",
        body={
            "id": page_id,
            "name": name,
            "source_query": source_query,
            "max_tokens": 4096,
            "trigger": PAGE_DEFAULTS,
        },
        timeout=15,
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_update_page(page_id: str, name: str = "", source_query: str = "", bank_id: str = "") -> str:
    """Update a page's name or source query. The content will re-synthesize on next consolidation."""
    body = {}
    if name:
        body["name"] = name
    if source_query:
        body["source_query"] = source_query
    if not body:
        return json.dumps({"error": "Provide name or source_query to update"})
    bid = _bank(bank_id)
    resp = _client.request(
        "PATCH", f"/v1/default/banks/{_encode_bank(bid)}/mental-models/{page_id}", body=body, timeout=10
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_delete_page(page_id: str, bank_id: str = "") -> str:
    """Permanently delete a knowledge page."""
    bid = _bank(bank_id)
    resp = _client.request("DELETE", f"/v1/default/banks/{_encode_bank(bid)}/mental-models/{page_id}", timeout=10)
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_recall(query: str, max_results: int = 10, bank_id: str = "") -> str:
    """Search across all retained conversations and documents for specific facts, numbers, or details not covered by your knowledge pages."""
    bid = _bank(bank_id)
    resp = _client.recall(bank_id=bid, query=query, max_tokens=max_results, budget="mid", timeout=10)
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_ingest(title: str, content: str, bank_id: str = "") -> str:
    """Upload a document into your memory bank. Pass the full raw content — never summarize before ingesting. The title becomes the document ID (re-ingesting replaces it)."""
    bid = _bank(bank_id)
    doc_id = title.lower().replace(" ", "-")
    resp = _client.retain(bank_id=bid, content=content, document_id=doc_id, timeout=15)
    return json.dumps(resp, indent=2)


# ── Entry point ─────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
