#!/usr/bin/env python3
"""Hindsight MCP server for the Devin CLI plugin.

Runs as a stdio subprocess, registered in `~/.config/devin/mcp_config.json` by
scripts/install.py. Exposes knowledge tools (list/get/create/update/delete
pages, recall, ingest). Port of the Claude Code plugin's mcp_server.py —
identical tool surface, so `hindsight-devin-cli:create-agent`-generated
subagents behave the same as their Claude Code counterparts.

Each tool accepts no bank_id parameter — bank_id is always resolved from
plugin config plus the project working directory at server startup.
"""

import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Launched via scripts/run_mcp.sh, which execs this plugin's persistent venv
# interpreter, so `mcp` resolves from ~/.hindsight/devin-cli/venv/site-packages.
from lib.bank import derive_bank_id
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.daemon import get_api_url
from mcp.server.fastmcp import FastMCP

# ── Server setup ────────────────────────────────────────

mcp = FastMCP("hindsight")

_config = load_config()


def _dbg(*a):
    debug_log(_config, *a)


if not _config.get("enableKnowledgeTools"):
    # Knowledge tools are opt-out. Don't exit: the MCP config registers this
    # server unconditionally, and a process that exits at startup gets
    # retried and surfaces a reconnect error on every prompt. Stay alive as
    # an empty MCP server instead.
    _dbg("Knowledge tools disabled (enableKnowledgeTools=false) — running empty MCP server")
    mcp.run(transport="stdio")
    sys.exit(0)

try:
    _api_url = get_api_url(_config, debug_fn=_dbg, allow_daemon_start=True)
except Exception as e:
    print(f"[Hindsight MCP] Failed to resolve API URL: {e}", file=sys.stderr)
    sys.exit(1)

_project_cwd = os.environ.get("HINDSIGHT_MCP_PROJECT_CWD", os.getcwd())
_hook_input = {"cwd": _project_cwd, "session_id": ""}
_default_bank_id = derive_bank_id(_hook_input, _config)
_client = HindsightClient(
    _api_url,
    _config.get("hindsightApiToken"),
    request_timeout_override=_config.get("requestTimeoutSeconds"),
)

_dbg(f"MCP server starting — API: {_api_url}, bank: {_default_bank_id}")


def _encode_bank(bank_id: str) -> str:
    return urllib.parse.quote(bank_id, safe="")


def _encode_page(page_id: str) -> str:
    """Page IDs are agent-supplied tool arguments, so they can contain anything.

    An unescaped `/` re-routes the request and a `?` or `#` truncates the path,
    which would make such a page unreachable through every tool below.
    """
    return urllib.parse.quote(page_id, safe="")


# ── Mental model defaults ───────────────────────────────

PAGE_DEFAULTS = {
    "mode": "delta",
    "refresh_after_consolidation": True,
    "fact_types": ["observation"],
    "exclude_mental_models": True,
}

# ── Tools ───────────────────────────────────────────────
# bank_id is never exposed as a parameter — it's always resolved from plugin
# config + project cwd at server startup.


@mcp.tool()
def agent_knowledge_get_current_bank() -> str:
    """Get the current memory bank ID. This is the bank where conversations are retained and pages are stored. Use this to tell the user which bank their agent will be bound to."""
    return json.dumps({"bank_id": _default_bank_id})


@mcp.tool()
def agent_knowledge_list_pages() -> str:
    """List all your knowledge pages (IDs and names only). Use agent_knowledge_get_page to read the full content of a specific page."""
    resp = _client.request(
        "GET",
        f"/v1/default/banks/{_encode_bank(_default_bank_id)}/mental-models?detail=metadata",
        timeout=10,
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_get_page(page_id: str) -> str:
    """Read a specific knowledge page by its ID. Returns the full synthesized content."""
    resp = _client.request(
        "GET",
        f"/v1/default/banks/{_encode_bank(_default_bank_id)}/mental-models/{_encode_page(page_id)}?detail=content",
        timeout=10,
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_create_page(page_id: str, name: str, source_query: str) -> str:
    """Create a new knowledge page. The source_query is a question the system re-asks after each consolidation to rebuild the page from conversation observations. Pages auto-update as you have more conversations."""
    resp = _client.request(
        "POST",
        f"/v1/default/banks/{_encode_bank(_default_bank_id)}/mental-models",
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
def agent_knowledge_update_page(page_id: str, name: str = "", source_query: str = "") -> str:
    """Update a page's name or source query. The content will re-synthesize on next consolidation."""
    body = {}
    if name:
        body["name"] = name
    if source_query:
        body["source_query"] = source_query
    if not body:
        return json.dumps({"error": "Provide name or source_query to update"})
    resp = _client.request(
        "PATCH",
        f"/v1/default/banks/{_encode_bank(_default_bank_id)}/mental-models/{_encode_page(page_id)}",
        body=body,
        timeout=10,
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_delete_page(page_id: str) -> str:
    """Permanently delete a knowledge page."""
    resp = _client.request(
        "DELETE",
        f"/v1/default/banks/{_encode_bank(_default_bank_id)}/mental-models/{_encode_page(page_id)}",
        timeout=10,
    )
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_recall(query: str, max_tokens: int = 1024) -> str:
    """Search across all retained conversations and documents for specific facts, numbers, or details not covered by your knowledge pages. max_tokens is the result token budget (server returns whatever fits)."""
    resp = _client.recall(bank_id=_default_bank_id, query=query, max_tokens=max_tokens, budget="mid", timeout=10)
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_ingest(title: str, content: str) -> str:
    """Upload text content into your memory bank. Pass the full raw content — never summarize before ingesting. The title becomes the document ID (re-ingesting replaces it)."""
    doc_id = title.lower().replace(" ", "-")
    resp = _client.retain(bank_id=_default_bank_id, content=content, document_id=doc_id, timeout=15)
    return json.dumps(resp, indent=2)


@mcp.tool()
def agent_knowledge_ingest_file(file_path: str) -> str:
    """Ingest a file from disk into your memory bank. Reads the file and uploads its full content. The filename becomes the document ID."""
    if not os.path.isfile(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    content = open(file_path, encoding="utf-8").read()
    if not content.strip():
        return json.dumps({"error": f"File is empty: {file_path}"})

    # Extension included. Stripping it collided distinct files — README.md and
    # README.txt both became "readme", and retain replaces by document_id, so
    # ingesting the second silently overwrote the first. It also makes the id
    # match what this tool's docstring promises: the filename.
    doc_id = os.path.basename(file_path).lower().replace(" ", "-")
    resp = _client.retain(bank_id=_default_bank_id, content=content, document_id=doc_id, timeout=15)
    return json.dumps(resp, indent=2)


# ── Entry point ─────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
