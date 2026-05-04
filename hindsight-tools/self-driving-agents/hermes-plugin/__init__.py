"""Self-driving agents Hindsight plugin for Hermes.

Registers agent_knowledge_* tools as a regular plugin (not a memory provider),
so it coexists with the bundled hindsight memory provider or any other provider.

Config read from ~/.self-driving-agents/hermes/<agent>/config.json:
  { "api_url": "...", "api_token": "...", "bank_id": "..." }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PAGE_DEFAULTS = {
    "mode": "delta",
    "refresh_after_consolidation": True,
    "exclude_mental_models": True,
    "fact_types": ["observation"],
}


def _get_hermes_home() -> Path:
    """Get the active HERMES_HOME (respects profile isolation)."""
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _load_config() -> dict | None:
    """Load Hindsight config from the active Hermes profile.

    Reads the same config.json the bundled hindsight provider uses,
    so both share the same bank, API URL, and credentials.
    """
    cfg_path = _get_hermes_home() / "hindsight" / "config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
        # Normalize field names (bundled provider uses api_url/api_key,
        # we expose as api_url/api_token for consistency with other harnesses)
        return {
            "api_url": cfg.get("api_url", ""),
            "api_token": cfg.get("api_key", ""),
            "bank_id": cfg.get("bank_id", "hermes"),
        }
    except Exception:
        return None


def _api(
    api_url: str,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> Any:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.request(
        method,
        f"{api_url}{path}",
        json=body,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def _is_available() -> bool:
    return _load_config() is not None


# ── Tool handlers ───────────────────────────────────────


def _handle_list_pages(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        result = _api(api_url, f"/v1/default/banks/{bank_id}/mental-models?detail=metadata", "GET", token=token)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_get_page(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        result = _api(api_url, f"/v1/default/banks/{bank_id}/mental-models/{args['page_id']}", "GET", token=token)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_create_page(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        result = _api(
            api_url,
            f"/v1/default/banks/{bank_id}/mental-models",
            "POST",
            body={
                "id": args["page_id"],
                "name": args["name"],
                "source_query": args["source_query"],
                "max_tokens": 4096,
                "trigger": PAGE_DEFAULTS,
            },
            token=token,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_update_page(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        body: dict[str, str] = {}
        if args.get("name"):
            body["name"] = args["name"]
        if args.get("source_query"):
            body["source_query"] = args["source_query"]
        result = _api(api_url, f"/v1/default/banks/{bank_id}/mental-models/{args['page_id']}", "PATCH", body=body, token=token)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_delete_page(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        _api(api_url, f"/v1/default/banks/{bank_id}/mental-models/{args['page_id']}", "DELETE", token=token)
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_recall(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        result = _api(
            api_url,
            f"/v1/default/banks/{bank_id}/memories/recall",
            "POST",
            body={"query": args["query"], "max_results": args.get("max_results", 10)},
            token=token,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _handle_ingest(args: dict, **kwargs: Any) -> str:
    config = _load_config()
    if not config:
        return json.dumps({"error": "Plugin not configured"})
    api_url = config["api_url"].rstrip("/")
    token = config.get("api_token")
    bank_id = config["bank_id"]
    try:
        doc_id = args["title"].lower().replace(" ", "-")
        result = _api(
            api_url,
            f"/v1/default/banks/{bank_id}/memories",
            "POST",
            body={"items": [{"content": args["content"], "document_id": doc_id}], "async": True},
            token=token,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool schemas ────────────────────────────────────────

_TOOLS = [
    (
        "agent_knowledge_list_pages",
        {
            "name": "agent_knowledge_list_pages",
            "description": "List all your knowledge pages (IDs and names only). Use agent_knowledge_get_page to read the full content of a specific page.",
            "parameters": {"type": "object", "properties": {}},
        },
        _handle_list_pages,
        "📚",
    ),
    (
        "agent_knowledge_get_page",
        {
            "name": "agent_knowledge_get_page",
            "description": "Read a specific knowledge page by its ID. Returns the full synthesized content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID (e.g. 'user-preferences')"},
                },
                "required": ["page_id"],
            },
        },
        _handle_get_page,
        "📖",
    ),
    (
        "agent_knowledge_create_page",
        {
            "name": "agent_knowledge_create_page",
            "description": (
                "Create a new knowledge page. The source_query is a question the system "
                "re-asks after each consolidation to rebuild the page from conversation observations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Unique page ID, lowercase with hyphens"},
                    "name": {"type": "string", "description": "Human-readable page name"},
                    "source_query": {"type": "string", "description": "The question that rebuilds this page"},
                },
                "required": ["page_id", "name", "source_query"],
            },
        },
        _handle_create_page,
        "📝",
    ),
    (
        "agent_knowledge_update_page",
        {
            "name": "agent_knowledge_update_page",
            "description": "Update a page's name or source query. Content re-synthesizes on next consolidation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID to update"},
                    "name": {"type": "string", "description": "New name (optional)"},
                    "source_query": {"type": "string", "description": "New source query (optional)"},
                },
                "required": ["page_id"],
            },
        },
        _handle_update_page,
        "✏️",
    ),
    (
        "agent_knowledge_delete_page",
        {
            "name": "agent_knowledge_delete_page",
            "description": "Permanently delete a knowledge page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID to delete"},
                },
                "required": ["page_id"],
            },
        },
        _handle_delete_page,
        "🗑️",
    ),
    (
        "agent_knowledge_recall",
        {
            "name": "agent_knowledge_recall",
            "description": "Search across all retained conversations and documents for specific facts, numbers, or details not covered by your knowledge pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "max_results": {"type": "number", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
        },
        _handle_recall,
        "🔍",
    ),
    (
        "agent_knowledge_ingest",
        {
            "name": "agent_knowledge_ingest",
            "description": (
                "Upload a document into your memory bank. Pass the full raw content — "
                "never summarize before ingesting. The title becomes the document ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title (becomes the document ID)"},
                    "content": {"type": "string", "description": "Full raw document content"},
                },
                "required": ["title", "content"],
            },
        },
        _handle_ingest,
        "📥",
    ),
]


# ── Registration ────────────────────────────────────────


def register(ctx: Any) -> None:
    """Register agent_knowledge_* tools. Called once by the Hermes plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="hindsight-sda",
            schema=schema,
            handler=handler,
            check_fn=_is_available,
            emoji=emoji,
        )
    logger.info("[hindsight-sda] registered %d tools", len(_TOOLS))
