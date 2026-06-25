"""Wire Hindsight into Devin Desktop's MCP config (``~/.codeium/windsurf/mcp_config.json``).

Devin Desktop (formerly Windsurf) reads MCP servers from
``~/.codeium/windsurf/mcp_config.json`` under the ``mcpServers`` key — the path
still carries the legacy ``windsurf`` segment, which is the app's on-disk data
directory and is unchanged by the rebrand. For a remote server it uses a
``serverUrl`` field (plus optional ``headers``), so the Hindsight MCP endpoint
connects with no bridge::

    {
      "mcpServers": {
        "hindsight": {
          "serverUrl": "https://api.hindsight.vectorize.io/mcp/<bank>/",
          "headers": { "Authorization": "Bearer hsk_..." }
        }
      }
    }

Devin Desktop has no project-local MCP file — ``mcp_config.json`` is a single
global file. We only edit it in place when it parses as strict JSON; otherwise
we return the exact snippet to paste, never risking the user's file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SERVER_NAME = "hindsight"


def default_mcp_path() -> Path:
    """The global Devin Desktop MCP config (``~/.codeium/windsurf/mcp_config.json``)."""
    return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"


def mcp_endpoint_url(api_url: str, bank_id: str) -> str:
    """The Hindsight MCP endpoint for a bank (bank is the last path segment)."""
    return f"{api_url.rstrip('/')}/mcp/{bank_id}/"


def build_http_server(api_url: str, api_token: Optional[str], bank_id: str) -> dict[str, Any]:
    """Build the ``mcpServers.hindsight`` entry for ``mcp_config.json``.

    A remote MCP server pointing at the Hindsight endpoint via ``serverUrl``,
    with a Bearer auth header when a token is set (omitted for an open
    self-hosted server).
    """
    server: dict[str, Any] = {"serverUrl": mcp_endpoint_url(api_url, bank_id)}
    if api_token:
        server["headers"] = {"Authorization": f"Bearer {api_token}"}
    return server


def render_snippet(server: dict[str, Any]) -> str:
    """Render the snippet a user can paste into ``mcp_config.json``."""
    return json.dumps({"mcpServers": {SERVER_NAME: server}}, indent=2)


@dataclass
class McpResult:
    """Outcome of editing ``mcp_config.json``.

    ``action`` is one of ``created``, ``merged``, ``unchanged``, ``removed``, or
    ``manual`` (file isn't strict JSON we'll rewrite — ``snippet`` holds what to
    paste).
    """

    action: str
    path: Path
    snippet: Optional[str] = None


def _load_strict(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def apply_to_mcp(path: Path, server: dict[str, Any]) -> McpResult:
    """Add/update ``mcpServers.hindsight`` in ``mcp_config.json`` at ``path``."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mcpServers": {SERVER_NAME: server}}, indent=2) + "\n", encoding="utf-8")
        return McpResult("created", path)

    data = _load_strict(path)
    if data is None:
        return McpResult("manual", path, snippet=render_snippet(server))

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    if servers.get(SERVER_NAME) == server:
        return McpResult("unchanged", path)
    servers[SERVER_NAME] = server
    data["mcpServers"] = servers
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return McpResult("merged", path)


def remove_from_mcp(path: Path) -> McpResult:
    """Remove ``mcpServers.hindsight`` from ``mcp_config.json`` at ``path``."""
    data = _load_strict(path)
    if data is None:
        return McpResult("manual" if path.is_file() else "unchanged", path)

    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return McpResult("unchanged", path)
    del servers[SERVER_NAME]
    if servers:
        data["mcpServers"] = servers
    else:
        data.pop("mcpServers", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return McpResult("removed", path)


def is_installed(path: Path) -> bool:
    """Whether our server is present in ``mcp_config.json`` at ``path``."""
    data = _load_strict(path)
    if data is None:
        return False
    servers = data.get("mcpServers")
    return isinstance(servers, dict) and SERVER_NAME in servers
