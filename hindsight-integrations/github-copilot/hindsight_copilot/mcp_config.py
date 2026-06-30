"""Wire Hindsight into VS Code's MCP config (``.vscode/mcp.json``).

VS Code's Copilot agent mode reads MCP servers from ``.vscode/mcp.json`` under
the ``servers`` key, and supports HTTP servers directly — so the Hindsight MCP
endpoint connects with no bridge::

    {
      "servers": {
        "hindsight": {
          "type": "http",
          "url": "https://api.hindsight.vectorize.io/mcp/<bank>/",
          "headers": { "Authorization": "Bearer hsk_..." }
        }
      }
    }

``.vscode/mcp.json`` may contain comments (JSONC), which the stdlib JSON parser
can't round-trip. So we only edit in place when the file parses as strict JSON;
otherwise we return the exact snippet to paste, never risking the user's file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SERVER_NAME = "hindsight"


def default_mcp_path() -> Path:
    """The workspace ``.vscode/mcp.json`` (project-shared MCP config)."""
    return Path.cwd() / ".vscode" / "mcp.json"


def mcp_endpoint_url(api_url: str, bank_id: str) -> str:
    """The Hindsight MCP endpoint for a bank (bank is the last path segment)."""
    return f"{api_url.rstrip('/')}/mcp/{bank_id}/"


def build_http_server(api_url: str, api_token: Optional[str], bank_id: str) -> dict[str, Any]:
    """Build the ``servers.hindsight`` entry for ``.vscode/mcp.json``.

    An HTTP MCP server pointing at the Hindsight endpoint, with a Bearer auth
    header when a token is set (omitted for an open self-hosted server).
    """
    server: dict[str, Any] = {"type": "http", "url": mcp_endpoint_url(api_url, bank_id)}
    if api_token:
        server["headers"] = {"Authorization": f"Bearer {api_token}"}
    return server


def render_snippet(server: dict[str, Any]) -> str:
    """Render the snippet a user can paste into ``.vscode/mcp.json``."""
    return json.dumps({"servers": {SERVER_NAME: server}}, indent=2)


@dataclass
class McpResult:
    """Outcome of editing ``.vscode/mcp.json``.

    ``action`` is one of ``created``, ``merged``, ``unchanged``, ``removed``, or
    ``manual`` (file is JSONC we won't rewrite — ``snippet`` holds what to paste).
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
    """Add/update ``servers.hindsight`` in ``.vscode/mcp.json`` at ``path``."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"servers": {SERVER_NAME: server}}, indent=2) + "\n", encoding="utf-8")
        return McpResult("created", path)

    data = _load_strict(path)
    if data is None:
        return McpResult("manual", path, snippet=render_snippet(server))

    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    if servers.get(SERVER_NAME) == server:
        return McpResult("unchanged", path)
    servers[SERVER_NAME] = server
    data["servers"] = servers
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return McpResult("merged", path)


def remove_from_mcp(path: Path) -> McpResult:
    """Remove ``servers.hindsight`` from ``.vscode/mcp.json`` at ``path``."""
    data = _load_strict(path)
    if data is None:
        return McpResult("manual" if path.is_file() else "unchanged", path)

    servers = data.get("servers")
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return McpResult("unchanged", path)
    del servers[SERVER_NAME]
    if servers:
        data["servers"] = servers
    else:
        data.pop("servers", None)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return McpResult("removed", path)


def is_installed(path: Path) -> bool:
    """Whether our server is present in ``.vscode/mcp.json`` at ``path``."""
    data = _load_strict(path)
    if data is None:
        return False
    servers = data.get("servers")
    return isinstance(servers, dict) and SERVER_NAME in servers
