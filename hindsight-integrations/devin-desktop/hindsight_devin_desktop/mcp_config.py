"""Wire Hindsight into Devin Desktop's MCP config (``~/.codeium/.../mcp_config.json``).

Devin Desktop (formerly Windsurf) reads MCP servers from a global
``mcp_config.json`` under the ``mcpServers`` key. The on-disk location still
carries the legacy ``.codeium`` root (unchanged by the rebrand), but Devin's own
docs disagree on the segment after it — the Cascade page says
``~/.codeium/windsurf/mcp_config.json`` while the FAQ/plugins pages say
``~/.codeium/mcp_config.json``. To be safe we write **both** (see
:func:`default_mcp_paths`).

We register Hindsight in **multi-bank mode** — a single remote ``serverUrl``
ending in ``/mcp/`` (no bank pinned in the path). In this mode every tool takes
an optional ``bank_id``, so one connection can address the user's global bank
*and* the per-project bank; the always-on rule tells the model which to use. An
``X-Bank-Id`` header names the global bank as the default when the model omits
``bank_id``::

    {
      "mcpServers": {
        "hindsight": {
          "serverUrl": "https://api.hindsight.vectorize.io/mcp/",
          "headers": {
            "Authorization": "Bearer hsk_...",
            "X-Bank-Id": "devin-desktop"
          }
        }
      }
    }

We only edit a file in place when it parses as strict JSON; otherwise we return
the exact snippet to paste, never risking the user's file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

SERVER_NAME = "hindsight"


def default_mcp_paths() -> List[Path]:
    """Both documented Devin Desktop MCP config locations (write to each).

    Devin's docs conflict on the path, so covering both guarantees the installed
    build reads our entry regardless of which location it honors.
    """
    codeium = Path.home() / ".codeium"
    return [codeium / "windsurf" / "mcp_config.json", codeium / "mcp_config.json"]


def default_mcp_path() -> Path:
    """The primary Devin Desktop MCP config (``~/.codeium/windsurf/mcp_config.json``)."""
    return default_mcp_paths()[0]


def mcp_endpoint_url(api_url: str) -> str:
    """The Hindsight multi-bank MCP endpoint (no bank pinned in the path)."""
    return f"{api_url.rstrip('/')}/mcp/"


def build_http_server(api_url: str, api_token: Optional[str], default_bank: Optional[str]) -> dict[str, Any]:
    """Build the multi-bank ``mcpServers.hindsight`` entry for ``mcp_config.json``.

    A remote MCP server at the ``/mcp/`` endpoint, with a Bearer auth header when
    a token is set and an ``X-Bank-Id`` header naming ``default_bank`` as the
    fallback bank for any call that omits ``bank_id``.
    """
    server: dict[str, Any] = {"serverUrl": mcp_endpoint_url(api_url)}
    headers: dict[str, str] = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    if default_bank:
        headers["X-Bank-Id"] = default_bank
    if headers:
        server["headers"] = headers
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
