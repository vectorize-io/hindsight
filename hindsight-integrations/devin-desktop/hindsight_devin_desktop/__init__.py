"""Hindsight memory integration for Devin Desktop (formerly Windsurf / Codeium).

Wires the Hindsight MCP server into Devin Desktop's ``~/.codeium/windsurf/mcp_config.json``
and writes an always-on recall/retain rule into ``.devin/rules/hindsight.md``,
so Devin has ``recall``/``retain``/``reflect`` tools and uses them automatically.

CLI::

    hindsight-devin-desktop init --api-token hsk_... --bank-id my-project
"""

__version__ = "0.1.0"
