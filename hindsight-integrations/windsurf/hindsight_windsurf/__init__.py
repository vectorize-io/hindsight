"""Hindsight memory integration for Windsurf (Codeium).

Wires the Hindsight MCP server into Windsurf's ``~/.codeium/windsurf/mcp_config.json``
and writes an always-on recall/retain rule into ``.windsurf/rules/hindsight.md``,
so Cascade has ``recall``/``retain``/``reflect`` tools and uses them automatically.

CLI::

    hindsight-windsurf init --api-token hsk_... --bank-id my-project
"""

__version__ = "0.1.0"
