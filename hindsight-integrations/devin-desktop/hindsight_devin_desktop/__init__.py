"""Hindsight memory integration for Devin Desktop (formerly Windsurf / Codeium).

Wires the Hindsight MCP server (multi-bank mode) into Devin Desktop's global
``mcp_config.json`` and writes always-on memory rules — a per-project rule naming
this repo's bank plus a global rule for cross-project memory — so Devin has
``recall``/``retain``/``reflect`` tools and uses them automatically, scoped per
project.

CLI::

    cd your-project
    hindsight-devin-desktop init --api-token hsk_...
"""

__version__ = "0.2.0"
