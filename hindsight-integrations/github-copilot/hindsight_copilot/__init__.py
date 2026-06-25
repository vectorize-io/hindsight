"""Hindsight memory integration for GitHub Copilot (VS Code).

Wires the Hindsight MCP server into VS Code's ``.vscode/mcp.json`` and writes a
recall/retain rule into ``.github/copilot-instructions.md``, so Copilot's agent
mode has ``recall``/``retain``/``reflect`` tools and uses them automatically.

CLI::

    hindsight-copilot init --api-token hsk_... --bank-id my-project
"""

__version__ = "0.1.0"
