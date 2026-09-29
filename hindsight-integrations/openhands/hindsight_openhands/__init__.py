"""Hindsight memory integration for OpenHands (formerly OpenDevin).

Wires the Hindsight MCP server into OpenHands' ``config.toml`` and writes a
recall/retain rule into the project's ``AGENTS.md``, so the agent has
``recall``/``retain``/``reflect`` tools and uses them automatically.

CLI::

    hindsight-openhands init --api-token hsk_... --bank-id my-project
"""

__version__ = "0.1.0"
