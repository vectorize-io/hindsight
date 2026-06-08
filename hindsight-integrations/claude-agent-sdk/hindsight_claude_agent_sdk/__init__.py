"""Hindsight-Claude-Agent-SDK: Persistent memory for Claude agents.

Provides an in-process MCP server with Hindsight memory tools and
automatic memory hooks for the Claude Agent SDK.

Basic usage with tools::

    from claude_agent_sdk import query, ClaudeAgentOptions
    from hindsight_claude_agent_sdk import create_hindsight_server

    server = create_hindsight_server(
        bank_id="my-agent",
        hindsight_api_url="http://localhost:8888",
    )

    async for msg in query(
        prompt="What do you remember about my preferences?",
        options=ClaudeAgentOptions(
            mcp_servers={"hindsight": server},
            allowed_tools=["mcp__hindsight__*"],
        ),
    ):
        print(msg)

With automatic memory hooks::

    from hindsight_claude_agent_sdk import create_hindsight_server, create_memory_hooks

    server = create_hindsight_server(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
    hooks = create_memory_hooks(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

    async for msg in query(
        prompt="Help me refactor the auth module.",
        options=ClaudeAgentOptions(
            mcp_servers={"hindsight": server},
            allowed_tools=["mcp__hindsight__*"],
            hooks=hooks,
        ),
    ):
        print(msg)
"""

from ._version import __version__
from .config import (
    HindsightClaudeAgentSDKConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .hooks import MemoryHookConfig, create_memory_hooks
from .tools import create_hindsight_server, create_hindsight_tools

__all__ = [
    "__version__",
    "configure",
    "get_config",
    "reset_config",
    "HindsightClaudeAgentSDKConfig",
    "HindsightError",
    "create_hindsight_server",
    "create_hindsight_tools",
    "create_memory_hooks",
    "MemoryHookConfig",
]
