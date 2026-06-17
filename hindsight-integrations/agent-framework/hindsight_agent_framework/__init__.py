"""Hindsight memory integration for Microsoft Agent Framework.

Provides ``HindsightProvider``, an Agent Framework ``ContextProvider`` that
automatically recalls relevant memories into an agent's context before each run
and retains the conversation afterward — persistent long-term memory with no MCP
and no tools the model must remember to call.

Usage::

    from agent_framework.openai import OpenAIChatClient
    from hindsight_agent_framework import HindsightProvider

    agent = OpenAIChatClient().as_agent(
        name="assistant",
        context_providers=[HindsightProvider(bank_id="user-123")],
    )
"""

from .config import (
    HindsightAgentFrameworkConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .provider import HindsightProvider

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightAgentFrameworkConfig",
    "HindsightError",
    "HindsightProvider",
]
