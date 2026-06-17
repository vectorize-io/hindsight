"""Hindsight memory integration for Haystack agents.

Provides Haystack-compatible ``Tool`` instances backed by Hindsight's
retain/recall/reflect APIs. Use ``create_hindsight_tools()`` to create
tools for any Haystack ``Agent``.

Usage::

    from hindsight_haystack import create_hindsight_tools
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    tools = create_hindsight_tools(bank_id="user-123", client=client)
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=tools)
"""

from .config import (
    HindsightHaystackConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import HindsightMemoryWrapper, create_hindsight_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightHaystackConfig",
    "HindsightError",
    "create_hindsight_tools",
    "HindsightMemoryWrapper",
]
