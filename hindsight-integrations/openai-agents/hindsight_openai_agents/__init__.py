"""Hindsight-OpenAI-Agents: Persistent memory tools for OpenAI Agents SDK.

Provides ``FunctionTool`` instances that give OpenAI agents long-term memory
via Hindsight's retain/recall/reflect APIs.

Basic usage::

    from hindsight_client import Hindsight
    from hindsight_openai_agents import create_hindsight_tools

    client = Hindsight(base_url="http://localhost:8888")
    tools = create_hindsight_tools(client=client, bank_id="user-123")

    agent = Agent(name="assistant", tools=tools)
"""

from ._version import __version__
from .config import (
    HindsightOpenAIAgentsConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import create_hindsight_tools, memory_instructions

__all__ = [
    "__version__",
    "configure",
    "get_config",
    "reset_config",
    "HindsightOpenAIAgentsConfig",
    "HindsightError",
    "create_hindsight_tools",
    "memory_instructions",
]
