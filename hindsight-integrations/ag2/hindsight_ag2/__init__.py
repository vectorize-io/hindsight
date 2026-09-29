"""Hindsight-AG2: Persistent memory tools for AG2 agents.

Provides Hindsight-backed tool functions that give AG2 agents long-term
memory across conversations via retain/recall/reflect operations.

Basic usage::

    from hindsight_ag2 import register_hindsight_tools

    register_hindsight_tools(
        assistant, user_proxy,
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )

Manual registration::

    from hindsight_ag2 import create_hindsight_tools

    tools = create_hindsight_tools(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )
    for tool_fn in tools:
        assistant.register_for_llm(description=tool_fn.__doc__)(tool_fn)
        user_proxy.register_for_execution()(tool_fn)
"""

from .config import (
    HindsightAG2Config,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import create_hindsight_tools, register_hindsight_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightAG2Config",
    "HindsightError",
    "create_hindsight_tools",
    "register_hindsight_tools",
]
