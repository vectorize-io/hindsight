"""Hindsight-AutoGen: Persistent memory tools for AutoGen agents.

Provides ``FunctionTool`` instances that give AutoGen agents long-term memory
via Hindsight's retain/recall/reflect APIs.

Basic usage::

    from hindsight_client import Hindsight
    from hindsight_autogen import create_hindsight_tools

    client = Hindsight(base_url="http://localhost:8888")
    tools = create_hindsight_tools(client=client, bank_id="user-123")

    # Use with an AutoGen AssistantAgent
    agent = AssistantAgent(name="assistant", model_client=model, tools=tools)
"""

from .config import (
    HindsightAutoGenConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import create_hindsight_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightAutoGenConfig",
    "HindsightError",
    "create_hindsight_tools",
]
