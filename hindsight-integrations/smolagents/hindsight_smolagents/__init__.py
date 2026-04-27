"""Hindsight-SmolAgents: Persistent memory tools for AI agents.

Provides Hindsight-backed Tool subclasses for SmolAgents agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from smolagents import CodeAgent, HfApiModel
    from hindsight_smolagents import create_hindsight_tools

    tools = create_hindsight_tools(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )

    agent = CodeAgent(
        tools=tools,
        model=HfApiModel(),
    )

    agent.run("Remember that I prefer dark mode")
"""

from .config import (
    HindsightSmolAgentsConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import (
    HindsightRecallTool,
    HindsightReflectTool,
    HindsightRetainTool,
    create_hindsight_tools,
    memory_instructions,
)

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightSmolAgentsConfig",
    "HindsightError",
    "HindsightRetainTool",
    "HindsightRecallTool",
    "HindsightReflectTool",
    "create_hindsight_tools",
    "memory_instructions",
]
