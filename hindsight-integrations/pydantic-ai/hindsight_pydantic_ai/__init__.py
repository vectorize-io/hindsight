"""Hindsight-Pydantic AI: Persistent memory tools for AI agents.

Provides Hindsight-backed tools and instructions for Pydantic AI agents,
giving them long-term memory across runs.

Basic usage::

    from hindsight_client import Hindsight
    from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions
    from pydantic_ai import Agent

    client = Hindsight(base_url="https://api.hindsight.vectorize.io", api_key="hsk_...")

    agent = Agent(
        "openai:gpt-4o",
        tools=create_hindsight_tools(client=client, bank_id="user-123"),
        instructions=[memory_instructions(client=client, bank_id="user-123")],
    )

    result = await agent.run("What do you remember about my preferences?")
"""

from .config import (
    HindsightPydanticAIConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import create_hindsight_tools, memory_instructions

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightPydanticAIConfig",
    "HindsightError",
    "create_hindsight_tools",
    "memory_instructions",
]
