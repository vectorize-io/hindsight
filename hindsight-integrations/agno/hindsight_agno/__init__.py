"""Hindsight-Agno: Persistent memory tools for AI agents.

Provides a Hindsight-backed Toolkit for Agno agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from agno.agent import Agent
    from agno.models.openai import OpenAIChat
    from hindsight_agno import HindsightTools, memory_instructions

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[HindsightTools(
            bank_id="user-123",
            hindsight_api_url="https://api.hindsight.vectorize.io",
        )],
        instructions=[memory_instructions(
            bank_id="user-123",
            hindsight_api_url="https://api.hindsight.vectorize.io",
        )],
    )

    agent.print_response("What do you remember about my preferences?")
"""

from .config import (
    HindsightAgnoConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import HindsightTools, memory_instructions

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightAgnoConfig",
    "HindsightError",
    "HindsightTools",
    "memory_instructions",
]
