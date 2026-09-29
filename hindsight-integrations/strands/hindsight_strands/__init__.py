"""Hindsight-Strands: Persistent memory tools for AI agents.

Provides Hindsight-backed tool functions for Strands agents,
giving them long-term memory via retain, recall, and reflect tools.

Basic usage::

    from strands import Agent
    from hindsight_strands import create_hindsight_tools

    tools = create_hindsight_tools(
        bank_id="user-123",
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",
    )

    agent = Agent(tools=tools)
    agent("Remember that I prefer dark mode")
"""

from .config import (
    HindsightStrandsConfig,
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
    "HindsightStrandsConfig",
    "HindsightError",
    "create_hindsight_tools",
    "memory_instructions",
]
