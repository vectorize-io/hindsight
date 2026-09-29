"""Hindsight-Composio: Persistent memory custom tools for AI agents.

Exposes Hindsight's retain, recall, and reflect operations as Composio
in-process custom tools. The Hindsight bank for each call is the Composio
session's ``user_id``, so one registered tool set isolates memory per user.

Basic usage::

    from composio import Composio
    from hindsight_composio import register_hindsight_tools

    composio = Composio()
    tools = register_hindsight_tools(
        composio,
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",
    )

    session = composio.create(
        user_id="user-123",  # becomes the Hindsight bank_id
        experimental={"custom_tools": tools},
    )
"""

from .config import (
    HindsightComposioConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import (
    RecallInput,
    ReflectInput,
    RetainInput,
    memory_instructions,
    register_hindsight_tools,
)

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightComposioConfig",
    "HindsightError",
    "register_hindsight_tools",
    "memory_instructions",
    "RetainInput",
    "RecallInput",
    "ReflectInput",
]
