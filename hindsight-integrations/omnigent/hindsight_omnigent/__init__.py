"""Hindsight-Omnigent: persistent memory tools for Omnigent agents.

Exposes Hindsight's retain, recall, and reflect operations as Omnigent
``type: function`` tools — plain Python callables referenced by dotted path from
an agent YAML. The Hindsight bank and connection come from :func:`configure` (or
``HINDSIGHT_*`` env vars), since Omnigent passes tool callables no session
context.

Basic usage (in the module that hosts your agent's tools)::

    from hindsight_omnigent import configure, tools_yaml

    configure(
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",   # or HINDSIGHT_API_KEY
        bank_id="user-123",  # or HINDSIGHT_BANK_ID
    )

    print(tools_yaml())  # paste into your agent.yaml under `tools:`
"""

from .config import (
    HindsightOmnigentConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .tools import (
    OmnigentToolSpec,
    memory_instructions,
    recall,
    reflect,
    retain,
    tool_specs,
    tools_yaml,
)

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightOmnigentConfig",
    "HindsightError",
    "retain",
    "recall",
    "reflect",
    "memory_instructions",
    "tool_specs",
    "tools_yaml",
    "OmnigentToolSpec",
]
