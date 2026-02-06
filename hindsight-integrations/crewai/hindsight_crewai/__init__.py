"""Hindsight-CrewAI: Persistent memory for AI agent crews.

Provides a Hindsight-backed Storage implementation for CrewAI's
ExternalMemory system, giving your crews long-term memory across runs.

Basic usage::

    from hindsight_crewai import configure, HindsightStorage
    from crewai.memory.external.external_memory import ExternalMemory
    from crewai import Crew

    configure(hindsight_api_url="http://localhost:8888")

    crew = Crew(
        agents=[...],
        tasks=[...],
        external_memory=ExternalMemory(
            storage=HindsightStorage(bank_id="my-crew")
        ),
    )

Per-agent banks::

    storage = HindsightStorage(
        bank_id="crew-shared",
        per_agent_banks=True,
    )
"""

from .config import (
    HindsightCrewAIConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .storage import HindsightStorage
from .tools import HindsightReflectTool

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightCrewAIConfig",
    "HindsightStorage",
    "HindsightReflectTool",
    "HindsightError",
]
