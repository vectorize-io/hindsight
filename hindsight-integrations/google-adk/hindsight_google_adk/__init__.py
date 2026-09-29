"""Hindsight memory integration for Google ADK agents.

Provides two complementary patterns:

- **Automatic memory** (``HindsightMemoryService``):
  Implements ADK's ``BaseMemoryService``. Pass an instance to
  ``Runner(memory_service=...)`` and sessions are retained on completion,
  with ``search_memory`` queries served from Hindsight.

- **Explicit tools** (``create_hindsight_tools``):
  ADK ``FunctionTool`` wrappers for retain / recall / reflect, so agents can
  call Hindsight directly from inside a turn.

Usage::

    from hindsight_google_adk import HindsightMemoryService, create_hindsight_tools
"""

from .config import (
    HindsightAdkConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .memory import HindsightMemoryService
from .tools import create_hindsight_tools

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightAdkConfig",
    "HindsightError",
    "HindsightMemoryService",
    "create_hindsight_tools",
]
