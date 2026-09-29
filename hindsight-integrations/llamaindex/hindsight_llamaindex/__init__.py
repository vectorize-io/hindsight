"""Hindsight memory integration for LlamaIndex agents.

Provides two complementary patterns:

- **Tools** (``HindsightToolSpec``, ``create_hindsight_tools``):
  Agent-driven memory via LlamaIndex's ``BaseToolSpec``.
  The agent decides when to retain/recall/reflect.

- **Memory** (``HindsightMemory``):
  Automatic memory via LlamaIndex's ``BaseMemory`` interface.
  Messages are stored on ``put()`` and recalled on ``get()``.

Usage::

    from hindsight_llamaindex import HindsightToolSpec, create_hindsight_tools
    from hindsight_llamaindex import HindsightMemory
"""

from .config import (
    HindsightLlamaIndexConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .memory import HindsightMemory
from .tools import HindsightToolSpec, create_hindsight_tools

__version__ = "0.1.2"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightLlamaIndexConfig",
    "HindsightError",
    "HindsightToolSpec",
    "create_hindsight_tools",
    "HindsightMemory",
]
