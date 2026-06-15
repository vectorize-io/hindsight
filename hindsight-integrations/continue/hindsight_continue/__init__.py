"""Hindsight-Continue: persistent memory for Continue.dev.

Exposes Hindsight memory to Continue through its native ``http`` context
provider. Run the adapter server and point Continue at it; typing
``@hindsight <query>`` (or a bare ``@hindsight``) recalls relevant long-term
memory and injects it into the model's context at query time.

Basic usage::

    from hindsight_continue import configure, run

    configure(bank_id="my-project", api_key="hsk_...")
    run()  # serves the context-provider endpoint on 127.0.0.1:8123

See the README for the matching Continue ``config.yaml`` snippet, plus the
optional MCP-server + rules setup for automatic recall/retain in agent mode.
"""

from .config import (
    HindsightContinueConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightError
from .provider import ContextItem, build_context_items, serialize
from .server import build_server, make_handler, run

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightContinueConfig",
    "HindsightError",
    "ContextItem",
    "build_context_items",
    "serialize",
    "build_server",
    "make_handler",
    "run",
]
