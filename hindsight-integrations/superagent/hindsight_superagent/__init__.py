"""Hindsight-Superagent: Safety middleware for AI agent memory.

Wraps Hindsight memory operations with Superagent Guard (prompt injection
detection) and Redact (PII removal) for secure memory storage and retrieval.

Basic usage::

    from hindsight_superagent import SafeHindsight

    safe = SafeHindsight(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
        redact_model="openai/gpt-4o-mini",
    )

    # Content is guarded and redacted before storage
    await safe.retain("John's email is john@acme.com")

    # Queries are guarded before recall
    results = await safe.recall("What do I know about John?")
"""

from .config import (
    HindsightSuperagentConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import GuardBlockedError, HindsightError
from .middleware import SafeHindsight

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("hindsight-superagent")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightSuperagentConfig",
    "HindsightError",
    "GuardBlockedError",
    "SafeHindsight",
]
