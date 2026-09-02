"""
The retain path must read its chunking parameters off the *resolved* config.

Regression guard for #3584. Both the streaming write path and the delta compare
chunk the same content, and delta detects unchanged chunks by comparing content
hashes at equal chunk_index — so the two must agree on chunk boundaries. They do
only as long as both derive chunk_size/structured_chunk_size from the same
resolved HindsightConfig.

These used to be read as ``getattr(config, "retain_chunk_size", 3000)``. Because
``StaticConfigProxy`` signals "this field is bank-configurable" by raising
``ConfigFieldAccessError`` — an ``AttributeError`` subclass — that default
swallowed the guard and silently substituted the global value for the bank's.
A bank configured at 12000 would then be re-chunked at 3000, no chunk index
could match, and every append fell back to a full re-extraction of the whole
document. Direct attribute access makes a wrong config object fail loudly.
"""

import dataclasses

import pytest

from hindsight_api.config import ConfigFieldAccessError, StaticConfigProxy, _get_raw_config
from hindsight_api.engine.retain.orchestrator import _chunk_contents_for_delta
from hindsight_api.engine.retain.types import RetainContent


def _contents(text: str) -> list[RetainContent]:
    return [RetainContent(content=text)]


def test_delta_chunking_rejects_global_config():
    """Handed global config, the delta chunker raises instead of defaulting to 3000."""
    proxy = StaticConfigProxy(_get_raw_config())

    with pytest.raises(ConfigFieldAccessError, match="retain_chunk_size"):
        _chunk_contents_for_delta(_contents("word. " * 5000), proxy)


def test_delta_chunking_uses_resolved_chunk_size():
    """Boundaries come from the resolved value, not from a hardcoded default."""
    text = "word. " * 5000

    base = _get_raw_config()
    at_12k = _chunk_contents_for_delta(_contents(text), dataclasses.replace(base, retain_chunk_size=12000))
    at_3k = _chunk_contents_for_delta(_contents(text), dataclasses.replace(base, retain_chunk_size=3000))

    assert max(len(c) for c in at_12k.values()) > 3000
    assert len(at_12k) < len(at_3k)
