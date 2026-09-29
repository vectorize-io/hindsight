"""Canonical handling of user-supplied memory metadata (issue #3209).

Metadata is accepted as arbitrary JSON at ingest (file retain, the MCP tools and
direct engine calls all take ``dict[str, Any]``) but is stored in a JSONB column
and read back through models that declare ``dict[str, str]``. A JSON ``null``
value therefore sailed through the write path and then failed validation on
every read that returned the affected rows.

Both ends normalize through the helpers here so the rule lives in one place:

* ``drop_null_values`` — the write contract. Null-valued keys are dropped;
  every other value is stored as given (the read contract stringifies).
* ``as_string_metadata`` — the read contract. Null-valued keys are dropped and
  the rest are coerced to strings, so rows written before this normalization
  existed stay readable without a data migration.
"""

from collections.abc import Mapping
from typing import Any


def drop_null_values(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return ``metadata`` without its null-valued keys (empty dict for no metadata)."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if v is not None}


def as_string_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Coerce a stored metadata bag to the ``dict[str, str]`` read contract.

    Drops null-valued keys and stringifies the rest (JSONB round-trips integers
    as integers, e.g. ``{"original_id": 348}``).
    """
    return {str(k): str(v) for k, v in drop_null_values(metadata).items()}
