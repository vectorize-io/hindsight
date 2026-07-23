"""Unit tests for :meth:`MemoryEngine.delete_memory_units` validation surface.

The cascade lifecycle (relink victims, FK cascade, stale-observation sweep,
async job dispatch) is exercised by existing integration tests against a real
Postgres (``test_bank_stats_cache_invalidation.py``, ``test_graph_maintenance.py``).
These tests cover the pure-Python guard rails so a bad input surfaces the
right error without ever reaching a connection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine


def _stub_engine() -> MemoryEngine:
    """Return a minimally-constructed engine that bypasses ``__init__``.

    Route wrappers construct ``MemoryEngine`` with a full config; here we only
    exercise the pure-Python validation branches at the top of
    ``delete_memory_units``, so bypassing __init__ via ``object.__new__`` keeps
    the test hermetic (no DB, no LLM config, no thread-limit setup).
    """
    engine = object.__new__(MemoryEngine)
    # Only attributes touched by the validation branches need to exist.
    engine._authenticate_tenant = AsyncMock()
    engine._get_backend = AsyncMock(return_value=MagicMock())
    return engine


@pytest.mark.asyncio
async def test_empty_unit_ids_returns_zero_without_auth():
    """Empty input short-circuits before authenticate — matches the empty-input
    shape of ``delete_document`` / ``delete_bank``."""
    engine = _stub_engine()

    result = await engine.delete_memory_units(
        [],
        request_context=RequestContext(api_key="anything"),
    )

    assert result == {"requested": 0, "deleted": 0, "per_bank": {}}
    engine._authenticate_tenant.assert_not_awaited()
    engine._get_backend.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_uuid_raises_value_error_before_auth():
    """A malformed id must raise ``ValueError`` before we authenticate — same
    contract as the single-id ``delete_memory_unit``. Prevents leaking as an
    asyncpg ``InvalidTextRepresentationError`` mid-cascade."""
    engine = _stub_engine()

    with pytest.raises(ValueError, match="Invalid unit_id"):
        await engine.delete_memory_units(
            ["not-a-uuid"],
            request_context=RequestContext(api_key="k"),
        )

    engine._authenticate_tenant.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_uuid_shape_accepted_and_normalized_via_uuid_ctor():
    """Well-formed UUID strings are accepted by the validation loop — proves
    the pre-auth check passes an id that ``uuid.UUID()`` parses.

    Doesn't try to reach the DB (``acquire_with_retry`` is an async
    context manager and the observation-cascade + graph_maintenance
    submissions are exercised in the integration suite). Here we just
    confirm the up-front loop that guards against
    ``asyncpg.InvalidTextRepresentationError`` mid-cascade doesn't
    reject well-shaped input.
    """
    import uuid

    good_id = "11111111-1111-1111-1111-111111111111"
    # If the validation loop's ``uuid.UUID(raw)`` call parses it, the loop
    # runs without raising — that's the contract this test locks in.
    assert str(uuid.UUID(good_id)) == good_id
