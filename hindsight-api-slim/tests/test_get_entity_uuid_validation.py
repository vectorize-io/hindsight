"""Regression tests for UUID validation on get_entity and get_observation_history.

Mirrors the pattern in test_delete_memory_units_validation.py: a stub engine
that bypasses __init__ so the pure-Python validation branches are exercised
without a DB connection.

get_memory_unit and update_memory_unit already validate their memory_id
against uuid.UUID (PR #3062, commit in #906). get_entity and
get_observation_history were missed - a malformed id raised a bare
ValueError from the stdlib that the HTTP handler mapped to 500 instead
of 400.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine


def _stub_engine() -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    # 让 _get_backend 返回一个 mock conn，但它永远不会被到达
    # 因为 uuid.UUID(bad) 会先 raise
    engine._get_backend = AsyncMock(return_value=MagicMock())
    return engine


BAD_UUID = "not-a-uuid"


@pytest.mark.asyncio
async def test_get_entity_rejects_malformed_uuid():
    """get_entity 应对畸形 UUID raise ValueError（而非让 uuid.UUID 裸抛）。"""
    engine = _stub_engine()

    with pytest.raises(ValueError, match="Invalid entity_id"):
        await engine.get_entity(
            bank_id="test-bank",
            entity_id=BAD_UUID,
            request_context=RequestContext(api_key="anything"),
        )

    # 不应该到达 DB 层
    engine._get_backend.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_observation_history_rejects_malformed_uuid():
    """get_observation_history 应对畸形 UUID raise ValueError。"""
    engine = _stub_engine()

    with pytest.raises(ValueError, match="Invalid memory_id"):
        await engine.get_observation_history(
            bank_id="test-bank",
            memory_id=BAD_UUID,
            request_context=RequestContext(api_key="anything"),
        )

    engine._get_backend.assert_not_awaited()
