"""Unit tests for memory unit invalidation (valid_to) — see issue #1391.

Covers:

  * `MemoryEngine.invalidate_memory_unit` issues the correct UPDATE,
    sets `valid_to` to the requested timestamp (or `now()` by default),
    and threads the invalidation `reason` into the row's `metadata` JSONB.
  * Recall SQL builders in the postgres dialect always emit
    `(valid_to IS NULL OR valid_to > now())` so invalidated rows are
    filtered out of semantic / BM25 arms.

Integration coverage that exercises a real Postgres + the new alembic
migration lives in the existing recall integration tests; this file is
unit-test scope so it runs in plain pytest without a live DB.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect as PostgresDialect
from hindsight_api.models import RequestContext


@pytest.mark.asyncio
async def test_invalidate_memory_unit_runs_update_with_default_now():
    """Calling without an explicit valid_to defaults to now()-ish UTC."""
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None

    fake_id = "0c14e4f1-9eb6-4dde-b0a4-c2e8b3a3e5f1"
    fake_row = {
        "id": fake_id,
        "valid_to": datetime(2026, 5, 2, 17, 30, tzinfo=timezone.utc),
        "fact_type": "world",
        "preview": "Server srv-04 runs PostgreSQL 17.",
    }

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=fake_row)

    # acquire_with_retry is an async context manager — wire that explicitly.
    class _CM:
        async def __aenter__(self_inner):
            return mock_conn

        async def __aexit__(self_inner, *exc):
            return False

    mock_pool = MagicMock()
    engine._get_backend = AsyncMock(return_value=mock_pool)

    import hindsight_api.engine.memory_engine as me

    me.acquire_with_retry = lambda _backend: _CM()  # type: ignore[assignment]

    rc = RequestContext(tenant_id="tenant-a", api_key_id="key-a")
    result = await engine.invalidate_memory_unit(
        bank_id="bank-1",
        memory_id=fake_id,
        reason="Server decommissioned",
        request_context=rc,
    )

    assert result is not None
    assert result["id"] == fake_id
    assert result["fact_type"] == "world"
    assert result["preview"].startswith("Server srv-04")

    # Confirm the UPDATE was called with the right shape: the third positional
    # parameter is the timestamp (defaulted to ~now()), the fourth is the reason.
    mock_conn.fetchrow.assert_awaited_once()
    args = mock_conn.fetchrow.await_args.args
    sql = args[0]
    assert "UPDATE" in sql
    assert "SET valid_to = $3::timestamptz" in sql
    assert "invalidation_reason" in sql
    # bank_id, valid_to, reason positions 2, 3, 4
    assert args[1] == fake_id
    assert args[2] == "bank-1"
    assert isinstance(args[3], datetime)
    # Defaulted to "now"-ish — must be UTC and within a few seconds of test start.
    assert args[3].tzinfo is not None
    assert abs((args[3] - datetime.now(timezone.utc)).total_seconds()) < 5
    assert args[4] == "Server decommissioned"


@pytest.mark.asyncio
async def test_invalidate_memory_unit_returns_none_when_not_found():
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None

    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=None)

    class _CM:
        async def __aenter__(self_inner):
            return mock_conn

        async def __aexit__(self_inner, *exc):
            return False

    engine._get_backend = AsyncMock(return_value=MagicMock())

    import hindsight_api.engine.memory_engine as me

    me.acquire_with_retry = lambda _backend: _CM()  # type: ignore[assignment]

    rc = RequestContext(tenant_id="tenant-a", api_key_id="key-a")
    result = await engine.invalidate_memory_unit(
        bank_id="bank-1",
        memory_id="0c14e4f1-9eb6-4dde-b0a4-c2e8b3a3e5f1",
        request_context=rc,
    )
    assert result is None


@pytest.mark.asyncio
async def test_invalidate_memory_unit_rejects_non_uuid():
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None

    rc = RequestContext(tenant_id="tenant-a", api_key_id="key-a")
    with pytest.raises(ValueError, match="not a valid UUID"):
        await engine.invalidate_memory_unit(
            bank_id="bank-1",
            memory_id="this-is-not-a-uuid",
            request_context=rc,
        )


def test_postgres_semantic_arm_filters_invalidated_rows():
    """Recall must skip memories whose valid_to has elapsed."""
    dialect = PostgresDialect()
    sql = dialect.build_semantic_arm(
        table="memory_units",
        cols="id, text",
        fact_type="world",
        embedding_param="$1",
        bank_id_param="$2",
        fetch_limit=20,
        extra_where=" AND (valid_to IS NULL OR valid_to > now())",
    )
    assert "valid_to IS NULL OR valid_to > now()" in sql
    # Sanity: still has the per-fact_type predicate so the partial HNSW index applies.
    assert "fact_type = 'world'" in sql


def test_postgres_bm25_arm_filters_invalidated_rows():
    dialect = PostgresDialect()
    sql = dialect.build_bm25_arm(
        table="memory_units",
        cols="id, text",
        fact_type="experience",
        bank_id_param="$2",
        limit_param="$3",
        text_param="$4",
        text_search_extension="native",
        extra_where=" AND (valid_to IS NULL OR valid_to > now())",
    )
    assert "valid_to IS NULL OR valid_to > now()" in sql
