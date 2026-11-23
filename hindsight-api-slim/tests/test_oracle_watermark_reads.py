from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.db.oracle import _convert_row_from_oracle, _rewrite_pg_to_oracle
from hindsight_api.engine.memories.pg.reads import (
    latest_memory_write_at,
    memories_updated_between,
    newest_memory_updated_at,
)


def _make_conn(backend_type: str, result: datetime | None) -> MagicMock:
    conn = MagicMock()
    conn.backend_type = backend_type
    conn.fetchval = AsyncMock(return_value=result)
    return conn


@pytest.mark.asyncio
async def test_newest_memory_updated_at_keeps_postgres_max_query_unchanged() -> None:
    conn = _make_conn("postgresql", None)
    until = datetime(2026, 11, 13, tzinfo=timezone.utc)
    since = datetime(2026, 11, 12, tzinfo=timezone.utc)

    result = await newest_memory_updated_at(
        conn=conn,
        fq_table=lambda table: f'"S1".{table}',
        bank_id="bank-1",
        until=until,
        since=since,
    )

    assert result is None
    conn.fetchval.assert_awaited_once_with(
        'SELECT MAX(updated_at) FROM "S1".memory_units WHERE bank_id = $1 AND updated_at <= $2 AND updated_at > $3',
        "bank-1",
        until,
        since,
    )


@pytest.mark.asyncio
async def test_newest_memory_updated_at_uses_utc_index_ordering_on_oracle() -> None:
    expected = datetime(2026, 11, 13, 12, 30, tzinfo=timezone.utc)
    conn = _make_conn("oracle", expected)

    result = await newest_memory_updated_at(
        conn=conn,
        fq_table=lambda table: f'"S1".{table}',
        bank_id="bank-1",
        until=datetime(2026, 11, 14, tzinfo=timezone.utc),
        since=datetime(2026, 11, 12, tzinfo=timezone.utc),
    )

    assert result == expected
    query = conn.fetchval.await_args.args[0]
    assert "SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ updated_at FROM" in query
    assert "SYS_EXTRACT_UTC(updated_at) <= SYS_EXTRACT_UTC(CAST($2 AS TIMESTAMP WITH TIME ZONE))" in query
    assert "SYS_EXTRACT_UTC(updated_at) > SYS_EXTRACT_UTC(CAST($3 AS TIMESTAMP WITH TIME ZONE))" in query
    assert "ORDER BY SYS_EXTRACT_UTC(updated_at) DESC FETCH FIRST 1 ROW ONLY" in query
    rewritten, _, _ = _rewrite_pg_to_oracle(query)
    assert "FETCH FIRST 1 ROW ONLY" in rewritten
    assert "LIMIT" not in rewritten


@pytest.mark.parametrize(
    ("until", "since"),
    [
        (datetime(2026, 11, 14), datetime(2026, 11, 12)),
        (
            datetime(2026, 11, 14, tzinfo=timezone.utc),
            datetime(2026, 11, 12, tzinfo=timezone(timedelta(hours=2))),
        ),
        (
            datetime(2026, 11, 14, tzinfo=timezone(timedelta(hours=2))),
            datetime(2026, 11, 12),
        ),
        (
            datetime(2026, 11, 14, tzinfo=timezone(timedelta(hours=2))),
            datetime(2026, 11, 12, tzinfo=timezone(timedelta(hours=2))),
        ),
    ],
)
@pytest.mark.asyncio
async def test_newest_memory_updated_at_casts_oracle_bounds_to_timezone_aware(until: datetime, since: datetime) -> None:
    conn = _make_conn("oracle", None)

    await newest_memory_updated_at(
        conn=conn,
        fq_table=lambda table: table,
        bank_id="bank-1",
        until=until,
        since=since,
    )

    query, bank_id, actual_until, actual_since = conn.fetchval.await_args.args
    assert bank_id == "bank-1"
    assert "SYS_EXTRACT_UTC(CAST($2 AS TIMESTAMP WITH TIME ZONE))" in query
    assert "SYS_EXTRACT_UTC(CAST($3 AS TIMESTAMP WITH TIME ZONE))" in query
    expected_until = until if until.tzinfo is not None else until.replace(tzinfo=timezone.utc)
    expected_since = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
    for actual, expected in ((actual_until, expected_until), (actual_since, expected_since)):
        assert actual.tzinfo is not None
        assert actual == expected
        assert actual.utcoffset() == expected.utcoffset()


@pytest.mark.asyncio
async def test_newest_memory_updated_at_returns_none_for_empty_oracle_scope() -> None:
    conn = _make_conn("oracle", None)

    result = await newest_memory_updated_at(
        conn=conn,
        fq_table=lambda table: table,
        bank_id="bank-1",
        until=datetime(2026, 11, 14, tzinfo=timezone.utc),
    )

    assert result is None


@pytest.mark.asyncio
async def test_latest_memory_write_at_keeps_postgres_max_query_unchanged() -> None:
    expected = datetime(2026, 11, 13, 12, 30, tzinfo=timezone.utc)
    conn = _make_conn("postgresql", expected)

    result = await latest_memory_write_at(
        conn=conn,
        fq_table=lambda table: f'"S1".{table}',
        bank_id="bank-1",
    )

    assert result == expected
    conn.fetchval.assert_awaited_once_with(
        'SELECT MAX(updated_at) FROM "S1".memory_units WHERE bank_id = $1',
        "bank-1",
    )


@pytest.mark.asyncio
async def test_latest_memory_write_at_uses_utc_index_ordering_on_oracle() -> None:
    expected = datetime(2026, 11, 13, 12, 30, tzinfo=timezone.utc)
    conn = _make_conn("oracle", expected)

    result = await latest_memory_write_at(
        conn=conn,
        fq_table=lambda table: f'"S1".{table}',
        bank_id="bank-1",
    )

    assert result == expected
    query = conn.fetchval.await_args.args[0]
    assert "SELECT /*+ INDEX(m idx_mu_bank_updated_at) */ updated_at FROM" in query
    assert "ORDER BY SYS_EXTRACT_UTC(updated_at) DESC FETCH FIRST 1 ROW ONLY" in query
    rewritten, _, _ = _rewrite_pg_to_oracle(query)
    assert "FETCH FIRST 1 ROW ONLY" in rewritten
    assert "LIMIT" not in rewritten


@pytest.mark.asyncio
async def test_latest_memory_write_at_returns_none_for_empty_oracle_bank() -> None:
    conn = _make_conn("oracle", None)

    result = await latest_memory_write_at(
        conn=conn,
        fq_table=lambda table: table,
        bank_id="bank-1",
    )

    assert result is None


def test_oracle_row_conversion_preserves_timestamptz_datetime_exactly() -> None:
    value = datetime(2026, 11, 13, 12, 30, 45, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    assert _convert_row_from_oracle(["UPDATED_AT"], (value,))[0] is value


@pytest.mark.asyncio
async def test_memories_updated_between_shares_oracle_utc_bounds_and_order() -> None:
    conn = _make_conn("oracle", None)
    conn.fetch = AsyncMock(return_value=[])
    naive_until = datetime(2026, 11, 14)
    naive_since = datetime(2026, 11, 12)

    result = await memories_updated_between(
        conn=conn,
        fq_table=lambda table: table,
        bank_id="bank-1",
        until=naive_until,
        since=naive_since,
        limit=50,
    )

    assert result == []
    query, bank_id, actual_until, actual_since, limit = conn.fetch.await_args.args
    assert bank_id == "bank-1"
    assert limit == 50
    assert actual_until == datetime(2026, 11, 14, tzinfo=timezone.utc)
    assert actual_since == datetime(2026, 11, 12, tzinfo=timezone.utc)
    assert "SYS_EXTRACT_UTC(updated_at) <= SYS_EXTRACT_UTC(CAST($2 AS TIMESTAMP WITH TIME ZONE))" in query
    assert "SYS_EXTRACT_UTC(updated_at) > SYS_EXTRACT_UTC(CAST($3 AS TIMESTAMP WITH TIME ZONE))" in query
    assert "ORDER BY SYS_EXTRACT_UTC(updated_at), id LIMIT $4" in query
