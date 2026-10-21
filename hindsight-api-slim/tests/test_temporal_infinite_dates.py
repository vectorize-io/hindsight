"""Temporal recall over a memory whose stored date is Postgres `infinity`.

asyncpg returns a timestamptz `infinity`/`-infinity` as a *naive* datetime.max/min, while
every other timestamptz comes back aware. A memory with one finite and one infinite bound
therefore reached the midpoint arithmetic with mixed awareness and raised TypeError (#5820),
failing the whole recall — and the consolidation that runs it.

Pure mechanics (no LLM), so these assert directly.
"""

import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.consolidation.consolidator import _aggregate_source_fields, _TemporalBounds
from hindsight_api.engine.memories.pg.recall import retrieve_temporal_combined_sql
from hindsight_api.engine.retain.fact_extraction import _parse_datetime
from hindsight_api.engine.schema import fq_store_table_explicit as fq_table

EMBED_DIM = 384
_QUERY = "[" + ",".join(["1.0"] + ["0.0"] * (EMBED_DIM - 1)) + "]"


async def _insert_unit(
    conn, bank_id: str, start_sql: str, end_sql: str, event_sql: str = "2025-01-16T00:00:00Z"
) -> uuid.UUID:
    row = await conn.fetchrow(
        f"""
        INSERT INTO {fq_table("memory_units")} (bank_id, text, fact_type, embedding, event_date,
                                                occurred_start, occurred_end)
        VALUES ($1, 'unit', 'world', $2::vector, '{event_sql}'::timestamptz,
                '{start_sql}'::timestamptz, '{end_sql}'::timestamptz)
        RETURNING id
        """,
        bank_id,
        _QUERY,
    )
    return row["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_sql", "end_sql"),
    [("2025-01-16T00:00:00Z", "infinity"), ("-infinity", "2025-01-16T00:00:00Z")],
    ids=["open-end", "open-start"],
)
async def test_infinite_bound_does_not_break_temporal_recall(memory, start_sql, end_sql):
    bank_id = f"test_temporal_infinite_{uuid.uuid4().hex[:8]}"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        entry = await _insert_unit(conn, bank_id, "2025-01-16T00:00:00Z", "2025-01-16T00:00:00Z")
        # Reached both as an entry point (its finite bound is in the window) and as a neighbour.
        odd = await _insert_unit(conn, bank_id, start_sql, end_sql)
        await conn.execute(
            f"""INSERT INTO {fq_table("memory_links")} (from_unit_id, to_unit_id, link_type, weight, bank_id)
                VALUES ($1, $2, 'temporal', 0.9, $3)""",
            entry,
            odd,
            bank_id,
        )
        try:
            results = await retrieve_temporal_combined_sql(conn, _QUERY, bank_id, ["world"], start, end, budget=10)
        finally:
            await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    ids = {r.id for r in results["world"]}
    assert str(odd) in ids


@pytest.mark.parametrize(
    "placeholder", ["0001-01-01T00:00:00", "9999-12-31T23:59:59.999999", "9999-12-31T23:59:59Z", "9999-12-31"]
)
def test_extraction_drops_placeholder_years(placeholder):
    """An open-ended range written by the model as year 1 / 9999 is an unknown bound, not a
    date: kept, it is stored as infinity (or overflows the per-fact ordering offset)."""
    assert _parse_datetime(placeholder) is None


def test_extraction_keeps_real_dates():
    assert _parse_datetime("2015-12-31T23:59:59") == datetime(2015, 12, 31, 23, 59, 59)


@pytest.mark.asyncio
async def test_consolidation_merges_a_stored_placeholder_date_as_unknown(memory):
    """Rows stored before #5820 keep their infinity dates (no migration rewrites them), so
    consolidation's date merging must read them as unknown rather than compare naive vs aware."""
    bank_id = f"test_placeholder_merge_{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    # Raw SQL on both sides: the API refuses a placeholder date, so only a direct INSERT can
    # forge a pre-#5820 row, and only a direct read hands back the naive datetime asyncpg
    # produces for infinity — the exact value consolidation receives.
    async with pool.acquire() as conn:
        await _insert_unit(conn, bank_id, "-infinity", "infinity")
        await _insert_unit(conn, bank_id, "2025-01-16T00:00:00Z", "2025-01-17T00:00:00Z")
        try:
            rows = await conn.fetch(
                f"SELECT occurred_start, occurred_end FROM {fq_table('memory_units')} "
                "WHERE bank_id = $1 ORDER BY occurred_start",  # -infinity row first
                bank_id,
            )
        finally:
            await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    source_mems = [dict(r) for r in rows]
    agg = _aggregate_source_fields(source_mems, tags=[])
    assert agg.occurred_start == datetime(2025, 1, 16, tzinfo=UTC)
    assert agg.occurred_end == datetime(2025, 1, 17, tzinfo=UTC)

    bounds = _TemporalBounds(occurred_start=rows[0]["occurred_start"], occurred_end=rows[0]["occurred_end"])
    merged = bounds.merged_with(_TemporalBounds(occurred_start=rows[1]["occurred_start"]))
    assert merged.occurred_start == datetime(2025, 1, 16, tzinfo=UTC)
    assert merged.occurred_end is None


@pytest.mark.asyncio
async def test_migration_clears_placeholder_event_dates_only(memory, monkeypatch):
    """c7e2f9a4d1b8 nulls a placeholder event_date (PG <= 16 cannot subtract it in retain's
    temporal-link SQL) and leaves real dates and the other columns, which readers ignore, alone."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).parent.parent / "hindsight_api/alembic/versions/c7e2f9a4d1b8_clear_placeholder_memory_dates.py"
    )
    spec = importlib.util.spec_from_file_location("clear_placeholder_dates", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statements: list[str] = []
    monkeypatch.setattr(migration.op, "execute", statements.append)
    monkeypatch.setattr(migration, "_pg_schema_prefix", lambda: fq_table("memory_units").removesuffix("memory_units"))
    migration._pg_upgrade()

    bank_id = f"test_placeholder_migration_{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        infinite = await _insert_unit(conn, bank_id, "2025-01-16T00:00:00Z", "infinity", event_sql="infinity")
        minus_infinite = await _insert_unit(conn, bank_id, "-infinity", "2025-01-16T00:00:00Z", event_sql="-infinity")
        year_9999 = await _insert_unit(conn, bank_id, "2025-01-16T00:00:00Z", "2025-01-17T00:00:00Z", "9999-12-31")
        real = await _insert_unit(conn, bank_id, "2025-01-16T00:00:00Z", "2025-01-17T00:00:00Z")
        try:
            for sql in statements:
                await conn.execute(sql)
            # Read back with SQL: the public reads normalize these dates, so they could not show
            # whether the stored value itself was cleared.
            rows = {
                r["id"]: r
                for r in await conn.fetch(
                    f"SELECT id, event_date, occurred_end FROM {fq_table('memory_units')} WHERE bank_id = $1",
                    bank_id,
                )
            }
        finally:
            await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    assert rows[infinite]["event_date"] is None
    assert rows[minus_infinite]["event_date"] is None
    assert rows[year_9999]["event_date"] is None
    assert rows[real]["event_date"] == datetime(2025, 1, 16, tzinfo=UTC)
    # Not cleared: no index serves occurred_end, and every reader treats it as unknown.
    assert rows[infinite]["occurred_end"] == datetime.max


@pytest.mark.parametrize("value", ["9999-12-31T23:59:59.999999Z", "0001-01-01T00:00:00"])
def test_retain_and_mcp_reject_placeholder_timestamps(value):
    """A caller that sends year 1 / 9999 gets told so, instead of a memory that breaks recall."""
    from pydantic import ValidationError

    from hindsight_api.api.http import FileRetainMetadata, MemoryItem
    from hindsight_api.engine.retain.orchestrator import parse_datetime_flexible
    from hindsight_api.mcp_tools import parse_timestamp

    with pytest.raises(ValidationError, match="placeholder"):
        MemoryItem(content="x", timestamp=value)
    with pytest.raises(ValueError, match="placeholder"):
        parse_timestamp(value)
    with pytest.raises(ValidationError, match="placeholder"):
        FileRetainMetadata(timestamp=value)
    # Every retain path — JSON, files, MCP, a direct engine call — parses its date here.
    with pytest.raises(ValueError, match="placeholder"):
        parse_datetime_flexible(value)
    assert MemoryItem(content="x", timestamp="2025-01-16T00:00:00Z").timestamp == datetime(2025, 1, 16, tzinfo=UTC)


@pytest.mark.asyncio
async def test_memory_edit_rejects_placeholder_date(memory, request_context):
    bank_id = f"test-placeholder-edit-{uuid.uuid4().hex[:8]}"
    await memory.submit_async_retain(
        bank_id, [{"content": "Alice started physio on Monday."}], request_context=request_context
    )
    page = await memory.list_memory_units(bank_id, request_context=request_context, fact_type=["world", "experience"])
    fact_id = page["items"][0]["id"]

    with pytest.raises(ValueError, match="placeholder"):
        await memory.update_memory_unit(
            bank_id, fact_id, occurred_end="9999-12-31T23:59:59", request_context=request_context
        )


@pytest.mark.asyncio
async def test_reprocess_replays_a_stored_placeholder_date_as_undated(memory, request_context):
    """A document retained before the fix kept its year-9999 timestamp in retain_params;
    retain now refuses it, so reprocess must replay it as undated instead of failing."""
    from unittest.mock import AsyncMock, patch

    bank_id = f"test-placeholder-reprocess-{uuid.uuid4().hex[:8]}"
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Alice works at Google.", "document_id": "doc"}],
        request_context=request_context,
    )
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        # No API can store this any more, so plant the legacy value directly.
        await conn.execute(
            f"""UPDATE {fq_table("documents")}
                SET retain_params = COALESCE(retain_params, '{{}}'::jsonb) || '{{"event_date": "9999-12-31T23:59:59.999999+00:00"}}'::jsonb
                WHERE bank_id = $1 AND id = 'doc'""",
            bank_id,
        )
    try:
        with patch.object(memory, "submit_async_retain", new=AsyncMock(return_value={"operation_id": "x"})) as submit:
            await memory.reprocess_document(bank_id, "doc", request_context=request_context)
        (replayed,) = submit.call_args.args[1]
        assert replayed["event_date"] is None
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
