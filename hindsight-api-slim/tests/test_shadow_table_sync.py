"""Runtime guard: the curation archive must carry every ``memory_units`` column.

``invalidated_memory_units`` was created with ``LIKE memory_units`` — a
point-in-time snapshot that does not follow later ALTERs. The curation row-move
(``_memory_unit_columns`` in memory_engine.py) builds its column list from the
live ``memory_units`` catalog and INSERTs into the archive **by name**, so any
column added to the main table but not the archive breaks invalidation at
runtime. ``test_migration_shape.py`` lints migration files for this statically;
this test verifies the actual migrated schema, catching drift the lint can't
see (e.g. a column added outside ``op.execute`` strings).

The archive may legitimately have *extra* columns (``invalidation_reason``,
``invalidated_at``, ``entity_ids``) — the requirement is superset, not equality.
"""

import asyncpg
import pytest


async def _column_names(conn: asyncpg.Connection, table_name: str) -> set[str]:
    rows = await conn.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        """,
        table_name,
    )
    return {r["column_name"] for r in rows}


@pytest.mark.asyncio
async def test_archive_columns_superset_of_memory_units(pg0_db_url: str) -> None:
    conn = await asyncpg.connect(pg0_db_url)
    try:
        main_columns = await _column_names(conn, "memory_units")
        archive_columns = await _column_names(conn, "invalidated_memory_units")
    finally:
        await conn.close()

    assert main_columns, "memory_units not found — did migrations run?"
    missing = main_columns - archive_columns
    assert not missing, (
        f"invalidated_memory_units is missing columns present on memory_units: {sorted(missing)}. "
        "A migration altered the main table without altering the curation archive; "
        "the row-move INSERT will fail. Add the columns to the archive in the same migration."
    )
