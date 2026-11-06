"""The pending-work probe agrees with the full consolidation statistics."""

import asyncpg

from hindsight_api.engine.memories.pg.counts import consolidation_freshness, has_pending_consolidation


async def test_pending_probe_ignores_failed_and_nonconsolidatable_facts(pg0_db_url):
    # Direct SQL on a temp table on purpose: the probe is a store-level query with no
    # engine read API, and the check is that it agrees row-for-row with the full count.
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute("""
            CREATE TEMP TABLE memory_units (
                bank_id text, fact_type text, created_at timestamptz DEFAULT now(),
                updated_at timestamptz DEFAULT now(),
                consolidated_at timestamptz, consolidation_failed_at timestamptz
            );
            INSERT INTO memory_units (bank_id, fact_type, consolidated_at, consolidation_failed_at) VALUES
                ('bank', 'observation', NULL, NULL),
                ('bank', 'experience', NULL, now()),
                ('bank', 'world', now(), NULL),
                ('other', 'world', NULL, NULL);
        """)
        kwargs = {"conn": conn, "fq_table": lambda name: f"pg_temp.{name}", "bank_id": "bank"}
        assert not (await consolidation_freshness(**kwargs))["pending"]
        assert not await has_pending_consolidation(**kwargs)

        await conn.execute("INSERT INTO memory_units (bank_id, fact_type) VALUES ('bank', 'world')")
        assert (await consolidation_freshness(**kwargs))["pending"] == 1
        assert await has_pending_consolidation(**kwargs)
    finally:
        await conn.close()
