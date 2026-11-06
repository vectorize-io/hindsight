"""Scope counts preserve normalization, filtering, and pagination."""

import asyncpg

from hindsight_api.engine.memories.pg.counts import observation_scope_counts
from hindsight_api.engine.search.tags import TagGroupLeaf


async def test_observation_scope_counts_with_shared_tags(pg0_db_url):
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute("""
            CREATE TEMP TABLE memory_units (bank_id text, fact_type text, tags varchar[]);
            INSERT INTO memory_units VALUES
                ('bank', 'observation', '{a,b}'),
                ('bank', 'observation', '{b,a}'),
                ('bank', 'observation', '{}'),
                ('bank', 'observation', NULL),
                ('bank', 'observation', '{a,a}'),
                ('bank', 'world', '{a,b}'),
                ('other', 'observation', '{a,b}');
        """)
        kwargs = {"conn": conn, "fq_table": lambda name: f"pg_temp.{name}", "bank_id": "bank"}
        assert await observation_scope_counts(**kwargs) == {
            "scopes": [{"tags": [], "count": 2}, {"tags": ["a", "b"], "count": 2}, {"tags": ["a", "a"], "count": 1}],
            "total": 3,
            "limit": 100,
            "offset": 0,
        }
        assert await observation_scope_counts(**kwargs, tag_groups=[TagGroupLeaf(tags=["b"])], limit=1) == {
            "scopes": [{"tags": ["a", "b"], "count": 2}],
            "total": 1,
            "limit": 1,
            "offset": 0,
        }
        assert await observation_scope_counts(**kwargs, limit=1, offset=3) == {
            "scopes": [],
            "total": 3,
            "limit": 1,
            "offset": 3,
        }
    finally:
        await conn.close()
