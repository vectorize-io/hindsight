"""Every pooled connection runs with JIT off (#6306).

The planner prices run-time-valued probes such as
``source_memory_ids @> ARRAY[cs.source_id]`` at a fixed fraction of the table,
which pushed expand_observations past the JIT cost thresholds and spent ~1 s
compiling a ~100 ms query on every recall.
"""

import pytest


@pytest.mark.asyncio
async def test_pooled_connections_run_with_jit_off(memory):
    pool = await memory._get_pool()
    # Twice: the second acquire may reuse the first connection, which must keep it.
    for _ in range(2):
        async with pool.acquire() as conn:
            assert await conn.fetchval("SHOW jit") == "off"
