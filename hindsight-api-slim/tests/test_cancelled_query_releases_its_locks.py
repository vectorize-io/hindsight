"""A query cancelled right after its prepare must not leave the server in a transaction (#6613).

asyncpg prepares with Parse/Describe/Flush and no Sync. A cancel delivered as the prepare
returns (an ``asyncio.wait_for`` timeout, as link expansion uses) leaves the server session in
an implicit transaction holding the statement's table locks, while asyncpg believes the
connection is idle. Parked in the pool, it made every DROP INDEX CONCURRENTLY on
``memory_units`` wait for it, and every bank create and delete on the server queued behind.
"""

import asyncio

import asyncpg
import pytest

from hindsight_api.engine.db.postgresql import _NoResetConnection


class _CancelAfterPrepare(_NoResetConnection):
    """Delivers the cancel at the exact point a short timeout can: the prepare has returned."""

    cancel_next = False

    async def _get_statement(self, *args, **kwargs):
        statement = await super()._get_statement(*args, **kwargs)
        if self.cancel_next:
            self.cancel_next = False
            raise asyncio.CancelledError
        return statement


async def _locks_held(observer: asyncpg.Connection, pid: int) -> int:
    return await observer.fetchval(
        "SELECT count(*) FROM pg_locks WHERE pid = $1 AND relation = 'memory_units'::regclass", pid
    )


@pytest.mark.asyncio
async def test_a_query_cancelled_after_its_prepare_leaves_no_locks_behind(pg0_db_url):
    pool = await asyncpg.create_pool(pg0_db_url, min_size=1, max_size=1, connection_class=_CancelAfterPrepare)
    observer = await asyncpg.connect(pg0_db_url)
    try:
        async with pool.acquire() as conn:
            pid = conn.get_server_pid()
            conn._con.cancel_next = True  # the pool hands out a proxy
            with pytest.raises(asyncio.CancelledError):
                await conn.fetch("SELECT count(*) FROM memory_units WHERE bank_id = $1", "cancelled-after-prepare")
            # The server is mid-transaction although asyncpg thinks it is idle.
            assert not conn.is_in_transaction()
            assert await _locks_held(observer, pid) > 0

        # Released to the pool: the reset ended that transaction.
        assert await _locks_held(observer, pid) == 0
        async with pool.acquire() as conn:
            assert conn.get_server_pid() == pid  # the same connection, still usable
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await observer.close()
        await pool.close()
