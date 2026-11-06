"""``HINDSIGHT_API_DATABASE_CONNECT_HOOK`` opens every new connection (#6163).

asyncpg reuses the pool's DSN for each new physical connection, so a rotated
password broke pool growth and reconnects until restart. The hook runs on every
new connection instead, so it can fetch the current credential. Runs against the
real test database: the hook here rewrites a *wrong* password to the right one,
which is exactly what a rotation hook does.
"""

from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest

from hindsight_api.config import ENV_DATABASE_CONNECT_HOOK, clear_config_cache
from hindsight_api.engine.db.postgresql import PostgreSQLBackend, connect_direct

calls: list[str] = []
_real_dsn: str = ""


async def rotating_connect(dsn: str, **kwargs):
    """Test hook: ignore the (stale) DSN password, connect with the current one."""
    calls.append(dsn)
    return await asyncpg.connect(_real_dsn, **kwargs)


def _with_password(dsn: str, password: str) -> str:
    u = urlparse(dsn)
    netloc = f"{u.username}:{password}@{u.hostname}:{u.port}"
    return urlunparse(u._replace(netloc=netloc))


@pytest.fixture
def hook(monkeypatch, pg0_db_url):
    global _real_dsn
    _real_dsn = pg0_db_url
    calls.clear()

    def _set(value: str):
        monkeypatch.setenv(ENV_DATABASE_CONNECT_HOOK, value)
        clear_config_cache()

    yield _set
    monkeypatch.delenv(ENV_DATABASE_CONNECT_HOOK, raising=False)
    clear_config_cache()


async def test_pool_and_direct_connections_go_through_the_hook(hook, pg0_db_url):
    hook(f"{__name__}:rotating_connect")
    stale = _with_password(pg0_db_url, "rotated-away")

    backend = PostgreSQLBackend()
    await backend.initialize(stale, min_size=1, max_size=2)
    try:
        async with backend.acquire() as conn:
            assert await conn.fetchval("SELECT 1") == 1
    finally:
        await backend.shutdown()
    assert calls and all("rotated-away" in c for c in calls)

    calls.clear()
    conn = await connect_direct(stale)
    try:
        assert await conn.fetchval("SELECT 1") == 1
    finally:
        await conn.close()
    assert len(calls) == 1


async def test_unset_hook_uses_plain_asyncpg(hook, pg0_db_url):
    conn = await connect_direct(pg0_db_url)
    await conn.close()
    assert calls == []


@pytest.mark.parametrize("value", ["no_colon", f"{__name__}:missing", "no.such.module:fn", f"{__name__}:calls"])
async def test_bad_hook_fails_fast_naming_the_env_var(hook, value):
    hook(value)
    with pytest.raises(ValueError, match=ENV_DATABASE_CONNECT_HOOK):
        await connect_direct("postgresql://u:p@h/db")
