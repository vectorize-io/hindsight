"""``HINDSIGHT_API_DB_SESSION_SETUP_ON_ACQUIRE`` controls the per-acquire hook.

asyncpg runs ``RESET ALL`` on release, so the session GUCs the init callback
applies are gone by the next acquire — hence the callback is wired as ``setup=``
as well as ``init=`` by default. Deployments that pin those GUCs on the role or
database get them back from ``RESET ALL`` anyway, making the re-apply a pure
round trip per acquire (and, behind a transaction-mode pooler, a transaction of
its own — #3499). This flag lets them drop it.

``application_name`` is deliberately *not* part of the trade-off: pgbouncer never
re-issues it after ``RESET ALL`` (#3491), so its per-acquire hook stays either
way.

Deterministic (no DB): asyncpg.create_pool is monkeypatched to capture kwargs.
"""

import pytest

from hindsight_api.config import (
    DEFAULT_DB_SESSION_SETUP_ON_ACQUIRE,
    ENV_DB_SESSION_SETUP_ON_ACQUIRE,
    HindsightConfig,
    clear_config_cache,
)
from hindsight_api.engine.db import postgresql as pg_mod
from hindsight_api.engine.db.postgresql import PostgreSQLBackend

_DSN = "postgresql://u:p@h:5432/db"
_DSN_NAMED = "postgresql://u:p@h:5432/db?application_name=worker-3"


class _FakePool:
    def get_size(self):
        return 0

    def get_idle_size(self):
        return 0


class _RecordingConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []

    async def execute(self, query: str, *args) -> None:
        self.statements.append((query, args))


@pytest.fixture
def captured_pool_kwargs(monkeypatch):
    captured: dict = {}

    async def fake_create_pool(dsn, **kwargs):
        captured["dsn"] = dsn
        captured.update(kwargs)
        return _FakePool()

    monkeypatch.setattr(pg_mod.asyncpg, "create_pool", fake_create_pool)
    return captured


@pytest.fixture
def setup_on_acquire(monkeypatch):
    """Set the flag for one test, rebuilding the cached config around it."""

    def _set(value: str):
        monkeypatch.setenv(ENV_DB_SESSION_SETUP_ON_ACQUIRE, value)
        clear_config_cache()

    yield _set
    # monkeypatch restores the env var; drop the config built from it so the
    # next test doesn't inherit this one's value.
    clear_config_cache()


class TestConfigField:
    def test_defaults_to_on(self):
        assert DEFAULT_DB_SESSION_SETUP_ON_ACQUIRE is True

    def test_parses_env(self, monkeypatch):
        monkeypatch.setenv(ENV_DB_SESSION_SETUP_ON_ACQUIRE, "false")
        assert HindsightConfig.from_env().db_session_setup_on_acquire is False
        monkeypatch.setenv(ENV_DB_SESSION_SETUP_ON_ACQUIRE, "true")
        assert HindsightConfig.from_env().db_session_setup_on_acquire is True

    def test_rejects_ambiguous_value(self, monkeypatch):
        monkeypatch.setenv(ENV_DB_SESSION_SETUP_ON_ACQUIRE, "maybe")
        with pytest.raises(ValueError, match=ENV_DB_SESSION_SETUP_ON_ACQUIRE):
            HindsightConfig.from_env()


class TestPoolWiring:
    @pytest.mark.asyncio
    async def test_enabled_reapplies_the_callback_on_every_acquire(self, captured_pool_kwargs, setup_on_acquire):
        setup_on_acquire("true")

        async def init_callback(conn):
            pass

        backend = PostgreSQLBackend()
        await backend.initialize(_DSN, init_callback=init_callback)

        assert captured_pool_kwargs["init"] is init_callback
        assert captured_pool_kwargs["setup"] is init_callback

    @pytest.mark.asyncio
    async def test_disabled_drops_the_per_acquire_hook(self, captured_pool_kwargs, setup_on_acquire):
        setup_on_acquire("false")

        async def init_callback(conn):
            pass

        backend = PostgreSQLBackend()
        await backend.initialize(_DSN, init_callback=init_callback)

        # Still applied once, when the connection is opened...
        assert captured_pool_kwargs["init"] is init_callback
        # ...but not re-applied on acquire, which is the whole point.
        assert captured_pool_kwargs["setup"] is None

    @pytest.mark.asyncio
    async def test_disabled_still_reasserts_application_name(self, captured_pool_kwargs, setup_on_acquire):
        setup_on_acquire("false")
        seen: list[object] = []

        async def init_callback(conn):
            seen.append(conn)

        backend = PostgreSQLBackend()
        await backend.initialize(_DSN_NAMED, init_callback=init_callback)

        conn = _RecordingConnection()
        await captured_pool_kwargs["setup"](conn)

        # pgbouncer clears application_name on RESET ALL and never re-issues it,
        # so this hook survives the flag (#3491) — but it must not drag the
        # session GUCs back in with it.
        assert conn.statements == [("SELECT set_config('application_name', $1, false)", ("worker-3",))]
        assert seen == []

        # The open-time hook still applies both.
        await captured_pool_kwargs["init"](conn)
        assert seen == [conn]

    @pytest.mark.asyncio
    async def test_enabled_with_application_name_applies_both(self, captured_pool_kwargs, setup_on_acquire):
        setup_on_acquire("true")
        seen: list[object] = []

        async def init_callback(conn):
            seen.append(conn)

        backend = PostgreSQLBackend()
        await backend.initialize(_DSN_NAMED, init_callback=init_callback)

        conn = _RecordingConnection()
        await captured_pool_kwargs["setup"](conn)

        assert conn.statements == [("SELECT set_config('application_name', $1, false)", ("worker-3",))]
        assert seen == [conn]
