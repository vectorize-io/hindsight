"""The DSN's ``application_name`` must survive connection reuse, not just the
first acquire.

asyncpg forwards the DSN's ``application_name`` in the startup packet, so a
direct connection is labelled correctly. But asyncpg also runs ``RESET ALL``
when a connection is released to the pool, and behind pgbouncer the server
connection's startup packet is the *pooler's* — pgbouncer applies the client's
name with a ``SET`` at link time, so ``RESET ALL`` clears it and pgbouncer does
not re-issue it. Only the first acquire on each server connection ends up
attributed in ``pg_stat_activity``; the rest report an empty name. The backend
therefore re-asserts the name on every acquire via the pool's setup hook.

Deterministic (no DB): asyncpg.create_pool is monkeypatched to capture kwargs.
"""

import pytest

from hindsight_api.engine.db import postgresql as pg_mod
from hindsight_api.engine.db.postgresql import PostgreSQLBackend, application_name_from_dsn


class TestApplicationNameFromDsn:
    def test_present(self):
        assert application_name_from_dsn("postgresql://u:p@h:5432/db?application_name=worker-0") == "worker-0"

    def test_absent(self):
        assert application_name_from_dsn("postgresql://u:p@h:5432/db") is None

    def test_other_params_only(self):
        assert application_name_from_dsn("postgresql://u:p@h:5432/db?sslmode=disable") is None

    def test_alongside_other_params(self):
        dsn = "postgresql://u:p@h:5432/db?application_name=api-1&sslmode=disable"
        assert application_name_from_dsn(dsn) == "api-1"

    def test_last_occurrence_wins(self):
        # libpq semantics for repeated URL parameters.
        dsn = "postgresql://u:p@h/db?application_name=a&application_name=b"
        assert application_name_from_dsn(dsn) == "b"

    def test_empty_value_is_none(self):
        assert application_name_from_dsn("postgresql://u:p@h/db?application_name=") is None

    def test_unparseable_dsn_is_none(self):
        assert application_name_from_dsn("not a dsn at all ::::") is None


class _FakePool:
    def get_size(self):
        return 0

    def get_idle_size(self):
        return 0


class _RecordingConnection:
    """Captures the statements the pool's setup hook issues on acquire."""

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


@pytest.mark.asyncio
async def test_setup_hook_reasserts_application_name_on_every_acquire(captured_pool_kwargs):
    backend = PostgreSQLBackend()
    await backend.initialize("postgresql://u:p@h:5432/db?application_name=worker-3")

    # asyncpg runs `setup` on every acquire (after the release-time RESET ALL),
    # which is the only hook that survives pgbouncer clearing the value.
    setup = captured_pool_kwargs["setup"]
    assert setup is captured_pool_kwargs["init"]

    conn = _RecordingConnection()
    await setup(conn)
    assert conn.statements == [("SELECT set_config('application_name', $1, false)", ("worker-3",))]

    # A second acquire re-applies it rather than assuming it stuck.
    await setup(conn)
    assert len(conn.statements) == 2


@pytest.mark.asyncio
async def test_setup_hook_still_runs_the_callers_init_callback(captured_pool_kwargs):
    seen: list[object] = []

    async def init_callback(conn):
        seen.append(conn)

    backend = PostgreSQLBackend()
    await backend.initialize(
        "postgresql://u:p@h:5432/db?application_name=worker-3",
        init_callback=init_callback,
    )

    conn = _RecordingConnection()
    await captured_pool_kwargs["setup"](conn)
    # The session GUCs (hnsw.ef_search, statement_timeout, ...) must still be
    # applied — wrapping the callback must not displace it.
    assert seen == [conn]


@pytest.mark.asyncio
async def test_without_application_name_the_callback_is_passed_through(captured_pool_kwargs):
    async def init_callback(conn):
        pass

    backend = PostgreSQLBackend()
    await backend.initialize("postgresql://u:p@h:5432/db", init_callback=init_callback)

    # No name to assert: no wrapper, no extra statement per acquire.
    assert captured_pool_kwargs["setup"] is init_callback
    assert captured_pool_kwargs["init"] is init_callback


@pytest.mark.asyncio
async def test_no_application_name_and_no_callback_leaves_hooks_unset(captured_pool_kwargs):
    backend = PostgreSQLBackend()
    await backend.initialize("postgresql://u:p@h:5432/db")

    assert captured_pool_kwargs["setup"] is None
    assert captured_pool_kwargs["init"] is None
