"""Regression tests for the migration advisory-lock leak (#4611).

A failure between ``pg_try_advisory_lock`` and the end of the migration used to
leave the session-level advisory lock behind: the ``finally`` block ran
``pg_advisory_unlock`` on a connection whose transaction had already aborted, so
the unlock raised ``InFailedSqlTransaction``, replaced the original error in the
log, and the lock stayed on the backend. Behind a transaction-mode pooler
(PgBouncer) that leaked lock blocks every later migrator until the pooler
recycles the backend.

These tests drive the real ``run_migrations`` lock/unlock path against a fake
SQLAlchemy connection and engine, mirroring the read-only ``CREATE EXTENSION``
failure from the issue.
"""

import contextlib
import logging
from typing import Any

import pytest
from psycopg2 import errors as pg_errors

from hindsight_api import migrations as migrations_module
from tests.pg_extension_fakes import FakePgConnection, Result


class FakeMigrationConnection(FakePgConnection):
    """``FakePgConnection`` plus the advisory-lock body of ``run_migrations``.

    Adds the advisory lock statements and PostgreSQL's aborted-transaction
    behaviour: once a statement fails, every further execute raises
    ``InFailedSqlTransaction`` until the transaction is ended, and a COMMIT on
    an aborted transaction acts as a ROLLBACK.
    """

    def __init__(
        self,
        fail_on: str | None = None,
        installed: set[str] | None = None,
    ) -> None:
        # The base class raises a bare RuntimeError on ``fail_on``; this one
        # models PostgreSQL instead, so it owns the field and leaves the base
        # one unset.
        super().__init__(extensions={name: ("public", True) for name in (installed or set())})
        self.fail_on = fail_on
        self.lock_acquired = False
        self.lock_released = False
        self.lock_released_after_abort = False
        self.open_txn = False
        self._aborted = False

    @property
    def in_aborted_transaction(self) -> bool:
        """Whether a statement has failed and the transaction is not yet ended."""
        return self._aborted

    def _record(self, sql: str) -> None:
        self.statements.append(sql)
        self.open_txn = True

    def execute(self, statement: Any, params: dict | None = None, *args: Any, **kwargs: Any) -> Any:
        sql = str(statement)
        if self._aborted:
            self._record(sql)
            raise pg_errors.InFailedSqlTransaction(
                "current transaction is aborted, commands ignored until end of transaction block"
            )
        if self.fail_on and self.fail_on in sql:
            self._record(sql)
            self._aborted = True
            raise pg_errors.ReadOnlySqlTransaction("cannot execute CREATE EXTENSION in a read-only transaction")
        if "pg_try_advisory_lock" in sql:
            self._record(sql)
            self.lock_acquired = True
            return Result(True)
        if "pg_advisory_unlock" in sql:
            self._record(sql)
            self.lock_released = True
            self.lock_released_after_abort = self._aborted
            return Result(True)
        result = super().execute(statement, params, *args, **kwargs)
        self.open_txn = True
        return result

    def commit(self) -> None:
        self.open_txn = False
        # A COMMIT on an aborted transaction is a ROLLBACK in PostgreSQL.
        if self._aborted:
            self._aborted = False
            self.rollbacks += 1
        else:
            self.commits += 1

    def rollback(self) -> None:
        self.open_txn = False
        self._aborted = False
        super().rollback()


class _WaitingConnection(FakeMigrationConnection):
    """A migrator that finds the lock taken by another worker at first."""

    def __init__(self, locks_error: Exception | None = None) -> None:
        super().__init__(installed={"vector", "pg_trgm"})
        self.other_holder = True
        self.polls = 0
        self.holder_lookups = 0
        self.locks_error = locks_error

    def execute(self, statement: Any, params: dict | None = None, *args: Any, **kwargs: Any) -> Any:
        sql = str(statement)
        if "pg_try_advisory_lock" in sql and not self._aborted:
            self.polls += 1
            if self.other_holder:
                self._record(sql)
                return Result(False)
        if "FROM pg_locks" in sql:
            self.holder_lookups += 1
            self._record(sql)
            if self.locks_error is not None:
                self._aborted = True
                raise self.locks_error
            return Result((4242, "hindsight-api", "10.0.0.7", "CREATE INDEX CONCURRENTLY ..."))
        return super().execute(statement, params, *args, **kwargs)


class _FakeEngine:
    """Replaces create_engine(...) so run_migrations uses the fake connection."""

    def __init__(self, conn: FakeMigrationConnection) -> None:
        self._conn = conn
        self.disposed = False

    def connect(self) -> "_FakeCtx":
        return _FakeCtx(self._conn)

    def dispose(self) -> None:
        self.disposed = True


class _FakeCtx:
    def __init__(self, conn: FakeMigrationConnection) -> None:
        self._conn = conn

    def __enter__(self) -> FakeMigrationConnection:
        return self._conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _patch_engine(monkeypatch: pytest.MonkeyPatch, conn: FakeMigrationConnection) -> _FakeEngine:
    engine = _FakeEngine(conn)
    monkeypatch.setattr(migrations_module, "create_engine", lambda url, poolclass=None: engine)
    return engine


def _isolate_run_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run_migrations off the subprocess-isolation and real-migration paths."""
    monkeypatch.setattr(migrations_module, "_should_isolate_migrations", lambda: False)
    monkeypatch.setattr(migrations_module, "is_oracle_url", lambda url: False)
    monkeypatch.setattr(migrations_module, "to_libpq_url", lambda url: url)
    monkeypatch.setattr(migrations_module, "configured_vector_extension", lambda: "pgvector")
    monkeypatch.setattr(migrations_module, "_run_migrations_internal", lambda *a, **k: None)


def _report_every_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report from the first failed poll, so a fake (instant) wait still logs."""
    monkeypatch.setattr(migrations_module, "_LOCK_WAIT_REPORT_AFTER_SECS", 0.0)


def _error_chain(excinfo: pytest.ExceptionInfo) -> list[str]:
    names = []
    err = excinfo.value
    while err is not None:
        names.append(type(err).__name__)
        err = err.__cause__
    return names


def _failing_migration(monkeypatch: pytest.MonkeyPatch) -> FakeMigrationConnection:
    """A migrator whose migration fails with its transaction already aborted.

    ``fail_on`` has to match the statement below: without it the fake never
    aborts, the unlock succeeds with or without the fix, and the tests that
    check the lock is still released prove nothing.
    """
    conn = FakeMigrationConnection(installed={"vector", "pg_trgm"}, fail_on="nonexistent_table")
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def failing_internal(*a, **k):
        with contextlib.suppress(Exception):
            conn.execute("SELECT count(*) FROM nonexistent_table")
        assert conn.in_aborted_transaction, "the fake must model the aborted transaction"
        raise pg_errors.InternalError('relation "nonexistent_table" does not exist')

    monkeypatch.setattr(migrations_module, "_run_migrations_internal", failing_internal)
    return conn


def test_lock_released_when_a_migration_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #4611 core: any failure inside the lock body must still release the lock.

    A failed statement aborts the transaction; the pre-fix finally ran
    pg_advisory_unlock on the aborted transaction, which raised
    InFailedSqlTransaction and left the lock on the backend. The fix rolls back
    first, so the unlock succeeds.
    """
    conn = _failing_migration(monkeypatch)

    with pytest.raises(RuntimeError):
        migrations_module.run_migrations("postgresql://user:***@host/db")

    assert conn.lock_acquired, "the lock must be acquired before the failure"
    assert conn.rollbacks >= 1, "a rollback must run before the unlock"
    assert conn.lock_released, "the advisory lock must be released despite the failure"
    assert not conn.lock_released_after_abort, "the unlock must not run on an aborted txn"


def test_original_error_is_not_masked_by_the_unlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The migration's own error must surface, not InFailedSqlTransaction."""
    conn = _failing_migration(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        migrations_module.run_migrations("postgresql://user:***@host/db")

    causes = _error_chain(excinfo)
    assert any("InternalError" in c for c in causes), causes
    assert not any("InFailedSqlTransaction" in c for c in causes), causes


def test_read_only_backend_skips_noop_ddl_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The #4611 trigger: installed extension on a read-only session.

    CREATE EXTENSION IF NOT EXISTS on an installed extension is a no-op on a
    writable session but an error on a read-only one (PgBouncer transaction
    mode carries default_transaction_read_only across clients). With the fix
    the pointless DDL is skipped and the migration run succeeds.
    """
    conn = FakeMigrationConnection(installed={"vector"}, fail_on="CREATE EXTENSION IF NOT EXISTS vector")
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    migrations_module.run_migrations("postgresql://user:***@host/db")

    assert conn.lock_acquired
    assert conn.lock_released
    assert not any("CREATE EXTENSION" in s for s in conn.statements), (
        "no CREATE EXTENSION may be issued when the extension is already installed"
    )


def test_unlock_failure_does_not_mask_the_original_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the rollback path itself fails, the unlock failure is logged, not raised.

    The surfaced error is the one the lock body raised first — here the
    OperationalError from the broken rollback that pgvector's install
    error-handler runs — never an InFailedSqlTransaction from the finally
    block.
    """
    conn = FakeMigrationConnection(fail_on="CREATE EXTENSION IF NOT EXISTS vector")
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def broken_rollback():
        conn.rollbacks += 1
        raise pg_errors.OperationalError("connection already closed")

    conn.rollback = broken_rollback

    with pytest.raises(RuntimeError) as excinfo:
        migrations_module.run_migrations("postgresql://user:***@host/db")

    causes = _error_chain(excinfo)
    assert any("OperationalError" in c for c in causes), causes
    assert not any("InFailedSqlTransaction" in c for c in causes), causes
    release_failures = [r for r in caplog.records if "Failed to release migration advisory lock" in r.getMessage()]
    assert release_failures, "an unlock failure must be logged so the lost lock is discoverable"


def test_run_migrations_succeeds_on_a_healthy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path still works: lock acquired, migrations run, lock released."""
    conn = FakeMigrationConnection(installed={"vector", "pg_trgm"})
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.lock_acquired
    assert conn.lock_released
    assert not conn.lock_released_after_abort


def test_waiting_worker_polls_and_commits_between_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that can't get the lock keeps polling, without an open txn."""
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.polls >= 2, "the worker must poll until the lock is free"
    assert conn.commits >= 1, "the worker must commit between polls"


def test_waiting_worker_logs_while_polling(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A worker waiting on a held lock must say so, not look hung."""
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)
    _report_every_wait(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    with caplog.at_level(logging.INFO, logger="hindsight_api.migrations"):
        migrations_module.run_migrations("postgresql://user:pass@host/db")

    waiting_logs = [r for r in caplog.records if "waiting" in r.getMessage().lower()]
    assert waiting_logs, "a worker polling for the lock must log while waiting"
    message = waiting_logs[0].getMessage()
    assert "pid=4242" in message, "the log must name the backend holding the lock"
    # The remedy it points at has to be a knob that exists.
    assert "HINDSIGHT_API_MIGRATION_DATABASE_URL" in message


def test_a_brief_wait_is_not_reported(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Queuing briefly behind another worker is routine and must stay quiet.

    Without a threshold every schema that waits one poll emits a WARNING and an
    extra pg_locks lookup, which is noise on every rolling deploy.
    """
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    with caplog.at_level(logging.INFO, logger="hindsight_api.migrations"):
        migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert not [r for r in caplog.records if "waiting" in r.getMessage().lower()]
    assert conn.holder_lookups == 0, "a brief wait must not cost a catalog lookup either"


def test_waiting_worker_holds_no_snapshot_across_the_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The holder lookup must not reopen a txn that spans the poll sleep.

    The commit in the wait loop exists so this connection holds no snapshot
    while waiting, which would otherwise block a CREATE INDEX CONCURRENTLY in
    the migration worker.  A diagnostic SELECT issued after that commit
    quietly undoes it.
    """
    conn = _WaitingConnection()
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)
    _report_every_wait(monkeypatch)

    seen = []

    def fake_sleep(_seconds):
        seen.append(conn.open_txn)
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.holder_lookups >= 1, "the holder lookup must actually run"
    assert seen and not any(seen), "no transaction may be open while the worker sleeps"


def test_failed_holder_lookup_does_not_break_the_wait_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A best-effort diagnostic must never turn a wait into a failed migration.

    A failing pg_locks lookup leaves the transaction aborted; unless the loop
    ends that transaction before its next statement, the following
    pg_try_advisory_lock raises InFailedSqlTransaction and the migration fails
    although the lock was merely busy.
    """
    conn = _WaitingConnection(locks_error=pg_errors.InsufficientPrivilege("permission denied for view pg_locks"))
    _patch_engine(monkeypatch, conn)
    _isolate_run_migrations(monkeypatch)
    _report_every_wait(monkeypatch)

    def fake_sleep(_seconds):
        conn.other_holder = False

    monkeypatch.setattr(migrations_module.time, "sleep", fake_sleep)

    with caplog.at_level(logging.DEBUG, logger="hindsight_api.migrations"):
        migrations_module.run_migrations("postgresql://user:pass@host/db")

    assert conn.lock_acquired and conn.lock_released
    waiting_logs = [r for r in caplog.records if "waiting" in r.getMessage().lower()]
    assert waiting_logs, "the wait is still reported when the holder cannot be identified"
