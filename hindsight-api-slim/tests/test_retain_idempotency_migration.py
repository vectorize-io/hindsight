"""Focused migration contract tests for Retain idempotency."""

import importlib


def test_oracle_idempotency_key_uses_character_semantics(monkeypatch):
    migration = importlib.import_module("hindsight_api.alembic.versions.e8f1a2b3c4d5_add_retain_idempotency")
    executed = []

    monkeypatch.setattr(migration.op, "execute", executed.append)
    monkeypatch.setattr(migration.op, "add_column", lambda *args, **kwargs: None)
    monkeypatch.setattr(migration.op, "create_unique_constraint", lambda *args, **kwargs: None)

    migration._oracle_upgrade()

    assert executed == ["ALTER TABLE async_operations ADD idempotency_key VARCHAR2(256 CHAR)"]


def test_oracle_serialization_dependency_uses_raw_uuid(monkeypatch):
    migration = importlib.import_module("hindsight_api.alembic.versions.f9a2b3c4d5e6_add_retain_serialization")
    executed = []

    monkeypatch.setattr(migration.op, "execute", executed.append)
    monkeypatch.setattr(migration.op, "add_column", lambda *args, **kwargs: None)
    monkeypatch.setattr(migration.op, "create_index", lambda *args, **kwargs: None)

    migration._oracle_upgrade()

    assert executed == ["ALTER TABLE async_operations ADD blocked_by_operation_id RAW(16)"]


def test_oracle_rewrites_serialization_bank_lock():
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    query, _, _ = _rewrite_pg_to_oracle("SELECT 1 FROM banks WHERE bank_id = $1 FOR NO KEY UPDATE")

    assert "FOR UPDATE" in query
    assert "NO KEY" not in query


def test_serialization_downgrade_unparks_pending_tasks(monkeypatch):
    migration = importlib.import_module("hindsight_api.alembic.versions.f9a2b3c4d5e6_add_retain_serialization")
    executed = []
    dropped = []

    monkeypatch.setattr(migration.op, "execute", executed.append)
    monkeypatch.setattr(
        migration.op,
        "drop_index",
        lambda name, **kwargs: dropped.append(("index", name)),
    )
    monkeypatch.setattr(
        migration.op,
        "drop_column",
        lambda table, column: dropped.append(("column", column)),
    )

    migration._pg_downgrade()

    assert "SET next_retry_at = NULL" in executed[0]
    assert dropped[-2:] == [
        ("column", "blocked_by_operation_id"),
        ("column", "serialization_key"),
    ]
