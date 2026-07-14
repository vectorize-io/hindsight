from pathlib import Path

MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "hindsight_api"
    / "alembic"
    / "versions"
    / "a8c1e4f7b0d3_add_operation_retention_indexes.py"
)


def test_operation_retention_migration_follows_current_head_and_covers_both_dialects():
    source = MIGRATION.read_text()

    assert 'down_revision: str | Sequence[str] | None = "f2a4b6c8d0e2"' in source
    assert "def _pg_upgrade()" in source
    assert "def _oracle_upgrade()" in source
    assert "run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)" in source


def test_operation_retention_migration_adds_cleanup_and_newest_first_indexes_with_downgrade():
    source = MIGRATION.read_text()

    assert "idx_async_operations_terminal_cleanup" in source
    assert "(updated_at, operation_id)" in source
    assert "WHERE status IN ('completed', 'failed', 'cancelled')" in source
    assert "(updated_at, operation_id, status)" in source
    assert "idx_async_operations_bank_created_desc" in source
    assert "(bank_id, created_at DESC)" in source
    assert "def _pg_downgrade()" in source
    assert "def _oracle_downgrade()" in source
    assert source.count("DROP INDEX") >= 4
