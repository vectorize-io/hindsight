"""Drop materialized entity rows from memory_links.

Entity edges are no longer stored in ``memory_links``. The /graph endpoint
derives them on demand from ``unit_entities``, and recall already used the
``unit_entities`` self-join. Storing entity rows duplicated state we never
read from the link table — on a 10k-unit benchmark bank, entity rows were
53% of all link rows (~190 MB after indexes) and recall never touched them.

This migration deletes ``memory_links`` rows with ``link_type = 'entity'``.
``idx_memory_links_entity_covering`` was already dropped by migration
``e1b2c3d4f5a6``; we still issue ``DROP INDEX IF EXISTS`` defensively in case
this migration runs against an older snapshot that predates that one.

Revision ID: e9b2c7d1f3a4
Revises: e1b2c3d4f5a6
Create Date: 2026-05-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e9b2c7d1f3a4"
down_revision: str | Sequence[str] | None = "e1b2c3d4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()

    # Drop the partial covering index first so the bulk DELETE doesn't churn it.
    # CREATE/DROP INDEX CONCURRENTLY must run outside a transaction block.
    op.execute("COMMIT")
    op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity_covering")

    # Delete entity rows. Chunked to keep individual transactions small on
    # large banks (the perf-medium bench had ~345k entity rows; production
    # banks can be much larger).
    op.execute(
        f"""
        DO $$
        DECLARE
            deleted INTEGER;
        BEGIN
            LOOP
                DELETE FROM {schema}memory_links
                WHERE ctid IN (
                    SELECT ctid FROM {schema}memory_links
                    WHERE link_type = 'entity'
                    LIMIT 50000
                );
                GET DIAGNOSTICS deleted = ROW_COUNT;
                EXIT WHEN deleted = 0;
                COMMIT;
            END LOOP;
        END$$;
        """
    )


def _pg_downgrade() -> None:
    # Cannot reconstruct deleted entity links — the writer was path-dependent
    # on retain order. New retains will not produce entity rows either, so the
    # partial index would stay empty. Leave both no-op.
    pass


def _oracle_upgrade() -> None:
    op.execute("DELETE FROM memory_links WHERE link_type = 'entity'")


def _oracle_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
