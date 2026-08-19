"""Add last_memory_seen_at to mental_models, splitting it from last_refreshed_at.

``last_refreshed_at`` carried two meanings at once: the wall-clock time of the
last refresh, and the source-data watermark (the newest in-scope memory the
refresh saw) that staleness keys off. A refresh persisted the watermark into it,
and the watermark is clamped so it never regresses — so on a model whose scope
gained no new memories the refresh wrote back the value already there. The
document was rewritten, the timestamp never moved, and a client asking
"have I already refreshed this?" refreshed it again on every tick.

``last_memory_seen_at`` takes over the watermark meaning; ``last_refreshed_at``
goes back to being what its name says. The new column is backfilled from
``last_refreshed_at`` — which today holds the watermark — so staleness decides
exactly as it did before the migration and no bank mass-refreshes on deploy.
Nullable, so consumers COALESCE back to ``last_refreshed_at`` for any row a
refresh has not stamped yet.

Revision ID: e7c3a91f4b62
Revises: c4f7a91b2d38
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e7c3a91f4b62"
down_revision: str | Sequence[str] | None = "c4f7a91b2d38"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        ALTER TABLE {schema}mental_models
        ADD COLUMN IF NOT EXISTS last_memory_seen_at TIMESTAMP WITH TIME ZONE
        """
    )
    # last_refreshed_at currently holds the watermark, so copying it carries each
    # model's staleness decision across the cutover unchanged. Only stamp rows
    # still NULL, so re-running the migration is a no-op rather than a rollback of
    # watermarks that refreshes have since advanced.
    op.execute(
        f"""
        UPDATE {schema}mental_models
        SET last_memory_seen_at = last_refreshed_at
        WHERE last_memory_seen_at IS NULL
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}mental_models DROP COLUMN IF EXISTS last_memory_seen_at")


def _oracle_upgrade() -> None:
    # Oracle has no ADD COLUMN IF NOT EXISTS; guard on the data dictionary so a
    # re-run doesn't fail with ORA-01430 (column already exists).
    op.get_bind().exec_driver_sql(
        """
        DECLARE
            n NUMBER;
        BEGIN
            SELECT COUNT(*) INTO n FROM user_tab_columns
            WHERE table_name = 'MENTAL_MODELS'
              AND column_name = 'LAST_MEMORY_SEEN_AT';
            IF n = 0 THEN
                EXECUTE IMMEDIATE
                    'ALTER TABLE mental_models ADD (last_memory_seen_at TIMESTAMP WITH TIME ZONE)';
            END IF;
        END;
        """
    )
    op.get_bind().exec_driver_sql(
        "UPDATE mental_models SET last_memory_seen_at = last_refreshed_at WHERE last_memory_seen_at IS NULL"
    )


def _oracle_downgrade() -> None:
    op.get_bind().exec_driver_sql("ALTER TABLE mental_models DROP COLUMN last_memory_seen_at")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
