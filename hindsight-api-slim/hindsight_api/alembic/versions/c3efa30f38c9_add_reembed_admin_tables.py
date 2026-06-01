"""Add reembed admin metadata tables.

Revision ID: c3efa30f38c9
Revises: c1d2e3f4a5b6
Create Date: 2026-06-01
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c3efa30f38c9"
down_revision: str | Sequence[str] | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}embedding_reembed_migrations (
            migration_id UUID PRIMARY KEY,
            schema_name TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector_extension TEXT NOT NULL,
            config_fingerprint TEXT NOT NULL,
            model_identity JSONB NOT NULL,
            embedding_state TEXT NOT NULL DEFAULT 'pending',
            shadow_indexes_state TEXT NOT NULL DEFAULT 'pending',
            semantic_links_state TEXT NOT NULL DEFAULT 'pending',
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            error_message TEXT,
            CHECK (embedding_state IN ('pending', 'complete')),
            CHECK (shadow_indexes_state IN ('pending', 'ready', 'deferred')),
            CHECK (semantic_links_state IN ('pending', 'complete')),
            CHECK (status IN ('running', 'prepared', 'completed', 'failed', 'abandoned'))
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}embedding_reembed_semantic_links (
            migration_id UUID NOT NULL,
            bank_id TEXT NOT NULL,
            from_unit_id UUID NOT NULL,
            to_unit_id UUID NOT NULL,
            weight DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (migration_id, from_unit_id, to_unit_id),
            FOREIGN KEY (migration_id)
                REFERENCES {schema}embedding_reembed_migrations(migration_id)
                ON DELETE CASCADE,
            CHECK (weight >= 0.0 AND weight <= 1.0)
        )
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_embedding_reembed_migrations_status
        ON {schema}embedding_reembed_migrations (status, updated_at)
        """
    )
    op.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_reembed_migrations_one_active
        ON {schema}embedding_reembed_migrations (schema_name)
        WHERE status IN ('running', 'prepared', 'failed')
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_embedding_reembed_semantic_links_bank
        ON {schema}embedding_reembed_semantic_links (migration_id, bank_id)
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_embedding_reembed_semantic_links_bank")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_embedding_reembed_migrations_one_active")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_embedding_reembed_migrations_status")
    op.execute(f"DROP TABLE IF EXISTS {schema}embedding_reembed_semantic_links")
    op.execute(f"DROP TABLE IF EXISTS {schema}embedding_reembed_migrations")


def _oracle_upgrade() -> None:
    # Re-embedding migration is PostgreSQL-only; Oracle admin CLI does not
    # expose this command path.
    pass


def _oracle_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
