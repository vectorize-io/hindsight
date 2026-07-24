"""Add ``causal_links`` to the curation archive (invalidated_memory_units).

Causal edges (``caused_by`` and the historical ``causes``/``enables``/
``prevents``) are retain-time extraction output: unlike temporal and semantic
links they cannot be recomputed from dates or embeddings, and graph maintenance
never rebuilds them. Invalidation MOVES a fact out of ``memory_units``, so the
``memory_links → memory_units`` FK cascade deletes every incident edge — and
revert had no way to bring the causal ones back (#2864).

This column parks the descriptors of the causal edges incident to an archived
fact — ``[{"from_unit_id", "to_unit_id", "link_type", "weight"}, ...]`` — so
revert can rematerialize them. It is deliberately unindexed and lives only on
the archive: live facts keep their causal edges in ``memory_links`` (curation
edits no longer delete them), and the archive is small, cold, and only read by
low-frequency curation operations.

Revision ID: c7d1e9a4b3f2
Revises: d7b2f8a1c934
Create Date: 2026-07-24
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7d1e9a4b3f2"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # NOT NULL DEFAULT is metadata-only on PG 11+, so this is cheap even on a
    # large archive. Existing rows read as "no causal edges captured" — edges
    # lost before this migration cannot be reconstructed and are not guessed.
    op.execute(
        f"ALTER TABLE {schema}invalidated_memory_units "
        f"ADD COLUMN IF NOT EXISTS causal_links JSONB NOT NULL DEFAULT '[]'::jsonb"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP COLUMN IF EXISTS causal_links")


def _oracle_upgrade() -> None:
    # Kept in sync with PG for schema parity (curation itself is PostgreSQL-only
    # today — it introspects pg_attribute to move rows between the two tables).
    # Swallow ORA-01430 (column already exists) so the migration is idempotent.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units ADD (causal_links CLOB DEFAULT ''[]''
                CONSTRAINT imu_causal_links_json CHECK (causal_links IS JSON))';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN RAISE; END IF;
        END;
        """
    )


def _oracle_downgrade() -> None:
    # Swallow ORA-00904 (column does not exist).
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units DROP COLUMN causal_links';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -904 THEN RAISE; END IF;
        END;
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
