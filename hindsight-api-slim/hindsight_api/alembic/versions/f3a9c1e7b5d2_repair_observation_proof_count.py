"""Repair proof_count on observations created from several sources (#4955).

Consolidation's create path wrote a literal ``proof_count = 1`` however many
source facts the new observation came from. Only a later update recounted it,
so observations that were never updated kept 1 and ranked lower in recall than
equally supported ones. The writer is fixed in the same change; this recounts
the existing rows from their distinct sources.

Only rows whose stored count differs from their distinct-source count are
written, so it is idempotent. Observations with no recorded sources keep their
count.

Revision ID: f3a9c1e7b5d2
Revises: e5b1c7d3a902
Create Date: 2026-10-05
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a9c1e7b5d2"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
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
        UPDATE {schema}memory_units mu
        SET proof_count = src.n
        FROM (
            SELECT id, (SELECT count(DISTINCT e) FROM unnest(source_memory_ids) e) AS n
            FROM {schema}memory_units
            WHERE fact_type = 'observation' AND cardinality(source_memory_ids) > 0
        ) src
        WHERE mu.id = src.id AND mu.proof_count IS DISTINCT FROM src.n
        """
    )


def _oracle_upgrade() -> None:
    # Oracle keeps an observation's sources in the observation_sources junction table.
    op.execute(
        """
        UPDATE memory_units mu
        SET proof_count = (SELECT COUNT(*) FROM observation_sources os WHERE os.observation_id = mu.id)
        WHERE mu.fact_type = 'observation'
          AND EXISTS (SELECT 1 FROM observation_sources os WHERE os.observation_id = mu.id)
          AND (mu.proof_count IS NULL
               OR mu.proof_count <> (SELECT COUNT(*) FROM observation_sources os WHERE os.observation_id = mu.id))
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def _noop_downgrade() -> None:
    # A data repair: the undercounted values it replaced were wrong, so there is nothing to restore.
    pass


def downgrade() -> None:
    run_for_dialect(pg=_noop_downgrade, oracle=_noop_downgrade)
