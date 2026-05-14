"""Repair mental_models.subtype on databases stuck at m3rg3h3ad5f6

Three production deployments reported `column "subtype" of relation
"mental_models" does not exist` on `create_mental_model` even after their
container reported `Database migrations completed successfully` and
`alembic_version` advanced to `m3rg3h3ad5f6` (see issue #1553, #1553#1
confirmations from @4Lienau and @khanhduyvt0101).

Both `h3c4d5e6f7g8_mental_models_v4` and `d5y6z7a8b9c0_backfill_mental_models_subtype`
were meant to ensure `subtype` exists, but on databases that came through the
`reflections -> mental_models` rename chain *and* whose alembic_version
advanced past `d5y6z7a8b9c0` along an alternate path during the divergent-heads
reorganization, neither column-add actually fired. The result is a head-tagged
database with a v3-shaped `mental_models` table missing six columns:
``subtype``, ``description``, ``entity_id``, ``observations``, ``links``,
``last_updated``.

This migration sits at the current head (`m3rg3h3ad5f6`) so every affected
deployment will pick it up on next container start. It mirrors the column-add
block from `d5y6z7a8b9c0_backfill_mental_models_subtype` using
``ADD COLUMN IF NOT EXISTS`` so it is a no-op on databases where the columns
are already present.

Revision ID: 86f7a033d372
Revises: m3rg3h3ad5f6
Create Date: 2026-05-14
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "86f7a033d372"
down_revision: str | Sequence[str] | None = "m3rg3h3ad5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    """Idempotently ensure mental_models has the v4 column set.

    Safe to re-apply on databases that already received the columns via
    `h3c4d5e6f7g8_mental_models_v4` or `d5y6z7a8b9c0_backfill_mental_models_subtype` —
    every column-add uses ``IF NOT EXISTS`` and the constraint is recreated
    from scratch with the canonical v4 allowlist.
    """
    schema = _pg_schema_prefix()
    bare_schema = schema.strip(".").strip('"') if schema else ""
    schema_clause = f"AND table_schema = '{bare_schema}'" if bare_schema else ""

    # Wrapped in a DO block so the existence check skips databases that
    # predate the reflections -> mental_models rename chain (no table to
    # repair). On those, every ALTER below would error.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'mental_models'
                  {schema_clause}
            ) THEN
                -- Add the six v4 columns idempotently.
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS subtype VARCHAR(32) NOT NULL DEFAULT 'structural';
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT '';
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS entity_id UUID;
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS observations JSONB DEFAULT '{{"observations": []}}'::jsonb;
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS links VARCHAR[];
                ALTER TABLE {schema}mental_models
                    ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP WITH TIME ZONE;

                -- Recreate the CHECK constraint with the canonical v4 allowlist.
                -- Existing rows with subtype = 'directive' (possible on databases
                -- that ran the o0j1k2l3m4n5 directive-only path) are rewritten to
                -- 'structural' first so the constraint add succeeds.
                UPDATE {schema}mental_models SET subtype = 'structural' WHERE subtype = 'directive';

                ALTER TABLE {schema}mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype;
                ALTER TABLE {schema}mental_models
                    ADD CONSTRAINT ck_mental_models_subtype
                    CHECK (subtype IN ('structural', 'emergent', 'pinned', 'learned'));

                CREATE INDEX IF NOT EXISTS idx_mental_models_subtype
                    ON {schema}mental_models(bank_id, subtype);
            END IF;
        END$$;
        """
    )


def _pg_downgrade() -> None:
    """No-op: dropping these columns would corrupt v4 application code."""
    pass


def upgrade() -> None:
    # PG-only: Oracle's baseline (o1a2b3c4d5e6) creates mental_models with its
    # own subtype shape (chk_mm_subtype IN ('directive', 'pinned')) and a
    # different table topology, so this PG-shaped repair does not apply.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
