"""Add fact-level supersession (bi-temporal ledger) columns to memory_units.

Adds three columns implementing temporal supersession of facts (design:
graphify-out/HINDSIGHT_GRAPHITI_DEEPDIVE_B1_C2.md, Part 1):

* ``valid_until``   — when the fact stopped being true in the real world
                      (the temporal counterpart of ``occurred_start``).
* ``superseded_at`` — when the system recorded the supersession (audit time).
* ``superseded_by`` — the fact that replaced this one. ``ON DELETE SET NULL``:
                      delta-retain replace can delete the superseding fact;
                      the supersession verdict (``valid_until``) survives with
                      the pointer cleared.

Naming deliberately avoids the "invalidated" word family: that verb belongs to
reversible curation (``invalidated_memory_units`` cold archive — rows are MOVED
out of the hot table). Supersession is the opposite: rows STAY in
``memory_units``, default retrieval filters them with ``valid_until IS NULL``,
and as-of queries can still see them.

``invalidated_memory_units`` was created with ``LIKE memory_units`` — a
point-in-time column snapshot that does NOT follow later ALTERs. The curation
row-move would silently drop (or fail on) the new columns, so this migration
alters BOTH tables in lockstep. The archive copy of ``superseded_by`` carries
no FK: its target row may itself be archived.

CHECK constraints encode the design rules:
* supersession requires a real-world start time (facts without
  ``occurred_start`` never participate in automatic interval algebra),
* ``valid_until`` must be after ``occurred_start``,
* a superseding pointer requires a ``valid_until`` (the reverse is allowed —
  manual curation may set only ``valid_until``).

Index note: the per-(bank, fact_type) partial vector indexes are intentionally
NOT touched. ``valid_until IS NULL`` participates as a residual filter after
the HNSW scan; the existing 5x over-fetch absorbs it (see design doc F3). The
only new index is the tiny partial ``superseded_by`` reverse-lookup index. On
Oracle a plain single-column B-tree index skips all-NULL keys, which is the
partial-index equivalent for free.

Revision ID: e7f8a9b0c1d2
Revises: c3e5a7b9d1f4
Create Date: 2026-06-11
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e7f8a9b0c1d2"
down_revision: str | Sequence[str] | None = "c3e5a7b9d1f4"
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
        ALTER TABLE {schema}memory_units
          ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS superseded_by UUID
            REFERENCES {schema}memory_units(id) ON DELETE SET NULL
        """
    )

    # CHECK constraints: drop-then-add for idempotency across per-schema reruns
    # (PG has no ADD CONSTRAINT IF NOT EXISTS).
    for name, expr in (
        (
            "chk_mu_supersession_needs_occurred",
            "valid_until IS NULL OR occurred_start IS NOT NULL",
        ),
        (
            "chk_mu_valid_until_after_start",
            "valid_until IS NULL OR valid_until > occurred_start",
        ),
        (
            "chk_mu_superseded_by_needs_until",
            "superseded_by IS NULL OR valid_until IS NOT NULL",
        ),
    ):
        op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS {name}")
        op.execute(f"ALTER TABLE {schema}memory_units ADD CONSTRAINT {name} CHECK ({expr})")

    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_mu_superseded_by "
        f"ON {schema}memory_units (superseded_by) WHERE superseded_by IS NOT NULL"
    )

    # Shadow-table sync (curation cold archive): LIKE snapshot does not follow
    # ALTERs, so the row-move INSERT...SELECT needs these columns here too.
    op.execute(
        f"""
        ALTER TABLE {schema}invalidated_memory_units
          ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
          ADD COLUMN IF NOT EXISTS superseded_by UUID
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_mu_superseded_by")
    for name in (
        "chk_mu_supersession_needs_occurred",
        "chk_mu_valid_until_after_start",
        "chk_mu_superseded_by_needs_until",
    ):
        op.execute(f"ALTER TABLE {schema}memory_units DROP CONSTRAINT IF EXISTS {name}")
    op.execute(
        f"""
        ALTER TABLE {schema}memory_units
          DROP COLUMN IF EXISTS superseded_by,
          DROP COLUMN IF EXISTS superseded_at,
          DROP COLUMN IF EXISTS valid_until
        """
    )
    op.execute(
        f"""
        ALTER TABLE {schema}invalidated_memory_units
          DROP COLUMN IF EXISTS superseded_by,
          DROP COLUMN IF EXISTS superseded_at,
          DROP COLUMN IF EXISTS valid_until
        """
    )


def _oracle_upgrade() -> None:
    # Alembic tracks revisions per schema, so each statement runs once —
    # no IF NOT EXISTS needed on ADD (Oracle errors on duplicate columns,
    # which would indicate a genuinely broken version table).
    op.execute(
        """
        ALTER TABLE memory_units ADD (
            valid_until   TIMESTAMP WITH TIME ZONE,
            superseded_at TIMESTAMP WITH TIME ZONE,
            superseded_by RAW(16),
            CONSTRAINT fk_mu_superseded_by FOREIGN KEY (superseded_by)
                REFERENCES memory_units(id) ON DELETE SET NULL,
            CONSTRAINT chk_mu_supersession_needs_occurred
                CHECK (valid_until IS NULL OR occurred_start IS NOT NULL),
            CONSTRAINT chk_mu_valid_until_after_start
                CHECK (valid_until IS NULL OR valid_until > occurred_start),
            CONSTRAINT chk_mu_superseded_by_needs_until
                CHECK (superseded_by IS NULL OR valid_until IS NOT NULL)
        )
        """
    )
    # Plain single-column B-tree: Oracle omits all-NULL keys, so this is the
    # partial-index equivalent of the PG index above.
    op.execute("CREATE INDEX idx_mu_superseded_by ON memory_units (superseded_by)")

    op.execute(
        """
        ALTER TABLE invalidated_memory_units ADD (
            valid_until   TIMESTAMP WITH TIME ZONE,
            superseded_at TIMESTAMP WITH TIME ZONE,
            superseded_by RAW(16)
        )
        """
    )


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_mu_superseded_by")
    op.execute("ALTER TABLE memory_units DROP (superseded_by, superseded_at, valid_until)")
    op.execute("ALTER TABLE invalidated_memory_units DROP (superseded_by, superseded_at, valid_until)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
