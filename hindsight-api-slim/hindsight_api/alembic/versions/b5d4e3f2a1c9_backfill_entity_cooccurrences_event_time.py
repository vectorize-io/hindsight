"""Backfill entity_cooccurrences.last_cooccurred from memory_units event time

Revision ID: b5d4e3f2a1c9
Revises: o1a2b3c4d5e6
Create Date: 2026-04-24

The writer path in `entity_resolver.link_units_to_entities_batch` historically
stamped `entity_cooccurrences.last_cooccurred` with `datetime.now(UTC)` at
flush time, ignoring the source memory unit's event date. For normal online
retains that's fine (now ≈ event time), but for any corpus that was
backfilled in a single session — migrating from another memory system, for
example — every co-occurrence collapsed to the import moment, which hid the
underlying knowledge timeline from the dashboard's entity graph recency heat
and from any downstream consumer of the column.

The writer is fixed in the same change set to propagate the unit's event_date;
this migration repairs historical rows by reading the true event time off
`unit_entities × memory_units` (falling back to `created_at` when
`mentioned_at` / `occurred_start` are NULL, so rows never regress).

Oracle slot is intentionally absent: the Oracle baseline (`o1a2b3c4d5e6`)
landed days before this fix, so any Oracle deployment runs the corrected
writer against an effectively empty `entity_cooccurrences` — there is no
historical residue on Oracle to repair. PG-only matches the asymmetry of
the data, not negligence.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b5d4e3f2a1c9"
down_revision: str | Sequence[str] | None = "o1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # Recompute last_cooccurred from the true event time per entity pair.
    # COALESCE picks the first non-null of mentioned_at / occurred_start /
    # created_at so banks without event-time metadata still see a sane value
    # (equivalent to the pre-fix behaviour) instead of NULL.
    #
    # The self-join on `unit_entities` is O(k²) per memory_unit in the number
    # of distinct entities mentioned (k). For typical units k is small (single
    # digits), but a bank with units containing hundreds of entities and tens
    # of millions of co-occurrence rows may want to run this off-hours — the
    # whole UPDATE is one statement, so it locks every targeted ec row for
    # the duration. The migration is one-time; subsequent online writes
    # already carry event time via the writer fix.
    op.execute(
        f"""
        UPDATE {schema}entity_cooccurrences ec
        SET last_cooccurred = sub.event_time
        FROM (
            SELECT
                LEAST(ue1.entity_id, ue2.entity_id) AS e1,
                GREATEST(ue1.entity_id, ue2.entity_id) AS e2,
                MAX(COALESCE(mu.mentioned_at, mu.occurred_start, mu.created_at)) AS event_time
            FROM {schema}memory_units mu
            JOIN {schema}unit_entities ue1 ON ue1.unit_id = mu.id
            JOIN {schema}unit_entities ue2 ON ue2.unit_id = mu.id AND ue1.entity_id <> ue2.entity_id
            GROUP BY 1, 2
        ) sub
        WHERE ec.entity_id_1 = sub.e1 AND ec.entity_id_2 = sub.e2
        """
    )


def _pg_downgrade() -> None:
    # No-op: the previous column value was `now()` at the time of write and
    # isn't recoverable. Rolling back the code is sufficient — new writes will
    # revert to the old behaviour for subsequent retains.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent — see header


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
