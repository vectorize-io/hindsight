"""Repair: widen the remaining live ``bank_id`` columns from VARCHAR(64) to TEXT on PostgreSQL.

Follow-up to ``c3e5a7b9d1f4`` (issue #2106), which widened the two *history*
tables (``observation_history``, ``mental_model_history``) to ``TEXT`` after the
narrow ``VARCHAR(64)`` declaration bricked startup. The same VARCHAR(64) / TEXT
inconsistency still affects the live tables that store a user-supplied
``bank_id``:

* ``directives``    -- created VARCHAR(64) in ``p1k2l3m4n5o6``
* ``mental_models`` -- VARCHAR(64) (origin ``pinned_reflections`` in
                       ``n9i0j1k2l3m4``; recreated in ``h3c4d5e6f7g8``)

``mental_model_versions`` is intentionally *not* widened here: it is created in
``j5e6f7g8h9i0`` but dropped (``DROP TABLE ... CASCADE``) in ``o0j1k2l3m4n5`` and
never recreated on the upgrade path, so it does not exist at head. Issuing
``ALTER TABLE mental_model_versions ...`` would raise ``UndefinedTable`` and --
because migrations run inside the lifespan-startup transaction -- roll the whole
migration back, bricking the API. (It is unrelated to the live
``mental_model_history`` table widened by ``c3e5a7b9d1f4``.)

``banks.bank_id`` is ``TEXT`` (unbounded), so a deployment can create a bank
whose id exceeds 64 chars -- the 78-char hierarchical org-unit shape reported in
issue #2106 -- and the bank insert succeeds. The next write that propagates that
id (``create_directive``, ``create_mental_model`` / consolidation, or
mental-model versioning) then aborts with::

    psycopg2.errors.StringDataRightTruncation: value too long for type
    character varying(64)

i.e. a 500 on core write endpoints, instead of the startup brick that
``c3e5a7b9d1f4`` already repaired.

``ALTER COLUMN ... TYPE TEXT`` is a no-op on a column that is already ``TEXT``,
so every upgrade path converges on ``TEXT``. These tables are per-tenant (they
live in each tenant schema, not ``public``), so this runs for every migrated
schema via the search-path-aware prefix -- the same mechanism as
``c3e5a7b9d1f4``.

PostgreSQL only: these tables are created by PostgreSQL-only migrations
(``run_for_dialect(pg=...)``); on Oracle they are absent or already
``VARCHAR2(256)`` (consistent, never truncates), so the Oracle slot is
intentionally absent -- mirroring ``c3e5a7b9d1f4``.

Revision ID: a1d3f5b7c9e2
Revises: c3e5a7b9d1f4
Create Date: 2026-06-13
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a1d3f5b7c9e2"
down_revision: str | Sequence[str] | None = "c3e5a7b9d1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}directives ALTER COLUMN bank_id TYPE TEXT")
    op.execute(f"ALTER TABLE {schema}mental_models ALTER COLUMN bank_id TYPE TEXT")


def _pg_downgrade() -> None:
    # No-op: narrowing back to VARCHAR(64) could truncate real data and would
    # re-introduce the bug this migration repairs. The column types are owned by
    # the migrations that created the tables.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)
