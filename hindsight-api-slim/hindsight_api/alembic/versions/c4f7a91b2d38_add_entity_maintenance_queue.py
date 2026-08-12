"""Add entity_maintenance_queue table (+ seed it with every existing entity)

Queue of entities whose unit references may have gone away — the input to the
graph_maintenance job's orphan-entity and stale-cooccurrence prunes.

Those two prunes used to be bank-wide single statements re-evaluated on every
run: the orphan prune probed once per entity in the bank, and the cooccurrence
prune evaluated an INTERSECT per cooccurrence row in the bank, whether or not
anything had changed. Their cost tracked the size of the bank rather than the
size of the delete, so past a few million rows they blew asyncpg's command
timeout on every run and the job could never complete (#3222).

With a queue the prunes only examine entities a delete actually touched, the
same way ``graph_maintenance_queue`` already scopes the relink pass.

Deliberately NOT seeded with the existing entities. Backfilling them would
reclaim whatever a bank accumulated while its sweep was failing, but it writes
one row per entity inside a migration that runs at API startup, and then charges
a prune check for every one of them — a slow upgrade plus a large self-inflicted
backlog, to collect rows that cost a bank nothing. The queue starts empty and
fills from real deletes; historical strays stay until something touches them.

Revision ID: c4f7a91b2d38
Revises: d9c1a7b4e2f6
Create Date: 2026-08-11
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c4f7a91b2d38"
down_revision: str | Sequence[str] | None = "d9c1a7b4e2f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Composite PK gives ON CONFLICT DO NOTHING dedup when the same entity is
    # enqueued from overlapping deletes. No FK to entities: the prune's whole
    # job is to delete the entity, and a cascade would race it away mid-drain.
    # A queue row naming an entity that no longer exists is a no-op.
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}entity_maintenance_queue (
            bank_id     TEXT NOT NULL,
            entity_id   UUID NOT NULL,
            enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, entity_id)
        )
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_entity_maintenance_queue_bank_enqueued
        ON {schema}entity_maintenance_queue (bank_id, enqueued_at)
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_entity_maintenance_queue_bank_enqueued")
    op.execute(f"DROP TABLE IF EXISTS {schema}entity_maintenance_queue")


def _oracle_execute_ignoring_955(sql: str) -> None:
    """Run a CREATE statement and swallow ORA-00955 (object already exists).

    Mirrors the helper in the graph_maintenance_queue migration so reruns stay
    safe on a database where the table was created by an earlier partial run.
    """
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql.strip()})


def _oracle_upgrade() -> None:
    _oracle_execute_ignoring_955(
        """
        CREATE TABLE entity_maintenance_queue (
            bank_id     VARCHAR2(256) NOT NULL,
            entity_id   RAW(16)       NOT NULL,
            enqueued_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_entity_maintenance_queue PRIMARY KEY (bank_id, entity_id)
        )
        """
    )
    _oracle_execute_ignoring_955(
        "CREATE INDEX idx_entity_maintenance_queue_bank_enqueued ON entity_maintenance_queue (bank_id, enqueued_at)"
    )


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX idx_entity_maintenance_queue_bank_enqueued")
    op.execute("DROP TABLE entity_maintenance_queue")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
