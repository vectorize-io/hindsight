"""Add graphiti_outbox + entities.graphiti_uuid — C1 forwarder queue (PG + Oracle).

Two artifacts for the Graphiti federation forwarder (design:
``graphify-out/HINDSIGHT_GRAPHITI_DEEPDIVE_B1_WORKER_C1_FORWARDER.md`` §2.2-2.3):

* ``graphiti_outbox`` — one row per retained world fact that the
  ``graphiti_forward`` worker still has to forward to a Graphiti world graph.
  Enqueued inside the retain Phase 2 transaction; drained bank-wide with
  row-level backoff (``attempts``/``next_attempt_at``) — distinct from
  ``supersession_queue`` which is claim-and-delete, because the outbox targets
  an external system that can stay down for hours and "retry the whole batch"
  would burn external quota (already-idempotent edges but wasted calls).
  Filtered at enqueue time on bank ``graphiti_group_id`` set, ``fact_type =
  'world'``, and ``validate_relations`` non-empty — the worker never sees
  rows it should not have.

* ``entities.graphiti_uuid`` — single-column write-back target for the
  add_triplet response's resolved node UUIDs. Partial index
  (``WHERE graphiti_uuid IS NOT NULL``) keeps the index small while the
  cold-mapping phase (deep-dive 2 §2.5) is in progress; flips to a full index
  implicitly once the cold phase completes since most rows have the value.

Revision ID: g3h4i5j6k7l8
Revises: f3a4b5c6d7e8
Create Date: 2026-06-12
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "g3h4i5j6k7l8"
down_revision: str | Sequence[str] | None = "f3a4b5c6d7e8"
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
        CREATE TABLE IF NOT EXISTS {schema}graphiti_outbox (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            bank_id         TEXT NOT NULL,
            memory_id       UUID NOT NULL,
            group_id        TEXT NOT NULL,
            fact_text       TEXT NOT NULL,
            entities        JSONB NOT NULL,
            relations       JSONB NOT NULL,
            tags            JSONB,
            attempts        INT NOT NULL DEFAULT 0,
            last_error      TEXT,
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_graphiti_outbox_drain ON {schema}graphiti_outbox (bank_id, next_attempt_at)"
    )

    op.execute(f"ALTER TABLE {schema}entities ADD COLUMN IF NOT EXISTS graphiti_uuid UUID")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_entities_graphiti_uuid ON {schema}entities (graphiti_uuid) "
        f"WHERE graphiti_uuid IS NOT NULL"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_entities_graphiti_uuid")
    op.execute(f"ALTER TABLE {schema}entities DROP COLUMN IF EXISTS graphiti_uuid")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_graphiti_outbox_drain")
    op.execute(f"DROP TABLE IF EXISTS {schema}graphiti_outbox")


def _oracle_upgrade() -> None:
    # Oracle 23ai: NUMBER GENERATED ALWAYS AS IDENTITY for BIGINT surrogate,
    # RAW(16) for UUID (matches supersession_queue precedent), CLOB IS JSON
    # (with the check constraint) for the JSON payload columns, VARCHAR2(255)
    # for the short text columns. fact_text is CLOB because long-form facts
    # can exceed VARCHAR2(4000).
    op.execute(
        """
        CREATE TABLE graphiti_outbox (
            id              NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            bank_id         VARCHAR2(255) NOT NULL,
            memory_id       RAW(16) NOT NULL,
            group_id        VARCHAR2(255) NOT NULL,
            fact_text       CLOB NOT NULL,
            entities        CLOB NOT NULL
                CONSTRAINT graphiti_outbox_entities_json CHECK (entities IS JSON),
            relations       CLOB NOT NULL
                CONSTRAINT graphiti_outbox_relations_json CHECK (relations IS JSON),
            tags            CLOB
                CONSTRAINT graphiti_outbox_tags_json CHECK (tags IS JSON),
            attempts        NUMBER(10) DEFAULT 0 NOT NULL,
            last_error      CLOB,
            next_attempt_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX idx_graphiti_outbox_drain ON graphiti_outbox (bank_id, next_attempt_at)")

    op.execute("ALTER TABLE entities ADD graphiti_uuid RAW(16)")
    op.execute("CREATE INDEX idx_entities_graphiti_uuid ON entities (graphiti_uuid) WHERE graphiti_uuid IS NOT NULL")


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX idx_entities_graphiti_uuid")
    op.execute("ALTER TABLE entities DROP COLUMN graphiti_uuid")
    op.execute("DROP INDEX idx_graphiti_outbox_drain")
    op.execute("DROP TABLE graphiti_outbox")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
