"""Add entities.entity_kind and exclude label entities from the trigram index.

Label entities (values of ``entity_labels`` config groups, stored as
``key:value`` canonical names) resolve by exact match only — fuzzy resolution
must never merge distinct label values (#1558), and since #3187 they are looked
up via the exact-match unique index rather than probed through pg_trgm. Their
rows were still covered by the shared trigram index, so every fuzzy probe for a
*regular* entity name pulled them into its candidate set only to discard them
in the bitmap recheck. On banks where a free-text label group accumulated tens
of thousands of mutually-similar values this recheck-discard overhead dominated
database CPU under ingest bursts (#3208).

"Is this row a label" was previously derived at runtime from the bank's
``entity_labels`` config, which an index predicate cannot reference — so the
classification is now materialised on the row:

1. Add ``entity_kind`` ("regular"/"label", CHECK-constrained) on both dialects.
   A kind column rather than a boolean so future entity kinds don't need
   another column.
2. Backfill per bank by classifying ``canonical_name`` against the bank's
   ``entity_labels`` config with the same ``is_label_entity()`` the resolver
   uses at insert time — a SQL reimplementation would be a second source of
   truth (and the map-group recursion doesn't translate). Banks hold at most
   tens of thousands of entities, so the synchronous per-bank backfill is fine.
   Label configs supplied only by a tenant extension (not stored in
   ``banks.config``) can't be seen here; their rows stay "regular", which
   costs index size but never correctness — label *texts* still resolve via
   the exact-match unique index.
3. Rebuild the PG trigram index as a partial index excluding label rows.
   Built CONCURRENTLY (autocommit block, invalid-leftover sweep, IF NOT
   EXISTS — same shape as 2071c7518f88) and only then drop the old full
   index, so fuzzy probes never lose index coverage. Skipped entirely when
   pg_trgm is absent (the resolver falls back to the "full" strategy, #626).

Oracle has no trigram index — it fuzzy-matches with a UTL_MATCH scan — so it
only gets the column + backfill; the resolver adds the matching
``entity_kind != 'label'`` filter to that scan.

Revision ID: b3e8d1c6f4a9
Revises: f2a6d8c4b1e9
Create Date: 2026-08-06
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b3e8d1c6f4a9"
down_revision: str | Sequence[str] | None = "f2a6d8c4b1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_INDEX = "entities_canonical_name_lower_trgm_idx"
_NEW_INDEX = "entities_canonical_name_lower_trgm_nonlabel_idx"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _backfill_entity_kind(schema: str) -> None:
    """Set entity_kind='label' on rows matching their bank's entity_labels config.

    Runs the resolver's own classification (``is_label_entity``) per bank in
    Python rather than reimplementing the enum/text/map prefix rules in SQL.
    Shared by both dialects: plain SELECT/UPDATE with expanding IN binds.
    """
    from hindsight_api.engine.retain.entity_labels import (
        build_labels_lookup,
        is_label_entity,
        parse_entity_labels,
    )

    bind = op.get_bind()
    banks = bind.execute(sa.text(f"SELECT bank_id, config FROM {schema}banks")).fetchall()
    for bank_id, raw_config in banks:
        # PG JSONB arrives as a dict; Oracle CLOB arrives as a LOB object on
        # raw text() fetches (oracledb's fetch_lobs default) — read it into a
        # JSON string first.
        if raw_config is not None and not isinstance(raw_config, (str, dict)):
            raw_config = raw_config.read()
        config = json.loads(raw_config) if isinstance(raw_config, str) else (raw_config or {})
        labels_cfg = parse_entity_labels(config.get("entity_labels"))
        if labels_cfg is None:
            continue
        lookup = build_labels_lookup(labels_cfg)
        rows = bind.execute(
            sa.text(f"SELECT id, canonical_name FROM {schema}entities WHERE bank_id = :bank_id"),
            {"bank_id": bank_id},
        ).fetchall()
        label_ids = [entity_id for entity_id, name in rows if is_label_entity(name, labels_cfg, lookup)]
        # Chunked to stay under Oracle's 1000-element IN limit; also keeps PG
        # bind arrays bounded.
        for start in range(0, len(label_ids), 500):
            chunk = label_ids[start : start + 500]
            stmt = sa.text(f"UPDATE {schema}entities SET entity_kind = 'label' WHERE id IN :ids").bindparams(
                sa.bindparam("ids", expanding=True)
            )
            bind.execute(stmt, {"ids": chunk})


def _pg_upgrade() -> None:
    bind = op.get_bind()
    schema = _pg_schema_prefix()
    # `or None` collapses an unset option and an explicit empty string into NULL
    # so the COALESCE below falls back to current_schema() in both cases.
    target_schema = context.config.get_main_option("target_schema") or None

    # IF NOT EXISTS: the transactional part below commits when the autocommit
    # block is entered, so a failure during the CONCURRENTLY build leaves the
    # revision unstamped with the column already added — the retry must not
    # trip over it. The constant default is a metadata-only change on PG 11+.
    op.execute(
        f"ALTER TABLE {schema}entities ADD COLUMN IF NOT EXISTS entity_kind TEXT DEFAULT 'regular' NOT NULL "
        f"CONSTRAINT chk_entities_entity_kind CHECK (entity_kind IN ('regular', 'label'))"
    )
    _backfill_entity_kind(schema)

    # Without pg_trgm neither the old index nor the extension's opclass exists;
    # the resolver already runs the "full" strategy there (#626).
    has_trgm = bind.execute(sa.text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")).scalar()
    if not has_trgm:
        return

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block; the
    # autocommit_block runs each statement outside Alembic's migration
    # transaction. Build the partial index first and drop the old full index
    # only afterwards, so fuzzy probes never lose index coverage.
    with op.get_context().autocommit_block():
        # A CONCURRENTLY build that errored on a previous run leaves an INVALID
        # index of this name behind, which IF NOT EXISTS would skip forever.
        leftover_invalid = bind.execute(
            sa.text(
                "SELECT NOT i.indisvalid "
                "FROM pg_class c "
                "JOIN pg_index i ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE c.relname = :index_name "
                "  AND n.nspname = COALESCE(:target_schema, current_schema())"
            ),
            {"index_name": _NEW_INDEX, "target_schema": target_schema},
        ).scalar()
        if leftover_invalid:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_NEW_INDEX}")

        # The predicate must textually match the resolver's candidate query
        # (`entity_kind != 'label'`) for the planner to choose this index.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_NEW_INDEX} "
            f"ON {schema}entities USING GIN (LOWER(canonical_name) gin_trgm_ops) "
            f"WHERE entity_kind != 'label'"
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_OLD_INDEX}")


def _pg_downgrade() -> None:
    bind = op.get_bind()
    schema = _pg_schema_prefix()

    has_trgm = bind.execute(sa.text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")).scalar()
    if has_trgm:
        # Restore the full index before dropping the partial one so fuzzy
        # probes keep index coverage throughout.
        with op.get_context().autocommit_block():
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_OLD_INDEX} "
                f"ON {schema}entities USING GIN (LOWER(canonical_name) gin_trgm_ops)"
            )
    # Dropping the column also drops the partial index and CHECK constraint.
    op.execute(f"ALTER TABLE {schema}entities DROP COLUMN IF EXISTS entity_kind")


def _oracle_upgrade() -> None:
    # Swallow ORA-01430 (column already exists) so a retry after a mid-run
    # failure is idempotent — Oracle DDL auto-commits statement by statement.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE entities ADD (entity_kind VARCHAR2(16) DEFAULT ''regular'' NOT NULL
                CONSTRAINT chk_entities_entity_kind CHECK (entity_kind IN (''regular'', ''label'')))';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN RAISE; END IF;
        END;
        """
    )
    _backfill_entity_kind("")


def _oracle_downgrade() -> None:
    # Swallow ORA-00904 (column does not exist).
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE entities DROP COLUMN entity_kind';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -904 THEN RAISE; END IF;
        END;
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
