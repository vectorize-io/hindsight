"""Drop the deprecated entity schema from ``memory_links``.

Entity edges are no longer materialized in ``memory_links``. Retain stores
memory-to-entity associations in ``unit_entities``, and both read paths derive
entity edges on demand from that table — the /graph endpoint builds them from
shared ``unit_entities`` rows, and recall expands via the ``unit_entities``
self-join. Migration ``e9b2c7d1f3a4`` deleted the materialized entity rows and
current writers only ever pass ``entity_id = NULL``, so the entity-specific
schema on ``memory_links`` is now dead weight:

- the ``entity_id`` column and its FK to ``entities``
- ``link_type = 'entity'`` in the table CHECK constraint
- the entity index (``idx_memory_links_entity`` on PG, ``idx_ml_entity`` on Oracle)
- the ``entity_id`` term in the function-based ``idx_memory_links_unique``

Once entity edges are gone, every remaining row (temporal, semantic, and the
causal types) has a single meaningful identity — ``(from_unit_id, to_unit_id,
link_type)`` — so the unique index collapses to those three columns and
preserves the existing effective uniqueness semantics.

Production ``memory_links`` tables and the entity index can be very large, so
this migration is written to avoid long exclusive locks: the residual delete is
chunked with per-batch commits, every index is swapped with ``CONCURRENTLY``,
and the new CHECK is added ``NOT VALID`` then validated separately so writers are
never blocked on a full-table scan.

Downgrade restores the former schema *shape* (column, FK, index, CHECK, and the
expression unique index) but cannot reconstruct the historical entity rows that
``e9b2c7d1f3a4`` already deleted — new retains do not produce them either, so the
restored entity index would simply stay empty.

Revision ID: c1e7a9d3f5b2
Revises: e4a7c1b9d2f6
Create Date: 2026-08-04
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c1e7a9d3f5b2"
down_revision: str | Sequence[str] | None = "e4a7c1b9d2f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NIL_ENTITY_UUID = "00000000-0000-0000-0000-000000000000"
_LINK_TYPES_WITHOUT_ENTITY = "'temporal', 'semantic', 'causes', 'caused_by', 'enables', 'prevents'"
_LINK_TYPES_WITH_ENTITY = "'temporal', 'semantic', 'entity', 'causes', 'caused_by', 'enables', 'prevents'"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()

    # CONCURRENTLY index swaps and the DO block's per-batch COMMIT both require
    # running outside Alembic's migration transaction — an autocommit_block
    # commits it and switches the connection to autocommit for the duration.
    with op.get_context().autocommit_block():
        # 1. Defensively delete any residual entity rows. e9b2c7d1f3a4 already
        #    did this, but a bank that predates its deployment can still carry
        #    them, and they must be gone before the three-column unique index
        #    (which no longer distinguishes entity rows) and the entity-free
        #    CHECK are created. Chunked with per-batch commits so a very large
        #    table drains in bounded transactions instead of one long-locking
        #    delete.
        op.execute(
            f"""
            DO $$
            DECLARE
                deleted INTEGER;
            BEGIN
                LOOP
                    DELETE FROM {schema}memory_links
                    WHERE ctid IN (
                        SELECT ctid FROM {schema}memory_links
                        WHERE link_type = 'entity'
                        LIMIT 50000
                    );
                    GET DIAGNOSTICS deleted = ROW_COUNT;
                    EXIT WHEN deleted = 0;
                    COMMIT;
                END LOOP;
            END$$;
            """
        )

        # 2. Drop the entity indexes. idx_memory_links_entity_covering was
        #    already removed by e1b2c3d4f5a6; dropped again defensively.
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_entity_covering")

        # 3. Replace the expression unique index with the three-column form.
        #    Build the new one first (under a temporary name) so uniqueness is
        #    never unprotected, then drop the old expression index and rename.
        #    Non-entity rows already collapse to (from, to, link_type) under the
        #    old COALESCE(entity_id, nil) expression, so this cannot find a
        #    duplicate once the entity rows are gone.
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_unique_new")
        op.execute(
            f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_unique_new "
            f"ON {schema}memory_links (from_unit_id, to_unit_id, link_type)"
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_unique")
        op.execute(f"ALTER INDEX IF EXISTS {schema}idx_memory_links_unique_new RENAME TO idx_memory_links_unique")

        # 4. Drop the FK and the now-unreferenced column. Both are metadata-only
        #    on PostgreSQL (DROP COLUMN marks the attribute dropped, no rewrite).
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS fk_memory_links_entity_id_entities")
        op.execute(f"ALTER TABLE {schema}memory_links DROP COLUMN IF EXISTS entity_id")

        # 5. Recreate the link_type CHECK without 'entity'. Added NOT VALID then
        #    validated separately: the ADD takes a brief lock without scanning,
        #    and VALIDATE takes only SHARE UPDATE EXCLUSIVE, so concurrent reads
        #    and writes are never blocked on the scan.
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS memory_links_link_type_check")
        op.execute(
            f"ALTER TABLE {schema}memory_links ADD CONSTRAINT memory_links_link_type_check "
            f"CHECK (link_type IN ({_LINK_TYPES_WITHOUT_ENTITY})) NOT VALID"
        )
        op.execute(f"ALTER TABLE {schema}memory_links VALIDATE CONSTRAINT memory_links_link_type_check")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()

    with op.get_context().autocommit_block():
        # Restore the column and FK (nullable, as in the initial schema).
        op.execute(f"ALTER TABLE {schema}memory_links ADD COLUMN IF NOT EXISTS entity_id uuid")
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS fk_memory_links_entity_id_entities")
        op.execute(
            f"ALTER TABLE {schema}memory_links ADD CONSTRAINT fk_memory_links_entity_id_entities "
            f"FOREIGN KEY (entity_id) REFERENCES {schema}entities (id) ON DELETE CASCADE"
        )

        # Restore the entity partial index.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_entity "
            f"ON {schema}memory_links (entity_id) WHERE entity_id IS NOT NULL"
        )

        # Restore the expression unique index (entity_id back in the key).
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_unique_old")
        op.execute(
            f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_memory_links_unique_old "
            f"ON {schema}memory_links (from_unit_id, to_unit_id, link_type, "
            f"COALESCE(entity_id, '{_NIL_ENTITY_UUID}'::uuid))"
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_memory_links_unique")
        op.execute(f"ALTER INDEX IF EXISTS {schema}idx_memory_links_unique_old RENAME TO idx_memory_links_unique")

        # Restore 'entity' as a permitted link_type.
        op.execute(f"ALTER TABLE {schema}memory_links DROP CONSTRAINT IF EXISTS memory_links_link_type_check")
        op.execute(
            f"ALTER TABLE {schema}memory_links ADD CONSTRAINT memory_links_link_type_check "
            f"CHECK (link_type IN ({_LINK_TYPES_WITH_ENTITY})) NOT VALID"
        )
        op.execute(f"ALTER TABLE {schema}memory_links VALIDATE CONSTRAINT memory_links_link_type_check")


def _oracle_exec_ignoring(sql: str, *ignore_codes: int) -> None:
    """Run a DDL statement, swallowing the given ORA-NNNNN codes for idempotency.

    Oracle has no ``IF EXISTS``/``IF NOT EXISTS`` for most objects; the standard
    pattern (see e4a7c1b9d2f6) is EXECUTE IMMEDIATE inside a PL/SQL block that
    re-raises anything but the expected "already absent"/"already present" code.
    """
    conditions = " AND ".join(f"SQLCODE != {code}" for code in ignore_codes)
    escaped = sql.replace("'", "''")
    op.execute(
        f"""
        BEGIN
            EXECUTE IMMEDIATE '{escaped}';
        EXCEPTION WHEN OTHERS THEN
            IF {conditions} THEN RAISE; END IF;
        END;
        """
    )


def _oracle_upgrade() -> None:
    # 1. Defensively drain residual entity rows in bounded, committed chunks so a
    #    large table doesn't delete under one long-held lock / huge undo segment.
    op.execute(
        """
        BEGIN
            LOOP
                DELETE FROM memory_links WHERE link_type = 'entity' AND ROWNUM <= 50000;
                EXIT WHEN SQL%ROWCOUNT = 0;
                COMMIT;
            END LOOP;
        END;
        """
    )

    # 2. Build the three-column unique index under a temporary name before
    #    dropping the old expression index, then rename. Oracle DDL implicitly
    #    commits, so ordering it this way keeps duplicate protection continuous
    #    instead of leaving a gap between drop and create. ONLINE avoids blocking
    #    concurrent DML during the (potentially large) index builds/drops.
    #    ORA-00955: name already in use; ORA-01418: index does not exist.
    _oracle_exec_ignoring("DROP INDEX idx_memory_links_unique_new", -1418)
    _oracle_exec_ignoring(
        "CREATE UNIQUE INDEX idx_memory_links_unique_new ON memory_links (from_unit_id, to_unit_id, link_type) ONLINE",
        -955,
    )
    _oracle_exec_ignoring("DROP INDEX idx_memory_links_unique", -1418)
    _oracle_exec_ignoring("ALTER INDEX idx_memory_links_unique_new RENAME TO idx_memory_links_unique", -1418)

    # 3. Drop the entity index. ORA-01418: index does not exist.
    _oracle_exec_ignoring("DROP INDEX idx_ml_entity ONLINE", -1418)

    # 4. Drop the entity FK, then the column. ORA-02443: constraint does not
    #    exist; ORA-00904: column does not exist.
    _oracle_exec_ignoring("ALTER TABLE memory_links DROP CONSTRAINT fk_ml_entity", -2443)
    _oracle_exec_ignoring("ALTER TABLE memory_links DROP COLUMN entity_id", -904)

    # 5. Recreate the link_type CHECK without 'entity'. ORA-02443: constraint
    #    does not exist; ORA-02264: name already used by an existing constraint.
    _oracle_exec_ignoring("ALTER TABLE memory_links DROP CONSTRAINT chk_ml_link_type", -2443)
    _oracle_exec_ignoring(
        f"ALTER TABLE memory_links ADD CONSTRAINT chk_ml_link_type CHECK (link_type IN ({_LINK_TYPES_WITHOUT_ENTITY}))",
        -2264,
    )


def _oracle_downgrade() -> None:
    # Restore the column, FK, entity index, expression unique index, and the
    # 'entity'-permitting CHECK. ORA-01430: column already exists; ORA-00955:
    # name already in use; ORA-01418: index does not exist; ORA-02443/-2264:
    # constraint absent / name in use.
    _oracle_exec_ignoring("ALTER TABLE memory_links ADD (entity_id RAW(16))", -1430)
    _oracle_exec_ignoring(
        "ALTER TABLE memory_links ADD CONSTRAINT fk_ml_entity "
        "FOREIGN KEY (entity_id) REFERENCES entities (id) ON DELETE CASCADE",
        -2264,
    )
    _oracle_exec_ignoring("CREATE INDEX idx_ml_entity ON memory_links (entity_id)", -955)

    _oracle_exec_ignoring("DROP INDEX idx_memory_links_unique_old", -1418)
    _oracle_exec_ignoring(
        "CREATE UNIQUE INDEX idx_memory_links_unique_old ON memory_links ("
        "from_unit_id, to_unit_id, link_type, "
        "NVL(entity_id, HEXTORAW('00000000000000000000000000000000'))) ONLINE",
        -955,
    )
    _oracle_exec_ignoring("DROP INDEX idx_memory_links_unique", -1418)
    _oracle_exec_ignoring("ALTER INDEX idx_memory_links_unique_old RENAME TO idx_memory_links_unique", -1418)

    _oracle_exec_ignoring("ALTER TABLE memory_links DROP CONSTRAINT chk_ml_link_type", -2443)
    _oracle_exec_ignoring(
        f"ALTER TABLE memory_links ADD CONSTRAINT chk_ml_link_type CHECK (link_type IN ({_LINK_TYPES_WITH_ENTITY}))",
        -2264,
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
