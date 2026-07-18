"""Background reconciliation for per-bank vector index coverage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name

logger = logging.getLogger(__name__)

# Session-level lock serializes reconciliation across API replicas while allowing
# CREATE INDEX CONCURRENTLY to run outside a transaction on the same connection.
_VECTOR_INDEX_RECONCILE_LOCK_ID = 0x48494E4453494748

# Per-(bank, fact_type) indexes share a consistent partial predicate shape:
# `CREATE INDEX ... ON memory_units USING ... WHERE fact_type = '<ft>' AND bank_id = '<id>'`.
# We pin the predicate prefix in health checks so a name collision against an
# unrelated index never silently satisfies reconciliation.
_BANK_INDEX_PARTIAL_SUFFIX = " WHERE (fact_type = "


# Postgres access methods we recognize as supporting per-(bank, fact_type)
# partial indexes. A legitimate HNSW/IVFFLAT/DiskANN/etc. index must match one
# of these; an index whose access method drifts after a backend switch does
# not, so the health check refuses to mark it healthy.
_SUPPORTED_INDEX_AM: tuple[str, ...] = (
    "btree",
    "gin",
    "gist",
    "hnsw",
    "ivfflat",
    "diskann",
    "vchordrq",
)


@dataclass
class SchemaVectorIndexReconcileResult:
    """Outcome of reconciling one PostgreSQL schema."""

    schema: str
    banks_scanned: int = 0
    already_present: int = 0
    created: int = 0
    skipped: int = 0
    failed: int = 0
    failed_indexes: list[str] = field(default_factory=list)
    # Set when reconciliation was skipped because another instance holds the
    # advisory lock — distinguishes "nothing to do" from "another holder is
    # already working".
    skipped_lock_busy: bool = False


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def bank_vector_indexes_healthy(conn: Any, schema: str, bank_id: str) -> bool:
    """Check one bank's expected index set with a single catalog query."""
    qschema = _quote_identifier(schema)
    suffixes = ", ".join(f"('{suffix}')" for suffix in _BANK_INDEX_FACT_TYPES.values())
    healthy = await conn.fetchval(
        f"""
        WITH expected AS (
            SELECT 'idx_mu_emb_' || spec.suffix || '_' ||
                   left(replace(b.internal_id::text, '-', ''), 16) AS index_name
            FROM {qschema}.banks b
            CROSS JOIN (VALUES {suffixes}) AS spec(suffix)
            WHERE b.bank_id = $1
        )
        SELECT count(i.indexrelid) = $3
               AND coalesce(bool_and(
                   (i.indisvalid AND i.indisready)
                   AND am.amname = ANY($4::text[])
                   AND pg_get_indexdef(i.indexrelid) LIKE $5
               ), false)
        FROM expected e
        LEFT JOIN pg_namespace n ON n.nspname = $2
        LEFT JOIN pg_class c ON c.relnamespace = n.oid AND c.relname = e.index_name
        LEFT JOIN pg_index i ON i.indexrelid = c.oid
        LEFT JOIN pg_am am ON am.oid = c.relam
        """,
        bank_id,
        schema,
        len(_BANK_INDEX_FACT_TYPES),
        list(_SUPPORTED_INDEX_AM),
        "%" + _BANK_INDEX_PARTIAL_SUFFIX + "%",
    )
    return bool(healthy)


async def _index_health(conn: Any, schema: str, index_names: list[str]) -> dict[str, bool]:
    """Return valid-and-ready state for the requested indexes in one query.

    Health requires the index to be valid, ready, defined over the expected
    `memory_units` table, to use a supported access method, and to carry our
    partial predicate. A name-only match (e.g. an unrelated index with the
    same relname) is *not* enough — backend switches would otherwise silently
    be marked healthy.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname AS index_name,
               (i.indisvalid AND i.indisready
                AND t.relname = 'memory_units'
                AND am.amname = ANY($3::text[])
                AND pg_get_indexdef(i.indexrelid) LIKE $4
               ) AS healthy
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index i ON i.indexrelid = c.oid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_am am ON am.oid = c.relam
        WHERE n.nspname = $1 AND c.relname = ANY($2::text[])
        """,
        schema,
        index_names,
        list(_SUPPORTED_INDEX_AM),
        "%" + _BANK_INDEX_PARTIAL_SUFFIX + "%",
    )
    return {row["index_name"]: bool(row["healthy"]) for row in rows}


async def _reconcile_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    *,
    dry_run: bool,
) -> SchemaVectorIndexReconcileResult:
    result = SchemaVectorIndexReconcileResult(schema=schema)
    qschema = _quote_identifier(schema)
    banks = await conn.fetch(f"SELECT bank_id, internal_id FROM {qschema}.banks ORDER BY bank_id")
    result.banks_scanned = len(banks)

    bank_specs: list[tuple[Any, dict[str, str]]] = []
    all_index_names: list[str] = []
    for bank in banks:
        expected = {
            fact_type: _bank_index_name(fact_type, str(bank["internal_id"])) for fact_type in _BANK_INDEX_FACT_TYPES
        }
        bank_specs.append((bank, expected))
        all_index_names.extend(expected.values())
    health = await _index_health(conn, schema, all_index_names) if all_index_names else {}

    for bank, expected in bank_specs:
        bank_id = bank["bank_id"]
        bank_id_literal = await conn.fetchval("SELECT quote_literal($1::text)", bank_id)
        for fact_type in _BANK_INDEX_FACT_TYPES:
            index_name = expected[fact_type]
            validity = health.get(index_name)
            if validity is True:
                result.already_present += 1
                continue

            qindex = _quote_identifier(index_name)
            qualified_index = f"{qschema}.{qindex}"
            if dry_run:
                result.skipped += 1
                continue
            try:
                if validity is False:
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified_index}")
                await conn.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {qindex} "
                    f"ON {qschema}.memory_units {index_clause} "
                    f"WHERE fact_type = '{fact_type}' AND bank_id = {bank_id_literal}"
                )
                result.created += 1
            except Exception as exc:
                result.failed += 1
                result.failed_indexes.append(f"{schema}.{index_name}")
                logger.warning(
                    "Failed to reconcile vector index %s (bank=%s, fact_type=%s): %s",
                    qualified_index,
                    bank_id,
                    fact_type,
                    exc,
                )
                try:
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified_index}")
                except Exception as cleanup_exc:
                    logger.warning("Failed to clean up vector index %s: %s", qualified_index, cleanup_exc)

    return result


async def _safe_reconcile_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    *,
    dry_run: bool,
) -> SchemaVectorIndexReconcileResult:
    try:
        return await _reconcile_schema(conn, schema, index_clause, dry_run=dry_run)
    except Exception as exc:
        logger.warning("Vector index reconcile aborted for schema %s: %s", schema, exc)
        return SchemaVectorIndexReconcileResult(
            schema=schema,
            failed=1,
            failed_indexes=[f"{schema}.<schema-error>"],
        )


async def reconcile_vector_indexes(
    conn: Any,
    schemas: list[str],
    index_clause: str,
    *,
    dry_run: bool = False,
) -> list[SchemaVectorIndexReconcileResult]:
    """Rebuild missing/invalid per-bank vector indexes without blocking writes.

    The caller must provide a raw autocommit PostgreSQL connection because
    ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction block.
    """
    acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", _VECTOR_INDEX_RECONCILE_LOCK_ID)
    if not acquired:
        logger.info("Vector index reconcile skipped: another instance holds the advisory lock")
        busy = SchemaVectorIndexReconcileResult(schema="", skipped_lock_busy=True)
        return [busy]

    try:
        return [await _safe_reconcile_schema(conn, schema, index_clause, dry_run=dry_run) for schema in schemas]
    finally:
        await conn.fetchval("SELECT pg_advisory_unlock($1)", _VECTOR_INDEX_RECONCILE_LOCK_ID)
