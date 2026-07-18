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
               AND coalesce(bool_and(i.indisvalid AND i.indisready), false)
        FROM expected e
        LEFT JOIN pg_namespace n ON n.nspname = $2
        LEFT JOIN pg_class c ON c.relnamespace = n.oid AND c.relname = e.index_name
        LEFT JOIN pg_index i ON i.indexrelid = c.oid
        """,
        bank_id,
        schema,
        len(_BANK_INDEX_FACT_TYPES),
    )
    return bool(healthy)


async def _index_health(conn: Any, schema: str, index_names: list[str]) -> dict[str, bool]:
    """Return valid-and-ready state for the requested indexes in one query."""
    rows = await conn.fetch(
        """
        SELECT c.relname AS index_name, i.indisvalid AND i.indisready AS healthy
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index i ON i.indexrelid = c.oid
        WHERE n.nspname = $1 AND c.relname = ANY($2::text[])
        """,
        schema,
        index_names,
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
        escaped_bank_id = bank_id.replace("'", "''")
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
                    f"WHERE fact_type = '{fact_type}' AND bank_id = '{escaped_bank_id}'"
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
        return []

    try:
        return [await _reconcile_schema(conn, schema, index_clause, dry_run=dry_run) for schema in schemas]
    finally:
        await conn.fetchval("SELECT pg_advisory_unlock($1)", _VECTOR_INDEX_RECONCILE_LOCK_ID)
