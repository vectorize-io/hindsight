"""Per-bank vector index coverage checks and repair.

Per-(bank, fact_type) partial vector indexes are created only when a bank is
first created (instant on an empty bank). A bank that becomes *populated*
outside that fresh-INSERT path — via a logical restore, a cross-version upgrade,
or a vector-extension switch (e.g. ScaNN→pgvector) — never gets them, so its
bank-scoped recall silently falls back to the global index + post-filter, which
is both slower and under-returns results. See issue #2645.

This module is the shared engine for detecting and repairing that gap. It is
driven by the ``repair-bank`` admin command; the build always uses
``CREATE INDEX CONCURRENTLY`` on a raw autocommit connection so it never takes
``ACCESS EXCLUSIVE`` on the shared ``memory_units`` table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name

logger = logging.getLogger(__name__)

# Postgres renders the partial predicate of an indexdef with parenthesized
# comparison operands and an explicit ::text cast, e.g.
# `... WHERE ((fact_type = 'world'::text) AND (bank_id = 'b1'::text))`.
# fact_type is emitted first (it is written first in the CREATE INDEX). Match
# that exact rendering so a mere name collision never counts as healthy.
_BANK_INDEX_PARTIAL_SUFFIX = " WHERE ((fact_type = "

# Access methods that legitimately back a per-(bank, fact_type) partial index.
# An index whose access method drifted after a backend switch does not match,
# so the health check treats it as unhealthy (rebuild).
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
class SchemaVectorIndexResult:
    """Per-schema outcome of a vector-index repair pass."""

    schema: str
    banks_scanned: int = 0
    already_present: int = 0
    created: int = 0
    skipped: int = 0  # would-create, reported under --dry-run
    failed: int = 0
    failed_indexes: list[str] = field(default_factory=list)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _index_health(conn: Any, schema: str, index_names: list[str]) -> dict[str, bool]:
    """Return valid-and-usable state for each requested index in one query.

    Health requires the index to be valid AND ready, defined over the expected
    ``memory_units`` table, to use a supported access method, and to carry our
    partial predicate. A name-only match is *not* enough: an INVALID leftover
    (from an interrupted concurrent build) or an index whose access method
    drifted after a backend switch must count as unhealthy so it is rebuilt —
    ``pg_indexes``/``IF NOT EXISTS`` alone would silently treat those as present.
    """
    if not index_names:
        return {}
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


async def _repair_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    *,
    dry_run: bool,
    bank_id: str | None,
) -> SchemaVectorIndexResult:
    result = SchemaVectorIndexResult(schema=schema)
    qschema = _quote_identifier(schema)

    if bank_id is not None:
        banks = await conn.fetch(
            f"SELECT bank_id, internal_id FROM {qschema}.banks WHERE bank_id = $1",  # noqa: S608 — schema is a quoted identifier
            bank_id,
        )
    else:
        banks = await conn.fetch(f"SELECT bank_id, internal_id FROM {qschema}.banks ORDER BY bank_id")  # noqa: S608
    result.banks_scanned = len(banks)

    # Resolve expected index names for every bank, then check them all in one
    # catalog query rather than one round-trip per index.
    expected_by_bank: list[tuple[str, dict[str, str]]] = []
    all_index_names: list[str] = []
    for bank in banks:
        expected = {ft: _bank_index_name(ft, str(bank["internal_id"])) for ft in _BANK_INDEX_FACT_TYPES}
        expected_by_bank.append((bank["bank_id"], expected))
        all_index_names.extend(expected.values())
    health = await _index_health(conn, schema, all_index_names)

    for bid, expected in expected_by_bank:
        # Render the bank_id literal server-side so escaping does not depend on
        # standard_conforming_strings (the predicate is inlined into the DDL).
        bank_id_literal = await conn.fetchval("SELECT quote_literal($1::text)", bid)
        for ft in _BANK_INDEX_FACT_TYPES:
            index_name = expected[ft]
            healthy = health.get(index_name)
            if healthy is True:
                result.already_present += 1
                continue
            if dry_run:
                result.skipped += 1
                continue

            qindex = _quote_identifier(index_name)
            qualified = f"{qschema}.{qindex}"
            try:
                # An unhealthy-but-present index (INVALID leftover, wrong access
                # method) must be dropped first — IF NOT EXISTS cannot repair it.
                if healthy is False:
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
                await conn.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {qindex} "
                    f"ON {qschema}.memory_units {index_clause} "
                    f"WHERE fact_type = '{ft}' AND bank_id = {bank_id_literal}"
                )
                result.created += 1
            except Exception as exc:  # noqa: BLE001 — one failed index must not abort the rest
                result.failed += 1
                result.failed_indexes.append(qualified)
                logger.warning(
                    "Failed to repair vector index %s (bank=%s, fact_type=%s): %s — "
                    "dropping the invalid leftover so a re-run can retry.",
                    qualified,
                    bid,
                    ft,
                    exc,
                )
                # A failed concurrent build leaves an INVALID index behind that
                # would shadow the good one; drop it so a re-run retries cleanly.
                try:
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.warning("Cleanup DROP INDEX for %s also failed: %s", qualified, cleanup_exc)

    return result


async def _safe_repair_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    *,
    dry_run: bool,
    bank_id: str | None,
) -> SchemaVectorIndexResult:
    try:
        return await _repair_schema(conn, schema, index_clause, dry_run=dry_run, bank_id=bank_id)
    except Exception as exc:  # noqa: BLE001 — one bad schema must not abort the whole sweep
        logger.warning("Vector index repair aborted for schema %s: %s", schema, exc)
        return SchemaVectorIndexResult(schema=schema, failed=1, failed_indexes=[f"{schema}.<schema-error>"])


async def repair_vector_indexes(
    conn: Any,
    schemas: list[str],
    index_clause: str,
    *,
    dry_run: bool = False,
    bank_id: str | None = None,
) -> list[SchemaVectorIndexResult]:
    """Rebuild missing or invalid per-bank vector indexes across ``schemas``.

    ``conn`` must be a raw autocommit PostgreSQL connection: ``CREATE INDEX
    CONCURRENTLY`` cannot run inside a transaction block. When ``bank_id`` is
    given, only that bank is reconciled (in each schema); otherwise every bank
    is scanned.

    Concurrency is handled by idempotency, not a lock (project rule: no advisory
    locks — they are unreliable behind connection poolers). Every build is
    ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` guarded by a valid/ready health
    check, so a second concurrent run is a no-op on already-built indexes; if two
    runs race the *same* missing index, Postgres rejects one build and the
    per-index handler drops the leftover so a re-run converges cleanly.
    """
    return [
        await _safe_repair_schema(conn, schema, index_clause, dry_run=dry_run, bank_id=bank_id) for schema in schemas
    ]
