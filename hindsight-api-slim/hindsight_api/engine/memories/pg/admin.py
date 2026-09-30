"""Operator-side SQL on the Postgres store's tables: gauges, index sizing, and the admin CLI.

None of this serves a request. The consolidation gauges count every bank of a schema
at once, the vector-index policy sizes one bank's partitions, and the admin CLI's
backup / restore / rename-bank walk every table of a schema — store tables
included — on one raw connection. Each statement is lifted verbatim from the caller
that used to carry it; the schema-qualified resolver is a parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# --------------------------------------------------------------- consolidation gauges


async def count_consolidation_backlog(*, conn: Any, schema: str, per_bank: bool) -> dict[str | None, int]:
    """Unconsolidated, not-failed experience/world memories of every bank in ``schema``.

    Runs with seqscan disabled in a scoped transaction. The partial index
    ``idx_memory_units_unconsolidated`` matches its predicate, but
    ``consolidated_at IS NULL`` is true for a large fraction of the table (every
    observation has a null consolidated_at), so the planner misjudges selectivity
    and otherwise seq-scans the whole (largest) table on every refresh — verified on
    a 114k-row table via EXPLAIN: seq scan ~92 ms vs index scan ~0.1 ms. SET LOCAL
    forces the index path and resets at transaction end.

    The ``consolidation_failed_at IS NULL`` term keeps the backlog disjoint from the
    failed gauge (see ``reads.find_unconsolidated``); it is not in the partial
    index's predicate, so it is a cheap recheck on the rows the index returned.
    """
    bank_sel = "bank_id, " if per_bank else ""
    bank_grp = " GROUP BY bank_id" if per_bank else ""
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_seqscan = off")
        rows = await conn.fetch(
            f"SELECT {bank_sel}COUNT(*) AS count "
            f'FROM "{schema}".memory_units '
            "WHERE consolidated_at IS NULL AND consolidation_failed_at IS NULL "
            "AND fact_type IN ('experience', 'world')"
            f"{bank_grp}"
        )
    return _by_bank(rows, per_bank)


async def count_consolidation_failed(*, conn: Any, schema: str, per_bank: bool) -> dict[str | None, int]:
    """Experience/world memories of every bank in ``schema`` whose consolidation permanently failed.

    ``consolidation_failed_at IS NOT NULL`` is rare, so ``idx_memory_units_consolidation_failed``
    is chosen on cost and needs no planner nudge.
    """
    bank_sel = "bank_id, " if per_bank else ""
    bank_grp = " GROUP BY bank_id" if per_bank else ""
    rows = await conn.fetch(
        f"SELECT {bank_sel}COUNT(*) AS count "
        f'FROM "{schema}".memory_units '
        "WHERE consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')"
        f"{bank_grp}"
    )
    return _by_bank(rows, per_bank)


def _by_bank(rows: list, per_bank: bool) -> dict[str | None, int]:
    counts: dict[str | None, int] = {}
    for row in rows:
        bank = row["bank_id"] if per_bank else None
        counts[bank] = counts.get(bank, 0) + int(row["count"])
    return counts


# --------------------------------------------------------------- per-bank vector indexes


async def capped_memory_counts(
    *, conn: Any, schema: str, bank_id: str, fact_types: list[str], cap: int
) -> dict[str, int]:
    """Rows each fact-type partition of the bank holds, counted no further than ``cap``.

    The cap is a query parameter rather than an outer-column reference on
    purpose: PostgreSQL rejects a LIMIT/OFFSET expression containing a variable
    from an outer query level. See ``vector_index_health._capped_row_counts``.
    """
    from ...vector_index_health import _quote_identifier

    if not fact_types:
        return {}
    qschema = _quote_identifier(schema)
    rows = await conn.fetch(
        f"""
        SELECT t.fact_type,
               (
                   SELECT count(*) FROM (
                       SELECT 1 FROM {qschema}.memory_units
                       WHERE bank_id = $1 AND fact_type = t.fact_type
                       LIMIT $2
                   ) capped
               ) AS row_count
        FROM unnest($3::text[]) AS t(fact_type)
        """,  # noqa: S608 — schema is a quoted identifier
        bank_id,
        cap,
        fact_types,
    )
    return {row["fact_type"]: int(row["row_count"]) for row in rows}


# --------------------------------------------------------------- admin CLI (whole schema)


async def count_rows(*, conn: Any, fq_table_explicit: Callable[..., str], schema: str, table: str) -> int:
    """Row count of one table of ``schema``, for the backup manifest."""
    qualified_table = fq_table_explicit(table, schema)
    return await conn.fetchval(f"SELECT COUNT(*) FROM {qualified_table}")


async def truncate_tables(*, conn: Any, fq_table_explicit: Callable[..., str], schema: str, tables: list[str]) -> None:
    """``TRUNCATE ... CASCADE`` each table of ``schema``, in the order given."""
    for table in tables:
        qualified_table = fq_table_explicit(table, schema)
        await conn.execute(f"TRUNCATE TABLE {qualified_table} CASCADE")


async def move_bank_id(
    *,
    conn: Any,
    fq_table_explicit: Callable[..., str],
    schema: str,
    tables: list[str],
    old_bank_id: str,
    new_bank_id: str,
) -> dict[str, int]:
    """Rewrite ``bank_id`` from old to new in each table, in order; rows moved per table (zeros omitted)."""
    moved: dict[str, int] = {}
    for table in tables:
        status = await conn.execute(
            f"UPDATE {fq_table_explicit(table, schema)} SET bank_id = $1 WHERE bank_id = $2",
            new_bank_id,
            old_bank_id,
        )
        count = int(status.split()[-1])
        if count:
            moved[table] = count
    return moved


async def bank_file_keys(
    *, conn: Any, fq_table_explicit: Callable[..., str], schema: str, bank_id: str, prefix: str
) -> list:
    """The bank's stored-file keys under ``prefix``: ``(table_name, key)`` rows from attachments and documents.

    starts_with, not LIKE: key segments are percent-encoded, so a prefix can contain
    '%' and would read as a wildcard.
    """
    return await conn.fetch(
        f"SELECT 'attachments' AS table_name, storage_key AS key FROM {fq_table_explicit('attachments', schema)} "
        f"WHERE bank_id = $1 AND starts_with(storage_key, $2) "
        f"UNION ALL "
        f"SELECT 'documents', file_storage_key FROM {fq_table_explicit('documents', schema)} "
        f"WHERE bank_id = $1 AND file_storage_key IS NOT NULL AND starts_with(file_storage_key, $2)",
        bank_id,
        prefix,
    )


async def repoint_file_key(
    *,
    conn: Any,
    fq_table_explicit: Callable[..., str],
    schema: str,
    table: str,
    column: str,
    bank_id: str,
    old_key: str,
    new_key: str,
) -> None:
    """Point one row of ``table`` from its old stored-file key to the new one."""
    await conn.execute(
        f"UPDATE {fq_table_explicit(table, schema)} SET {column} = $1 WHERE bank_id = $2 AND {column} = $3",
        new_key,
        bank_id,
        old_key,
    )
