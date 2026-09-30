"""The bank-level statements that reach into `documents` and `memory_units`.

A bank's own row lives in `banks`, which is not the store's; what is, is the per-bank
vector indexes on `memory_units` and the watermarks and counts the bank list reads off
`documents` and `memory_units`. The SQL is lifted verbatim from ``retain.bank_utils``.
"""

from __future__ import annotations

from collections.abc import Callable


async def create_bank_vector_indexes(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    internal_id: str,
    index_clause: str,
    fact_types: dict[str, str],
) -> None:
    """Build the bank's per-fact_type partial vector indexes on `memory_units`."""
    await ops.create_bank_vector_indexes(
        conn,
        fq_table("memory_units"),
        bank_id,
        internal_id,
        index_clause,
        fact_types,
    )


async def list_bank_rows(*, conn, fq_table: Callable[[str], str], where_clause: str, params: list[str]) -> list:
    """Every matching bank with its document and memory write watermarks, by bank_id."""
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    mu_table = fq_table("memory_units")
    return await conn.fetch(
        f"""
            SELECT
                b.bank_id, b.name, b.disposition, b.mission,
                b.created_at, b.updated_at,
                d.last_document_at,
                d.last_document_write_at,
                -- Per bank, off `idx_memory_units_bank_updated_at`: one index entry read instead of
                -- the GROUP BY over all of memory_units this replaced. `updated_at`, not
                -- `created_at`, because that index is the one that exists — and it is the right
                -- column anyway: this feeds `last_write_at` only, and every write to a fact bumps
                -- `updated_at`, so the watermark is the same or newer, which is what "last written"
                -- means. The documents half stays a GROUP BY: correlating it would be random reads
                -- over the same rows, since this query lists every bank before paging, and
                -- `last_document_at` must stay ingestion time (frozen when a document is rewritten).
                (SELECT MAX(m.updated_at) FROM {mu_table} m WHERE m.bank_id = b.bank_id) AS last_fact_at
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            {where_clause}
            ORDER BY b.bank_id
            """,
        *params,
    )


async def bank_page_rows(*, conn, fq_table: Callable[[str], str], bank_ids: list[str], sql_owned: list[str]) -> list:
    """The named banks' rows with their watermarks; `memory_units` is joined only for ``sql_owned``.

    Not named at all when ``sql_owned`` is empty, for planning cost — see ``bank_utils._bank_rows``.
    """
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    if sql_owned:
        mu_table = fq_table("memory_units")
        fact_select = "f.last_fact_at"
        fact_join = f"""
            LEFT JOIN (
                SELECT bank_id, MAX(updated_at) AS last_fact_at
                FROM {mu_table}
                WHERE bank_id = ANY($2::text[])
                GROUP BY bank_id
            ) f ON f.bank_id = b.bank_id"""
        params = (bank_ids, sql_owned)
    else:
        # Not `NULL::timestamptz` off a join that is simply empty — the table must not appear.
        fact_select = "NULL::timestamptz AS last_fact_at"
        fact_join = ""
        params = (bank_ids,)
    return await conn.fetch(
        f"""
            SELECT b.bank_id, b.name, b.disposition, b.mission, b.created_at, b.updated_at,
                   d.last_document_at, d.last_document_write_at,
                   {fact_select}
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                WHERE bank_id = ANY($1::text[])
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id{fact_join}
            WHERE b.bank_id = ANY($1::text[])
            """,
        *params,
    )


async def bank_fact_counts(*, conn, fq_table: Callable[[str], str], bank_ids: list[str]) -> dict[str, int]:
    """``memory_units`` rows per bank, for the banks named."""
    rows = await conn.fetch(
        f"""
            SELECT bank_id, COUNT(*) AS fact_count
            FROM {fq_table("memory_units")}
            WHERE bank_id = ANY($1)
            GROUP BY bank_id
            """,
        bank_ids,
    )
    return {row["bank_id"]: row["fact_count"] for row in rows}
