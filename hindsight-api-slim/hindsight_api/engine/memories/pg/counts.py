"""The count/aggregate surfaces: consolidation freshness, per-document counts,
ingestion over time, observation scopes.

Each is one ``GROUP BY`` (or filtered ``COUNT``) over `memory_units`. They back
the stats and admin views, not retrieval, so they are grouped here away from the
addressed reads. The SQL is lifted verbatim from the engine methods that used to
carry it; only the connection and ``fq_table`` resolver are now parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any


async def consolidation_freshness(*, conn, fq_table: Callable[[str], str], bank_id: str) -> dict[str, Any]:
    """Last consolidation time plus the pending / failed fact counts, in one scan.

    All three come from a single pass so keeping ``failed`` — part of the
    published contract — costs nothing over reflect()'s ``pending`` read.
    """
    row = await conn.fetchrow(
        f"""
        SELECT
            MAX(consolidated_at) AS last_consolidated_at,
            COUNT(*) FILTER (WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')) AS pending,
            COUNT(*) FILTER (WHERE consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')) AS failed
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1
        """,
        bank_id,
    )
    if row is None:
        return {"last_consolidated_at": None, "pending": 0, "failed": 0}
    return {
        "last_consolidated_at": row["last_consolidated_at"],
        "pending": row["pending"] or 0,
        "failed": row["failed"] or 0,
    }


async def document_memory_counts(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_ids: list[str]
) -> dict[str, int]:
    """Live memory count per document id, for the ids given."""
    if not document_ids:
        return {}
    rows = await conn.fetch(
        f"""
        SELECT document_id, COUNT(*) AS unit_count
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND document_id = ANY($2::text[])
        GROUP BY document_id
        """,
        bank_id,
        list(document_ids),
    )
    return {row["document_id"]: row["unit_count"] for row in rows}


async def memories_timeseries(
    *, conn, fq_table: Callable[[str], str], bank_id: str, time_field: str, trunc: str, since: datetime
) -> list[dict[str, Any]]:
    """Memories bucketed by ``time_field`` (truncated to ``trunc``) and fact_type.

    ``time_field`` is whitelisted by the caller before it reaches here — it is
    interpolated into SQL. Event-time fields fall back to ``created_at`` per row so
    rows without an event timestamp still appear.
    """
    bucket_expr = time_field if time_field == "created_at" else f"COALESCE({time_field}, created_at)"
    rows = await conn.fetch(
        f"""
        SELECT date_trunc('{trunc}', {bucket_expr} AT TIME ZONE 'UTC') AS bucket,
               fact_type, COUNT(*) AS count
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND {bucket_expr} >= $2
        GROUP BY bucket, fact_type
        ORDER BY bucket
        """,
        bank_id,
        since,
    )
    return [{"bucket": r["bucket"], "fact_type": r["fact_type"], "count": r["count"]} for r in rows]


async def observation_scope_counts(*, conn, fq_table: Callable[[str], str], bank_id: str) -> list[dict[str, Any]]:
    """Observations grouped by scope (their sorted tag set), most-populous first."""
    rows = await conn.fetch(
        f"""
        SELECT scope, COUNT(*) AS count
        FROM (
            SELECT COALESCE(ARRAY(SELECT unnest(tags) ORDER BY 1), '{{}}'::text[]) AS scope
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1 AND fact_type = 'observation'
        ) s
        GROUP BY scope
        ORDER BY count DESC, scope
        """,
        bank_id,
    )
    return [{"tags": list(r["scope"]), "count": r["count"]} for r in rows]


__all__ = [
    "consolidation_freshness",
    "document_memory_counts",
    "memories_timeseries",
    "observation_scope_counts",
]
