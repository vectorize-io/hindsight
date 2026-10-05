"""The consolidation writes and checks: source liveness, observation insert / rewrite, dedup folds.

The SQL is lifted verbatim from ``engine/consolidation/consolidator.py``, which now makes one
store call per site. Each runs on the caller's connection, inside the consolidation batch's
transaction, so every write derived from one LLM response commits or rolls back together
(#3876), and the ``FOR SHARE`` locks hold until that transaction ends.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from ....config import get_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...consolidation.consolidator import _TemporalBounds


def _native_search_vector_update(config, param: str) -> str:
    """UPDATE-clause fragment that repopulates ``search_vector`` inline, or ''
    when the backend does not maintain a native tsvector column that way.

    ``to_tsvector(...)::regconfig`` is PostgreSQL-only. On Oracle ``search_vector``
    is a CLOB maintained by Oracle's own text index rather than an inline
    tsvector, so emit nothing there (mirrors the insert path, which gates
    ``search_vector`` on the PG-only ``pg_search_vector_expr``). Without this
    guard the PG expression reaches Oracle and fails with DPY-4010 (the
    ``::regconfig`` cast becomes an unbound ``:REGCONFIG`` placeholder).
    """
    from ...schema import _is_oracle  # noqa: PLC0415

    if config.text_search_extension != "native" or _is_oracle():
        return ""
    lang = config.text_search_extension_native_language
    return f",\n            search_vector = to_tsvector('{lang}'::regconfig, COALESCE({param}, ''))"


async def lock_live_memory_ids(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[uuid.UUID]
) -> set[str]:
    """Which of ``unit_ids`` still exist, ``FOR SHARE``-locked until the transaction ends.

    (Oracle has no ``FOR SHARE``; the SQL rewriter promotes it to ``FOR UPDATE`` — more
    conservative, still correct.)
    """
    rows = await conn.fetch(
        f"SELECT id FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
        unit_ids,
        bank_id,
    )
    return {str(r["id"]) for r in rows}


async def lock_observation_tags(
    *, conn, fq_table: Callable[[str], str], bank_id: str, observation_id: str
) -> list[str] | None:
    """The observation's current tags, row-locked until the transaction ends; None if it is gone.

    ``FOR NO KEY UPDATE`` still lets other rows take FK references to it. Callers lock the
    source rows first (``lock_live_memory_ids``), keeping the sources-before-observation order.
    """
    row = await conn.fetchrow(
        f"SELECT tags FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 FOR NO KEY UPDATE",
        uuid.UUID(observation_id),
        bank_id,
    )
    return None if row is None else list(row["tags"] or [])


async def memories_changed_since(
    *, conn, fq_table: Callable[[str], str], bank_id: str, read_at: dict[str, datetime]
) -> list[str]:
    """Ids in ``read_at`` whose ``updated_at`` moved since it was read, ``FOR SHARE``-locked."""
    rows = await conn.fetch(
        f"SELECT id, updated_at FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 FOR SHARE",
        [uuid.UUID(mid) for mid in read_at],
        bank_id,
    )
    return [str(r["id"]) for r in rows if r["updated_at"] != read_at[str(r["id"])]]


async def any_memory_exists(*, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[uuid.UUID]) -> bool:
    """Whether any of ``unit_ids`` still exists. Non-locking."""
    found = await conn.fetchval(
        f"SELECT 1 FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 LIMIT 1",
        unit_ids,
        bank_id,
    )
    return found is not None


async def count_observations_with_tags(*, conn, fq_table: Callable[[str], str], bank_id: str, tags: list[str]) -> int:
    """Observations whose tags contain every one of ``tags``."""
    return await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND fact_type = 'observation' AND tags @> $2::varchar[]",
        bank_id,
        tags,
    )


async def fold_sources_into_observation(
    *,
    conn,
    fq_table: Callable[[str], str],
    observation_id: str,
    expected_text: str,
    merged_text: str,
    source_memory_ids: list[uuid.UUID],
    bounds: _TemporalBounds,
) -> bool:
    """Fold ``source_memory_ids`` and ``merged_text`` into the observation, widening its dates.

    Keeps the observation's existing embedding (the merged text is >= threshold similar, so it
    stays representative and avoids a re-embed + a dialect-specific vector UPDATE). False when
    the row vanished or was rewritten since ``expected_text`` was read.
    """
    # Oracle-safe: _native_search_vector_update emits the to_tsvector clause only for a
    # native PG tsvector column, "" otherwise (see #3021 — the raw ::regconfig cast
    # breaks Oracle). RETURNING-gate on the twin's probe-time text so a concurrent
    # survivor rewrite during the connection-free LLM window can't be clobbered.
    search_vector_clause = _native_search_vector_update(get_config(), "$1")
    folded = await conn.fetchval(
        f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                source_memory_ids = (SELECT array_agg(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                proof_count = (SELECT count(DISTINCT e) FROM unnest(source_memory_ids || $2::uuid[]) e),
                event_date = LEAST(event_date, COALESCE($5, event_date)),
                occurred_start = LEAST(occurred_start, COALESCE($6, occurred_start)),
                occurred_end = GREATEST(occurred_end, COALESCE($7, occurred_end)),
                mentioned_at = GREATEST(mentioned_at, COALESCE($8, mentioned_at)),
                updated_at = now(){search_vector_clause}
            WHERE id = $3::uuid AND text = $4
            RETURNING id
            """,
        merged_text,
        source_memory_ids,
        uuid.UUID(observation_id),
        expected_text,
        bounds.event_date,
        bounds.occurred_start,
        bounds.occurred_end,
        bounds.mentioned_at,
    )
    return folded is not None


async def fold_observation_into_twin(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: str,
    observation_text: str,
    twin_id: str,
    twin_text: str,
    merged_text: str,
) -> bool:
    """Fold the observation's live sources and dates into its twin, with ``merged_text``.

    False when there is nothing to fold (the observation or all its sources are gone) or either
    row was rewritten since it was read. The caller deletes the folded observation.
    """
    # Snapshot the updated row's sources with a PLAIN read (no FOR UPDATE). Lock order
    # must be sources-before-observation: the liveness check below takes
    # FOR SHARE on the SOURCE rows first, then the fold UPDATE locks the observation
    # rows -- the same order as the create fold and the normal write paths
    # (insert_observation / rewrite_observation). Locking the observation
    # here (FOR UPDATE) would invert that against the invalidation path and deadlock.
    updated_row = await conn.fetchrow(
        f"""
            SELECT source_memory_ids
            FROM {fq_table("memory_units")}
            WHERE id = $1::uuid AND text = $2
            """,
        uuid.UUID(observation_id),
        observation_text,
    )
    if updated_row is None:
        return False
    sources = list(updated_row["source_memory_ids"] or [])
    if not sources:
        return False
    live = await lock_live_memory_ids(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sources)
    live_u_sources = [mid for mid in sources if str(mid) in live]
    if not live_u_sources:
        return False
    # Oracle-safe search_vector clause (#3021): "" unless a native PG tsvector column.
    # RETURNING-gate on both rows' probe-time text so a survivor/updated rewrite during
    # the connection-free LLM window can't be clobbered or fold a stale row.
    search_vector_clause = _native_search_vector_update(get_config(), "$1")
    folded = await conn.fetchval(
        f"""
            UPDATE {fq_table("memory_units")} t
            SET text = $1,
                source_memory_ids = (
                    SELECT array_agg(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                proof_count = (
                    SELECT count(DISTINCT e) FROM unnest(t.source_memory_ids || $6::uuid[]) e
                ),
                event_date = LEAST(t.event_date, COALESCE(u.event_date, t.event_date)),
                occurred_start = LEAST(t.occurred_start, COALESCE(u.occurred_start, t.occurred_start)),
                occurred_end = GREATEST(t.occurred_end, COALESCE(u.occurred_end, t.occurred_end)),
                mentioned_at = GREATEST(t.mentioned_at, COALESCE(u.mentioned_at, t.mentioned_at)),
                updated_at = now(){search_vector_clause}
            FROM {fq_table("memory_units")} u
            WHERE t.id = $2::uuid AND u.id = $3::uuid AND t.text = $4 AND u.text = $5
            RETURNING t.id
            """,
        merged_text,
        uuid.UUID(twin_id),
        uuid.UUID(observation_id),
        twin_text,
        observation_text,
        live_u_sources,
    )
    # None: twin or updated row vanished during the LLM window — the caller keeps the
    # updated row as a distinct observation instead of deleting it unfolded.
    return folded is not None


async def rewrite_observation(
    *,
    conn,
    fq_table: Callable[[str], str],
    observation_id: str,
    text: str,
    embedding: str | None,
    source_memory_ids: list[uuid.UUID],
    tags: list[str],
    bounds: _TemporalBounds,
) -> bool:
    """Rewrite an observation in place, widening its dates by ``bounds``. False if the row is gone."""
    search_vector_clause = _native_search_vector_update(get_config(), "$1")
    # Unlike the dedup folds this statement also runs on Oracle, where LEAST/GREATEST
    # return NULL as soon as ANY argument is NULL (PostgreSQL ignores NULL arguments).
    # The inner COALESCE covers a NULL *parameter*; the outer one covers a NULL
    # *column* — an observation with no occurred interval yet, which is precisely the
    # #3477 case. Without it Oracle would compute LEAST(NULL, <source date>) = NULL and
    # silently drop the date it was told to inherit. Keep the inner
    # ``COALESCE($n, col)`` spelled exactly like this: the Oracle driver shim keys its
    # TIMESTAMP-TZ input-size hint off that pattern (db/oracle.py::_apply_clob_input_sizes),
    # and a NULL parameter binds as VARCHAR2 (ORA-00932) without it.
    updated_rows = await conn.execute_rows_affected(
        f"""
            UPDATE {fq_table("memory_units")}
            SET text = $1,
                embedding = $2::vector,
                source_memory_ids = $3,
                proof_count = $4,
                tags = $10,
                updated_at = now(),
                event_date = COALESCE(LEAST(event_date, COALESCE($6, event_date)), $6),
                occurred_start = COALESCE(LEAST(occurred_start, COALESCE($7, occurred_start)), $7),
                occurred_end = COALESCE(GREATEST(occurred_end, COALESCE($8, occurred_end)), $8),
                mentioned_at = COALESCE(GREATEST(mentioned_at, COALESCE($9, mentioned_at)), $9){search_vector_clause}
            WHERE id = $5
            """,
        text,
        embedding,
        source_memory_ids,
        len(source_memory_ids),
        uuid.UUID(observation_id),
        bounds.event_date,
        bounds.occurred_start,
        bounds.occurred_end,
        bounds.mentioned_at,
        tags,
    )
    return updated_rows != 0


async def insert_observation(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    observation_id: uuid.UUID,
    text: str,
    embedding: str | None,
    source_memory_ids: list[uuid.UUID],
    tags: list[str],
    event_date: datetime | None,
    occurred_start: datetime | None,
    occurred_end: datetime | None,
    mentioned_at: datetime | None,
) -> str:
    """Insert a new observation as a `memory_units` row (and its Oracle source postings)."""
    # Query varies based on text search backend.
    from ...schema import _is_oracle  # noqa: PLC0415

    config = get_config()
    if config.text_search_extension == "vchord":
        # VectorChord: manually tokenize and insert search_vector
        query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, $11, $5, $6, $7, $8, $9, $10,
                        tokenize($3, 'llmlingua2')::bm25_catalog.bm25vector)
                RETURNING id
            """
    elif config.text_search_extension == "native" and not _is_oracle():
        # Native (PostgreSQL): search_vector is populated with to_tsvector()
        # using the configured native language dictionary, matching the batch
        # insert path in ops_postgresql.insert_facts_batch. On Oracle this falls
        # through to the no-search_vector branch below (Oracle maintains its text
        # index separately; to_tsvector/::regconfig is PG-only — see #3021).
        query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at, search_vector
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, $11, $5, $6, $7, $8, $9, $10,
                        to_tsvector('{config.text_search_extension_native_language}'::regconfig, COALESCE($3, '')))
                RETURNING id
            """
    else:  # pg_textsearch, pgroonga, pg_search, and Oracle: base text columns / separate index
        query = f"""
                INSERT INTO {fq_table("memory_units")} (
                    id, bank_id, text, fact_type, embedding, proof_count, source_memory_ids,
                    tags, event_date, occurred_start, occurred_end, mentioned_at
                )
                VALUES ($1, $2, $3, 'observation', $4::vector, $11, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """

    row = await conn.fetchrow(
        query,
        observation_id,
        bank_id,
        text,
        embedding,
        source_memory_ids,
        tags,
        event_date,
        occurred_start,
        occurred_end,
        mentioned_at,
        len(source_memory_ids),  # proof_count: the caller passes each live source once
    )

    # Populate observation_sources junction table (Oracle only — PG uses native array ops).
    if ops.uses_observation_sources_table and source_memory_ids:
        await conn.executemany(
            f"""
                INSERT INTO {fq_table("observation_sources")} (observation_id, source_id)
                VALUES ($1, $2)
                ON CONFLICT (observation_id, source_id) DO NOTHING
                """,
            [(observation_id, sid) for sid in dict.fromkeys(source_memory_ids)],
        )
    return str(row["id"])


async def delete_observation(*, conn, fq_table: Callable[[str], str], bank_id: str, observation_id: str) -> None:
    """Delete one observation row (the caller drops its history)."""
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2 AND fact_type = 'observation'",
        uuid.UUID(observation_id),
        bank_id,
    )
