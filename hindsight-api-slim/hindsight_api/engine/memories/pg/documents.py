"""The document, chunk and recall-enrichment reads and writes the engine used to issue inline.

Lifted verbatim from ``memory_engine`` (``get_document``, ``delete_document``,
``update_document``, ``list_documents``, ``get_chunk``, ``list_document_chunks``,
``record_document_file``, the attachment lookups and the recall ``include_chunks`` /
``include_source_facts`` enrichment) so
:class:`~hindsight_api.engine.memories.postgres.PostgresMemories` can delegate to them
rather than the engine naming `documents`, `chunks` and `memory_units` itself.

A function that takes ``conn`` runs on the caller's connection, inside whatever
transaction it holds. One that takes ``backend`` acquires its own, exactly where the
engine used to: those are the reads a store that owns its memories answers without
Postgres, so the caller no longer acquires a connection it would not use.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from ...db_utils import acquire_with_retry
from ...search.tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause
from ...time_filter import DOCUMENT_TIME_FIELDS, build_time_clause
from ..base import (
    AttachmentRef,
    DeletedDocument,
    DocumentSourceUnits,
    DocumentTags,
    ObservationChunkIds,
    StoredMemory,
)
from . import counts

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ uploaded file


async def record_document_file(
    *,
    backend,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    storage_key: str,
    original_name: str,
    content_type: str,
) -> bool:
    """Put the uploaded-file reference on the `documents` row. ``False`` if there is no row."""
    async with acquire_with_retry(backend) as conn:
        updated = await conn.fetchval(
            f"""
            UPDATE {fq_table("documents")}
            SET file_storage_key = $3,
                file_original_name = $4,
                file_content_type = $5,
                updated_at = NOW()
            WHERE id = $1 AND bank_id = $2
            RETURNING id
            """,
            document_id,
            bank_id,
            storage_key,
            original_name,
            content_type,
        )
    return updated is not None


# ------------------------------------------------------------------ attachments


def _attachment_ids_of(value: Any) -> list[str]:
    """Read `memory_units.attachment_ids` back on either backend.

    Postgres stores it as TEXT[] and hands back a list; Oracle has no array type
    in this tree's dialect surface, so it is a JSON CLOB — the same shape `tags`
    and `observation_scopes` already take there — and arrives as a string. A
    reader that assumed one of the two worked on that backend and silently
    returned nothing on the other.
    """
    if not value:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return []
        return [str(v) for v in decoded] if isinstance(decoded, list) else []
    return [str(v) for v in value]


async def memory_attachment_refs(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[str]
) -> dict[str, AttachmentRef]:
    """Each memory's own attachment ids, read from `memory_units`; memories with none are omitted."""
    rows = await conn.fetch(
        # No `cardinality(...)` filter: it is a Postgres collection
        # function, and Oracle stores this column as a JSON CLOB, where
        # it raises ORA-00932. The rows are being fetched anyway for
        # their document_id, so the empty ones are dropped below.
        f"SELECT id::text AS id, document_id, attachment_ids FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $1 AND id = ANY($2::uuid[])",
        bank_id,
        list(unit_ids),
    )
    return {
        row["id"]: AttachmentRef(row["document_id"], ids)
        for row in rows
        if (ids := _attachment_ids_of(row["attachment_ids"]))
    }


# ------------------------------------------------------------------ recall enrichment


async def recall_observation_chunk_ids(
    *, backend, ops, fq_table: Callable[[str], str], observation_ids: list[uuid.UUID]
) -> ObservationChunkIds:
    """The chunk ids of each observation's sources, in observation-rank order, in one query."""
    async with acquire_with_retry(backend) as obs_conn:
        if ops.uses_observation_sources_table:
            obs_source_rows = await obs_conn.fetch(
                f"""
                SELECT os.observation_id AS obs_id, mu.chunk_id
                FROM {fq_table("observation_sources")} os
                JOIN {fq_table("memory_units")} mu
                  ON mu.id = os.source_id
                WHERE os.observation_id = ANY($1::uuid[])
                  AND mu.chunk_id IS NOT NULL
                ORDER BY array_position($1::uuid[], os.observation_id)
                """,
                observation_ids,
            )
        else:
            obs_source_rows = await obs_conn.fetch(
                f"""
                SELECT obs.id AS obs_id, mu.chunk_id
                FROM {fq_table("memory_units")} obs
                JOIN {fq_table("memory_units")} mu
                  ON mu.id = ANY(obs.source_memory_ids)
                WHERE obs.id = ANY($1::uuid[])
                  AND mu.chunk_id IS NOT NULL
                ORDER BY array_position($1::uuid[], obs.id)
                """,
                observation_ids,
            )
    # Rows are ordered by observation, so each observation's run is contiguous and the
    # dict's insertion order is the rows' order.
    by_obs: dict[str, list[str]] = {}
    for row in obs_source_rows:
        by_obs.setdefault(str(row["obs_id"]), []).append(row["chunk_id"])
    return ObservationChunkIds(chunk_ids_by_observation=by_obs)


async def recall_chunks(*, backend, fq_table: Callable[[str], str], chunk_ids: list[str]) -> dict[str, Any]:
    """The candidate chunks by chunk_id, in a single query.

    The asyncpg Records are returned as-is — no per-chunk ``dict`` copy — since
    ``chunk_text`` is already in the row.
    """
    async with acquire_with_retry(backend) as conn:
        chunks_rows = await conn.fetch(
            f"""
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1::text[])
            """,
            chunk_ids,
        )
    return {row["chunk_id"]: row for row in chunks_rows}


async def recall_observation_sources(
    *, conn, fq_table: Callable[[str], str], observation_ids: list[uuid.UUID]
) -> dict[str, Any]:
    """Each observation's ``source_memory_ids``, in observation-rank order.

    Reads only the two columns it needs rather than a full memory row: this is a
    recall hot path.
    """
    return {
        str(r["id"]): r["source_memory_ids"]
        for r in await conn.fetch(
            f"SELECT id, source_memory_ids FROM {fq_table('memory_units')} "
            f"WHERE id = ANY($1::uuid[]) AND fact_type = 'observation' "
            f"ORDER BY array_position($1::uuid[], id)",
            observation_ids,
        )
    }


async def recall_source_facts(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[str]
) -> dict[str, StoredMemory]:
    """The display columns of the given source facts, by id.

    Only those columns, bank-scoped, instead of the full 17-column memory row — the
    difference is measurable on this hot path.
    """
    return {
        str(r["id"]): StoredMemory(
            unit_id=str(r["id"]),
            text=r["text"],
            fact_type=r["fact_type"],
            context=r["context"],
            occurred_start=r["occurred_start"],
            occurred_end=r["occurred_end"],
            mentioned_at=r["mentioned_at"],
            document_id=r["document_id"],
            chunk_id=r["chunk_id"],
            tags=list(r["tags"] or []),
            metadata=r["metadata"],
        )
        for r in await conn.fetch(
            f"SELECT id, text, fact_type, context, occurred_start, occurred_end, "
            f"mentioned_at, document_id, chunk_id, tags, metadata "
            f"FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[]) AND bank_id = $2",
            [uuid.UUID(s) for s in unit_ids],
            bank_id,
        )
    }


# ------------------------------------------------------------------ documents


def _observations_via_source_match_sql(
    ops,
    fq_table: Callable[[str], str],
    source_column: str,
    source_placeholder: int,
    bank_placeholder: int | None,
) -> str:
    """SQL predicate matching `memory_units` rows that are observations
    whose source memories satisfy ``<source_column> = $source_placeholder``.

    Observations have no `document_id` / `chunk_id` of their own; the link
    to a source row lives in `source_memory_ids` (PG) or the
    `observation_sources` junction (Oracle).
    """
    if source_column not in ("document_id", "chunk_id"):
        raise ValueError(f"Unsupported source_column: {source_column!r}")
    if ops.uses_observation_sources_table:
        bank_clause = f" AND src.bank_id = ${bank_placeholder}" if bank_placeholder else ""
        return (
            f"id IN (SELECT os.observation_id "
            f"FROM {fq_table('observation_sources')} os "
            f"JOIN {fq_table('memory_units')} src ON src.id = os.source_id "
            f"WHERE src.{source_column} = ${source_placeholder}{bank_clause})"
        )
    bank_clause = f" AND bank_id = ${bank_placeholder}" if bank_placeholder else ""
    return (
        f"source_memory_ids && (SELECT array_agg(id) "
        f"FROM {fq_table('memory_units')} "
        f"WHERE {source_column} = ${source_placeholder}{bank_clause})"
    )


async def get_document_with_counts(*, conn, ops, fq_table: Callable[[str], str], bank_id: str, document_id: str):
    """The `documents` row with its per-fact-type memory counts, or ``None``."""
    obs_match = _observations_via_source_match_sql(
        ops, fq_table, "document_id", source_placeholder=1, bank_placeholder=2
    )
    observation_count_sql = (
        f"(SELECT COUNT(*) FROM {fq_table('memory_units')} "
        f"WHERE bank_id = $2 AND fact_type = 'observation' AND {obs_match})"
    )
    # Use a subquery for counts to avoid GROUP BY on CLOB columns
    # (Oracle cannot use CLOB types as comparison keys in GROUP BY).
    return await conn.fetchrow(
        f"""
        SELECT d.id, d.bank_id, d.original_text, d.content_hash,
               d.created_at, d.updated_at, d.tags, d.retain_params,
               COALESCE(stats.unit_count, 0) as unit_count,
               COALESCE(stats.world_count, 0) as world_count,
               COALESCE(stats.experience_count, 0) as experience_count,
               COALESCE({observation_count_sql}, 0) as observation_count
        FROM {fq_table("documents")} d
        LEFT JOIN (
            SELECT mu.document_id, mu.bank_id,
                   COUNT(mu.id) as unit_count,
                   COUNT(CASE WHEN mu.fact_type = 'world' THEN 1 END) as world_count,
                   COUNT(CASE WHEN mu.fact_type = 'experience' THEN 1 END) as experience_count
            FROM {fq_table("memory_units")} mu
            WHERE mu.document_id = $1 AND mu.bank_id = $2
            GROUP BY mu.document_id, mu.bank_id
        ) stats ON stats.document_id = d.id AND stats.bank_id = d.bank_id
        WHERE d.id = $1 AND d.bank_id = $2
        """,
        document_id,
        bank_id,
    )


async def document_source_units(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> DocumentSourceUnits:
    """The document's experience/world unit ids, and its total unit count, before a delete."""
    unit_rows = await conn.fetch(
        f"SELECT id FROM {fq_table('memory_units')} WHERE document_id = $1 AND bank_id = $2 AND fact_type IN ('experience', 'world')",
        document_id,
        bank_id,
    )
    unit_ids = [str(row["id"]) for row in unit_rows]
    units_count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE document_id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    return DocumentSourceUnits(unit_ids=unit_ids, units_count=units_count)


async def delete_document_rows(
    *, conn, ops, fq_table: Callable[[str], str], bank_id: str, document_id: str, unit_ids: list[str]
) -> DeletedDocument:
    """Delete the `documents` row (its FK cascade takes the memory units), links first."""
    # The uploaded original a file retain kept, if any. Only this row
    # knows its key, so it must be read before the row goes.
    file_storage_key = await conn.fetchval(
        f"SELECT file_storage_key FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )

    # Drop the facts' links in lock order before the cascade reaches them (#4251).
    await ops.delete_unit_links(conn, fq_table("memory_links"), bank_id, unit_ids)

    # Delete document first (cascades to memory_units and all their links).
    # Running the stale-observation sweep AFTER the delete ensures we also
    # catch observations inserted concurrently by consolidation — otherwise
    # an insert that commits between the sweep and the delete would leave an
    # orphan referencing the just-deleted source memory.
    deleted = await conn.fetchval(
        f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id",
        document_id,
        bank_id,
    )
    return DeletedDocument(deleted=bool(deleted), file_storage_key=file_storage_key)


async def current_document_tags(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> DocumentTags:
    """The tags the `documents` row carries; ``found=False`` when there is no row."""
    _doc_row = await conn.fetchrow(
        f"SELECT tags FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    if _doc_row is None:
        return DocumentTags(found=False, tags=None)
    return DocumentTags(found=True, tags=list(_doc_row["tags"] or []))


async def documents_tags(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_ids: list[str]
) -> dict[str, list[str]]:
    """document id -> its `documents.tags`, for the ids that have a row in this bank."""
    rows = await conn.fetch(
        f"SELECT id, tags FROM {fq_table('documents')} WHERE id = ANY($1) AND bank_id = $2",
        document_ids,
        bank_id,
    )
    return {row["id"]: list(row["tags"] or []) for row in rows}


async def update_document_tags(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str, tags: list[str] | None
) -> bool:
    """Stamp the `documents` row (and set its tags when given). ``False`` when there is no row."""
    set_parts: list[str] = ["updated_at = now()"]
    params: list[Any] = []
    p = 1

    if tags is not None:
        set_parts.append(f"tags = ${p}")
        params.append(tags)
        p += 1

    params.extend([document_id, bank_id])
    doc_id_found = await conn.fetchval(
        f"""
        UPDATE {fq_table("documents")}
        SET {", ".join(set_parts)}
        WHERE id = ${p} AND bank_id = ${p + 1}
        RETURNING id
        """,
        *params,
    )
    return bool(doc_id_found)


async def retag_document_memories(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    tags: list[str],
    retagged: Callable[[list[str] | None], list[str]],
    rescoped: Callable[[Any], str | None],
) -> int:
    """Retag the document's units, then delete the observations built on them and requeue
    their sources (the document's and every co-source) for re-consolidation.

    Returns how many observations were deleted.
    """
    invalidated_obs = 0
    # `tags` as well as `id`: the projection each unit must keep is read
    # here, before the blanket write below overwrites it.
    unit_rows = await conn.fetch(
        f"SELECT id, tags, fact_type, observation_scopes FROM {fq_table('memory_units')} "
        f"WHERE document_id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    _by_scopes: dict[str, list] = {}
    for _row in unit_rows:
        _scopes_json = rescoped(_row["observation_scopes"])
        if _scopes_json is not None:
            _by_scopes.setdefault(_scopes_json, []).append(_row["id"])
    for _scopes_json, _ids in _by_scopes.items():
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET observation_scopes = $1 "
            f"WHERE bank_id = $2 AND id = ANY($3::uuid[])",
            _scopes_json,
            bank_id,
            _ids,
        )
    unit_ids = [str(row["id"]) for row in unit_rows if row["fact_type"] in ("experience", "world")]

    await conn.execute(
        f"UPDATE {fq_table('memory_units')} SET tags = $1, updated_at = now() WHERE document_id = $2 AND bank_id = $3",
        tags,
        document_id,
        bank_id,
    )

    # Restore each unit's own label projection over that blanket write.
    # A follow-up rather than one combined statement so a unit created
    # concurrently still lands on the document tags exactly as before;
    # a bank with no `tag: true` group issues nothing here at all.
    # Grouped by the FINAL array `retagged` computed, not by the
    # projection alone: a unit whose label tag is also a document tag
    # would otherwise be written `[..., 'category:durable',
    # 'category:durable']`, since the two would be concatenated here
    # after `retagged` had already deduped them.
    _by_final: dict[tuple[str, ...], list] = {}
    for _row in unit_rows:
        _final = retagged(_row["tags"])
        if _final != list(tags):
            _by_final.setdefault(tuple(_final), []).append(_row["id"])
    for _final, _ids in _by_final.items():
        await conn.execute(
            f"UPDATE {fq_table('memory_units')} SET tags = $1, updated_at = now() "
            f"WHERE document_id = $2 AND bank_id = $3 AND id = ANY($4::uuid[])",
            list(_final),
            document_id,
            bank_id,
            _ids,
        )

    if unit_ids:
        unit_uuids = [uuid.UUID(uid) for uid in unit_ids]
        unit_uuid_set = {str(u) for u in unit_uuids}
        if ops.uses_observation_sources_table:
            affected_obs = await conn.fetch(
                f"""
                SELECT mu.id, mu.source_memory_ids
                FROM {fq_table("memory_units")} mu
                WHERE mu.bank_id = $1
                  AND mu.fact_type = 'observation'
                  AND EXISTS (
                      SELECT 1 FROM {fq_table("observation_sources")} os
                      WHERE os.observation_id = mu.id
                        AND os.source_id = ANY($2::uuid[])
                  )
                """,
                bank_id,
                unit_uuids,
            )
        else:
            affected_obs = await conn.fetch(
                f"""
                SELECT id, source_memory_ids
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND fact_type = 'observation'
                  AND source_memory_ids && $2::uuid[]
                """,
                bank_id,
                unit_uuids,
            )
        if affected_obs:
            obs_ids = [obs["id"] for obs in affected_obs]

            seen: set[str] = set()
            other_source_uuids: list[uuid.UUID] = []
            for obs in affected_obs:
                for src_id in obs["source_memory_ids"] or []:
                    src_str = str(src_id)
                    if src_str not in unit_uuid_set and src_str not in seen:
                        other_source_uuids.append(src_id)
                        seen.add(src_str)

            await conn.execute(
                f"DELETE FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
                obs_ids,
            )
            # Requeue the sources: bookkeeping only, so `updated_at`
            # stays put (see META_UPDATED_AT). The tag change above is
            # what stamped these rows.
            await conn.execute(
                f"""
                UPDATE {fq_table("memory_units")}
                SET consolidated_at = NULL
                WHERE id = ANY($1::uuid[])
                  AND fact_type IN ('experience', 'world')
                """,
                unit_uuids,
            )
            if other_source_uuids:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("memory_units")}
                    SET consolidated_at = NULL
                    WHERE id = ANY($1::uuid[])
                      AND fact_type IN ('experience', 'world')
                    """,
                    other_source_uuids,
                )
            invalidated_obs = len(obs_ids)
            logger.info(
                f"[OBSERVATIONS] Deleted {invalidated_obs} observations, reset "
                f"{len(unit_ids)} document source memories and "
                f"{len(other_source_uuids)} co-source memories for re-consolidation "
                f"after document update on '{document_id}' in bank {bank_id}"
            )
    return invalidated_obs


async def list_documents(
    *,
    backend,
    fq_table: Callable[[str], str],
    bank_id: str,
    search_query: str | None,
    tags: list[str] | None,
    tags_match: TagsMatch,
    tag_groups: list[TagGroup] | None,
    time_field: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """A page of the bank's `documents` rows with their memory counts — ``{items, total, limit, offset}``."""
    async with acquire_with_retry(backend) as conn:
        # Build query conditions
        query_conditions = []
        query_params = []
        param_count = 0

        param_count += 1
        query_conditions.append(f"bank_id = ${param_count}")
        query_params.append(bank_id)

        if search_query:
            # Search in document ID
            param_count += 1
            query_conditions.append(f"id ILIKE ${param_count}")
            query_params.append(f"%{search_query}%")

        built = build_tags_where_clause(tags, param_offset=param_count + 1, match=tags_match)
        tags_clause = built.sql
        tags_params = built.params
        next_param = built.next_param_offset
        query_params.extend(tags_params)
        param_count = next_param - 1  # next_param is next available; convert to last used
        if tag_groups:
            groups = build_tag_groups_where_clause(tag_groups, param_count + 1)
            query_conditions.append(groups.sql.removeprefix("AND "))
            query_params.extend(groups.params)
            param_count = groups.next_param_offset - 1

        window = build_time_clause(
            time_field=time_field,
            start_date=start_date,
            end_date=end_date,
            allowed=DOCUMENT_TIME_FIELDS,
            default_field="updated_at",
            param_offset=param_count + 1,
        )
        query_conditions.extend(window.conditions)
        query_params.extend(window.params)
        param_count = window.next_param_offset - 1

        where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""
        if tags_clause:
            # tags_clause starts with "AND", append after WHERE conditions
            where_clause = where_clause + " " + tags_clause if where_clause else "WHERE " + tags_clause[4:].lstrip()

        # Get total count
        count_query = f"""
            SELECT COUNT(*) as total
            FROM {fq_table("documents")}
            {where_clause}
        """
        count_result = await conn.fetchrow(count_query, *query_params)
        total = count_result["total"]

        # Get documents with limit and offset (without original_text for performance)
        param_count += 1
        limit_param = f"${param_count}"
        query_params.append(limit)

        param_count += 1
        offset_param = f"${param_count}"
        query_params.append(offset)

        documents = await conn.fetch(
            f"""
            SELECT
                id,
                bank_id,
                content_hash,
                created_at,
                updated_at,
                LENGTH(original_text) as text_length,
                retain_params,
                tags
            FROM {fq_table("documents")}
            {where_clause}
            ORDER BY {window.order_by or "updated_at DESC, created_at DESC, id"}
            LIMIT {limit_param} OFFSET {offset_param}
        """,
            *query_params,
        )

        # Memory count per document.
        doc_ids = [row["id"] for row in documents]
        per_doc = (
            await counts.document_memory_counts(conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=doc_ids)
            if doc_ids
            else {}
        )
        count_map = {(doc_id, bank_id): count for doc_id, count in per_doc.items()}

        # Build result items
        items = []
        for row in documents:
            doc_id = row["id"]
            bank_id_val = row["bank_id"]
            unit_count = count_map.get((doc_id, bank_id_val), 0)

            retain_params_val = conn.parse_json(row["retain_params"])

            # document_metadata is sourced from retain_params.metadata
            document_metadata = retain_params_val.get("metadata") if retain_params_val else None

            items.append(
                {
                    "id": doc_id,
                    "bank_id": bank_id_val,
                    "content_hash": row["content_hash"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
                    "text_length": row["text_length"] or 0,
                    "memory_unit_count": unit_count,
                    "retain_params": retain_params_val or None,
                    "document_metadata": document_metadata or None,
                    "tags": row["tags"] if row["tags"] else [],
                }
            )

        return {"items": items, "total": total, "limit": limit, "offset": offset}


# ------------------------------------------------------------------ chunks


async def get_chunk_row(*, conn, fq_table: Callable[[str], str], chunk_id: str) -> Mapping[str, Any] | None:
    """One `chunks` row by id, or ``None``."""
    return await conn.fetchrow(
        f"""
        SELECT
            chunk_id,
            document_id,
            bank_id,
            chunk_index,
            chunk_text,
            created_at
        FROM {fq_table("chunks")}
        WHERE chunk_id = $1
    """,
        chunk_id,
    )


async def list_document_chunks(
    *, backend, fq_table: Callable[[str], str], bank_id: str, document_id: str, limit: int, offset: int
) -> dict[str, Any] | None:
    """A page of the document's `chunks` rows by index — ``{items, total, limit, offset}`` — or
    ``None`` when there is no `documents` row."""
    async with acquire_with_retry(backend) as conn:
        # Verify document exists
        doc = await conn.fetchrow(
            f"SELECT id FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
        if not doc:
            return None

        count_result = await conn.fetchrow(
            f"""
            SELECT COUNT(*) as total
            FROM {fq_table("chunks")}
            WHERE document_id = $1 AND bank_id = $2
            """,
            document_id,
            bank_id,
        )
        total = count_result["total"]

        chunks = await conn.fetch(
            f"""
            SELECT chunk_id, document_id, bank_id, chunk_index, chunk_text, created_at
            FROM {fq_table("chunks")}
            WHERE document_id = $1 AND bank_id = $2
            ORDER BY chunk_index ASC
            LIMIT $3 OFFSET $4
            """,
            document_id,
            bank_id,
            limit,
            offset,
        )

        items = [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "bank_id": row["bank_id"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else "",
            }
            for row in chunks
        ]

        return {"items": items, "total": total, "limit": limit, "offset": offset}
