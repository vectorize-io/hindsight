"""Curation reads: the memory list, the memory detail view, and the entity list.

These back the curation UI — the table of memories a bank holds, the detail panel
for one of them, and the entity roster beside it. They are paged and filtered
rather than ranked: nothing here scores anything, and nothing walks the corpus.

Two things separate them from the addressed reads in :mod:`reads`. They render
*view* dicts (ISO strings, joined entity names, a ``state`` discriminator) rather
than :class:`~hindsight_api.engine.memories.base.StoredMemory`, because the HTTP
layer serialises what comes back verbatim. And they read the archive as well as
the live table: curation moves an invalidated fact to `invalidated_memory_units`,
so "show me the invalidated ones" is a different table, not a different predicate.

Authentication, operation validation and audit stay with the engine methods that
call these — only the queries and their row rendering live here.
"""

from __future__ import annotations

import json
from typing import Any

from ...search.tags import build_tags_where_clause


def _entity_rows_for_units_sql(*, ops, fq_table, unit_ids_placeholder: int) -> str:
    """SQL SELECT producing ``(unit_id, entity_id, canonical_name)`` rows for
    the given unit IDs.

    Direct rows come from ``unit_entities``. Observations rarely carry
    direct rows there; their entity association lives transitively through
    their source memories (``source_memory_ids`` on PG, the
    ``observation_sources`` junction on Oracle). When an observation has
    no direct entity rows the SELECT inherits its source memories'
    entities, so the result is the same set callers would get from
    ``get_memory_unit``.

    ``unit_ids_placeholder`` is the 1-based parameter index that holds the
    ``uuid[]`` of unit IDs. The placeholder is referenced twice — both
    sides of the UNION need it — so callers should not reuse the slot.
    """
    ue = fq_table("unit_entities")
    ents = fq_table("entities")
    mu = fq_table("memory_units")
    p = unit_ids_placeholder

    direct = (
        f"SELECT ue.unit_id, e.id AS entity_id, e.canonical_name "
        f"FROM {ue} ue "
        f"JOIN {ents} e ON e.id = ue.entity_id "
        f"WHERE ue.unit_id = ANY(${p}::uuid[])"
    )

    if ops.uses_observation_sources_table:
        os_t = fq_table("observation_sources")
        inherited = (
            f"SELECT os.observation_id AS unit_id, e.id AS entity_id, e.canonical_name "
            f"FROM {os_t} os "
            f"JOIN {ue} src_ue ON src_ue.unit_id = os.source_id "
            f"JOIN {ents} e ON e.id = src_ue.entity_id "
            f"WHERE os.observation_id = ANY(${p}::uuid[]) "
            f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = os.observation_id)"
        )
    else:
        inherited = (
            f"SELECT obs.id AS unit_id, e.id AS entity_id, e.canonical_name "
            f"FROM {mu} obs "
            f"CROSS JOIN LATERAL unnest(obs.source_memory_ids) AS src_id "
            f"JOIN {ue} src_ue ON src_ue.unit_id = src_id "
            f"JOIN {ents} e ON e.id = src_ue.entity_id "
            f"WHERE obs.id = ANY(${p}::uuid[]) "
            f"AND obs.fact_type = 'observation' "
            f"AND obs.source_memory_ids IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = obs.id)"
        )

    return f"({direct}) UNION ({inherited})"


async def list_memory_units(
    *,
    conn,
    ops,
    fq_table,
    bank_id: str,
    fact_type: str | None = None,
    search_query: str | None = None,
    consolidation_state: str | None = None,
    state: str | None = None,
    document_id: str | None = None,
    tags: list[str] | None = None,
    tags_match: str = "any",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List memory units for table view with optional full-text search.

    Args:
        conn: Open database connection (the caller owns the transaction).
        ops: Dialect ops. Unused by this query; part of the interface signature.
        fq_table: Table-name resolver.
        bank_id: Filter by bank ID
        fact_type: Filter by fact type (world, experience)
        search_query: Full-text search query (searches text and context fields)
        document_id: Optional filter to a single source document.
        tags: Optional list of tag names to filter by. When omitted, no tag
            filtering is applied (except tags_match='exact', which then selects
            the untagged/global scope).
        tags_match: How to combine tags (same modes as recall): 'any' (OR,
            default) or 'all' (AND) both also include untagged units;
            'any_strict'/'all_strict' exclude untagged units; 'exact' matches
            units whose tag set equals the given tags exactly.
        state: Optional curation-state filter ('valid' or 'invalidated').
            Invalidated facts live in a separate archive table; 'invalidated'
            reads that archive. Omitted/('valid') lists live facts.
        consolidation_state: Optional filter on consolidation state. One of
            'failed' (consolidation permanently failed and awaiting recovery),
            'pending' (not yet consolidated, no failure), or
            'done' (successfully consolidated). Only applies to source memory
            types (world/experience).
        limit: Maximum number of results to return
        offset: Offset for pagination

    Returns:
        Dict with items (list of memory units) and total count
    """
    if state is not None and state not in ("valid", "invalidated"):
        raise ValueError(f"Invalid state '{state}': expected 'valid' or 'invalidated'.")
    # Invalidated facts live in a separate archive table; pick the source
    # accordingly. Default (state is None) lists live facts.
    is_archived = state == "invalidated"
    source_table = fq_table("invalidated_memory_units") if is_archived else fq_table("memory_units")

    # Build query conditions
    query_conditions = []
    query_params = []
    param_count = 0

    if bank_id:
        param_count += 1
        query_conditions.append(f"bank_id = ${param_count}")
        query_params.append(bank_id)

    if fact_type:
        param_count += 1
        query_conditions.append(f"fact_type = ${param_count}")
        query_params.append(fact_type)

    if document_id:
        param_count += 1
        query_conditions.append(f"document_id = ${param_count}")
        query_params.append(document_id)

    if search_query:
        # Full-text search on text and context fields using ILIKE
        param_count += 1
        query_conditions.append(f"(text ILIKE ${param_count} OR context ILIKE ${param_count})")
        query_params.append(f"%{search_query}%")

    if consolidation_state:
        # Named apart from `state`, which the engine method used to shadow here;
        # `is_archived` was already resolved above, so behaviour is unchanged.
        wanted = consolidation_state.lower()
        if wanted == "failed":
            query_conditions.append("consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')")
        elif wanted == "pending":
            query_conditions.append(
                "consolidated_at IS NULL AND consolidation_failed_at IS NULL AND fact_type IN ('experience', 'world')"
            )
        elif wanted == "done":
            query_conditions.append("consolidated_at IS NOT NULL AND fact_type IN ('experience', 'world')")
        else:
            raise ValueError(
                f"Invalid consolidation_state '{consolidation_state}': expected 'failed', 'pending', or 'done'."
            )

    if tags:
        tags_clause, tags_params, next_param = build_tags_where_clause(tags, param_count + 1, "", tags_match)
        if tags_clause:
            query_conditions.append(tags_clause.removeprefix("AND "))
            query_params.extend(tags_params)
            param_count = next_param - 1
    elif tags_match == "exact":
        # Exact match with no tags is the "global" scope: rows that carry no
        # tags at all. (Other match modes treat empty tags as "no filter".)
        query_conditions.append("(tags IS NULL OR tags = '{}')")

    where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total
        FROM {source_table}
        {where_clause}
    """
    count_result = await conn.fetchrow(count_query, *query_params)
    total = count_result["total"]

    # Get units with limit and offset
    param_count += 1
    limit_param = f"${param_count}"
    query_params.append(limit)

    param_count += 1
    offset_param = f"${param_count}"
    query_params.append(offset)

    # The archive carries invalidation bookkeeping; the live table doesn't.
    curation_cols = (
        "invalidation_reason, invalidated_at"
        if is_archived
        else "NULL::text AS invalidation_reason, NULL::timestamptz AS invalidated_at"
    )
    units = await conn.fetch(
        f"""
        SELECT id, text, event_date, context, fact_type, document_id,
               mentioned_at, occurred_start, occurred_end, chunk_id, proof_count,
               tags, metadata, consolidated_at, consolidation_failed_at, edited_at, {curation_cols}
        FROM {source_table}
        {where_clause}
        ORDER BY mentioned_at DESC NULLS LAST, created_at DESC
        LIMIT {limit_param} OFFSET {offset_param}
    """,
        *query_params,
    )

    # Get entity information for these units
    if units:
        unit_ids = [row["id"] for row in units]
        unit_entities = await conn.fetch(
            f"""
            SELECT ue.unit_id, e.canonical_name
            FROM {fq_table("unit_entities")} ue
            JOIN {fq_table("entities")} e ON ue.entity_id = e.id
            WHERE ue.unit_id = ANY($1::uuid[])
            ORDER BY ue.unit_id
        """,
            unit_ids,
        )
    else:
        unit_entities = []

    # Build entity mapping
    entity_map: dict[Any, list[str]] = {}
    for row in unit_entities:
        unit_id = row["unit_id"]
        entity_name = row["canonical_name"]
        if unit_id not in entity_map:
            entity_map[unit_id] = []
        entity_map[unit_id].append(entity_name)

    # Build result items
    items = []
    for row in units:
        unit_id = row["id"]
        entities = entity_map.get(unit_id, [])

        items.append(
            {
                "id": str(unit_id),
                "text": row["text"],
                "context": row["context"] if row["context"] else "",
                "date": row["event_date"].isoformat() if row["event_date"] else "",
                "fact_type": row["fact_type"],
                "document_id": row["document_id"],
                "mentioned_at": row["mentioned_at"].isoformat() if row["mentioned_at"] else None,
                "occurred_start": row["occurred_start"].isoformat() if row["occurred_start"] else None,
                "occurred_end": row["occurred_end"].isoformat() if row["occurred_end"] else None,
                "entities": ", ".join(entities) if entities else "",
                "chunk_id": row["chunk_id"] if row["chunk_id"] else None,
                "proof_count": row["proof_count"] if row["proof_count"] is not None else 1,
                "tags": list(row["tags"]) if row["tags"] else [],
                "metadata": conn.parse_json(row["metadata"]) if row["metadata"] is not None else {},
                "consolidated_at": row["consolidated_at"].isoformat() if row["consolidated_at"] else None,
                "consolidation_failed_at": (
                    row["consolidation_failed_at"].isoformat() if row["consolidation_failed_at"] else None
                ),
                "state": "invalidated" if is_archived else "valid",
                "invalidation_reason": row["invalidation_reason"],
                "invalidated_at": row["invalidated_at"].isoformat() if row["invalidated_at"] else None,
                "edited_at": row["edited_at"].isoformat() if row["edited_at"] else None,
            }
        )

    return {"items": items, "total": total, "limit": limit, "offset": offset}


async def get_memory_unit(*, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
    """
    Get a single memory unit by ID.

    Args:
        conn: Open database connection (the caller owns the transaction).
        ops: Dialect ops, for the observation→source entity inheritance shape.
        fq_table: Table-name resolver.
        bank_id: Bank ID
        unit_id: Memory unit ID (the caller validates it is a UUID)

    Returns:
        Dict with memory unit data or None if not found
    """
    # Get the memory unit (include source_memory_ids for mental models).
    # Curation moves invalidated facts to invalidated_memory_units, so fall
    # back to the archive (with its invalidation bookkeeping) on a miss.
    select_cols = (
        "id, text, context, event_date, occurred_start, occurred_end, "
        "mentioned_at, fact_type, document_id, chunk_id, tags, metadata, source_memory_ids, "
        "observation_scopes, edited_at"
    )
    row = await conn.fetchrow(
        f"SELECT {select_cols}, NULL::text AS invalidation_reason, NULL::timestamptz AS invalidated_at "
        f"FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2",
        unit_id,
        bank_id,
    )
    unit_state = "valid"
    if not row:
        row = await conn.fetchrow(
            f"SELECT {select_cols}, invalidation_reason, invalidated_at "
            f"FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
            unit_id,
            bank_id,
        )
        unit_state = "invalidated"

    if not row:
        return None

    # Get entity information. _entity_rows_for_units_sql handles the
    # observation→source_memory_ids inheritance fallback in SQL, so a
    # single query covers direct rows and inherited ones.
    entities_rows = await conn.fetch(
        _entity_rows_for_units_sql(ops=ops, fq_table=fq_table, unit_ids_placeholder=1),
        [row["id"]],
    )
    entities = [r["canonical_name"] for r in entities_rows]

    result: dict[str, Any] = {
        "id": str(row["id"]),
        "text": row["text"],
        "context": row["context"] if row["context"] else "",
        "date": row["event_date"].isoformat() if row["event_date"] else "",
        "type": row["fact_type"],
        "mentioned_at": row["mentioned_at"].isoformat() if row["mentioned_at"] else None,
        "occurred_start": row["occurred_start"].isoformat() if row["occurred_start"] else None,
        "occurred_end": row["occurred_end"].isoformat() if row["occurred_end"] else None,
        "entities": entities,
        "document_id": row["document_id"] if row["document_id"] else None,
        "chunk_id": str(row["chunk_id"]) if row["chunk_id"] else None,
        "tags": row["tags"] if row["tags"] else [],
        "metadata": conn.parse_json(row["metadata"]) if row["metadata"] is not None else {},
        "observation_scopes": (
            conn.parse_json(row["observation_scopes"]) if row["observation_scopes"] is not None else None
        ),
        "state": unit_state,
        "invalidation_reason": row["invalidation_reason"],
        "invalidated_at": row["invalidated_at"].isoformat() if row["invalidated_at"] else None,
        "edited_at": row["edited_at"].isoformat() if row["edited_at"] else None,
    }

    # For observations, include source_memory_ids
    # history is deprecated here - use GET /memories/{id}/history instead
    if row["fact_type"] == "observation":
        result["history"] = []

    if row["fact_type"] == "observation" and row["source_memory_ids"]:
        source_ids = row["source_memory_ids"]
        result["source_memory_ids"] = [str(sid) for sid in source_ids]

        # Fetch source memories
        source_rows = await conn.fetch(
            f"""
            SELECT id, text, fact_type, context, occurred_start, mentioned_at
            FROM {fq_table("memory_units")}
            WHERE id = ANY($1::uuid[])
            ORDER BY mentioned_at DESC NULLS LAST
            """,
            source_ids,
        )
        result["source_memories"] = [
            {
                "id": str(r["id"]),
                "text": r["text"],
                "type": r["fact_type"],
                "context": r["context"],
                "occurred_start": r["occurred_start"].isoformat() if r["occurred_start"] else None,
                "mentioned_at": r["mentioned_at"].isoformat() if r["mentioned_at"] else None,
            }
            for r in source_rows
        ]

    return result


async def list_entities(
    *,
    conn,
    fq_table,
    bank_id: str,
    search: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List all entities for a bank with pagination.

    Args:
        conn: Open database connection (the caller owns the transaction).
        fq_table: Table-name resolver.
        bank_id: bank IDentifier
        search: Optional case-insensitive substring match on canonical_name.
        limit: Maximum number of entities to return
        offset: Offset for pagination

    Returns:
        Dict with items, total, limit, offset
    """
    conditions = ["bank_id = $1"]
    params: list[Any] = [bank_id]
    if search:
        # Substring match, same ILIKE shape entity lookup uses elsewhere. Applied
        # to the count too, so the UI pages over the filtered set.
        params.append(f"%{search}%")
        conditions.append(f"canonical_name ILIKE ${len(params)}")
    where_clause = " AND ".join(conditions)

    # Get total count
    total_row = await conn.fetchrow(
        f"""
        SELECT COUNT(*) as total
        FROM {fq_table("entities")}
        WHERE {where_clause}
        """,
        *params,
    )
    total = total_row["total"] if total_row else 0

    # Get paginated entities
    rows = await conn.fetch(
        f"""
        SELECT id, canonical_name, mention_count, first_seen, last_seen, metadata
        FROM {fq_table("entities")}
        WHERE {where_clause}
        ORDER BY mention_count DESC, last_seen DESC, id ASC
        LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """,
        *params,
        limit,
        offset,
    )

    entities = []
    for row in rows:
        # Handle metadata - may be dict, JSON string, or None
        metadata = row["metadata"]
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        entities.append(
            {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "mention_count": row["mention_count"],
                "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
                "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
                "metadata": metadata,
            }
        )
    return {
        "items": entities,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


__all__ = ["get_memory_unit", "list_entities", "list_memory_units"]
