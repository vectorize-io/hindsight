"""The engine's curation and bank-admin SQL: deletes, requeues, history and entity lookups.

These back :class:`~hindsight_api.engine.memory_engine.MemoryEngine` methods that used to
run this SQL inline — deleting one or many memories, clearing or deleting a bank, clearing
observations, retrying failed consolidation, an observation's history, the entity graph and
the entity detail view. The SQL is lifted verbatim from those methods; only the connection,
the dialect ``ops`` and the ``fq_table`` resolver are now parameters. Authentication, the
transaction and the post-commit side effects stay with the engine.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...search.tags import TagsMatch, build_tag_filter_clause, tag_filter_active
from ..base import BankContentCounts, MemoryLocation, StoredMemory, TypedMemoryScope
from .curation import visible_entity_stats_sql

# Ids per DELETE in the bulk delete, so one statement's parameter array stays bounded.
_DELETE_CHUNK_SIZE = 10_000


async def locate_memory(*, conn, fq_table: Callable[[str], str], unit_id: str) -> MemoryLocation | None:
    """The bank and fact_type of one memory, found by its (globally unique) id."""
    row = await conn.fetchrow(
        f"SELECT bank_id, fact_type FROM {fq_table('memory_units')} WHERE id = $1",
        unit_id,
    )
    if not row:
        return None
    return MemoryLocation(unit_id=unit_id, bank_id=row["bank_id"], fact_type=row["fact_type"])


async def locate_memories(*, conn, fq_table: Callable[[str], str], unit_ids: list[str]) -> list[MemoryLocation]:
    """The bank and fact_type of each of ``unit_ids`` that exists, in one round trip."""
    rows = await conn.fetch(
        f"SELECT id, bank_id, fact_type FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
        unit_ids,
    )
    return [MemoryLocation(unit_id=str(row["id"]), bank_id=row["bank_id"], fact_type=row["fact_type"]) for row in rows]


async def delete_memory(*, conn, ops, fq_table: Callable[[str], str], bank_id: str | None, unit_id: str) -> str | None:
    """Delete one memory (its links first, in lock order); the deleted id, or None."""
    # Links in lock order before the cascade reaches them (see delete_unit_links).
    if bank_id:
        await ops.delete_unit_links(conn, fq_table("memory_links"), bank_id, [unit_id])
    deleted = await conn.fetchval(f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 RETURNING id", unit_id)
    return str(deleted) if deleted is not None else None


async def delete_memories(*, conn, ops, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[str]) -> int:
    """Delete one bank's ``unit_ids`` (links first, then chunked); how many rows went."""
    # Links in lock order before the cascade reaches them (see delete_unit_links).
    await ops.delete_unit_links(conn, fq_table("memory_links"), bank_id, unit_ids)

    # Chunked delete. Cascade handles unit_entities / memory_links / observation history via FK.
    deleted = 0
    for i in range(0, len(unit_ids), _DELETE_CHUNK_SIZE):
        chunk = unit_ids[i : i + _DELETE_CHUNK_SIZE]
        tag = await conn.execute(
            f"DELETE FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
            chunk,
        )
        # asyncpg tag: "DELETE N"
        parts = tag.split()
        if len(parts) >= 2:
            try:
                deleted += int(parts[-1])
            except ValueError:
                pass
    return deleted


async def bank_memories_of_type(
    *, conn, fq_table: Callable[[str], str], bank_id: str, fact_type: str
) -> TypedMemoryScope:
    """The ids (source types only) and the count of a bank's memories of one fact_type."""
    unit_ids: list[str] = []
    if fact_type in ("experience", "world"):
        unit_id_rows = await conn.fetch(
            f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
            bank_id,
            fact_type,
        )
        unit_ids = [str(row["id"]) for row in unit_id_rows]
    units_count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
        bank_id,
        fact_type,
    )
    return TypedMemoryScope(unit_ids=unit_ids, count=units_count)


async def delete_bank_memories_of_type(
    *, conn, ops, fq_table: Callable[[str], str], bank_id: str, fact_type: str, unit_ids: list[str]
) -> None:
    """Delete a bank's live and archived memories of one fact_type (links first)."""
    # Links in lock order before the cascade reaches them (see delete_unit_links).
    await ops.delete_unit_links(conn, fq_table("memory_links"), bank_id, unit_ids)
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
        bank_id,
        fact_type,
    )
    # Curation archive holds invalidated facts of the same types.
    await conn.execute(
        f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1 AND fact_type = $2",
        bank_id,
        fact_type,
    )


async def count_bank_contents(*, conn, fq_table: Callable[[str], str], bank_id: str) -> BankContentCounts:
    """How many memories, entities and documents a bank holds."""
    units_count = await conn.fetchval(f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)
    entities_count = await conn.fetchval(f"SELECT COUNT(*) FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
    documents_count = await conn.fetchval(f"SELECT COUNT(*) FROM {fq_table('documents')} WHERE bank_id = $1", bank_id)
    return BankContentCounts(memory_units=units_count, entities=entities_count, documents=documents_count)


async def purge_bank_rows(*, conn, fq_table: Callable[[str], str], bank_id: str) -> list[str]:
    """Delete every row a bank holds in these tables; the legacy file keys, read before they go.

    Files written before keys carried the tenant sit outside the bank's prefix, so the
    caller's post-commit sweep cannot find them: only these rows know their keys.
    """
    # ponytail: one unbatched list; only pre-prefix banks have any.
    legacy_files = [
        row["storage_key"]
        for row in await conn.fetch(
            f"SELECT storage_key FROM {fq_table('attachments')} "
            f"WHERE bank_id = $1 AND storage_key NOT LIKE 'tenants/%' "
            f"UNION ALL SELECT file_storage_key FROM {fq_table('documents')} "
            f"WHERE bank_id = $1 AND file_storage_key IS NOT NULL "
            f"AND file_storage_key NOT LIKE 'tenants/%'",
            bank_id,
        )
    ]

    # Delete documents (cascades to chunks)
    await conn.execute(f"DELETE FROM {fq_table('documents')} WHERE bank_id = $1", bank_id)
    # Attachments hang off the bank, not a document, so clearing a bank
    # that stays would otherwise keep every one of them.
    await conn.execute(f"DELETE FROM {fq_table('attachments')} WHERE bank_id = $1", bank_id)

    # Delete memory units (cascades to unit_entities, memory_links)
    await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

    # Observation history no longer cascades from memory_units (that FK was
    # dropped so history can be recorded for observations kept outside SQL), so
    # clear it by bank explicitly — otherwise every snapshot outlives the bank.
    await conn.execute(f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1", bank_id)

    # Curation archive (rows with NULL document_id aren't covered by
    # the documents cascade, so clear by bank explicitly).
    await conn.execute(f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1", bank_id)

    # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
    await conn.execute(f"DELETE FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
    return legacy_files


async def clear_observations_and_requeue(*, conn, fq_table: Callable[[str], str], bank_id: str) -> int:
    """Delete a bank's observations and requeue its sources; how many observations went."""
    # Count observations before deletion
    count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
        bank_id,
    )

    # Delete all observations
    await conn.execute(
        f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
        bank_id,
    )

    # Reset consolidated_at on source memories so they get re-consolidated.
    # Bookkeeping only: `updated_at` stays put (see META_UPDATED_AT).
    await conn.execute(
        f"UPDATE {fq_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND fact_type IN ('experience', 'world')",
        bank_id,
    )
    return count


async def requeue_failed_consolidation(*, conn, fq_table: Callable[[str], str], bank_id: str) -> int:
    """Clear the failure (and consolidated) marker on a bank's failed sources; how many."""
    count = await conn.fetchval(
        f"""
        SELECT COUNT(*) FROM {fq_table("memory_units")}
        WHERE bank_id = $1
          AND consolidation_failed_at IS NOT NULL
          AND fact_type IN ('experience', 'world')
        """,
        bank_id,
    )
    # Bookkeeping only: `updated_at` stays put (see META_UPDATED_AT).
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET consolidation_failed_at = NULL, consolidated_at = NULL
        WHERE bank_id = $1
          AND consolidation_failed_at IS NOT NULL
          AND fact_type IN ('experience', 'world')
        """,
        bank_id,
    )
    return count


async def requeue_source_memory(*, conn, fq_table: Callable[[str], str], bank_id: str, unit_id: uuid.UUID) -> None:
    """Clear one source memory's consolidated marker. Bookkeeping: `updated_at` stays put."""
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET consolidated_at = NULL
        WHERE id = $1
          AND bank_id = $2
          AND fact_type IN ('experience', 'world')
        """,
        unit_id,
        bank_id,
    )


async def entity_names_by_id(*, conn, fq_table: Callable[[str], str], bank_id: str, entity_ids: list[str]) -> list[str]:
    """Canonical names of ``entity_ids`` in this bank, ordered by entity id."""
    if not entity_ids:
        return []
    rows = await conn.fetch(
        f"SELECT canonical_name FROM {fq_table('entities')} WHERE id = ANY($1::uuid[]) AND bank_id = $2 ORDER BY id",
        entity_ids,
        bank_id,
    )
    return [r["canonical_name"] for r in rows]


async def observation_head(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_id: uuid.UUID
) -> MemoryLocation | None:
    """One memory's fact_type and current source ids, for its history view."""
    row = await conn.fetchrow(
        f"""
        SELECT fact_type, source_memory_ids
        FROM {fq_table("memory_units")}
        WHERE id = $1 AND bank_id = $2
        """,
        unit_id,
        bank_id,
    )
    if not row:
        return None
    return MemoryLocation(
        unit_id=str(unit_id),
        bank_id=bank_id,
        fact_type=row["fact_type"],
        source_memory_ids=[str(sid) for sid in (row["source_memory_ids"] or [])],
    )


async def source_fact_summaries(
    *, conn, fq_table: Callable[[str], str], unit_ids: list[uuid.UUID]
) -> list[StoredMemory]:
    """Text, fact_type and context of each of ``unit_ids`` that exists, in one query."""
    rows = await conn.fetch(
        f"""
        SELECT id, text, fact_type, context
        FROM {fq_table("memory_units")}
        WHERE id = ANY($1::uuid[])
        """,
        unit_ids,
    )
    return [
        StoredMemory(unit_id=str(r["id"]), text=r["text"], fact_type=r["fact_type"], context=r["context"]) for r in rows
    ]


@dataclass
class _EntityNode:
    id: str
    label: str
    mention_count: int


async def _tag_filtered_entity_graph_edges(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    limit: int,
    min_count: int,
    tags: list[str] | None,
    tags_match: TagsMatch,
    tag_groups: list | None,
) -> list[Any]:
    """Co-occurrence edges over the tag-matching memories only, in the row shape of the materialized query.

    ``entity_cooccurrences`` counts every memory in the bank and carries no tags,
    so it is recomputed here: one pair per two entities named by the same matching
    memory, the node counts from the same memories. Costs a self-join over the
    bank's postings, which is why only a tag-filtered read pays it.

    The tag clause filters an unaliased subquery for the reason given on
    :func:`visible_entity_stats_sql` (the Oracle rewriter needs a bare column).
    """
    built = build_tag_filter_clause(tags, tags_match, tag_groups, param_offset=2)
    params: list[Any] = [bank_id, *built.params, min_count, limit]
    min_count_param = len(params) - 1
    limit_param = len(params)
    return await conn.fetch(
        f"""
        WITH ve AS (
            SELECT ue.unit_id, ue.entity_id, vu.seen_at
            FROM {fq_table("unit_entities")} ue
            JOIN (
                SELECT id, COALESCE(occurred_start, mentioned_at, event_date) AS seen_at
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 {built.sql}
            ) vu ON vu.id = ue.unit_id
        ),
        pairs AS (
            -- One row per unordered pair, smaller id first, like the materialized table.
            SELECT a.entity_id AS entity_id_1,
                   b.entity_id AS entity_id_2,
                   COUNT(*) AS cooccurrence_count,
                   MAX(a.seen_at) AS last_cooccurred
            FROM ve a
            JOIN ve b ON b.unit_id = a.unit_id AND a.entity_id < b.entity_id
            GROUP BY a.entity_id, b.entity_id
            HAVING COUNT(*) >= ${min_count_param}
        ),
        stats AS (
            SELECT entity_id, COUNT(*) AS mention_count FROM ve GROUP BY entity_id
        )
        SELECT p.entity_id_1,
               p.entity_id_2,
               p.cooccurrence_count,
               p.last_cooccurred,
               e1.canonical_name AS name_1,
               s1.mention_count  AS mention_count_1,
               e2.canonical_name AS name_2,
               s2.mention_count  AS mention_count_2
        FROM pairs p
        JOIN {fq_table("entities")} e1 ON e1.id = p.entity_id_1
        JOIN {fq_table("entities")} e2 ON e2.id = p.entity_id_2
        JOIN stats s1 ON s1.entity_id = p.entity_id_1
        JOIN stats s2 ON s2.entity_id = p.entity_id_2
        WHERE e1.bank_id = $1
          AND e2.bank_id = $1
        ORDER BY p.cooccurrence_count DESC, p.last_cooccurred DESC
        LIMIT ${limit_param}
        """,
        *params,
    )


async def entity_graph(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    limit: int,
    min_count: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list | None = None,
) -> dict[str, Any]:
    """The entity co-occurrence graph from ``entity_cooccurrences``, strongest edges first.

    With a tag filter the materialized table cannot be used — it has no tags — so
    edges and node mention counts are recomputed from the matching memories.
    Keeping the stored edges and only dropping hidden nodes would still leak how
    often visible entities appear together out of scope (#5031).
    """
    if tag_filter_active(tags, tags_match, tag_groups):
        edge_rows = await _tag_filtered_entity_graph_edges(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            limit=limit,
            min_count=min_count,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )
        return _render_entity_graph(edge_rows, limit)
    edge_rows = await conn.fetch(
        f"""
        SELECT ec.entity_id_1,
               ec.entity_id_2,
               ec.cooccurrence_count,
               ec.last_cooccurred,
               e1.canonical_name AS name_1,
               e1.mention_count  AS mention_count_1,
               e2.canonical_name AS name_2,
               e2.mention_count  AS mention_count_2
        FROM {fq_table("entity_cooccurrences")} ec
        JOIN {fq_table("entities")} e1 ON e1.id = ec.entity_id_1
        JOIN {fq_table("entities")} e2 ON e2.id = ec.entity_id_2
        WHERE e1.bank_id = $1
          AND e2.bank_id = $1
          AND ec.cooccurrence_count >= $2
        ORDER BY ec.cooccurrence_count DESC, ec.last_cooccurred DESC
        LIMIT $3
        """,
        bank_id,
        min_count,
        limit,
    )
    return _render_entity_graph(edge_rows, limit)


def _render_entity_graph(edge_rows: list[Any], limit: int) -> dict[str, Any]:
    """Nodes and edges for the graph view; the stored and the tag-filtered query share the row shape."""
    nodes_by_id: dict[str, _EntityNode] = {}
    edges: list[dict[str, Any]] = []
    for row in edge_rows:
        for eid, name, mentions in (
            (row["entity_id_1"], row["name_1"], row["mention_count_1"]),
            (row["entity_id_2"], row["name_2"], row["mention_count_2"]),
        ):
            key = str(eid)
            if key not in nodes_by_id:
                nodes_by_id[key] = _EntityNode(id=key, label=name, mention_count=mentions or 0)

        from_id = str(row["entity_id_1"])
        to_id = str(row["entity_id_2"])
        count = row["cooccurrence_count"]
        edges.append(
            {
                "data": {
                    "id": f"{from_id}-{to_id}",
                    "source": from_id,
                    "target": to_id,
                    "linkType": "cooccurrence",
                    "weight": count,
                    "color": "#ffd700",
                    "lineStyle": "solid",
                    "lastCooccurred": row["last_cooccurred"].isoformat() if row["last_cooccurred"] else None,
                }
            }
        )

    nodes = [
        {
            "data": {
                "id": n.id,
                "label": n.label,
                "mentionCount": n.mention_count,
                "color": "#42a5f5" if n.mention_count > 1 else "#90caf9",
            }
        }
        for n in nodes_by_id.values()
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_entities": len(nodes),
        "total_edges": len(edges),
        "limit": limit,
    }


async def count_bank_documents(*, conn, fq_table: Callable[[str], str], bank_id: str) -> int:
    """How many documents a bank holds."""
    doc_count_row = await conn.fetchrow(
        f"SELECT COUNT(*) as count FROM {fq_table('documents')} WHERE bank_id = $1",
        bank_id,
    )
    return doc_count_row["count"] if doc_count_row else 0


async def get_entity_detail(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    entity_id: uuid.UUID,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list | None = None,
) -> dict[str, Any] | None:
    """One entity's registry row, rendered for the entity detail view; None if absent.

    With a tag filter the counts and dates come from the matching memories, and an
    entity no matching memory mentions is None — the same answer as an unknown id,
    so a scoped reader cannot tell it exists elsewhere (#5031).
    """
    if tag_filter_active(tags, tags_match, tag_groups):
        built = build_tag_filter_clause(tags, tags_match, tag_groups, param_offset=2)
        # Inner join on the stats: no matching memory, no row.
        entity_row = await conn.fetchrow(
            f"""
            SELECT e.id, e.canonical_name, s.mention_count, s.first_seen, s.last_seen, e.metadata
            FROM {fq_table("entities")} e
            JOIN ({visible_entity_stats_sql(fq_table, built.sql)}) s ON s.entity_id = e.id
            WHERE e.bank_id = $1 AND e.id = ${len(built.params) + 2}
            """,
            bank_id,
            *built.params,
            entity_id,
        )
    else:
        entity_row = await conn.fetchrow(
            f"""
            SELECT id, canonical_name, mention_count, first_seen, last_seen, metadata
            FROM {fq_table("entities")}
            WHERE bank_id = $1 AND id = $2
            """,
            bank_id,
            entity_id,
        )
    if not entity_row:
        return None
    return {
        "id": str(entity_row["id"]),
        "canonical_name": entity_row["canonical_name"],
        "mention_count": entity_row["mention_count"],
        "first_seen": entity_row["first_seen"].isoformat() if entity_row["first_seen"] else None,
        "last_seen": entity_row["last_seen"].isoformat() if entity_row["last_seen"] else None,
        "metadata": entity_row["metadata"] or {},
        "observations": [],
    }
