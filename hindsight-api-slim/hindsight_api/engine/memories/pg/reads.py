"""Addressed reads over `memory_units`: get, scan, count, tags, consolidation state.

Not retrieval — nothing here ranks. These are the queries behind the curation
detail view, export, the bank-stats panel and the consolidation queue, lifted out
of the call sites that used to issue them inline (``memory_engine``,
``transfer/export``, ``consolidation/consolidator``) so
:class:`~hindsight_api.engine.memories.postgres.PostgresMemories` can delegate
rather than embed SQL.

Every function takes the live connection and Hindsight's ``fq_table`` resolver, so
each one runs inside whatever transaction the caller already holds; none of them
acquires a connection of its own.

**Cursor semantics.** ``scan_memories``'s ``page_token`` is opaque to callers, and
for Postgres it is simply a *numeric offset rendered as a decimal string* against
the scan's fixed ``ORDER BY created_at, id``. An empty token means "start at the
beginning", and an empty token comes back once the walk is exhausted (i.e. the
final short page). An offset cursor is a position rather than a snapshot — exactly
the guarantee :class:`~hindsight_api.engine.memories.base.ScanPage` documents:
rows written or deleted mid-walk can shift later pages, so a scan is
eventually-complete browsing rather than a consistent iterator. ``skip`` is applied
*on top of* the decoded cursor, so a caller that pages with both should pass
``skip`` only on the first call — the returned token already accounts for it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ...search.tags import (
    build_tag_groups_where_clause,
    build_tags_where_clause,
    build_tags_where_clause_simple,
)
from ..base import ScanPage, StoredMemory

# The `memory_units` projection every read here shares. Superset of the by-id
# SELECT the recall source-facts path used (text/fact_type/context/timestamps/
# document_id/chunk_id/tags/metadata), plus the observation bookkeeping columns
# `StoredMemory` carries: source_memory_ids and consolidated_at.
_MEMORY_COLUMNS = """
    id, text, fact_type, context, document_id, chunk_id, tags, metadata,
    proof_count, event_date, occurred_start, occurred_end, mentioned_at,
    created_at, source_memory_ids, consolidated_at, observation_scopes
"""

# The scan's order. Fixed (created_at, id) like the export loader's, because an
# offset cursor is only meaningful against a total order.
_SCAN_ORDER = "ORDER BY created_at, id"


def _as_json(value: Any) -> Any:
    """Coerce an asyncpg JSONB column (str or already-decoded) to a Python object.

    Connections differ in whether a JSONB codec is registered, so the column
    arrives either as text or as the decoded object.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # A valid scalar such as `"combined"` arrives already decoded on
            # connections that do register a decoder.
            return value
    return value


def _as_uuids(unit_ids: list[Any]) -> list[uuid.UUID]:
    """Unit ids as UUIDs, dropping anything unparseable.

    A malformed id is treated the same way a deleted one is — simply absent from
    the result — rather than failing the whole read.
    """
    out: list[uuid.UUID] = []
    for unit_id in unit_ids or []:
        if isinstance(unit_id, uuid.UUID):
            out.append(unit_id)
            continue
        try:
            out.append(uuid.UUID(str(unit_id)))
        except (ValueError, AttributeError, TypeError):
            continue
    return out


def _column(row: Any, name: str, default: Any = None) -> Any:
    """One column of an asyncpg Record, tolerating a narrower projection."""
    try:
        return row[name]
    except (KeyError, IndexError):
        return default


def _stored_from_row(row: Any) -> StoredMemory:
    """Map a `memory_units` row onto :class:`StoredMemory`.

    Shared by every read in this module so the row → dataclass mapping exists
    once. ``entity_ids`` stays empty: the unit→entity posting lives in
    `unit_entities` and is served by ``entities_for_units``, not by a join here.
    """
    source_ids = _column(row, "source_memory_ids") or []
    return StoredMemory(
        unit_id=str(row["id"]),
        text=row["text"],
        fact_type=row["fact_type"],
        context=_column(row, "context"),
        document_id=_column(row, "document_id"),
        chunk_id=str(_column(row, "chunk_id")) if _column(row, "chunk_id") else None,
        tags=list(_column(row, "tags") or []),
        metadata=_as_json(_column(row, "metadata")),
        proof_count=_column(row, "proof_count") or 1,
        event_date=_column(row, "event_date"),
        occurred_start=_column(row, "occurred_start"),
        occurred_end=_column(row, "occurred_end"),
        mentioned_at=_column(row, "mentioned_at"),
        created_at=_column(row, "created_at"),
        source_memory_ids=[str(sid) for sid in source_ids],
        consolidated_at=_column(row, "consolidated_at"),
        # Consolidation routes a candidate by its scopes, so this has to survive
        # the trip through the store rather than being re-queried per memory.
        observation_scopes=_as_json(_column(row, "observation_scopes")),
    )


def _decode_page_token(page_token: str) -> int:
    """Decode the offset cursor. Empty, malformed or negative all mean "start"."""
    if not page_token:
        return 0
    try:
        offset = int(page_token)
    except (TypeError, ValueError):
        return 0
    return offset if offset > 0 else 0


async def get_memories(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[str]
) -> list[StoredMemory]:
    """Fetch memories by id. Missing or deleted ids are simply absent."""
    ids = _as_uuids(unit_ids)
    if not ids:
        return []
    rows = await conn.fetch(
        f"""
        SELECT {_MEMORY_COLUMNS}
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND id = ANY($2::uuid[])
        """,
        bank_id,
        ids,
    )
    return [_stored_from_row(row) for row in rows]


async def _semantic_edges(
    *, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[uuid.UUID]
) -> dict[str, list[tuple[str, float]]]:
    """Derived kNN edges for ``unit_ids``, keyed by unit id.

    Walked in both directions, like the graph arm's semantic expansion: a
    `memory_links` row is written once, so a unit's neighbourhood is the union of
    the edges leaving it and those arriving at it.
    """
    if not unit_ids:
        return {}
    rows = await conn.fetch(
        f"""
        SELECT from_unit_id AS unit_id, to_unit_id AS target_id, weight
        FROM {fq_table("memory_links")}
        WHERE bank_id = $1 AND link_type = 'semantic' AND from_unit_id = ANY($2::uuid[])
        UNION ALL
        SELECT to_unit_id AS unit_id, from_unit_id AS target_id, weight
        FROM {fq_table("memory_links")}
        WHERE bank_id = $1 AND link_type = 'semantic' AND to_unit_id = ANY($2::uuid[])
        """,
        bank_id,
        unit_ids,
    )
    edges: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        edges.setdefault(str(row["unit_id"]), []).append((str(row["target_id"]), float(row["weight"] or 0.0)))
    return edges


async def scan_memories(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    fact_types: list[str] | None = None,
    limit: int = 100,
    page_token: str = "",
    tags: list[str] | None = None,
    tags_match: str = "any",
    document_id: str | None = None,
    metadata_equals: dict[str, str] | None = None,
    skip: int = 0,
    include_edges: bool = False,
) -> ScanPage:
    """Page through stored memories. A full walk — for browsing and export only.

    See the module docstring for the ``page_token`` (offset) cursor semantics.
    """
    if limit is None or limit <= 0:
        return ScanPage()

    where: list[str] = ["bank_id = $1"]
    params: list[Any] = [bank_id]

    if fact_types:
        params.append(list(fact_types))
        where.append(f"fact_type = ANY(${len(params)})")

    if document_id is not None:
        # A real column here, which is why it is not folded into
        # `metadata_equals`: only a store without the column keeps it in the bag.
        params.append(document_id)
        where.append(f"document_id = ${len(params)}")

    if metadata_equals:
        # str→str equality across every key, which is exactly JSONB containment.
        params.append(json.dumps(metadata_equals))
        where.append(f"metadata @> ${len(params)}::jsonb")

    # The tags clause owns its own `AND` prefix and, per the helper's contract,
    # only consumes a bind param when `tags` is non-empty (match="exact" with no
    # tags is the untagged/global scope and needs none).
    tags_clause = build_tags_where_clause_simple(tags, len(params) + 1, match=tags_match)
    if tags:
        params.append(list(tags))

    offset = _decode_page_token(page_token) + max(int(skip or 0), 0)
    params.append(limit)
    limit_idx = len(params)
    params.append(offset)
    offset_idx = len(params)

    rows = await conn.fetch(
        f"""
        SELECT {_MEMORY_COLUMNS}
        FROM {fq_table("memory_units")}
        WHERE {" AND ".join(where)} {tags_clause}
        {_SCAN_ORDER}
        LIMIT ${limit_idx} OFFSET ${offset_idx}
        """,
        *params,
    )

    memories = [_stored_from_row(row) for row in rows]
    if include_edges and memories:
        edges = await _semantic_edges(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=_as_uuids([m.unit_id for m in memories])
        )
        for memory in memories:
            memory.semantic_edges = edges.get(memory.unit_id, [])

    # A short page means the walk is exhausted, so the cursor goes empty.
    next_token = str(offset + len(rows)) if len(rows) == limit else ""
    return ScanPage(memories=memories, next_page_token=next_token)


async def count_memories(*, conn, fq_table: Callable[[str], str], bank_id: str) -> dict[str, int]:
    """Live memory count per fact_type. The bank-stats node counts."""
    rows = await conn.fetch(
        f"""
        SELECT fact_type, COUNT(*) as count
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1
        GROUP BY fact_type
        """,
        bank_id,
    )
    return {row["fact_type"]: int(row["count"]) for row in rows}


async def list_tags(*, conn, fq_table: Callable[[str], str], bank_id: str) -> dict[str, int]:
    """Distinct tags in a bank and how many live memories carry each.

    The engine's generic tag histogram builds its fragments from
    ``ops.build_tag_listing_parts`` so it can serve any table on any dialect. This
    module's signature carries no ``ops``, and reaching one off the connection
    would be a back door into the dialect layer for a query that is already
    Postgres-specific by virtue of living under ``pg/`` — so the Postgres
    fragments (``unnest(tags)`` + the non-empty guard) are inlined verbatim here.
    Unpaged on purpose: the interface returns the whole histogram.
    """
    rows = await conn.fetch(
        f"""
        SELECT tag, COUNT(*) as count
        FROM {fq_table("memory_units")}, unnest(tags) AS tag
        WHERE bank_id = $1 AND tags IS NOT NULL AND tags != '{{}}'
        GROUP BY tag
        ORDER BY count DESC, tag ASC
        """,
        bank_id,
    )
    return {row["tag"]: int(row["count"]) for row in rows}


async def find_unconsolidated(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    fact_types: list[str],
    limit: int,
    scope_tags: list[str] | None = None,
) -> list[StoredMemory]:
    """Memories not yet folded into an observation, oldest first.

    The consolidator's candidate query: never consolidated, never *failed* to
    consolidate (a memory the LLM could not handle must not be retried forever),
    ordered by ``created_at`` so the queue drains in arrival order. ``scope_tags``
    is the same ``tags @> scope`` containment the job's scope filter uses — the
    job ORs several scopes together; one scope is passed here.
    """
    where = [
        "bank_id = $1",
        "consolidated_at IS NULL",
        "consolidation_failed_at IS NULL",
    ]
    params: list[Any] = [bank_id]
    if fact_types:
        params.append(list(fact_types))
        where.append(f"fact_type = ANY(${len(params)})")
    if scope_tags:
        params.append(list(scope_tags))
        where.append(f"tags @> ${len(params)}::varchar[]")
    params.append(limit)

    rows = await conn.fetch(
        f"""
        SELECT {_MEMORY_COLUMNS}
        FROM {fq_table("memory_units")}
        WHERE {" AND ".join(where)}
        ORDER BY created_at ASC
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [_stored_from_row(row) for row in rows]


async def mark_consolidated(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    unit_ids: list[str],
    when: datetime | None,
    failed: bool = False,
) -> None:
    """Stamp (or clear, with ``when=None``) the consolidated marker on sources.

    ``failed`` writes ``consolidation_failed_at`` instead of ``consolidated_at``,
    which is what keeps a memory the LLM could not consolidate out of the queue.

    ``when=None`` clears the column rather than stamping it — that is how a source
    is requeued once the observation built on it is deleted. The clear keeps the
    ``fact_type IN ('experience', 'world')`` guard the requeue sites carry:
    observations are never themselves consolidated, so nothing about them should
    be reset by a requeue.

    ``updated_at`` is deliberately left alone, matching the consolidator's own
    statements: consolidation bookkeeping is not an edit to the memory, and
    bumping it would make every consolidation pass look like a write to the
    staleness check below.
    """
    ids = _as_uuids(unit_ids)
    if not ids:
        return
    column = "consolidation_failed_at" if failed else "consolidated_at"
    guard = "" if when is not None else " AND fact_type IN ('experience', 'world')"
    await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET {column} = $1
        WHERE bank_id = $2 AND id = ANY($3::uuid[]){guard}
        """,
        when,
        bank_id,
        ids,
    )


async def any_memory_updated_since(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    since: datetime,
    fact_types: list[str] | None = None,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: list | None = None,
) -> bool:
    """Whether any memory in ``bank_id``'s scope was written after ``since``.

    Backs the mental-model staleness check, so it is a bounded existence test —
    ``LIMIT 1``, never a COUNT: the answer is "is there one", and the planner can
    stop at the first hit. The scope is the mental model's: its flat tags (or the
    compound ``tag_groups``) plus an optional ``fact_types`` restriction. This is
    where the staleness query's WHERE lives, so the same scope that gates a
    refresh decides whether one is due.
    """
    params: list[Any] = [bank_id, since]
    where = ["bank_id = $1", "updated_at > $2"]

    tag_clause, tag_params, next_param = build_tags_where_clause(tags, param_offset=len(params) + 1, match=tags_match)
    if tag_clause:
        where.append(tag_clause.removeprefix("AND "))
        params.extend(tag_params)

    group_clause, group_params, _ = build_tag_groups_where_clause(tag_groups, param_offset=next_param)
    if group_clause:
        where.append(group_clause.removeprefix("AND "))
        params.extend(group_params)
    # Untagged, no tag_groups → no tag constraint, matching any memory in the bank.

    if fact_types:
        params.append(list(fact_types))
        where.append(f"fact_type = ANY(${len(params)}::text[])")

    row = await conn.fetchval(
        f"SELECT 1 FROM {fq_table('memory_units')} WHERE {' AND '.join(where)} LIMIT 1",
        *params,
    )
    return row is not None


__all__ = [
    "any_memory_updated_since",
    "count_memories",
    "find_unconsolidated",
    "get_memories",
    "list_tags",
    "mark_consolidated",
    "scan_memories",
]
