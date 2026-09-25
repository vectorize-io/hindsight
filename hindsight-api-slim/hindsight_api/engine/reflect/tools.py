"""
Tool implementations for the reflect agent.

Implements hierarchical retrieval:
1. search_mental_models - User-curated stored reflect responses (highest quality)
2. search_observations - Consolidated knowledge with freshness
3. recall - Raw facts as ground truth
"""

import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..chunk_ids import resolve_chunk_id_in
from .tokenization import count_prompt_tokens

if TYPE_CHECKING:
    from asyncpg import Connection

    from ...api.http import RequestContext
    from ..memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

#: Snippet length for a mental-model search hit, matching the knowledge-page search
#: API's own ``LEFT(content, 280)`` so both surfaces show a page the same way.
_SNIPPET_CHARS = 280


#: Retrieval plumbing that the reflect agent never reads, dropped from tool
#: results before they reach the model.
#:
#: These are scoring and provenance internals, not evidence: the agent cites by
#: ``id``, ``based_on`` re-reads the cited memories' provenance from the store
#: (``MemoryEngine._evidence_as_stored``), and the expand tool takes
#: ``memory_ids`` and resolves chunks server-side -- so nothing downstream needs
#: them here, while on real banks they measure several times the size of the
#: observation text they accompany.
#:
#: Identity, text, dates, tags and ``source_fact_ids`` are deliberately kept.
#: So is ``entities``: it carries canonical entity *names* (not ids), which are
#: semantically useful retrieval handles -- the canonical name can differ from
#: the surface text ("Bob" in the text vs canonical "Robert Smith"). Reflect's
#: recalls don't populate it today (``include_entities`` defaults to False), but
#: trimming it would bake in dropping the names if that ever flips on.
_UNREAD_RESULT_FIELDS = ("scores", "metadata", "chunk_id", "document_id")


def _drop_unread_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Strip retrieval plumbing from one serialized tool result.

    Mutates and returns ``d``, which is always a fresh ``model_dump()`` by the
    time it gets here -- never a caller's dict.
    """
    for k in _UNREAD_RESULT_FIELDS:
        d.pop(k, None)
    return d


def _prune_nulls(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None or an empty collection (``""``, ``[]``, ``{}``).

    Reflect tools dump ``MemoryFact`` / ``ObservationResult`` via ``model_dump()``,
    which emits every field including the many that are typically null or empty
    (``context``, ``occurred_start``, ``metadata``, ``tags``, etc.). Stripping
    these before serializing to JSON for the LLM cuts token cost and removes
    fields that aren't telling the model anything.

    Callers that need the *presence* of a specific field as a signal (e.g.
    ``source_fact_ids`` for drill-down) must ensure the value is non-empty —
    pass the upstream flag that populates it (e.g. ``source_facts_max_tokens``
    > 0 on ``tool_search_observations``) rather than relying on Pydantic
    emitting ``None``.
    """
    return {k: v for k, v in d.items() if v is not None and v != "" and v != [] and v != {}}


def _document_metadata_from_retain_params(retain_params: Any) -> dict[str, Any] | None:
    """Return document metadata stored under retain_params.metadata."""
    if isinstance(retain_params, str):
        try:
            retain_params = json.loads(retain_params)
        except json.JSONDecodeError:
            return None

    if not isinstance(retain_params, dict):
        return None

    metadata = retain_params.get("metadata")
    return metadata if isinstance(metadata, dict) else None


async def tool_search_mental_models(
    memory_engine: "MemoryEngine",
    conn: "Connection",
    bank_id: str,
    query: str,
    query_embedding: list[float],
    max_results: int = 5,
    top_result_max_tokens: int = 4000,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    exclude_ids: list[str] | None = None,
    last_memory_write_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Search user-curated mental models by semantic similarity.

    Mental models are high-quality, manually created summaries about specific topics.
    They should be searched FIRST as they represent the most reliable synthesized knowledge.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        query: Search query (for logging/tracing)
        query_embedding: Pre-computed embedding for semantic search
        max_results: Maximum number of mental models to return
        top_result_max_tokens: Size limit for returning the best-ranked page in full; above it
            every result is a snippet and the model reads what it wants with read_mental_models
        tags: Optional tags to filter mental models
        tags_match: How to match tags - "any", "all", "any_strict", "all_strict", or "exact"
        exclude_ids: Optional list of mental model IDs to exclude (e.g., when refreshing a mental model)
        last_memory_write_at: The bank's newest memory write, resolved once per reflect. Skips the
            per-model staleness query for any model refreshed at or after it.

    Returns:
        Dict with matching mental models including content and freshness info
    """
    from ..memory_engine import _knowledge_snippet, _mental_model_stale_scope_from_row, fq_table
    from ..search.tags import build_tag_groups_where_clause, build_tags_where_clause

    # Build filters dynamically
    filters = ""
    params: list[Any] = [bank_id, str(query_embedding), max_results]
    next_param = 4

    # Exact matching treats absent or empty tags as the global scope. Do not
    # skip the filter, or mental models would see every scope while the other
    # reflect retrieval tools correctly see only untagged data.
    if tags or tags_match == "exact":
        built = build_tags_where_clause(tags, param_offset=next_param, match=tags_match)
        tag_clause = built.sql
        tag_params = built.params
        next_param = built.next_param_offset
        filters += f" {tag_clause}"
        params.extend(tag_params)

    if tag_groups:
        built = build_tag_groups_where_clause(tag_groups, next_param)
        groups_clause = built.sql
        groups_params = built.params
        next_param = built.next_param_offset
        filters += f" {groups_clause}"
        params.extend(groups_params)

    if exclude_ids:
        filters += f" AND id != ALL(${next_param}::text[])"
        params.append(exclude_ids)
        next_param += 1

    # Search mental models by embedding similarity.
    #
    # A store that indexes pages answers the ranking and the relevance; Postgres still hydrates the
    # rows, because the store holds only the searchable half. The tag scope and `exclude_ids` are
    # pushed down rather than applied afterwards: a hit that gets discarded here has already taken a
    # top-k slot from a page that would have qualified.
    from ..memories import get_memories

    store = get_memories()
    if store.store_owned_for(bank_id):
        matches = await store.search_knowledge_pages_semantic(
            bank_id,
            embedding=list(query_embedding),
            limit=max_results,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            exclude_ids=exclude_ids,
        )
        relevance_by_id = {m.page_id: m.score for m in matches}
        rows = (
            await conn.fetch(
                f"""
                SELECT
                    id, name, content,
                    tags, created_at, last_refreshed_at, last_memory_seen_at, trigger
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1 AND id = ANY($2::text[])
                """,
                bank_id,
                list(relevance_by_id),
            )
            if relevance_by_id
            else []
        )
        # The store ranked them; the SELECT did not preserve that, so restore it here rather than
        # returning whatever order the planner produced.
        rows = sorted(rows, key=lambda r: -relevance_by_id.get(str(r["id"]), 0.0))
    else:
        relevance_by_id = None
        rows = await conn.fetch(
            f"""
            SELECT
                id, name, content,
                tags, created_at, last_refreshed_at, last_memory_seen_at, trigger,
                1 - (embedding <=> $2::vector) as relevance
            FROM {fq_table("mental_models")}
            WHERE bank_id = $1 AND embedding IS NOT NULL {filters}
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            *params,
        )

    # Per-MM staleness: new in-scope memories since last refresh (includes pending).
    # Every model gets the exact, scoped answer — the agent trusts a model without a
    # verifying recall() only on `is_stale is False`, so guessing conservatively here
    # would buy LLM turns to save a query. One round-trip for the whole result set:
    # the models the bank-wide watermark already proves current are answered without
    # a query at all, the rest are asked together.
    staleness = await memory_engine.compute_mental_models_are_stale(
        conn,
        bank_id,
        {str(row["id"]): _mental_model_stale_scope_from_row(row, key=str(row["id"])) for row in rows},
        watermark=last_memory_write_at,
    )

    mental_models = []

    for row in rows:
        last_refreshed_at = row["last_refreshed_at"]
        if last_refreshed_at and last_refreshed_at.tzinfo is None:
            last_refreshed_at = last_refreshed_at.replace(tzinfo=timezone.utc)

        is_stale = staleness[str(row["id"])]
        staleness_reason = "new in-scope memories ingested since last refresh" if is_stale else None

        content = row["content"] or ""
        # The top-ranked page comes back whole, the rest as snippets. A model that
        # answers from a snippet without reading the page is a measured failure
        # (test_06), and the best hit is the one it is most likely to need; the
        # cost is one page instead of five.
        top_hit = not mental_models and count_prompt_tokens(content) <= top_result_max_tokens
        mental_models.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                # A snippet, like the knowledge-page search this mirrors — not the whole
                # page. Five whole pages measured 8.7-19k tokens and were re-sent on
                # every later turn of the loop, the largest single item in a reflect's
                # floor (#4533). The model reads the ones it wants with
                # ``read_mental_models``.
                **(
                    {"content": content}
                    if top_hit
                    # ``_knowledge_snippet`` so a page with no body reads as
                    # empty-on-purpose rather than as a blank match — the same
                    # wording the knowledge-page search gives for the same state.
                    else {
                        "snippet": _knowledge_snippet(content)[:_SNIPPET_CHARS].strip(),
                        "content_chars": len(content),
                    }
                ),
                "tags": row["tags"] or [],
                # The store path carries relevance beside the rows (its SELECT hydrates only what
                # the store does not hold); the SQL path has it as a computed column.
                "relevance": round(
                    relevance_by_id[str(row["id"])] if relevance_by_id is not None else row["relevance"],
                    4,
                ),
                "updated_at": last_refreshed_at.isoformat() if last_refreshed_at else None,
                "is_stale": is_stale,
                "staleness_reason": staleness_reason,
            }
        )

    return {
        "query": query,
        "count": len(mental_models),
        "mental_models": mental_models,
    }


async def tool_read_mental_models(
    conn: "Connection",
    bank_id: str,
    mental_model_ids: list[str],
    max_tokens: int = 6000,
) -> dict[str, Any]:
    """Read the full text of mental models the search returned as snippets.

    The budget is spent in the order asked for and stops at the first page that
    would cross it, so one enormous page cannot swallow the reflect's context —
    the failure ``search_mental_models`` used to have by returning five of them
    whole (#4533). A page that does not fit at all is reported by name rather
    than silently missing.
    """
    from ..memory_engine import fq_table

    if not mental_model_ids:
        return {"error": "read_mental_models requires mental_model_ids"}

    rows = await conn.fetch(
        f"""
        SELECT id, name, content, tags, last_refreshed_at
        FROM {fq_table("mental_models")}
        WHERE bank_id = $1 AND id = ANY($2::text[])
        """,
        bank_id,
        [str(i) for i in mental_model_ids],
    )
    by_id = {str(r["id"]): r for r in rows}

    pages: list[dict[str, Any]] = []
    omitted: list[str] = []
    spent = 0
    for wanted in mental_model_ids:
        row = by_id.get(str(wanted))
        if row is None:
            omitted.append(str(wanted))
            continue
        content = row["content"] or ""
        cost = count_prompt_tokens(content)
        if pages and spent + cost > max_tokens:
            # The id, not the name: ``not_read`` is a retry list, and the model can
            # only ask again with an id.
            omitted.append(str(row["id"]))
            continue
        spent += cost
        last_refreshed_at = row["last_refreshed_at"]
        pages.append(
            {
                "id": str(row["id"]),
                "name": row["name"],
                "content": content,
                "tags": row["tags"] or [],
                "updated_at": last_refreshed_at.isoformat() if last_refreshed_at else None,
            }
        )
    out: dict[str, Any] = {"mental_models": pages}
    if omitted:
        out["not_read"] = omitted
    return out


async def tool_search_observations(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    max_tokens: int = 5000,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    last_consolidated_at: datetime | None = None,
    pending_consolidation: int = 0,
    source_facts_max_tokens: int = -1,
    include_entities: bool = True,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> dict[str, Any]:
    """
    Search consolidated observations using recall.

    Observations are auto-generated from memories. Returns freshness info
    so the agent knows if it should also verify with recall().

    Args:
        memory_engine: Memory engine instance
        bank_id: Bank identifier
        query: Search query
        request_context: Request context for authentication
        max_tokens: Maximum tokens for results (default 5000)
        tags: Optional tags to filter observations
        tags_match: How to match tags - "any", "all", "any_strict", "all_strict", or "exact"
        last_consolidated_at: When consolidation last ran (for staleness check)
        pending_consolidation: Number of memories waiting to be consolidated
        source_facts_max_tokens: Token budget for source facts (-1 = disabled, 0+ = enabled with limit)
        include_entities: Attach resolved entity names to each observation (see below)

    Returns:
        Dict with matching observations including freshness info and source memories
    """
    include_source_facts = source_facts_max_tokens != -1
    recall_kwargs: dict[str, Any] = {}
    if include_source_facts and source_facts_max_tokens > 0:
        recall_kwargs["max_source_facts_tokens"] = source_facts_max_tokens

    # Use an internal request context so this recall is not billed as a
    # user-facing operation. The reflect caller is already billed for the
    # overall reflect operation; double-billing the sub-recalls would
    # overcharge the customer.
    internal_ctx = replace(request_context, internal=True)
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=["observation"],
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=internal_ctx,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        include_source_facts=include_source_facts,
        # Canonical entity names are semantic signal the surface text may lack
        # ("Bob" in the text vs canonical "Robert Smith"): they populate each
        # result's `entities` field, giving the agent resolved names to cite
        # and to pivot follow-up queries on. They are also a large share of the
        # serialized payload, so a bank that does not need them can turn them off
        # (reflect_default_options.reflect_search_observations_include_entities, #4483).
        include_entities=include_entities,
        created_after=created_after,
        created_before=created_before,
        _connection_budget=1,
        _quiet=True,
        **recall_kwargs,
    )

    is_stale = pending_consolidation > 0
    if pending_consolidation == 0:
        freshness = "up_to_date"
    elif pending_consolidation < 10:
        freshness = "slightly_stale"
    else:
        freshness = "stale"

    return {
        "query": query,
        "count": len(result.results),
        "observations": [_drop_unread_fields(_prune_nulls(m.model_dump())) for m in result.results],
        "source_facts": {
            k: _drop_unread_fields(_prune_nulls(v.model_dump())) for k, v in (result.source_facts or {}).items()
        },
        "is_stale": is_stale,
        "freshness": freshness,
    }


async def tool_recall(
    memory_engine: "MemoryEngine",
    bank_id: str,
    query: str,
    request_context: "RequestContext",
    max_tokens: int = 2048,
    tags: list[str] | None = None,
    tags_match: str = "any",
    tag_groups: "list | None" = None,
    connection_budget: int = 1,
    max_chunk_tokens: int = 1000,
    fact_types: list[str] | None = None,
    include_chunks: bool = True,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> dict[str, Any]:
    """
    Search memories using TEMPR retrieval.

    This is the ground truth - raw facts and experiences.
    Use when mental models/observations don't exist, are stale, or need verification.

    Args:
        memory_engine: Memory engine instance
        bank_id: Bank identifier
        query: Search query
        request_context: Request context for authentication
        max_tokens: Maximum tokens for results (default 2048)
        tags: Filter by tags (includes untagged memories)
        tags_match: How to match tags - "any", "all", "any_strict", "all_strict", or "exact"
        connection_budget: Max DB connections for this recall (default 1 for internal ops)
        max_chunk_tokens: Maximum tokens for raw source chunk text (default 1000)
        fact_types: Optional filter for fact types to retrieve. Defaults to ["experience", "world"].
        include_chunks: Whether to fetch raw chunk text alongside facts (default True).

    Returns:
        Dict with list of matching memories including raw chunk text (when include_chunks)
    """
    # Only world/experience are valid for raw recall (observation is handled by search_observations)
    recall_fact_type = [ft for ft in (fact_types or ["experience", "world"]) if ft in ("world", "experience")]
    internal_ctx = replace(request_context, internal=True)
    result = await memory_engine.recall_async(
        bank_id=bank_id,
        query=query,
        fact_type=recall_fact_type,
        max_tokens=max_tokens,
        enable_trace=False,
        request_context=internal_ctx,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        # See tool_search_observations: resolved entity names on each result
        # are worth the one extra lookup query.
        include_entities=True,
        _connection_budget=connection_budget,
        _quiet=True,  # Suppress logging for internal operations
        include_chunks=include_chunks,
        max_chunk_tokens=max_chunk_tokens,
    )

    return {
        "query": query,
        "memories": [_drop_unread_fields(_prune_nulls(m.model_dump())) for m in result.results],
        # ``chunks`` is deliberately not trimmed: ChunkInfo carries only
        # chunk_text / chunk_index / truncated, so it holds none of the fields
        # above and the call would be a no-op. Pinned by
        # test_chunk_info_carries_no_unread_fields.
        "chunks": {k: _prune_nulls(v.model_dump()) for k, v in (result.chunks or {}).items()},
    }


async def tool_expand(
    conn: "Connection",
    bank_id: str,
    memory_ids: list[str],
    depth: str,
) -> dict[str, Any]:
    """
    Expand multiple memories to get chunk or document context.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        memory_ids: List of memory unit IDs
        depth: "chunk" or "document"

    Returns:
        Dict with results array, each containing memory, chunk, and optionally document data
    """
    from ..memory_engine import fq_table

    if not memory_ids:
        return {"error": "memory_ids is required and must not be empty"}

    # Validate and convert UUIDs. Each id keeps a handle on its own UUID: a list of
    # only the valid ones no longer lines up with memory_ids once one id is invalid.
    uuid_by_id: dict[str, uuid.UUID] = {}
    errors: dict[str, str] = {}
    for mid in memory_ids:
        try:
            uuid_by_id[mid] = uuid.UUID(mid)
        except ValueError:
            errors[mid] = f"Invalid memory_id format: {mid}"

    if not uuid_by_id:
        return {"error": "No valid memory IDs provided", "details": errors}

    valid_uuids = list(uuid_by_id.values())

    # Batch fetch all memory units. A store that keeps memories outside SQL answers by id
    # through the store; normalize its records to the same UUID-keyed dict shape the SQL rows
    # have so the result-building below stays store-agnostic.
    from ..memories import get_memories

    _store = get_memories()
    if not _store.store_owned_for(bank_id):
        memories = await conn.fetch(
            f"""
            SELECT id, text, chunk_id, document_id, fact_type, context
            FROM {fq_table("memory_units")}
            WHERE id = ANY($1) AND bank_id = $2
            """,
            valid_uuids,
            bank_id,
        )
    else:
        stored = await _store.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(u) for u in valid_uuids]
        )
        memories = [
            {
                "id": uuid.UUID(s.unit_id),
                "text": s.text,
                "chunk_id": s.chunk_id,
                "document_id": s.document_id,
                "fact_type": s.fact_type,
                "context": s.context,
            }
            for s in stored
        ]
    memory_map = {row["id"]: row for row in memories}

    # Collect chunk_ids and document_ids for batch fetching
    chunk_ids = [m["chunk_id"] for m in memories if m["chunk_id"]]
    doc_ids_from_chunks: set[str] = set()
    doc_ids_direct: set[str] = set()

    # Batch fetch all chunks. A store that owns the document store leaves the `chunks` and
    # `documents` tables empty, so the SQL below would return nothing and `expand` would answer
    # without the chunk or document it was asked for — the memories read above was routed to the
    # store but these two were not.
    _docs_in_store = _store.store_owned_for(bank_id)
    chunk_map: dict[str, Any] = {}
    if chunk_ids and _docs_in_store:
        # The store addresses a chunk by (document_id, index), and the index is what remains
        # once the known bank/document prefix is removed (see `engine/chunk_ids.py`). Anchored
        # on the ids in hand rather than split on "_", which a bank or document id containing
        # one would break.
        # Deduped by chunk_id: co-located memories share one chunk, and the SQL branch collapses
        # them through `= ANY($1)`. Without this the store is asked for the same chunk once per
        # memory sitting in it.
        refs: list[tuple[str, int]] = []
        ref_owner: list[dict] = []
        _seen_chunks: set[str] = set()
        for m in memories:
            cid, did = m["chunk_id"], m["document_id"]
            if not cid or not did:
                continue
            if cid in _seen_chunks:
                continue
            ref = resolve_chunk_id_in(cid, bank_id)
            if ref is None or ref.document_id != did:
                continue
            index = ref.chunk_index
            _seen_chunks.add(cid)
            refs.append((did, index))
            ref_owner.append({"chunk_id": cid, "document_id": did, "chunk_index": index})
        if refs:
            texts = await _store.get_chunk_texts(bank_id=bank_id, refs=refs)
            for owner, text in zip(ref_owner, texts):
                if text is None:
                    continue
                chunk_map[owner["chunk_id"]] = {**owner, "chunk_text": text}
        if depth == "document":
            doc_ids_from_chunks = {c["document_id"] for c in chunk_map.values() if c["document_id"]}
    elif chunk_ids:
        chunks = await conn.fetch(
            f"""
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1)
            """,
            chunk_ids,
        )
        chunk_map = {row["chunk_id"]: row for row in chunks}
        if depth == "document":
            doc_ids_from_chunks = {c["document_id"] for c in chunks if c["document_id"]}

    # Collect direct document IDs (memories without chunks)
    if depth == "document":
        for m in memories:
            if not m["chunk_id"] and m["document_id"]:
                doc_ids_direct.add(m["document_id"])

    # Batch fetch all documents
    doc_map: dict[str, Any] = {}
    all_doc_ids = list(doc_ids_from_chunks | doc_ids_direct)
    if all_doc_ids and _docs_in_store:
        # One read per document: the store addresses a document by id and has no batch form here.
        # The set is the documents behind the memories being expanded, which is bounded by the
        # caller's own memory_ids rather than by corpus size.
        for did in all_doc_ids:
            record = await _store.get_document_record(bank_id=bank_id, document_id=did, include_text=True)
            if record is None:
                continue
            # The store has no `retain_params` column; it keeps the retain params inside the
            # document record's metadata bag, under that key and serialised as JSON. So they are
            # read back out of the bag rather than reconstructed — `_document_metadata_from_retain_params`
            # already parses the JSON form, which is the same thing Postgres hands it from JSONB.
            doc_map[did] = {
                "id": did,
                "original_text": record.get("original_text"),
                "retain_params": (record.get("metadata") or {}).get("retain_params"),
            }
    elif all_doc_ids:
        docs = await conn.fetch(
            f"""
            SELECT id, original_text, retain_params
            FROM {fq_table("documents")}
            WHERE id = ANY($1) AND bank_id = $2
            """,
            all_doc_ids,
            bank_id,
        )
        doc_map = {row["id"]: row for row in docs}

    # Build results
    results: list[dict[str, Any]] = []
    for mid in memory_ids:
        if mid in errors:
            results.append({"memory_id": mid, "error": errors[mid]})
            continue

        memory = memory_map.get(uuid_by_id[mid])
        if not memory:
            results.append({"memory_id": mid, "error": f"Memory not found: {mid}"})
            continue

        item: dict[str, Any] = {
            "memory_id": mid,
            "memory": {
                "id": str(memory["id"]),
                "text": memory["text"],
                "type": memory["fact_type"],
                "context": memory["context"],
            },
        }

        # Add chunk if available
        if memory["chunk_id"] and memory["chunk_id"] in chunk_map:
            chunk = chunk_map[memory["chunk_id"]]
            item["chunk"] = {
                "id": chunk["chunk_id"],
                "text": chunk["chunk_text"],
                "index": chunk["chunk_index"],
                "document_id": chunk["document_id"],
            }
            # Add document if depth=document
            if depth == "document" and chunk["document_id"] in doc_map:
                doc = doc_map[chunk["document_id"]]
                item["document"] = {
                    "id": doc["id"],
                    "full_text": doc["original_text"],
                    "metadata": _document_metadata_from_retain_params(doc["retain_params"]),
                    "retain_params": doc["retain_params"],
                }
        elif memory["document_id"] and depth == "document" and memory["document_id"] in doc_map:
            # No chunk, but has document_id
            doc = doc_map[memory["document_id"]]
            item["document"] = {
                "id": doc["id"],
                "full_text": doc["original_text"],
                "metadata": _document_metadata_from_retain_params(doc["retain_params"]),
                "retain_params": doc["retain_params"],
            }

        results.append(item)

    return {"results": results, "count": len(results)}
