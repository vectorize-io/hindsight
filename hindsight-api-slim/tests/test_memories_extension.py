"""The memories store is an extension point, and the default is Postgres.

Two things are worth pinning down here, and neither is about SQL:

1. **The default is unconditional.** With nothing configured the engine gets
   :class:`PostgresMemories`, so every other test in this suite is exercising the
   real store rather than a seam that happens to fall through to it.
2. **The interface is complete.** A store that implements
   :class:`MemoriesExtension` and touches no database at all can be installed and
   used. Most tests here drive the stub's methods directly with ``conn=None`` to
   prove each one answers from the dict without SQL. That is a check on the stub,
   not on the engine's routing — so
   :func:`test_engine_list_tags_routes_through_the_installed_store` additionally
   drives a real :class:`MemoryEngine` operation with the stub installed: the
   engine may hold a live Postgres connection, but if it reached past the
   interface to query ``memory_units`` it would miss the tags only the stub holds.

The stub is deliberately a dictionary. Anything cleverer would start re-testing
storage instead of the seam.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from hindsight_api.engine.memories import create_memories, get_memories, set_memories
from hindsight_api.engine.memories.base import (
    EntityPrunePassResult,
    MemoriesExtension,
    RecallArms,
    RelinkPassResult,
    RetainResult,
    RetainSession,
    ScanPage,
    StoredMemory,
)
from hindsight_api.engine.memories.postgres import PostgresMemories


class InMemoryMemories(MemoriesExtension):
    """A complete store that is a dict — no connection, no tables, no SQL.

    Every method that touches storage is answered from ``self.rows``. The
    Postgres handles (``conn``, ``ops``, ``fq_table``) are accepted and ignored,
    which is exactly what an implementation that owns the store does with them.
    """

    name = "in-memory"
    # This store keeps memory rows in its own dict, not in Postgres, so the engine must take the
    # store-delegating branches (never the inline-SQL fast paths that assume a live ``conn``).
    # It also owns the document/chunk bodies, so the retain and read paths route those through the
    # document methods below — exercising that half of the seam too.
    store_owned = True
    # When True the store carries each unit's entity ids on the recall result (a backend that
    # resolves the unit->entity posting inline), so recall builds the entity map from the result
    # and resolves only names via ``resolve_entity_names``. When False the result carries no ids
    # (the default store) and recall re-fetches via ``entity_map_for_units``. Both must produce
    # identical output — the parametrized entity tests pin that.
    carries_entity_ids_on_result = False
    # Likewise for an observation's sources. When False the result carries none, and recall
    # re-fetches the observation to read them (for the prefer_observations dedup and the chunk
    # walk); when True it reads them off the result. Both must produce identical output.
    carries_source_ids_on_result = False

    def __init__(self, config: dict[str, str] | None = None):
        super().__init__(config or {})
        self.rows: dict[str, StoredMemory] = {}
        # The curation archive is just a second dict — invalidation moves a memory
        # from `rows` to `archive`, exactly as it moves between tables/namespaces.
        self.archive: dict[str, StoredMemory] = {}
        self.invalidation_reason: dict[str, str | None] = {}
        self.embeddings: dict[str, object] = {}
        # What `apply_edit` was handed, so a test can tell which door the vector came through and
        # whether the caller supplied the pre-edit fact type.
        self.edit_embedding: object = None
        self.edit_current_fact_type: str | None = None
        # The document store: one record per document, each carrying its extracted text and the
        # ordered chunk texts — the bodies a store that owns them keeps out of Postgres.
        self.documents: dict[str, dict] = {}
        # The knowledge-page index, per bank. Unlike the dicts above this one is DERIVED — the
        # page's row lives in Postgres either way, so this holds only what a search needs.
        self.knowledge_pages: dict[str, dict] = {}
        # Proof the engine went through the interface rather than around it.
        self.calls: list[str] = []
        # Banks whose storage exists. A store that owns its storage creates it when the
        # bank is created and keeps it for as long as the bank lives.
        self.ensured: set[str] = set()
        # Every DeletePredicate handed to delete_where, so a test can assert WHICH delete
        # was asked for and not merely that one happened.
        self.predicates: list = []

    # -- writes --------------------------------------------------------------

    async def insert_facts(self, *, conn, ops, bank_id, facts, document_id=None, defer_index=False):
        self.calls.append("insert_facts")
        unit_ids = self.allocate_unit_ids(len(facts))
        if not defer_index:
            await self.index_facts(bank_id, unit_ids, facts, document_id)
        return unit_ids

    # The interface's newer members, implemented so the seam stays a COMPLETE store — the
    # structural test asserts every method is answered rather than inheriting the
    # NotImplementedError default, which is what stops a capability being added to the interface
    # and silently never exercised.
    async def set_document_tags(self, *, bank_id, document_id, tags):
        doc = self.documents.get(str(document_id))
        if doc is not None:
            doc["tags"] = list(tags)

    async def count_documents(self, *, bank_id):
        return len(self.documents)

    async def get_entity_graph(self, *, bank_id, limit=1000, min_count=1):
        return {"nodes": [], "edges": []}

    async def list_documents(
        self, *, bank_id, search_query=None, tags=None, tags_match="any_strict", limit=100, offset=0
    ):
        items = [{"id": doc_id, **doc} for doc_id, doc in self.documents.items()]
        return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}

    async def retain(
        self,
        bank_id,
        unit_ids,
        facts,
        document_id=None,
        unit_entity_names=None,
        replace_document_id="",
        replace_chunk_ids=None,
        replace_keep_chunk_ids=None,
        resolve_threshold=0.0,
        enable_text_search=True,
        enable_graph_retrieval=True,
    ):
        self.calls.append("retain")
        return None

    async def begin_retain(self, *, bank_id, config):
        """A session that buffers and commits once, which is the policy a store with no LLM in the
        loop should pick. It writes through the same ``index_facts`` the non-session path uses, so
        a test asserting on ``rows`` cannot tell which path produced them — the point being that the
        session is a different WHEN, not a different WHAT."""
        self.calls.append("begin_retain")
        return _InMemoryRetainSession(self, bank_id)

    async def index_facts(self, bank_id, unit_ids, facts, document_id=None, unit_entity_ids=None):
        self.calls.append("index_facts")
        for unit_id, fact in zip(unit_ids, facts):
            self.rows[unit_id] = StoredMemory(
                unit_id=unit_id,
                text=fact.fact_text,
                fact_type=fact.fact_type,
                document_id=document_id,
                tags=list(fact.tags or []),
                created_at=datetime.now(timezone.utc),
            )

    async def delete_facts(self, bank_id, unit_ids):
        self.calls.append("delete_facts")
        for unit_id in unit_ids:
            self.rows.pop(str(unit_id), None)

    async def delete_document(self, *, conn, fq_table, bank_id, document_id):
        self.calls.append("delete_document")
        for unit_id in [k for k, v in self.rows.items() if v.document_id == document_id]:
            del self.rows[unit_id]

    async def delete_observations(self, *, conn, fq_table, bank_id):
        self.calls.append("delete_observations")
        for unit_id in [k for k, v in self.rows.items() if v.fact_type == "observation"]:
            del self.rows[unit_id]

    # -- recall arms ---------------------------------------------------------

    async def search(self, *, conn, bank_id, fact_types, query_embedding, query_text, limit, **kwargs):
        self.calls.append("search")
        # The engine consumes the three-field result (semantic hits, bm25 hits, graph seeds) and
        # fuses + reranks it into ``top_scored``, so the seam has to hand back REAL candidates built
        # from the dict rows — not an empty shell. An empty result means recall returns nothing and
        # every enrichment arm downstream (chunks / source facts / entities) has nothing to hydrate,
        # which is exactly the blind spot these tests exist to close.
        from hindsight_api.engine.search.retrieval import SemanticBm25Result
        from hindsight_api.engine.search.types import RetrievalResult

        def _candidate(row, rank: int, *, semantic: bool) -> RetrievalResult:
            # One dict row becomes one ranked candidate. Score descends with insertion order so the
            # fusion/rerank keeps a stable, faithful order; the chunk_id/document_id ride along so the
            # chunk-overlay arm can find the row's chunk, exactly like a real semantic/BM25 hit.
            score = 1.0 - rank * 0.01
            return RetrievalResult(
                id=row.unit_id,
                text=row.text,
                fact_type=row.fact_type,
                context=row.context,
                document_id=row.document_id,
                chunk_id=row.chunk_id,
                tags=list(row.tags) or None,
                metadata=row.metadata,
                proof_count=row.proof_count,
                # A store that resolves the posting inline carries the ids on the result (a list,
                # possibly empty); otherwise leave it None so recall re-fetches via the map path.
                entity_ids=list(row.entity_ids) if self.carries_entity_ids_on_result else None,
                source_memory_ids=(list(row.source_memory_ids) if self.carries_source_ids_on_result else None),
                similarity=score if semantic else None,
                bm25_score=None if semantic else score,
            )

        out: dict[str, SemanticBm25Result] = {}
        for ft in fact_types:
            matches = [r for r in self.rows.values() if r.fact_type == ft][:limit]
            out[ft] = SemanticBm25Result(
                semantic=[_candidate(r, i, semantic=True) for i, r in enumerate(matches)],
                bm25=[_candidate(r, i, semantic=False) for i, r in enumerate(matches)],
                graph_seeds=[],
            )
        return out

    async def temporal_search(
        self, *, conn, bank_id, fact_types, query_embedding, start_date, end_date, limit, **kwargs
    ):
        self.calls.append("temporal_search")
        return {ft: [] for ft in fact_types}

    async def recall_unified(
        self,
        *,
        conn,
        bank_id,
        fact_types,
        query_embedding,
        query_text,
        limit,
        temporal_window=None,
        temporal_semantic_threshold=0.1,
        tags=None,
        tags_match="any",
        tag_groups=None,
        created_after=None,
        created_before=None,
        min_semantic=None,
        min_keyword=None,
        enable_text_search=True,
        enable_graph=True,
    ):
        # The one recall interface. This store owns its links (no separate graph arm), so it
        # answers dense/keyword from its own ``search`` and, when a window is given, temporal from
        # its own ``temporal_search`` — the same seam the engine drove per-arm before it unified.
        self.calls.append("recall_unified")
        sb = await self.search(
            conn=conn,
            bank_id=bank_id,
            fact_types=fact_types,
            query_embedding=query_embedding,
            query_text=query_text,
            limit=limit,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
            min_semantic=min_semantic,
            min_keyword=min_keyword,
        )
        temporal: dict[str, list] = {ft: [] for ft in fact_types}
        if temporal_window is not None:
            start_date, end_date = temporal_window
            temporal = await self.temporal_search(
                conn=conn,
                bank_id=bank_id,
                fact_types=fact_types,
                query_embedding=query_embedding,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                semantic_threshold=temporal_semantic_threshold,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                created_after=created_after,
                created_before=created_before,
            )
        return {
            ft: RecallArms(
                semantic=sb[ft].semantic,
                bm25=sb[ft].bm25,
                graph=[],
                temporal=temporal.get(ft, []),
            )
            for ft in fact_types
        }

    # -- addressed reads -----------------------------------------------------

    async def get_memories(self, *, conn, fq_table, bank_id, unit_ids):
        self.calls.append("get_memories")
        return [self.rows[str(u)] for u in unit_ids if str(u) in self.rows]

    async def scan_memories(self, *, conn, fq_table, bank_id, limit=100, page_token="", **kwargs):
        start = int(page_token or 0)
        ordered = list(self.rows.values())[start : start + limit]
        nxt = str(start + limit) if start + limit < len(self.rows) else ""
        return ScanPage(memories=ordered, next_page_token=nxt)

    async def count_memories(self, *, conn, fq_table, bank_id):
        self.calls.append("count_memories")
        counts: dict[str, int] = {}
        for row in self.rows.values():
            counts[row.fact_type] = counts.get(row.fact_type, 0) + 1
        return counts

    async def list_tags(self, *, conn, fq_table, bank_id, pattern=None, limit=100, offset=0):
        self.calls.append("list_tags")
        counts: dict[str, int] = {}
        for row in self.rows.values():
            for tag in row.tags:
                counts[tag] = counts.get(tag, 0) + 1
        items = [{"tag": tag, "count": count} for tag, count in counts.items()]
        if pattern:
            import re as _re

            regex = _re.compile("^" + ".*".join(_re.escape(p) for p in pattern.split("*")) + "$", _re.IGNORECASE)
            items = [it for it in items if regex.match(str(it["tag"]))]
        items.sort(key=lambda it: (-it["count"], it["tag"]))
        return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}

    async def find_unconsolidated(self, *, conn, fq_table, bank_id, fact_types, limit, scope_tags=None):
        out = [r for r in self.rows.values() if r.fact_type in fact_types and r.consolidated_at is None]
        if scope_tags:
            out = [r for r in out if set(scope_tags).issubset(set(r.tags))]
        return out[:limit]

    async def mark_consolidated(self, *, conn, fq_table, bank_id, unit_ids, when, failed=False):
        for unit_id in unit_ids:
            row = self.rows.get(str(unit_id))
            if row is not None:
                row.consolidated_at = when

    async def entity_memory_counts(self, *, conn, fq_table, bank_id, entity_ids=None):
        counts: dict[str, int] = {}
        for row in self.rows.values():
            for entity_id in row.entity_ids:
                counts[entity_id] = counts.get(entity_id, 0) + 1
        return counts if entity_ids is None else {k: v for k, v in counts.items() if k in set(entity_ids)}

    async def entities_for_units(self, *, conn, fq_table, bank_id, unit_ids):
        return {str(u): list(self.rows[str(u)].entity_ids) for u in unit_ids if str(u) in self.rows}

    async def entity_map_for_units(self, *, conn, fq_table, bank_id, unit_ids):
        # The memory carries its own entity ids inline, but the *names* live in the shared Postgres
        # entity registry — which this store does not stand in for. Resolve id -> canonical_name
        # through the connection recall hands us, exactly as a store that owns its rows (but not the
        # registry) would. Falls back to the id as the name when there is no connection or no
        # matching registry row, so the direct-call tests (conn=None) still get the recall shape.
        wanted = {str(u): list(self.rows[str(u)].entity_ids) for u in unit_ids if str(u) in self.rows}
        all_ids = {e for ids in wanted.values() for e in ids}
        names = await self.resolve_entity_names(conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=all_ids)
        # A unit with no entities is omitted, not mapped to [] — same as the default store, so
        # its fact keeps entities=None downstream rather than an empty list.
        return {
            uid: [{"entity_id": e, "canonical_name": names.get(e, e)} for e in ids]
            for uid, ids in wanted.items()
            if ids
        }

    async def resolve_entity_names(self, *, conn, fq_table, bank_id, entity_ids):
        # The store owns the memory rows (and their entity ids), but the *names* live in the
        # shared entity registry, which this store does not stand in for — resolve them through
        # the connection recall hands us, bank-scoped, exactly as a store that owns its rows (but
        # not the registry) would. No connection or no ids means no names to resolve.
        if not entity_ids or conn is None:
            return {}
        try:
            as_uuids = [uuid.UUID(str(e)) for e in set(entity_ids)]
        except (ValueError, AttributeError, TypeError):
            return {}
        rows = await conn.fetch(
            f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = ANY($1::uuid[]) AND bank_id = $2",
            as_uuids,
            bank_id,
        )
        return {str(r["id"]): r["canonical_name"] for r in rows}

    async def any_memory_updated_since(
        self, *, conn, fq_table, bank_id, since, fact_types=None, tags=None, tags_match="any", tag_groups=None
    ):
        rows = self.rows.values()
        if fact_types:
            rows = [r for r in rows if r.fact_type in fact_types]
        return any(r.created_at is not None and r.created_at > since for r in rows)

    async def live_memory_ids(self, *, conn, fq_table, bank_id, unit_ids):
        # "Live" for this stub is simply "present in self.rows" — the retraction
        # check only ever asks whether the id still resolves, so a store that keeps
        # memories itself answers from its own keyspace with no archive to consult.
        wanted = {str(u) for u in unit_ids}
        return {unit_id for unit_id in self.rows if unit_id in wanted}

    # -- observations --------------------------------------------------------

    async def observations_for_sources(self, *, conn, ops, fq_table, bank_id, unit_ids):
        wanted = {str(u) for u in unit_ids}
        return [r for r in self.rows.values() if wanted & set(r.source_memory_ids)]

    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id, fact_ids):
        stale = await self.observations_for_sources(
            conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=fact_ids
        )
        for obs in stale:
            self.rows.pop(obs.unit_id, None)
        return len(stale)

    # -- curation reads ------------------------------------------------------

    async def list_memory_units(self, *, conn, ops, fq_table, bank_id, limit=100, offset=0, **kwargs):
        self.calls.append("list_memory_units")
        ordered = list(self.rows.values())
        return {"items": ordered[offset : offset + limit], "total": len(ordered), "limit": limit, "offset": offset}

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id, unit_id):
        row = self.rows.get(str(unit_id))
        return None if row is None else {"id": row.unit_id, "text": row.text, "fact_type": row.fact_type}

    # -- curation archive: a second dict is the archive namespace ------------

    async def get_archived_memory(self, *, conn, fq_table, bank_id, unit_id):
        return self.archive.get(str(unit_id))

    async def invalidate_memory(self, *, conn, fq_table, bank_id, unit_id, reason):
        row = self.rows.pop(str(unit_id), None)
        if row is None:
            return False
        self.archive[str(unit_id)] = row
        self.invalidation_reason[str(unit_id)] = reason
        return True

    async def set_invalidation_reason(self, *, conn, fq_table, bank_id, unit_id, reason):
        self.invalidation_reason[str(unit_id)] = reason

    async def restore_memory(self, *, conn, fq_table, bank_id, unit_id):
        row = self.archive.pop(str(unit_id), None)
        if row is None:
            return None
        self.rows[str(unit_id)] = row
        self.invalidation_reason.pop(str(unit_id), None)
        return row

    async def set_memory_embedding(self, *, conn, fq_table, bank_id, unit_id, embedding):
        self.calls.append("set_memory_embedding")
        self.embeddings[str(unit_id)] = embedding

    async def list_entities(self, *, conn, fq_table, bank_id, search=None, limit=100, offset=0):
        self.calls.append("list_entities")
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    async def graph_units(self, *, conn, fq_table, bank_id, limit=1000, **kwargs):
        rows = list(self.rows.values())[:limit]
        return {"units": [{"id": r.unit_id, "fact_type": r.fact_type} for r in rows], "total": len(self.rows)}

    async def graph_entity_rows(self, *, conn, fq_table, bank_id, unit_ids):
        return []

    async def graph_direct_links(self, *, conn, fq_table, bank_id, unit_ids):
        return []

    # -- curation / bulk writes ----------------------------------------------

    async def ensure_bank_storage(self, bank_id):
        self.calls.append("ensure_bank_storage")
        self.ensured.add(bank_id)

    async def drop_bank_storage(self, bank_id):
        self.calls.append("drop_bank_storage")
        self.ensured.discard(bank_id)
        self.rows = {k: v for k, v in self.rows.items() if False}

    async def delete_where(self, bank_id, predicate):
        self.calls.append("delete_where")
        self.predicates.append(predicate)
        if predicate.delete_all:
            n = len(self.rows)
            self.rows.clear()
            return n
        victims = []
        for uid, row in self.rows.items():
            if predicate.fact_types and row.fact_type not in predicate.fact_types:
                continue
            if predicate.metadata_equals:
                bag = row.metadata or {}
                if any(bag.get(k) != v for k, v in predicate.metadata_equals.items()):
                    continue
            if predicate.tags and not set(predicate.tags).issubset(set(row.tags)):
                continue
            victims.append(uid)
        for uid in victims:
            del self.rows[uid]
        return len(victims)

    async def update_memories(self, bank_id, patches):
        self.calls.append("update_memories")
        for patch in patches:
            row = self.rows.get(str(patch.unit_id))
            if row is None:
                continue
            if patch.text is not None:
                row.text = patch.text
            if patch.tags is not None:
                row.tags = list(patch.tags)
            if patch.event_date is not None:
                row.event_date = patch.event_date
            if patch.proof_count_delta:
                row.proof_count += patch.proof_count_delta
            if patch.metadata:
                row.metadata = {**(row.metadata or {}), **patch.metadata}

    async def apply_edit(
        self,
        *,
        conn,
        fq_table,
        bank_id,
        unit_id,
        text,
        context,
        fact_type,
        occurred_start,
        occurred_end,
        event_date,
        mentioned_at,
        entity_ids,
        entity_names=None,
        embedding=None,
        current_fact_type=None,
    ):
        self.calls.append("apply_edit")
        self.edit_embedding = embedding
        self.edit_current_fact_type = current_fact_type
        row = self.rows.get(str(unit_id))
        if row is None:
            return
        if embedding is not None:
            # Applying an edit includes writing its vector — the same call, not a following one.
            self.embeddings[str(unit_id)] = embedding
        row.text = text
        row.context = context
        row.fact_type = fact_type
        row.occurred_start = occurred_start
        row.occurred_end = occurred_end
        row.event_date = event_date
        row.mentioned_at = mentioned_at
        if entity_ids is not None:
            row.entity_ids = list(entity_ids)

    async def upsert_observation(self, *, conn, bank_id, record):
        self.calls.append("upsert_observation")
        self.rows[str(record.unit_id)] = StoredMemory(
            unit_id=str(record.unit_id),
            text=record.text,
            fact_type=record.fact_type,
            tags=list(record.tags or []),
            proof_count=record.proof_count,
            source_memory_ids=[str(s) for s in (record.source_memory_ids or [])],
            event_date=record.event_date,
            created_at=record.created_at or datetime.now(timezone.utc),
        )

    # -- document store (store_owned=True) -----------------------------------

    async def put_document(
        self,
        *,
        bank_id,
        document_id,
        content_hash,
        original_text,
        chunk_texts,
        tags=None,
        metadata=None,
        file_bytes=None,
        file_content_type="",
        file_original_name="",
        expect_watermark=None,
    ):
        self.calls.append("put_document")
        self.documents[str(document_id)] = {
            "id": str(document_id),
            "content_hash": content_hash,
            "original_text": original_text,
            "chunk_texts": list(chunk_texts),
            "tags": list(tags or []),
            "metadata": dict(metadata or {}),
        }

    async def document_content_hash(self, *, bank_id, document_id):
        self.calls.append("document_content_hash")
        doc = self.documents.get(str(document_id))
        return doc["content_hash"] if doc else None

    async def get_document_record(self, *, bank_id, document_id, include_text=False):
        self.calls.append("get_document_record")
        doc = self.documents.get(str(document_id))
        if doc is None:
            return None
        record = {"id": doc["id"], "content_hash": doc["content_hash"], "tags": list(doc["tags"])}
        if include_text:
            record["original_text"] = doc["original_text"]
        return record

    async def get_chunk_text(self, *, bank_id, document_id, chunk_index):
        self.calls.append("get_chunk_text")
        doc = self.documents.get(str(document_id))
        if doc is None or not (0 <= chunk_index < len(doc["chunk_texts"])):
            return None
        return doc["chunk_texts"][chunk_index]

    async def list_chunk_texts(self, *, bank_id, document_id):
        self.calls.append("list_chunk_texts")
        doc = self.documents.get(str(document_id))
        return list(doc["chunk_texts"]) if doc else None

    async def count_chunks(self, *, bank_id, document_id):
        self.calls.append("count_chunks")
        doc = self.documents.get(str(document_id))
        return len(doc["chunk_texts"]) if doc else 0

    async def delete_document_record(self, *, bank_id, document_id):
        self.calls.append("delete_document_record")
        self.documents.pop(str(document_id), None)

    # -- stats reads (computed from the dicts) -------------------------------

    async def consolidation_freshness(self, *, conn, fq_table, bank_id):
        self.calls.append("consolidation_freshness")
        unconsolidated = sum(
            1 for r in self.rows.values() if r.fact_type in ("experience", "world") and r.consolidated_at is None
        )
        return {"unconsolidated_source_count": unconsolidated}

    async def document_memory_counts(self, *, conn, fq_table, bank_id, document_ids):
        self.calls.append("document_memory_counts")
        wanted = {str(d) for d in document_ids}
        counts = {d: 0 for d in wanted}
        for row in self.rows.values():
            if row.document_id in wanted:
                counts[row.document_id] += 1
        return counts

    async def link_counts(self, *, conn, fq_table, bank_id):
        self.calls.append("link_counts")
        # A store that carries links inline has no join table to tally.
        return {"temporal": 0, "semantic": 0, "causal": 0}

    async def memories_timeseries(self, *, conn, fq_table, bank_id, time_field, trunc, since):
        self.calls.append("memories_timeseries")
        return []

    async def observation_scope_counts(self, *, conn, fq_table, bank_id):
        self.calls.append("observation_scope_counts")
        return []

    # -- the knowledge-page index -------------------------------------------
    #
    # Kept as a dict beside the memory rows, so the engine's routing is exercised end to end
    # rather than mocked. The ranking is deliberately crude — substring for text, ordering by the
    # first embedding component for vectors — because what these tests assert is that the engine
    # ROUTES here and hydrates what comes back, not that a dict can rank.

    store_owned = True

    def _pages(self, bank_id):
        return self.knowledge_pages.setdefault(bank_id, {})

    async def index_knowledge_pages(self, bank_id, entries):
        self.calls.append("index_knowledge_pages")
        pages = self._pages(bank_id)
        for e in entries:
            pages[e.page_id] = e

    async def delete_knowledge_pages(self, bank_id, page_ids):
        self.calls.append("delete_knowledge_pages")
        pages = self._pages(bank_id)
        for pid in page_ids:
            pages.pop(pid, None)

    async def search_knowledge_pages(
        self, bank_id, *, embedding, text, limit, tags=None, tags_match="any", tag_groups=None
    ):
        self.calls.append("search_knowledge_pages")
        from hindsight_api.engine.memories.base import KnowledgePageMatch

        needle = (text or "").lower()
        hits = [
            e
            for e in self._pages(bank_id).values()
            if (not needle or needle in e.index_text.lower()) and (not tags or set(tags) & set(e.tags))
        ]
        return [KnowledgePageMatch(page_id=e.page_id, score=1.0) for e in hits][:limit]

    async def search_knowledge_pages_semantic(
        self,
        bank_id,
        *,
        embedding,
        limit,
        tags=None,
        tags_match="any",
        tag_groups=None,
        exclude_ids=None,
    ):
        self.calls.append("search_knowledge_pages_semantic")
        from hindsight_api.engine.memories.base import KnowledgePageMatch

        excluded = set(exclude_ids or ())
        hits = [
            e
            for e in self._pages(bank_id).values()
            if e.page_id not in excluded and (not tags or set(tags) & set(e.tags))
        ]
        return [KnowledgePageMatch(page_id=e.page_id, score=0.9) for e in hits][:limit]

    async def list_knowledge_pages(self, bank_id):
        self.calls.append("list_knowledge_pages")
        from hindsight_api.engine.memories.base import KnowledgePageRef

        return [KnowledgePageRef(page_id=e.page_id, updated_at=e.updated_at) for e in self._pages(bank_id).values()]


class _InMemoryRetainSession(RetainSession):
    """Buffer-then-commit. Deliberately the whole-retain policy rather than a per-part flush: it is
    the one that would break if `commit` ever stopped being called, so a leak in the orchestrator's
    session handling shows up as missing rows instead of passing on a partial write."""

    def __init__(self, store: "InMemoryMemories", bank_id: str):
        self._store = store
        self._bank_id = bank_id
        self._parts: list = []

    async def add(self, part) -> None:
        self._store.calls.append("session.add")
        self._parts.append(part)

    async def commit(self) -> RetainResult:
        self._store.calls.append("session.commit")
        unit_ids: dict[str, list[str]] = {}
        for part in self._parts:
            doc = self._store.documents.setdefault(part.document_id, {"chunks": [], "text": ""})
            if part.document_body is not None:
                doc["text"] = part.document_body
            if part.chunk_texts:
                # `chunk_offset` is per document, so a part is placed at its offset rather than
                # appended — two parts of one document can arrive in either order.
                needed = part.chunk_offset + len(part.chunk_texts)
                if len(doc["chunks"]) < needed:
                    doc["chunks"].extend([""] * (needed - len(doc["chunks"])))
                doc["chunks"][part.chunk_offset : needed] = list(part.chunk_texts)
            if part.facts:
                ids = self._store.allocate_unit_ids(len(part.facts))
                await self._store.index_facts(self._bank_id, ids, part.facts, part.document_id)
                unit_ids.setdefault(part.document_id, []).extend(ids)
        self._parts.clear()
        return RetainResult(unit_ids=unit_ids)

    async def abort(self) -> None:
        self._store.calls.append("session.abort")
        self._parts.clear()


@pytest.fixture
def restore_default_store():
    """Put the process-wide store back, whatever a test did to it."""
    yield
    set_memories(None)


def test_default_store_is_postgres(restore_default_store):
    """Nothing configured means the SQL path — which is what every other test runs."""
    set_memories(None)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HINDSIGHT_API_MEMORIES_EXTENSION", None)
        assert isinstance(create_memories(), PostgresMemories)
        assert get_memories().name == "postgres"


def test_a_configured_store_replaces_the_default(restore_default_store):
    """The ordinary extension env var selects the store, like every other extension."""
    set_memories(None)
    spec = f"{InMemoryMemories.__module__}:{InMemoryMemories.__name__}"
    with patch.dict(os.environ, {"HINDSIGHT_API_MEMORIES_EXTENSION": spec}):
        store = create_memories()
    assert isinstance(store, InMemoryMemories)
    assert store.name == "in-memory"


def test_the_store_receives_its_prefixed_config(restore_default_store):
    """`HINDSIGHT_API_MEMORIES_*` reaches the store, stripped and lowercased."""
    set_memories(None)
    spec = f"{InMemoryMemories.__module__}:{InMemoryMemories.__name__}"
    env = {
        "HINDSIGHT_API_MEMORIES_EXTENSION": spec,
        "HINDSIGHT_API_MEMORIES_TARGET": "example:50051",
        "HINDSIGHT_API_MEMORIES_NPROBE": "16",
    }
    with patch.dict(os.environ, env):
        store = create_memories()
    assert store.config["target"] == "example:50051"
    assert store.config["nprobe"] == "16"
    assert "extension" not in store.config, "the selector must not leak into the store's own config"


def test_the_interface_is_implementable_without_a_database(restore_default_store):
    """The point of the extraction: a store with no SQL behind it is a valid store.

    Instantiating an ABC with a missing method raises `TypeError` naming it, so a
    method added to the interface without a home here fails loudly rather than at
    the first call site that needs it.
    """
    store = InMemoryMemories({})
    assert isinstance(store, MemoriesExtension)


async def test_a_store_that_owns_its_rows_needs_no_postgres(restore_default_store):
    """Write, read back, and delete — with `conn` set to something unusable.

    Passing `None` where the Postgres store would expect a connection is the
    assertion: any code path that quietly reached for SQL would raise instead of
    returning the rows the store holds.
    """
    store = InMemoryMemories({})
    set_memories(store)

    class _Fact:
        fact_text = "the cat sat on the mat"
        fact_type = "world"
        tags = ["animals"]

    unit_ids = await store.insert_facts(conn=None, ops=None, bank_id="bank", facts=[_Fact()], document_id="doc-1")
    assert len(unit_ids) == 1

    got = await store.get_memories(conn=None, fq_table=None, bank_id="bank", unit_ids=unit_ids)
    assert [m.text for m in got] == ["the cat sat on the mat"]
    assert await store.count_memories(conn=None, fq_table=None, bank_id="bank") == {"world": 1}
    assert await store.list_tags(conn=None, fq_table=None, bank_id="bank") == {
        "items": [{"tag": "animals", "count": 1}],
        "total": 1,
        "limit": 100,
        "offset": 0,
    }

    # Deleting the document takes its memories with it, with no cascade to rely on.
    await store.delete_document(conn=None, fq_table=None, bank_id="bank", document_id="doc-1")
    assert await store.get_memories(conn=None, fq_table=None, bank_id="bank", unit_ids=unit_ids) == []
    assert "insert_facts" in store.calls and "get_memories" in store.calls


async def test_engine_list_tags_routes_through_the_installed_store(memory, request_context, restore_default_store):
    """Drive a real :class:`MemoryEngine` read with the dict store installed.

    Unlike the direct-call tests above, this goes through the engine: it acquires a live Postgres
    connection and hands it to the store. The tag lives only in the stub's dict and in no
    ``memory_units`` row, so the engine can only return it by routing through the interface — if it
    reached past the seam to query SQL it would come back empty.
    """
    store = InMemoryMemories({})
    set_memories(store)

    class _Fact:
        fact_text = "the cat sat on the mat"
        fact_type = "world"
        tags = ["only-in-the-store"]

    await store.insert_facts(conn=None, ops=None, bank_id="seam-bank", facts=[_Fact()], document_id="d")

    result = await memory.list_tags("seam-bank", request_context=request_context)

    assert result["items"] == [{"tag": "only-in-the-store", "count": 1}]
    assert "list_tags" in store.calls


async def test_maintenance_passes_are_optional(restore_default_store):
    """A store with inline links has nothing to relink and no join table to sweep.

    These have safe base implementations precisely so such a store does not have
    to write four no-op methods to be complete — and so the maintenance job can
    call them unconditionally.
    """
    store = InMemoryMemories({})
    assert await store.enqueue_relink_victims(conn=None, fq_table=None, bank_id="b", affected_unit_ids=["x"]) == 0
    assert await store.relink_pass(backend=None, fq_table=None, bank_id="b", config=None) == RelinkPassResult()
    assert (
        await store.enqueue_entity_prune_candidates(conn=None, fq_table=None, bank_id="b", affected_unit_ids=["x"]) == 0
    )
    assert await store.entity_prune_pass(backend=None, fq_table=None, bank_id="b") == EntityPrunePassResult()
    # And recording entity postings is a no-op rather than an error: the posting
    # travels on the memory for a store that owns it.
    await store.record_unit_entities(conn=None, ops=None, fq_table=None, unit_ids=["u"], entity_ids=["e"])


# ---------------------------------------------------------------------------
# Per-bank store capabilities. A store may route different banks to different
# backends, so every BANK-SCOPED call site asks per bank —
# store_owned_for(bank_id) —
# rather than reading the process-global class attribute. The class attribute
# stays the single-store default the _for methods fall back to.
# ---------------------------------------------------------------------------


def test_per_bank_capability_defaults_to_the_class_attribute():
    """A single-store extension needs no override: the _for methods return the class attr, so
    every existing store keeps its exact behaviour for every bank."""
    pg = PostgresMemories({})
    assert pg.store_owned is False
    assert pg.store_owned_for("any-bank") is False

    mem = InMemoryMemories({})  # owns its rows AND its document store
    assert mem.store_owned is True
    assert mem.store_owned_for("any-bank") is True


def test_a_store_answers_capabilities_per_bank():
    """The point of the _for methods: a store that keeps some banks in SQL and others in a
    separate store answers PER BANK, so mixed banks in one process each take the right path."""

    class PerBankStore(InMemoryMemories):
        name = "per-bank"
        # The process-level class attr stays False; the per-bank answer is what every
        # bank-scoped site consults.
        store_owned = False

        def __init__(self, config=None):
            super().__init__(config)
            self.sql_banks = {"legacy-bank"}

        def store_owned_for(self, bank_id):
            return bank_id not in self.sql_banks

    store = PerBankStore({})
    # A SQL-backed bank looks like Postgres: the caller writes the rows and the documents.
    assert store.store_owned_for("legacy-bank") is False
    # ...a store-backed bank owns its rows, its document store and its retain.
    assert store.store_owned_for("new-bank") is True
    # The process-level gate (the cross-store recovery loop) still reads the class attribute, which
    # stays False so that loop keeps running even though some banks answer True per bank.
    assert store.store_owned is False


def test_assert_writable_defaults_to_allowing_everything():
    """No existing store needs a change: the default is a no-op, so a store that has never heard
    of write windows keeps taking every write."""
    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        PostgresMemories({}).assert_writable("any-bank")
    )


async def test_retain_refuses_a_bank_the_store_has_closed_to_writes(restore_default_store):
    """A retain writes documents, chunks and entities through SQL paths that never reach the
    memories interface, so a store cannot close a bank from its own methods alone. `retain_batch`
    asks first — and asks BEFORE anything is written, which is what this pins: every other
    argument here is None, so a guard that ran even one step late would raise something else.
    """
    from hindsight_api.engine.memories.base import StoreWriteUnavailable
    from hindsight_api.engine.retain import orchestrator

    class ClosedForWrites(InMemoryMemories):
        name = "closed"

        async def assert_writable(self, bank_id):
            if bank_id == "frozen-bank":
                raise StoreWriteUnavailable(f"bank {bank_id} is mid-cutover")

    set_memories(ClosedForWrites({}))

    with pytest.raises(StoreWriteUnavailable, match="mid-cutover"):
        await orchestrator.retain_batch(None, None, None, None, None, "frozen-bank", [{"content": "hello"}], None)


# ---------------------------------------------------------------------------
# Interface conformance: the stub must stay a COMPLETE, signature-compatible
# implementation of every MemoriesExtension method. This is the guard that keeps
# a future change to a provider method (a new method, a renamed/added parameter)
# from silently slipping past the seam — add or change one on the interface and
# these fail until the stub (and therefore every real provider) is updated to match.
# ---------------------------------------------------------------------------

import inspect  # noqa: E402


def _iface_methods() -> dict:
    """Every public method MemoriesExtension declares in its own body (not properties)."""
    out = {}
    for name, obj in vars(MemoriesExtension).items():
        if name.startswith("_"):
            continue
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if inspect.isfunction(obj):
            out[name] = obj
    return out


def _param_info(func) -> tuple[set[str], bool]:
    names: set[str] = set()
    has_var_kw = False
    for pname, p in inspect.signature(func).parameters.items():
        if pname == "self":
            continue
        if p.kind is p.VAR_KEYWORD:
            has_var_kw = True
        elif p.kind is p.VAR_POSITIONAL:
            continue
        else:
            names.add(pname)
    return names, has_var_kw


def _raises_not_implemented(func) -> bool:
    try:
        return "raise NotImplementedError" in inspect.getsource(func)
    except (OSError, TypeError):
        return False


_IFACE = _iface_methods()


def test_stub_instantiates_all_abstract_methods_present():
    """A new @abstractmethod on the interface makes this fail — the stub can't be built."""
    InMemoryMemories({})


@pytest.mark.parametrize("method_name", sorted(_IFACE))
def test_stub_answers_every_interface_method(method_name):
    """The stub must not fall through to a ``NotImplementedError`` default for any method — a
    store that owns its rows has to answer them all. The interface's *safe* defaults (the no-op
    maintenance / lifecycle hooks that return None/0/{}) may be inherited; a method the
    interface leaves as ``raise NotImplementedError`` must be implemented here. Add such a method
    to the interface and this surfaces it, instead of a real provider discovering it in production."""
    if method_name in vars(InMemoryMemories):
        return  # explicitly implemented
    assert not _raises_not_implemented(_IFACE[method_name]), (
        f"InMemoryMemories inherits the NotImplementedError default for {method_name!r}; "
        f"implement it in the stub so the seam is a complete store."
    )


@pytest.mark.parametrize("method_name", sorted(_IFACE))
def test_stub_signature_accepts_interface_params(method_name):
    """The stub accepts every parameter the interface declares — a renamed/added provider
    parameter (the class of bug where the engine passes a kwarg the store never expected)
    fails here. Resolves inherited methods so safe-default hooks are checked too."""
    iface_params, _ = _param_info(_IFACE[method_name])
    stub_params, stub_var_kw = _param_info(getattr(InMemoryMemories, method_name))
    missing = iface_params - stub_params
    assert not missing or stub_var_kw, f"{method_name}: stub is missing interface params {missing}"


# ---------------------------------------------------------------------------
# Engine-driven flows: the same operations a client drives, with the stub
# installed. These prove the ENGINE routes each through the interface — if a
# call site reached past the seam to SQL it would miss the stub's dict entirely.
# ---------------------------------------------------------------------------


class _Fact:
    def __init__(self, text="a fact", fact_type="world", tags=None, document_id=None):
        self.fact_text = text
        self.fact_type = fact_type
        self.tags = tags or []
        self.document_id = document_id


async def _seed(store, bank_id, **fact_kwargs):
    return await store.insert_facts(
        conn=None, ops=None, bank_id=bank_id, facts=[_Fact(**fact_kwargs)], document_id=fact_kwargs.get("document_id")
    )


async def test_engine_recall_routes_search_through_store(memory, request_context, restore_default_store):
    store = InMemoryMemories({})
    set_memories(store)
    await memory.recall_async("seam-bank", "anything at all", request_context=request_context)
    assert "search" in store.calls  # the semantic/bm25 arm went through the interface, not SQL


async def test_engine_list_memory_units_routes_through_store(memory, request_context, restore_default_store):
    store = InMemoryMemories({})
    set_memories(store)
    await _seed(store, "seam-bank", text="only in the store", fact_type="world")
    res = await memory.list_memory_units("seam-bank", request_context=request_context)
    assert "list_memory_units" in store.calls
    assert res["total"] == 1  # the row exists only in the stub, so it can only have come from it


async def test_an_edit_writes_its_vector_in_the_edit_itself(memory, request_context, restore_default_store):
    """Applying an edit writes its re-embedded vector — with no second call to do it.

    The property is not "the vector ends up stored": that passed while the edit was making two
    writes to get it there, which is the whole cost of an edit again for a store whose write is a
    durable append rather than a row update. It is that the vector arrives through `apply_edit`
    and that nothing follows it to write the same row again.
    """
    store = InMemoryMemories({})
    set_memories(store)
    unit_ids = await _seed(store, "seam-bank", text="before the edit", fact_type="world")

    await memory.update_memory_unit("seam-bank", unit_ids[0], text="after the edit", request_context=request_context)

    assert "apply_edit" in store.calls
    assert store.edit_embedding is not None, "the store is handed the re-embedded vector"
    assert store.embeddings.get(unit_ids[0]) is not None, "…and applying the edit stored it"
    assert "set_memory_embedding" not in store.calls, "a second write of the row just written"


async def test_apply_edit_is_told_the_pre_edit_fact_type(memory, request_context, restore_default_store):
    """A store that can apply a type-preserving edit more cheaply needs to know the type did not
    change, and re-reading the memory to find that out costs the round trip the cheap path saves.
    The caller has it from the re-read it already did, so it passes it."""
    store = InMemoryMemories({})
    set_memories(store)
    unit_ids = await _seed(store, "seam-bank", text="before the edit", fact_type="world")

    await memory.update_memory_unit("seam-bank", unit_ids[0], text="after the edit", request_context=request_context)

    assert store.edit_current_fact_type == "world"


async def test_engine_list_entities_routes_through_store(memory, request_context, restore_default_store):
    store = InMemoryMemories({})
    set_memories(store)
    await memory.list_entities("seam-bank", request_context=request_context)
    assert "list_entities" in store.calls


async def test_engine_clear_observations_routes_through_store(memory, request_context, restore_default_store):
    store = InMemoryMemories({})
    set_memories(store)
    # Seed an observation directly through the interface, then clear via the engine.
    from hindsight_api.engine.memories.base import FactRecord

    await store.upsert_observation(
        conn=None,
        bank_id="seam-bank",
        record=FactRecord(
            unit_id="11111111-1111-1111-1111-111111111111", text="an obs", embedding=None, fact_type="observation"
        ),
    )
    assert any(r.fact_type == "observation" for r in store.rows.values())

    await memory.clear_observations("seam-bank", request_context=request_context)

    assert "delete_observations" in store.calls  # the engine's clear reached the store
    assert not any(r.fact_type == "observation" for r in store.rows.values())


# ---------------------------------------------------------------------------
# Recall enrichment through the seam. These lock the input combinations that a
# store owning its rows/bodies (store_owned=True) must hydrate FROM THE STORE,
# not from SQL. The class
# of bug they guard against: recall(include_chunks=True) through such a store
# returned EMPTY chunk text because the engine read the (empty) SQL chunks row
# and never overlaid the store's body. Each test seeds a memory in the stub,
# whatever SQL metadata row the enrichment reads, and the body in the store,
# then asserts the store's content is what comes back.
# ---------------------------------------------------------------------------


def _stored(unit_id, text, fact_type, **kw):
    """A ranked-search candidate as the store holds it (bypassing insert_facts so
    chunk_id / entity_ids / source_memory_ids can be set directly)."""
    return StoredMemory(unit_id=unit_id, text=text, fact_type=fact_type, created_at=datetime.now(timezone.utc), **kw)


async def test_store_document_bodies_carries_retain_params(restore_default_store):
    """A store that owns the whole retain has NO SQL `documents` row, so anything the write path
    does not put in the store record is lost outright.

    `retain_params` was passed as `metadata={}`, which is why `get_document` returned null
    `retain_params` / `document_metadata` / `observation_scopes` for such a bank. The store's
    metadata map is string -> string, so the params ride as one JSON value.
    """
    import json as _json

    from hindsight_api.engine.retain.orchestrator import _store_document_bodies

    store = InMemoryMemories({})
    set_memories(store)
    suffix = uuid.uuid4().hex[:8]
    bank_id = f"seam-params-{suffix}"
    doc_id = f"doc-{suffix}"
    params = {"metadata": {"source": "upload"}, "observation_scopes": ["team"], "chunk_size": 800}

    class _Cfg:
        store_document_text = True

    await _store_document_bodies(
        bank_id=bank_id,
        document_id=doc_id,
        combined_content="body text",
        chunk_texts=["body text"],
        merged_tags=["t"],
        config=_Cfg(),
        retain_params=params,
    )

    carried = store.documents[doc_id]["metadata"].get("retain_params")
    assert carried is not None, "retain_params must reach the store record, not be dropped"
    assert _json.loads(carried) == params, carried


async def test_store_document_bodies_omits_absent_retain_params(restore_default_store):
    """No params ⇒ no key, rather than a "null" string that the read would then hand back as the
    literal string. The read path json-parses whatever is there."""
    from hindsight_api.engine.retain.orchestrator import _store_document_bodies

    store = InMemoryMemories({})
    set_memories(store)
    suffix = uuid.uuid4().hex[:8]
    doc_id = f"doc-{suffix}"

    class _Cfg:
        store_document_text = True

    await _store_document_bodies(
        bank_id=f"seam-noparams-{suffix}",
        document_id=doc_id,
        combined_content="body",
        chunk_texts=["body"],
        merged_tags=None,
        config=_Cfg(),
        retain_params=None,
    )
    assert "retain_params" not in store.documents[doc_id]["metadata"]


async def test_recall_include_chunks_hydrates_body_from_store(memory, request_context, restore_default_store):
    """include_chunks must overlay chunk TEXT from the store, not the empty SQL chunks row.

    This is the regression guard: the SQL ``chunks`` row a store that owns bodies writes carries an
    EMPTY ``chunk_text``; the real text lives in the store. Remove the overlay in ``recall_async``
    (~line 5271, the ``store_owned`` block) and the returned text falls back to that empty
    string — this assertion then fails, which is exactly the bug that slipped through before.
    """
    store = InMemoryMemories({})
    set_memories(store)
    suffix = uuid.uuid4().hex[:8]
    bank_id = f"seam-chunks-{suffix}"
    doc_id = f"doc-{suffix}"
    chunk_id = f"chunk-{suffix}"
    fact_id = str(uuid.uuid4())
    body = "the topological qubit stayed coherent for a record two hundred microseconds"

    # The store owns the chunk body ...
    await store.put_document(
        bank_id=bank_id, document_id=doc_id, content_hash="h", original_text=body, chunk_texts=[body]
    )
    # ... while the SQL documents/chunks rows carry only metadata (empty text for this store).
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash) VALUES ($1, $2, '', 'h')",
            doc_id,
            bank_id,
        )
        await conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text) VALUES ($1, $2, $3, 0, '')",
            chunk_id,
            doc_id,
            bank_id,
        )
    # The fact the search returns carries the chunk_id + document_id linking to that row.
    store.rows[fact_id] = _stored(
        fact_id,
        "topological qubits show promise for scalable machines",
        "world",
        document_id=doc_id,
        chunk_id=chunk_id,
    )

    try:
        result = await memory.recall_async(
            bank_id=bank_id,
            query="topological qubit coherence",
            fact_type=["world"],
            max_tokens=4096,
            include_chunks=True,
            max_chunk_tokens=2000,
            request_context=request_context,
        )
        assert "search" in store.calls  # candidates came through the seam
        assert result.chunks and chunk_id in result.chunks, f"chunk missing from {result.chunks}"
        assert result.chunks[chunk_id].chunk_text == body, (
            "chunk text must be overlaid from the store's body, not the empty SQL chunks row"
        )
        # The overlay went through the interface. The METHOD is deliberately not pinned: hydration
        # moved from `list_chunk_texts` (which downloads a document's whole packed chunk blob) to
        # the addressed reads, because a many-chunk recall was paying that blob per document. What
        # matters is that the body came from the store rather than the empty SQL chunks row, which
        # the assertion above already proves; this checks it did not arrive by some path outside
        # the seam.
        assert {"list_chunk_texts", "get_chunk_texts", "get_chunk_text"} & set(store.calls), (
            f"no chunk read went through the interface: {store.calls}"
        )
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)


async def test_recall_include_source_facts_hydrates_from_store(memory, request_context, restore_default_store):
    """include_source_facts must resolve an observation's source facts FROM THE STORE."""
    store = InMemoryMemories({})
    set_memories(store)
    bank_id = f"seam-srcfacts-{uuid.uuid4().hex[:8]}"
    obs_id = str(uuid.uuid4())
    src_id = str(uuid.uuid4())
    src_text = "Alice deployed the hotfix on Tuesday afternoon"

    store.rows[src_id] = _stored(src_id, src_text, "world")
    store.rows[obs_id] = _stored(
        obs_id, "Alice ships fixes reliably", "observation", proof_count=1, source_memory_ids=[src_id]
    )

    result = await memory.recall_async(
        bank_id=bank_id,
        query="Alice reliable fixes",
        fact_type=["observation"],
        max_tokens=4096,
        include_source_facts=True,
        request_context=request_context,
    )

    by_id = {str(r.id): r for r in result.results}
    assert obs_id in by_id, f"observation missing from results {list(by_id)}"
    assert result.source_facts and src_id in result.source_facts, f"source fact missing from {result.source_facts}"
    assert result.source_facts[src_id].text == src_text
    assert by_id[obs_id].source_fact_ids == [src_id]


@pytest.mark.parametrize("carries_ids_on_result", [False, True], ids=["map-refetch", "ids-on-result"])
async def test_recall_include_entities_hydrates_names_from_registry(
    carries_ids_on_result, memory, request_context, restore_default_store
):
    """include_entities must resolve the memory's inline entity ids to NAMES from the registry.

    The store owns the memory rows (and the entity ids on them), but the entity name registry stays
    in Postgres — so the names can only come back by resolving through it, not from the dict.

    Parametrized over both recall paths, which MUST agree:
    - ``map-refetch``: the result carries no entity ids, so recall re-fetches the map via
      ``entity_map_for_units`` (the default store's behaviour);
    - ``ids-on-result``: the store carries the ids on the result, so recall builds the map from
      them and resolves only names via ``resolve_entity_names`` — no re-fetch.
    """
    store = InMemoryMemories({})
    store.carries_entity_ids_on_result = carries_ids_on_result
    set_memories(store)
    bank_id = f"seam-entities-{uuid.uuid4().hex[:8]}"
    fact_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        ent_id = await conn.fetchval(
            "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES ($1, $2, 1) RETURNING id",
            bank_id,
            "Quantum Lab",
        )
    store.rows[fact_id] = _stored(fact_id, "the quantum lab published its results", "world", entity_ids=[str(ent_id)])

    try:
        result = await memory.recall_async(
            bank_id=bank_id,
            query="quantum lab results",
            fact_type=["world"],
            max_tokens=4096,
            include_entities=True,
            max_entity_tokens=2000,
            request_context=request_context,
        )
        by_id = {str(r.id): r for r in result.results}
        assert fact_id in by_id, f"fact missing from results {list(by_id)}"
        assert by_id[fact_id].entities == ["Quantum Lab"], (
            f"entity id must resolve to the registry name; got {by_id[fact_id].entities}"
        )
        assert result.entities and "Quantum Lab" in result.entities
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.parametrize("carries_ids_on_result", [False, True], ids=["map-refetch", "ids-on-result"])
async def test_recall_include_entities_omits_entityless_unit(
    carries_ids_on_result, memory, request_context, restore_default_store
):
    """A unit with no entities is OMITTED, not emitted as ``entities=[]`` — in both recall paths.

    One entity-bearing unit and one entity-less unit in the same recall: the bearing unit resolves
    to its registry name, the entity-less unit comes back with ``entities is None`` (never ``[]``),
    and the aggregate ``entities`` dict holds only the one real entity. The ``ids-on-result`` fast
    path and the ``map-refetch`` path must agree exactly.
    """
    store = InMemoryMemories({})
    store.carries_entity_ids_on_result = carries_ids_on_result
    set_memories(store)
    bank_id = f"seam-entities-omit-{uuid.uuid4().hex[:8]}"
    with_entity_id = str(uuid.uuid4())
    without_entity_id = str(uuid.uuid4())

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        ent_id = await conn.fetchval(
            "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES ($1, $2, 1) RETURNING id",
            bank_id,
            "Quantum Lab",
        )
    store.rows[with_entity_id] = _stored(
        with_entity_id, "the quantum lab published its results", "world", entity_ids=[str(ent_id)]
    )
    store.rows[without_entity_id] = _stored(without_entity_id, "the cafeteria changed its menu", "world", entity_ids=[])

    try:
        result = await memory.recall_async(
            bank_id=bank_id,
            query="lab and cafeteria",
            fact_type=["world"],
            max_tokens=4096,
            include_entities=True,
            max_entity_tokens=2000,
            request_context=request_context,
        )
        by_id = {str(r.id): r for r in result.results}
        assert with_entity_id in by_id and without_entity_id in by_id, f"both facts expected; got {list(by_id)}"
        assert by_id[with_entity_id].entities == ["Quantum Lab"]
        assert by_id[without_entity_id].entities is None, (
            f"entity-less unit must be omitted (entities=None), not entities=[]; "
            f"got {by_id[without_entity_id].entities!r}"
        )
        assert result.entities and list(result.entities) == ["Quantum Lab"]
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


@pytest.mark.parametrize(
    "ids_by_unit, names, expected",
    [
        # entity-less unit is omitted, not mapped to []
        ({"u1": ["e1"], "u2": []}, {"e1": "Alpha"}, {"u1": [{"entity_id": "e1", "canonical_name": "Alpha"}]}),
        # a repeated id yields one row, first-seen order preserved
        (
            {"u1": ["e1", "e2", "e1"]},
            {"e1": "Alpha", "e2": "Beta"},
            {"u1": [{"entity_id": "e1", "canonical_name": "Alpha"}, {"entity_id": "e2", "canonical_name": "Beta"}]},
        ),
        # an id with no resolved name is dropped; a unit left with none is omitted
        ({"u1": ["e1", "eX"], "u2": ["eX"]}, {"e1": "Alpha"}, {"u1": [{"entity_id": "e1", "canonical_name": "Alpha"}]}),
        # nothing to resolve -> empty map
        ({"u1": []}, {}, {}),
    ],
)
def test_entity_map_from_results_shape(ids_by_unit, names, expected):
    """The pure fast-path helper mirrors entity_map_for_units: per-unit order-preserving dedupe,
    unresolved ids dropped, entity-less units omitted (never ``[]``). DB-free."""
    from hindsight_api.engine.memory_engine import _entity_map_from_results

    assert _entity_map_from_results(ids_by_unit, names) == expected


@pytest.mark.parametrize("carries_source_ids", [False, True], ids=["refetch", "ids-on-result"])
async def test_recall_all_enrichments_together_through_store(
    memory, request_context, restore_default_store, carries_source_ids
):
    """All three flags at once, with prefer_observations=True, through the owning store.

    The observation supersedes its raw source fact (prefer_observations drops it from results), so:
    - chunks resolve via the observation's ``source_memory_ids`` -> source fact's chunk_id -> the
      store's body (the O(documents) overlay path);
    - source_facts come back populated from the store's rows;
    - entities on the observation resolve to registry names.
    One recall exercises the whole enrichment surface for a store that owns its rows and bodies.

    Parametrized over the two ways recall can learn an observation's sources, because both consumers
    of that list are in this one recall:

    - ``refetch``: the result carries none, so the dedup and the chunk walk each re-fetch the
      observation to read them — an addressed read apiece, which for a store whose reads are round
      trips is most of what those steps cost;
    - ``ids-on-result``: the store carried them on the hydrated result and neither read happens.

    The assertions are the same for both. That is the point: the output must not depend on which
    path recall took, or the fast path is an approximation rather than an optimisation.
    """
    store = InMemoryMemories({})
    store.carries_source_ids_on_result = carries_source_ids
    set_memories(store)
    suffix = uuid.uuid4().hex[:8]
    bank_id = f"seam-all-{suffix}"
    doc_id = f"doc-{suffix}"
    chunk_id = f"chunk-{suffix}"
    src_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())
    body = "Alice migrated the billing service to the new cluster over the weekend"
    src_text = "Alice migrated the billing service to the new cluster"

    await store.put_document(
        bank_id=bank_id, document_id=doc_id, content_hash="h", original_text=body, chunk_texts=[body]
    )

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash) VALUES ($1, $2, '', 'h')",
            doc_id,
            bank_id,
        )
        await conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text) VALUES ($1, $2, $3, 0, '')",
            chunk_id,
            doc_id,
            bank_id,
        )
        ent_id = await conn.fetchval(
            "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES ($1, $2, 1) RETURNING id",
            bank_id,
            "billing service",
        )

    # Raw source fact carries the chunk; the observation consolidates it and carries the entity.
    store.rows[src_id] = _stored(src_id, src_text, "world", document_id=doc_id, chunk_id=chunk_id)
    store.rows[obs_id] = _stored(
        obs_id,
        "Alice handles infra migrations",
        "observation",
        proof_count=1,
        source_memory_ids=[src_id],
        entity_ids=[str(ent_id)],
    )

    try:
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Alice billing migration",
            fact_type=["world", "observation"],
            max_tokens=4096,
            prefer_observations=True,
            include_chunks=True,
            max_chunk_tokens=2000,
            include_source_facts=True,
            include_entities=True,
            max_entity_tokens=2000,
            request_context=request_context,
        )

        by_id = {str(r.id): r for r in result.results}
        assert obs_id in by_id, f"observation missing from results {list(by_id)}"
        assert src_id not in by_id, "prefer_observations should have dropped the superseded raw fact"

        # chunks: resolved through the observation's source and overlaid from the store body
        assert result.chunks and chunk_id in result.chunks
        assert result.chunks[chunk_id].chunk_text == body

        # source_facts: the raw fact, populated from the store
        # And the fast path really is one: carrying the sources removes THREE addressed reads
        # from this single recall — the prefer-observations dedup's, the chunk walk's and the
        # source-facts block's, each of which re-read an observation hydration had already read.
        # Asserted as a count rather than "it was faster", because the point is WHICH reads stopped
        # happening; the two that remain fetch the observation's SOURCES, memories recall never
        # retrieved, for their chunk ids and their text.
        assert store.calls.count("get_memories") == (2 if carries_source_ids else 5), store.calls

        assert result.source_facts and src_id in result.source_facts
        assert result.source_facts[src_id].text == src_text
        assert by_id[obs_id].source_fact_ids == [src_id]

        # entities: the observation's inline id resolved to the registry name
        assert by_id[obs_id].entities == ["billing service"]
        assert result.entities and "billing service" in result.entities
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM documents WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)


# ---------------------------------------------------------------------------
# Bank storage lifecycle. A store that owns its storage relies on exactly one
# thing: the storage exists for as long as the bank does. Both halves live here
# because neither is visible from the store's own suite — they are engine
# behaviour, and a store can only observe them through these calls.
#
# `_ensure_bank_exists` is called directly rather than through a public entry
# point: it IS the seam under test, and reaching it via retain would drag in
# embeddings and an LLM to assert something neither is involved in.
# ---------------------------------------------------------------------------


async def test_ensuring_a_bank_ensures_its_storage(memory, request_context, restore_default_store):
    """A bank that exists must have storage behind it.

    Without this the storage is only ever created implicitly by a write, so "bank exists, storage
    does not" is a routine state every read path has to decide what to do about. The store cannot
    tell that apart from an empty bank, so it reports empty — and a bank whose storage went
    missing then serves empty results, successfully, until somebody notices the data is gone.
    """
    store = InMemoryMemories({})
    set_memories(store)

    await memory._ensure_bank_exists("lifecycle-bank", request_context)

    assert "ensure_bank_storage" in store.calls
    assert "lifecycle-bank" in store.ensured


async def test_ensuring_an_existing_bank_still_ensures_its_storage(memory, request_context, restore_default_store):
    """Ensuring is not gated on the bank row having just been inserted.

    `created` is False for a bank that already exists in SQL but has no storage — one predating
    this call, or whose storage went away out of band. Those are exactly the banks that need it,
    so a second ensure must still reach the store. That is also what backfills them.
    """
    store = InMemoryMemories({})
    set_memories(store)
    await memory._ensure_bank_exists("twice-bank", request_context)
    store.calls.clear()
    store.ensured.clear()

    await memory._ensure_bank_exists("twice-bank", request_context)

    assert "ensure_bank_storage" in store.calls, "the second ensure must still reach the store"
    assert "twice-bank" in store.ensured


async def test_clearing_a_banks_memories_keeps_its_storage(memory, request_context, restore_default_store):
    """Clearing empties the bank; it does not delete the bank's storage.

    `delete_bank_profile=False` is the API's clear endpoint — the bank goes on existing. Dropping
    the storage there left it existing in SQL with nothing behind it, and discarded the
    namespace's declared metadata keys, which are fixed at creation: the facets then stayed empty
    even after the bank was written to again.
    """
    store = InMemoryMemories({})
    set_memories(store)
    await memory._ensure_bank_exists("clear-bank", request_context)
    await _seed(store, "clear-bank", text="wipe me")
    assert store.rows

    await memory.delete_bank("clear-bank", delete_bank_profile=False, request_context=request_context)

    assert "drop_bank_storage" not in store.calls, "a clear must not delete the bank's storage"
    assert "clear-bank" in store.ensured, "the storage outlives the memories it held"
    assert store.predicates and store.predicates[-1].delete_all, "an unfiltered clear is a delete-all"
    assert store.rows == {}, "and it really did empty the bank"


async def test_deleting_a_bank_drops_its_storage(memory, request_context, restore_default_store):
    """The other side of the same coin: when the BANK goes, its storage goes with it.

    This direction needs guarding as much as the first. Routing a real bank deletion to a
    delete-all would leave the namespace behind for a bank that no longer exists — storage nothing
    will ever read, delete, or account for.
    """
    store = InMemoryMemories({})
    set_memories(store)
    await memory._ensure_bank_exists("doomed-bank", request_context)
    await _seed(store, "doomed-bank", text="going away")

    await memory.delete_bank("doomed-bank", request_context=request_context)

    assert "drop_bank_storage" in store.calls, "deleting the bank must drop its storage, not orphan it"
    assert "doomed-bank" not in store.ensured


# ---------------------------------------------------------------------------
# Retain routing: which write path a bank takes
# ---------------------------------------------------------------------------


def test_every_bank_deltas_on_the_first_sub_batch():
    """Delta is not gated on the store any more.

    A store-owned bank's delta is one `retain` whose replace names the chunks that moved, so there
    is nothing left for a store to be unable to express. What remains is the sub-batch rule: delta
    runs on the FIRST sub-batch only, because the caller keeps one result list per sub-batch item
    and advances `chunk_index_offset` by the splitter's per-slice count rather than by what a delta
    wrote.
    """
    from hindsight_api.engine.retain.orchestrator import attempts_delta_retain

    class StoreOwned(InMemoryMemories):
        name = "store-owned"
        store_owned = True

    for store in (InMemoryMemories({}), StoreOwned({})):
        assert attempts_delta_retain(store, "b", True) is True
        assert attempts_delta_retain(store, "b", False) is False


async def test_store_owned_bank_stops_writing_the_postgres_page_search_columns(
    memory, request_context, restore_default_store
):
    """Postgres keeps the page ROW and stops carrying its two derived search columns.

    This is what makes leaving `idx_mental_models_embedding` / `idx_mental_models_text_search` in
    place cheap for a store-owned bank: the indexes still exist for the Postgres banks in the same
    schema, but nothing new is written into them here, so no ANN insert and (on VectorChord) no
    `tokenize()` call through the extension is paid per page write.

    Asserted on the columns rather than on a call count, because the cost being avoided is the
    index maintenance the column write triggers, not the code path.
    """
    from hindsight_api.engine.db_utils import acquire_with_retry
    from hindsight_api.engine.schema import fq_table

    store = InMemoryMemories({})
    set_memories(store)
    bank = "seam-kp-bank"
    try:
        page = await memory.create_knowledge_page(
            bank,
            name="Quarterly revenue",
            source_query="revenue",
            content="Revenue forecast for the platform team.",
            request_context=request_context,
        )
        # It reached the store's index -- otherwise "not in Postgres" would just mean "lost".
        assert "index_knowledge_pages" in store.calls
        assert store.knowledge_pages.get(bank), "the page must be searchable in the store"

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"SELECT name, content, embedding, search_vector "
                f"FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                bank,
                page["mental_model_id"],
            )
        assert row is not None, "the page row itself must still live in Postgres"
        # The readable half is untouched: this is not the row moving.
        assert row["name"] == "Quarterly revenue"
        assert row["content"]
        # The ANN column is not written, so `idx_mental_models_embedding` takes no insert for this
        # bank. True on every backend, because the application is what writes this column.
        assert row["embedding"] is None, "the ANN column must not be maintained for a store-owned bank"
        # `search_vector` is deliberately NOT asserted NULL. On vchord the application writes it and
        # the gate does skip it, but on the native backend the column is
        # `GENERATED ALWAYS AS (to_tsvector(...)) STORED`, so Postgres computes it and maintains the
        # GIN index on every write whatever the application does. Only dropping the column or the
        # index stops that, and both are per-SCHEMA, so they need the deployment-level decision this
        # per-bank flag cannot make. Asserting NULL here would encode a saving that does not exist.
    finally:
        await memory.delete_bank(bank, request_context=request_context)
