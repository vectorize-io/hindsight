"""The default memories store: Postgres holds the memories and the links.

This is the behaviour Hindsight has always had, stated as an implementation of
:class:`~hindsight_api.engine.memories.base.MemoriesExtension` rather than as the
absence of one. Rows go in `memory_units`, the joins around it are `memory_links`
and `unit_entities`, and every read is SQL — writing a row *is* indexing it, so
:meth:`index_facts` has nothing left to do.

The class is deliberately thin. Each method delegates to a plain function in
:mod:`hindsight_api.engine.memories.pg`, split by what calls it — curation,
graph, reads, writes — so a change to one area is a change to one file, and the
SQL is grouped by concern rather than piled behind a class. The two retrieval
arms delegate further out still, to the query functions that already own them in
:mod:`hindsight_api.engine.search.retrieval`.

Keeping this as an explicit store (rather than an ``if store is None`` branch at
each call site) means the default path is the one the whole test suite exercises,
and a second implementation cannot change it by accident.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .base import DeletePredicate, MemoriesExtension, MemoryPatch, ScanPage, StoredMemory
from .pg import curation, graph, reads, writes


class PostgresMemories(MemoriesExtension):
    """Memories in `memory_units`, links in `memory_links` / `unit_entities`."""

    name = "postgres"

    # ------------------------------------------------------------------ writes

    async def insert_facts(
        self,
        *,
        conn,
        ops,
        bank_id: str,
        facts: list,
        document_id: str | None = None,
        defer_index: bool = False,
    ) -> list[str]:
        # `defer_index` is meaningless here: the INSERT that returns the ids is
        # also what indexes the facts, so there is nothing to defer.
        return await writes.insert_facts(conn=conn, ops=ops, bank_id=bank_id, facts=facts, document_id=document_id)

    async def delete_facts(self, bank_id: str, unit_ids: list[str]) -> None:
        """No-op: the caller's `memory_units` DELETE (or its FK cascade) removed them."""

    async def delete_where(self, bank_id: str, predicate: DeletePredicate) -> int:
        """No-op: predicate deletes are issued as SQL by the caller that owns the transaction."""
        return 0

    async def delete_document(self, *, conn, fq_table, bank_id: str, document_id: str) -> None:
        await writes.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)

    async def delete_namespace(self, bank_id: str) -> None:
        """No-op: deleting the bank cascades to its memories."""

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        await writes.delete_observations(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch]) -> None:
        """No-op: the caller's UPDATE already wrote the row it holds open."""

    # ------------------------------------------------------------------ recall arms

    async def search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
    ) -> dict[str, tuple[list, list]]:
        # Imported here: retrieval imports this package, so a module-level import
        # would close the cycle.
        from ..search.retrieval import retrieve_semantic_bm25_combined_sql

        return await retrieve_semantic_bm25_combined_sql(
            conn,
            query_embedding,
            query_text,
            bank_id,
            fact_types,
            limit,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
            min_semantic=min_semantic,
            min_keyword=min_keyword,
        )

    async def temporal_search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list]:
        from ..search.retrieval import retrieve_temporal_combined_sql

        return await retrieve_temporal_combined_sql(
            conn,
            query_embedding,
            bank_id,
            fact_types,
            start_date,
            end_date,
            limit,
            semantic_threshold=semantic_threshold,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            created_after=created_after,
            created_before=created_before,
        )

    # ------------------------------------------------------------------ addressed reads

    async def get_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
        return await reads.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def scan_memories(
        self,
        *,
        conn,
        fq_table,
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
        return await reads.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            page_token=page_token,
            tags=tags,
            tags_match=tags_match,
            document_id=document_id,
            metadata_equals=metadata_equals,
            skip=skip,
            include_edges=include_edges,
        )

    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await reads.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def list_tags(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await reads.list_tags(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def find_unconsolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        scope_tags: list[str] | None = None,
    ) -> list[StoredMemory]:
        return await reads.find_unconsolidated(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_types=fact_types,
            limit=limit,
            scope_tags=scope_tags,
        )

    async def mark_consolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_ids: list[str],
        when: datetime | None,
        failed: bool = False,
    ) -> None:
        await reads.mark_consolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids, when=when, failed=failed
        )

    async def any_memory_updated_since(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        since: datetime,
        fact_types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
    ) -> bool:
        return await reads.any_memory_updated_since(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            since=since,
            fact_types=fact_types,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    # ------------------------------------------------------------------ observations

    async def upsert_observation(self, *, conn, bank_id: str, record) -> None:
        """No-op: the observation was written as a `memory_units` row by the caller."""

    async def observations_for_sources(
        self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]
    ) -> list[StoredMemory]:
        return await writes.observations_for_sources(
            conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id: str, fact_ids: list) -> int:
        return await writes.delete_stale_observations(
            conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, fact_ids=fact_ids
        )

    # ------------------------------------------------------------------ curation reads

    async def list_memory_units(
        self,
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
        return await curation.list_memory_units(
            conn=conn,
            ops=ops,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_type=fact_type,
            search_query=search_query,
            consolidation_state=consolidation_state,
            state=state,
            document_id=document_id,
            tags=tags,
            tags_match=tags_match,
            limit=limit,
            offset=offset,
        )

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        return await curation.get_memory_unit(conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    async def list_entities(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await curation.list_entities(
            conn=conn, fq_table=fq_table, bank_id=bank_id, search=search, limit=limit, offset=offset
        )

    # ------------------------------------------------------------------ graph

    async def graph_units(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_type: str | None = None,
        search_query: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "all_strict",
        limit: int = 1000,
    ) -> dict[str, Any]:
        return await graph.graph_units(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            fact_type=fact_type,
            search_query=search_query,
            document_id=document_id,
            chunk_id=chunk_id,
            tags=tags,
            tags_match=tags_match,
            limit=limit,
        )

    async def graph_entity_rows(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        return await graph.graph_entity_rows(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def graph_direct_links(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        return await graph.graph_direct_links(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def entity_memory_counts(
        self, *, conn, fq_table, bank_id: str, entity_ids: list[str] | None = None
    ) -> dict[str, int]:
        return await graph.entity_memory_counts(conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=entity_ids)

    async def entities_for_units(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> dict[str, list[str]]:
        return await graph.entities_for_units(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    async def entity_map_for_units(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        return await graph.entity_map_for_units(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    # ------------------------------------------------------------------ maintenance

    async def record_unit_entities(self, *, conn, ops, fq_table, unit_ids: list[Any], entity_ids: list[Any]) -> None:
        await ops.bulk_insert_unit_entities(conn, fq_table("unit_entities"), unit_ids, entity_ids)

    async def enqueue_relink_victims(self, *, conn, fq_table, bank_id: str, deleted_unit_ids: list) -> int:
        return await graph.enqueue_relink_victims(
            conn=conn, fq_table=fq_table, bank_id=bank_id, deleted_unit_ids=deleted_unit_ids
        )

    async def relink_pass(self, *, backend, fq_table, bank_id: str, config) -> dict:
        return await graph.relink_pass(backend=backend, fq_table=fq_table, bank_id=bank_id, config=config)

    async def prune_orphan_entities(self, *, conn, fq_table, bank_id: str) -> int:
        return await graph.prune_orphan_entities(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def prune_stale_cooccurrences(self, *, conn, fq_table, bank_id: str) -> int:
        return await graph.prune_stale_cooccurrences(conn=conn, fq_table=fq_table, bank_id=bank_id)


__all__ = ["PostgresMemories"]
