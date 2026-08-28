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

from .base import (
    DeletePredicate,
    EntityPrunePassResult,
    MemoriesExtension,
    MemoryPatch,
    MemoryScopeWatermark,
    RecallArms,
    RelinkPassResult,
    ScanPage,
    StoredMemory,
)
from .pg import counts, curation, graph, reads, writes


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
        txn=None,
    ) -> list[str]:
        # `txn` is ignored: Postgres memories live in the caller's own transaction, so the
        # write is already atomic with it — there is no separate store to hold invisible.
        # `defer_index` is meaningless here: the INSERT that returns the ids is
        # also what indexes the facts, so there is nothing to defer.
        return await writes.insert_facts(conn=conn, ops=ops, bank_id=bank_id, facts=facts, document_id=document_id)

    async def delete_facts(self, bank_id: str, unit_ids: list[str], *, txn=None) -> None:
        """No-op: the caller's `memory_units` DELETE (or its FK cascade) removed them."""

    async def delete_where(self, bank_id: str, predicate: DeletePredicate, txn=None) -> int:
        """No-op: predicate deletes are issued as SQL by the caller that owns the transaction."""
        return 0

    async def delete_document(self, *, conn, fq_table, bank_id: str, document_id: str, txn=None) -> None:
        # `txn` ignored: Postgres memories are covered by the caller's own transaction.
        await writes.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)

    async def drop_bank_storage(self, bank_id: str) -> None:
        """No-op: deleting the bank cascades to its memories."""

    async def delete_observations(self, *, conn, fq_table, bank_id: str, txn=None) -> None:
        await writes.delete_observations(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch], txn=None) -> None:
        """No-op: the caller's UPDATE already wrote the row it holds open."""

    # ------------------------------------------------------------------ recall

    async def recall_unified(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        temporal_window: "tuple[datetime, datetime] | None" = None,
        temporal_semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
        enable_text_search: bool = True,
        enable_graph: bool = True,
    ) -> "dict[str, RecallArms]":
        """Run every recall arm for Postgres by orchestrating the split per-arm SQL internally.

        The per-arm split is Postgres's own business, kept off the interface: this reproduces the
        exact orchestration recall used before it was unified — one dense+BM25 UNION query and the
        temporal query share a single connection, then the graph retriever runs per fact_type on the
        pool in parallel, seeded by the same dense results. Result is byte-identical to running the
        arms separately; fusion/rerank still happen downstream.
        """
        import asyncio

        from ..db_utils import acquire_with_retry
        from ..search.retrieval import get_default_graph_retriever

        # `conn` is the connection pool: this store owns the per-arm orchestration and acquires its
        # own connections from it (and runs the graph arm on it).
        pool = conn

        # graph_seed_min_similarity restricts which dense hits seed the graph arm; only the graph
        # arm consumes the seeds, so it is resolved only when that arm runs. It does not affect the
        # semantic/bm25 lists, so the dense+BM25 result is identical whether or not it is passed.
        graph_seed_min_similarity = None
        retriever = None
        if enable_graph:
            from ...config import get_config

            graph_seed_min_similarity = get_config().graph_seed_min_similarity
            # Resolving the retriever can lazily construct one, so only do it when the arm is on.
            retriever = get_default_graph_retriever()

        # Semantic + BM25 (+ temporal) share ONE connection, exactly as before: the dense/keyword
        # UNION runs first, then the temporal query on the same connection, which is then released
        # before the graph arm opens its own connections.
        async with acquire_with_retry(pool) as db_conn:
            semantic_bm25 = await self.search(
                conn=db_conn,
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
                graph_seed_min_similarity=graph_seed_min_similarity,
                enable_text_search=enable_text_search,
            )

            temporal_by_ft: dict[str, list] = {}
            if temporal_window is not None:
                start_date, end_date = temporal_window
                temporal_by_ft = await self.temporal_search(
                    conn=db_conn,
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

        # Graph per fact_type in parallel, on the pool, after the dense connection is released —
        # seeded by the dense results (preselected_semantic_seeds), matching the prior path.
        graph_by_ft: dict[str, list] = {ft: [] for ft in fact_types}
        if enable_graph:
            assert retriever is not None  # only resolved when the arm is on

            async def _run_graph(ft: str) -> list:
                results, _timing = await retriever.retrieve(
                    pool=pool,
                    query_embedding_str=query_embedding,
                    bank_id=bank_id,
                    fact_type=ft,
                    budget=limit,
                    query_text=query_text,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    created_after=created_after,
                    created_before=created_before,
                    preselected_semantic_seeds=semantic_bm25[ft].graph_seeds,
                )
                return results

            # gather preserves input order, so zip back onto fact_types positionally.
            graph_lists = await asyncio.gather(*[_run_graph(ft) for ft in fact_types])
            graph_by_ft = dict(zip(fact_types, graph_lists))

        return {
            ft: RecallArms(
                semantic=semantic_bm25[ft].semantic,
                bm25=semantic_bm25[ft].bm25,
                graph=graph_by_ft.get(ft, []),
                temporal=temporal_by_ft.get(ft, []),
            )
            for ft in fact_types
        }

    # ---- per-arm SQL helpers, private to Postgres (called only by recall_unified) ----

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
        graph_seed_min_similarity: float | None = None,
        enable_text_search: bool = True,
    ) -> "dict[str, SemanticBm25Result]":
        """The dense + keyword arms, as one UNION query.

        How deep the ANN scan goes is not decided here: the connection carries
        ``hnsw.iterative_scan``, which lets the scan resume until this query's own LIMIT
        is met (see ``_ANN_TUNING_HIGH_RECALL``). Before that was enabled the scan
        stopped at ``hnsw.ef_search`` rows — a fixed 200 — so a larger recall budget
        widened the SQL and changed nothing.
        """
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
            graph_seed_min_similarity=graph_seed_min_similarity,
            enable_text_search=enable_text_search,
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
        tag_groups: list | None = None,
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
            tag_groups=tag_groups,
            document_id=document_id,
            metadata_equals=metadata_equals,
            skip=skip,
            include_edges=include_edges,
        )

    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await reads.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def list_tags(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await reads.list_tags(
            conn=conn, fq_table=fq_table, bank_id=bank_id, pattern=pattern, limit=limit, offset=offset
        )

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

    async def count_unconsolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str],
        scopes: list[list[str] | None],
        limit: int,
    ) -> int:
        return await reads.count_unconsolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, scopes=scopes, limit=limit
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
        txn=None,
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

    async def any_memory_updated_since_batch(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        scopes: list[MemoryScopeWatermark],
    ) -> dict[str, bool]:
        return await reads.any_memory_updated_since_batch(conn=conn, fq_table=fq_table, bank_id=bank_id, scopes=scopes)

    async def live_memory_ids(self, *, conn, fq_table, bank_id: str, unit_ids: list[Any]) -> set[str]:
        return await reads.live_memory_ids(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)

    # -- count surfaces --

    async def consolidation_freshness(self, *, conn, fq_table, bank_id: str) -> dict[str, Any]:
        return await counts.consolidation_freshness(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def document_memory_counts(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, int]:
        return await counts.document_memory_counts(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=document_ids
        )

    async def link_counts(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await counts.link_counts(conn=conn, fq_table=fq_table, bank_id=bank_id)

    async def memories_timeseries(
        self, *, conn, fq_table, bank_id: str, time_field: str, trunc: str, since: datetime
    ) -> list[dict[str, Any]]:
        return await counts.memories_timeseries(
            conn=conn, fq_table=fq_table, bank_id=bank_id, time_field=time_field, trunc=trunc, since=since
        )

    async def observation_scope_counts(self, *, conn, fq_table, bank_id: str) -> list[dict[str, Any]]:
        return await counts.observation_scope_counts(conn=conn, fq_table=fq_table, bank_id=bank_id)

    # ------------------------------------------------------------------ observations

    async def upsert_observation(self, *, conn, bank_id: str, record, txn=None) -> None:
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
        fact_type: str | list[str] | None = None,
        search_query: str | None = None,
        consolidation_state: str | None = None,
        state: str | None = None,
        document_id: str | None = None,
        entity_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        created_before: datetime | None = None,
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
            entity_id=entity_id,
            tags=tags,
            tags_match=tags_match,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        return await curation.get_memory_unit(conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    # -- curation archive --

    async def get_archived_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        return await writes.get_archived_memory(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    async def invalidate_memory(
        self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None, txn=None
    ) -> bool:
        return await writes.invalidate_memory(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id, reason=reason
        )

    async def set_invalidation_reason(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> None:
        await writes.set_invalidation_reason(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id, reason=reason
        )

    async def restore_memory(self, *, conn, fq_table, bank_id: str, unit_id: str, txn=None) -> StoredMemory | None:
        return await writes.restore_memory(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding, txn=None) -> None:
        await writes.set_memory_embedding(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id, embedding=embedding
        )

    async def clear_unit_entities(self, *, conn, fq_table, bank_id: str, unit_id: str) -> None:
        await writes.clear_unit_entities(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_id=unit_id)

    async def apply_edit(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_id: str,
        text: str,
        context: str | None,
        fact_type: str,
        occurred_start,
        occurred_end,
        event_date,
        mentioned_at,
        entity_ids: list[str] | None,
        entity_names: list[str] | None = None,  # noqa: ARG002 — this store's registry is SQL; the host already minted+linked, so entity_ids is authoritative.
        txn=None,
    ) -> None:
        await writes.apply_edit(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            unit_id=unit_id,
            text=text,
            context=context,
            fact_type=fact_type,
            occurred_start=occurred_start,
            occurred_end=occurred_end,
            event_date=event_date,
            mentioned_at=mentioned_at,
            entity_ids=entity_ids,
        )

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

    async def resolve_entity_names(self, *, conn, fq_table, bank_id: str, entity_ids: list[str]) -> dict[str, str]:
        return await graph.resolve_entity_names(conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=entity_ids)

    # ------------------------------------------------------------------ maintenance

    async def record_unit_entities(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str | None = None,
        unit_ids: list[Any],
        entity_ids: list[Any],
        txn=None,
    ) -> None:
        # The join is keyed by global unit id, so bank_id is not needed here. `txn` is inert: this
        # posting is an ordinary INSERT in the caller's own transaction, which is already the unit
        # of atomicity — there is no second store to coordinate with.
        await ops.bulk_insert_unit_entities(conn, fq_table("unit_entities"), unit_ids, entity_ids)

    async def enqueue_relink_victims(
        self, *, conn, fq_table, bank_id: str, affected_unit_ids: list, include_affected_units: bool = False
    ) -> int:
        return await graph.enqueue_relink_victims(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            affected_unit_ids=affected_unit_ids,
            include_affected_units=include_affected_units,
        )

    async def relink_pass(
        self, *, backend, fq_table, bank_id: str, config, deadline: float | None = None
    ) -> RelinkPassResult:
        return await graph.relink_pass(
            backend=backend, fq_table=fq_table, bank_id=bank_id, config=config, deadline=deadline
        )

    async def enqueue_entity_prune_candidates(self, *, conn, fq_table, bank_id: str, affected_unit_ids: list) -> int:
        return await graph.enqueue_entity_prune_candidates(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            affected_unit_ids=affected_unit_ids,
        )

    async def entity_prune_pass(
        self, *, backend, fq_table, bank_id: str, deadline: float | None = None
    ) -> EntityPrunePassResult:
        return await graph.entity_prune_pass(backend=backend, fq_table=fq_table, bank_id=bank_id, deadline=deadline)


__all__ = ["PostgresMemories"]
