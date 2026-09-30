"""The default memories store: Postgres holds every store-owned table.

This is the behaviour Hindsight has always had, stated as an implementation of
:class:`~hindsight_api.engine.memories.base.MemoriesExtension` rather than as the
absence of one. Memories go in `memory_units`, the joins around it are `memory_links`
and `unit_entities`, documents, chunks and the entity registry are their own tables,
and every read is SQL — writing a row *is* indexing it, so :meth:`index_facts` has
nothing left to do. This class and :mod:`.pg` are the only code that names those
tables (through ``fq_store_table``; ``fq_table`` refuses them).

The class is deliberately thin. Each method delegates to a plain function in
:mod:`hindsight_api.engine.memories.pg`, split by what calls it — reads, writes,
counts, curation, engine_curation, graph, documents, retain, banks, consolidation,
expand, transfer, admin, the retain-time links and entity_resolver, and the recall
arms in recall / link_expansion — so a change to one area is a change to one file,
and the SQL is grouped by concern rather than piled behind a class.

Keeping this as an explicit store (rather than an ``if store is None`` branch at
each call site) means the default path is the one the whole test suite exercises,
and a second implementation cannot change it by accident.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..retain.types import EmbeddingLike, EntityResolutionResult
from ..schema import fq_store_table, fq_store_table_explicit
from ..search.tags import TagsMatch
from .base import (
    AttachmentRef,
    BankContentCounts,
    DeletedDocument,
    DeletePredicate,
    DocumentBase,
    DocumentChunkState,
    DocumentSourceUnits,
    DocumentTags,
    EntityPrunePassResult,
    EntityResolverHandle,
    ExistingChunk,
    MemoriesExtension,
    MemoryLocation,
    MemoryPatch,
    MemoryScopeWatermark,
    ObservationChunkIds,
    RecallArms,
    RelabelResult,
    RelinkPassResult,
    ScanPage,
    SemanticBm25Result,
    StoredMemory,
    TypedMemoryScope,
)
from .pg import admin as pg_admin
from .pg import banks as pg_banks
from .pg import consolidation as pg_consolidation
from .pg import counts, curation, documents, engine_curation, graph, reads, writes
from .pg import expand as pg_expand
from .pg import links as pg_links
from .pg import retain as pg_retain
from .pg import transfer as pg_transfer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..consolidation.consolidator import _TemporalBounds
    from ..embeddings import Embeddings
    from ..response_models import MemoryFact
    from ..retain.types import CausalRelation, ProcessedFact
    from ..search.graph_retrieval import GraphRetriever
    from ..transfer.export import _LoadedExport, _UnitLocation
    from ..transfer.importer import FactLifecycle, _ObservationOutcome
    from ..transfer.schema import TransferObservation
    from .pg.entity_resolver import EntityResolver

logger = logging.getLogger(__name__)


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
        await writes.delete_document(conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id)

    async def drop_bank_storage(self, bank_id: str) -> None:
        """No-op: deleting the bank cascades to its memories."""

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        await writes.delete_observations(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch]) -> None:
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
        tags_match: TagsMatch = "any",
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
                retrieved = await retriever.retrieve(
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
                # Timings are diagnostics for the perf harness; this path drops them.
                return retrieved.results

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

    def graph_retriever(self) -> GraphRetriever:
        from ...config import get_config
        from .pg.link_expansion import LinkExpansionRetriever

        retriever_type = get_config().graph_retriever.lower()
        if retriever_type == "link_expansion":
            logger.info("Using LinkExpansion graph retriever")
        else:
            logger.warning(f"Unknown graph retriever '{retriever_type}', falling back to link_expansion")
        return LinkExpansionRetriever()

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
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
        graph_seed_min_similarity: float | None = None,
        enable_text_search: bool = True,
    ) -> dict[str, SemanticBm25Result]:
        """The dense + keyword arms, as one UNION query.

        How deep the ANN scan goes is not decided here: the connection carries
        ``hnsw.iterative_scan``, which lets the scan resume until this query's own LIMIT
        is met (see ``_ANN_TUNING_HIGH_RECALL``). Before that was enabled the scan
        stopped at ``hnsw.ef_search`` rows — a fixed 200 — so a larger recall budget
        widened the SQL and changed nothing.
        """
        # Imported here: recall imports search.retrieval, which imports this package, so a
        # module-level import would close the cycle.
        from .pg.recall import retrieve_semantic_bm25_combined_sql

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
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list]:
        from .pg.recall import retrieve_temporal_combined_sql

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
        return await reads.get_memories(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

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
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        document_id: str | None = None,
        metadata_equals: dict[str, str] | None = None,
        skip: int = 0,
        include_edges: bool = False,
    ) -> ScanPage:
        return await reads.scan_memories(
            conn=conn,
            fq_table=fq_store_table,
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
        return await reads.count_memories(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, pattern=pattern, limit=limit, offset=offset
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
            fq_table=fq_store_table,
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, fact_types=fact_types, scopes=scopes, limit=limit
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids, when=when, failed=failed
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
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
    ) -> bool:
        return await reads.any_memory_updated_since(
            conn=conn,
            fq_table=fq_store_table,
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
        return await reads.any_memory_updated_since_batch(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, scopes=scopes
        )

    async def newest_memory_updated_at(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        until: datetime,
        since: datetime | None = None,
        fact_types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
    ) -> datetime | None:
        return await reads.newest_memory_updated_at(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            until=until,
            since=since,
            fact_types=fact_types,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    async def latest_memory_write_at(self, *, conn, fq_table, bank_id: str) -> datetime | None:
        return await reads.latest_memory_write_at(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def live_memory_ids(self, *, conn, fq_table, bank_id: str, unit_ids: list[Any]) -> set[str]:
        return await reads.live_memory_ids(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    # -- count surfaces --

    async def consolidation_freshness(self, *, conn, fq_table, bank_id: str) -> dict[str, Any]:
        return await counts.consolidation_freshness(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def document_memory_counts(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, int]:
        return await counts.document_memory_counts(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_ids=document_ids
        )

    async def link_counts(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        return await counts.link_counts(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def memories_timeseries(
        self, *, conn, fq_table, bank_id: str, time_field: str, trunc: str, since: datetime
    ) -> list[dict[str, Any]]:
        return await counts.memories_timeseries(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, time_field=time_field, trunc=trunc, since=since
        )

    async def observation_scope_counts(
        self, *, conn, fq_table, bank_id: str, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        return await counts.observation_scope_counts(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, limit=limit, offset=offset
        )

    # ------------------------------------------------------------------ observations

    async def upsert_observation(self, *, conn, bank_id: str, record) -> None:
        """No-op: the observation was written as a `memory_units` row by the caller."""

    async def observations_for_sources(
        self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]
    ) -> list[StoredMemory]:
        return await writes.observations_for_sources(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id: str, fact_ids: list) -> int:
        return await writes.delete_stale_observations(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, fact_ids=fact_ids
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
        tags_match: TagsMatch = "any",
        created_before: datetime | None = None,
        time_field: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        return await curation.list_memory_units(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
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
            time_field=time_field,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        return await curation.get_memory_unit(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id
        )

    # -- curation archive --

    async def get_archived_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        return await writes.get_archived_memory(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id)

    async def invalidate_memory(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> bool:
        return await writes.invalidate_memory(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id, reason=reason
        )

    async def set_invalidation_reason(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> None:
        await writes.set_invalidation_reason(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id, reason=reason
        )

    async def restore_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        return await writes.restore_memory(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id)

    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding) -> None:
        await writes.set_memory_embedding(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id, embedding=embedding
        )

    async def clear_unit_entities(self, *, conn, fq_table, bank_id: str, unit_id: str) -> None:
        await writes.clear_unit_entities(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id)

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
        embedding=None,
        current_fact_type: str | None = None,  # noqa: ARG002 — one UPDATE writes every field, so a fact-type change needs no different path.
    ) -> None:
        await writes.apply_edit(
            conn=conn,
            fq_table=fq_store_table,
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
            embedding=embedding,
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
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, search=search, limit=limit, offset=offset
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
        tags_match: TagsMatch = "all_strict",
        limit: int = 1000,
    ) -> dict[str, Any]:
        return await graph.graph_units(
            conn=conn,
            fq_table=fq_store_table,
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
        return await graph.graph_entity_rows(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def graph_direct_links(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        return await graph.graph_direct_links(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def entity_memory_counts(
        self, *, conn, fq_table, bank_id: str, entity_ids: list[str] | None = None
    ) -> dict[str, int]:
        return await graph.entity_memory_counts(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, entity_ids=entity_ids
        )

    async def entities_for_units(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> dict[str, list[str]]:
        return await graph.entities_for_units(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def entity_map_for_units(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        return await graph.entity_map_for_units(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def resolve_entity_names(self, *, conn, fq_table, bank_id: str, entity_ids: list[str]) -> dict[str, str]:
        return await graph.resolve_entity_names(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, entity_ids=entity_ids
        )

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
    ) -> None:
        # The join is keyed by global unit id, so bank_id is not needed here.
        await ops.bulk_insert_unit_entities(conn, fq_store_table("unit_entities"), unit_ids, entity_ids)

    async def enqueue_relink_victims(
        self, *, conn, fq_table, bank_id: str, affected_unit_ids: list, include_affected_units: bool = False
    ) -> int:
        return await graph.enqueue_relink_victims(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            affected_unit_ids=affected_unit_ids,
            include_affected_units=include_affected_units,
        )

    async def relink_pass(
        self, *, backend, fq_table, bank_id: str, config, deadline: float | None = None
    ) -> RelinkPassResult:
        return await graph.relink_pass(
            backend=backend, fq_table=fq_store_table, bank_id=bank_id, config=config, deadline=deadline
        )

    async def enqueue_entity_prune_candidates(self, *, conn, fq_table, bank_id: str, affected_unit_ids: list) -> int:
        return await graph.enqueue_entity_prune_candidates(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            affected_unit_ids=affected_unit_ids,
        )

    async def entity_prune_pass(
        self, *, backend, fq_table, bank_id: str, deadline: float | None = None
    ) -> EntityPrunePassResult:
        return await graph.entity_prune_pass(
            backend=backend, fq_table=fq_store_table, bank_id=bank_id, deadline=deadline
        )

    # ------------------------------------------------------------------ documents and chunks

    async def record_document_file(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        document_id: str,
        storage_key: str,
        original_name: str,
        content_type: str,
    ) -> bool:
        return await documents.record_document_file(
            backend=backend,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            storage_key=storage_key,
            original_name=original_name,
            content_type=content_type,
        )

    async def memory_attachment_refs(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, AttachmentRef]:
        return await documents.memory_attachment_refs(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def recall_observation_chunk_ids(
        self,
        *,
        backend,
        ops,
        fq_table,
        bank_id: str,
        observation_ids: list[uuid.UUID],
        carried_sources: Mapping[str, list[str] | None],
    ) -> ObservationChunkIds:
        return await documents.recall_observation_chunk_ids(
            backend=backend, ops=ops, fq_table=fq_store_table, observation_ids=observation_ids
        )

    async def recall_chunks(self, *, backend, fq_table, bank_id: str, chunk_ids: list[str]) -> dict[str, Any]:
        return await documents.recall_chunks(backend=backend, fq_table=fq_store_table, chunk_ids=chunk_ids)

    async def recall_observation_sources(
        self, *, conn, fq_table, bank_id: str, observation_ids: list[uuid.UUID]
    ) -> dict[str, Any]:
        return await documents.recall_observation_sources(
            conn=conn, fq_table=fq_store_table, observation_ids=observation_ids
        )

    async def recall_source_facts(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, StoredMemory]:
        return await documents.recall_source_facts(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def get_document_with_counts(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str
    ) -> Mapping[str, Any] | None:
        return await documents.get_document_with_counts(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def document_source_units(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentSourceUnits:
        return await documents.document_source_units(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def delete_document_rows(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str, unit_ids: list[str]
    ) -> DeletedDocument:
        return await documents.delete_document_rows(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id, unit_ids=unit_ids
        )

    async def current_document_tags(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentTags:
        return await documents.current_document_tags(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def update_document_tags(
        self, *, conn, fq_table, bank_id: str, document_id: str, tags: list[str] | None, found: bool
    ) -> bool:
        """``found`` is not needed: the UPDATE's RETURNING says whether the row exists."""
        return await documents.update_document_tags(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id, tags=tags
        )

    async def retag_document_memories(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        document_id: str,
        tags: list[str],
        retagged: Callable[[list[str] | None], list[str]],
        rescoped: Callable[[Any], str | None],
    ) -> int:
        return await documents.retag_document_memories(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            tags=tags,
            retagged=retagged,
            rescoped=rescoped,
        )

    async def list_documents_page(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        search_query: str | None,
        tags: list[str] | None,
        tags_match: TagsMatch,
        time_field: str | None,
        start_date: datetime | None,
        end_date: datetime | None,
        limit: int,
        offset: int,
    ) -> dict:
        return await documents.list_documents(
            backend=backend,
            fq_table=fq_store_table,
            bank_id=bank_id,
            search_query=search_query,
            tags=tags,
            tags_match=tags_match,
            time_field=time_field,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

    async def get_chunk_row(self, *, conn, fq_table, bank_id: str | None, chunk_id: str) -> Mapping[str, Any] | None:
        """Any id, parsed or not: the row carries the bank."""
        return await documents.get_chunk_row(conn=conn, fq_table=fq_store_table, chunk_id=chunk_id)

    async def list_document_chunks(
        self, *, backend, fq_table, bank_id: str, document_id: str, limit: int, offset: int
    ) -> dict | None:
        return await documents.list_document_chunks(
            backend=backend,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------ curation and bank admin

    async def locate_memory(self, *, conn, fq_table, bank_id: str | None, unit_id: str) -> MemoryLocation | None:
        return await engine_curation.locate_memory(conn=conn, fq_table=fq_store_table, unit_id=unit_id)

    async def locate_memories(
        self, *, conn, fq_table, bank_id: str | None, unit_ids: list[str]
    ) -> list[MemoryLocation]:
        return await engine_curation.locate_memories(conn=conn, fq_table=fq_store_table, unit_ids=unit_ids)

    async def delete_memory(self, *, conn, ops, fq_table, bank_id: str | None, unit_id: str) -> str | None:
        return await engine_curation.delete_memory(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id
        )

    async def delete_memories(self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]) -> int:
        return await engine_curation.delete_memories(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def bank_memories_of_type(self, *, conn, fq_table, bank_id: str, fact_type: str) -> TypedMemoryScope:
        return await engine_curation.bank_memories_of_type(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, fact_type=fact_type
        )

    async def delete_bank_memories_of_type(
        self, *, conn, ops, fq_table, bank_id: str, fact_type: str, unit_ids: list[str]
    ) -> None:
        await engine_curation.delete_bank_memories_of_type(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, fact_type=fact_type, unit_ids=unit_ids
        )

    async def count_bank_contents(self, *, conn, fq_table, bank_id: str) -> BankContentCounts:
        return await engine_curation.count_bank_contents(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def purge_bank_rows(self, *, conn, fq_table, bank_id: str) -> list[str]:
        return await engine_curation.purge_bank_rows(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def clear_observations_and_requeue(self, *, conn, fq_table, bank_id: str) -> int:
        return await engine_curation.clear_observations_and_requeue(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def requeue_failed_consolidation(self, *, conn, fq_table, bank_id: str) -> int:
        return await engine_curation.requeue_failed_consolidation(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def requeue_source_memory(self, *, conn, fq_table, bank_id: str, unit_id: uuid.UUID) -> None:
        await engine_curation.requeue_source_memory(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id
        )

    async def entity_names_by_id(self, *, conn, fq_table, bank_id: str, entity_ids: list[str]) -> list[str]:
        return await engine_curation.entity_names_by_id(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, entity_ids=entity_ids
        )

    async def observation_head(self, *, conn, fq_table, bank_id: str, unit_id: uuid.UUID) -> MemoryLocation | None:
        return await engine_curation.observation_head(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_id=unit_id
        )

    async def source_fact_summaries(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]
    ) -> list[StoredMemory]:
        # Not bank-scoped, exactly as before: the ids come from this bank's own observation.
        return await engine_curation.source_fact_summaries(conn=conn, fq_table=fq_store_table, unit_ids=unit_ids)

    async def entity_graph(self, *, conn, fq_table, bank_id: str, limit: int, min_count: int) -> dict:
        return await engine_curation.entity_graph(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, limit=limit, min_count=min_count
        )

    async def count_bank_documents(self, *, conn, fq_table, bank_id: str) -> int:
        return await engine_curation.count_bank_documents(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def get_entity_detail(self, *, conn, fq_table, bank_id: str, entity_id: uuid.UUID) -> dict[str, Any] | None:
        return await engine_curation.get_entity_detail(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, entity_id=entity_id
        )

    # ------------------------------------------------------------------ retain

    async def read_document_base(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentBase | None:
        return await pg_retain.read_document_base(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def document_updated_at(self, *, backend, fq_table, bank_id: str, document_id: str) -> datetime | None:
        return await pg_retain.document_updated_at(
            pool=backend, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def recovery_chunk_hashes(
        self, *, backend, fq_table, bank_id: str, document_id: str, content_hash: str
    ) -> set[str]:
        return await pg_retain.recovery_chunk_hashes(
            pool=backend, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id, content_hash=content_hash
        )

    async def lock_document_for_write(self, *, conn, ops, fq_table, bank_id: str, document_id: str) -> str | None:
        return await pg_retain.lock_document_for_write(
            conn=conn, ops=ops, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def lock_pending_document(self, *, conn, fq_table, bank_id: str, document_id: str) -> None:
        await pg_retain.lock_pending_document(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def lock_document_hash(self, *, conn, fq_table, bank_id: str, document_id: str) -> str | None:
        return await pg_retain.lock_document_hash(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def load_document_chunks(
        self, *, backend, fq_table, bank_id: str, document_id: str, include_text: bool
    ) -> DocumentChunkState:
        return await pg_retain.load_document_chunks(
            pool=backend, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id, include_text=include_text
        )

    async def load_existing_chunks(self, *, conn, fq_table, bank_id: str, document_id: str) -> list[ExistingChunk]:
        return await pg_retain.load_existing_chunks(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def current_document_hash(self, *, backend, fq_table, bank_id: str, document_id: str) -> str | None:
        return await pg_retain.current_document_hash(
            pool=backend, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def document_original_text(self, *, conn, fq_table, bank_id: str, document_id: str) -> str | None:
        return await pg_retain.document_original_text(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def count_document_memories(self, *, conn, fq_table, bank_id: str, document_id: str) -> int:
        return await pg_retain.count_document_memories(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def document_unit_ids(self, *, conn, fq_table, bank_id: str, document_id: str) -> list[str]:
        return await pg_retain.document_unit_ids(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def delete_document_for_replace(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str, outgoing_unit_ids: list[str]
    ) -> datetime | None:
        return await pg_retain.delete_document_for_replace(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            outgoing_unit_ids=outgoing_unit_ids,
        )

    async def upsert_document_row(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        document_id: str,
        original_text: str | None,
        content_hash: str,
        retain_params: dict | None,
        document_tags: list[str] | None,
        preserved_created_at: datetime | None,
    ) -> None:
        await pg_retain.upsert_document_row(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            original_text=original_text,
            content_hash=content_hash,
            retain_params=retain_params,
            document_tags=document_tags,
            preserved_created_at=preserved_created_at,
        )

    async def relabel_document_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        document_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        observation_scopes: list | str | None,
        final_tags: Callable[[list[str] | None], list[str]],
    ) -> RelabelResult:
        return await pg_retain.relabel_document_memories(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            tags=tags,
            metadata=metadata,
            observation_scopes=observation_scopes,
            final_tags=final_tags,
        )

    async def memory_ids_for_chunks(self, *, conn, fq_table, bank_id: str, chunk_ids: list[str]) -> list[str]:
        return await pg_retain.memory_ids_for_chunks(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, chunk_ids=chunk_ids
        )

    async def delete_chunks(self, *, conn, fq_table, bank_id: str, chunk_ids: list[str]) -> None:
        await pg_retain.delete_chunks(conn=conn, fq_table=fq_store_table, bank_id=bank_id, chunk_ids=chunk_ids)

    async def upsert_chunks(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        document_id: str,
        chunk_ids: list[str],
        chunk_texts: list[str],
        chunk_indices: list[int],
        content_hashes: list[str],
    ) -> None:
        await pg_retain.upsert_chunks(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_id=document_id,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            chunk_indices=chunk_indices,
            content_hashes=content_hashes,
        )

    async def unit_embeddings(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list:
        return await pg_retain.unit_embeddings(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def create_bank_vector_indexes(
        self, *, conn, ops, fq_table, bank_id: str, internal_id: str, index_clause: str, fact_types: dict[str, str]
    ) -> None:
        await pg_banks.create_bank_vector_indexes(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            internal_id=internal_id,
            index_clause=index_clause,
            fact_types=fact_types,
        )

    async def list_bank_rows(self, *, conn, fq_table, where_clause: str, params: list[str]) -> list:
        return await pg_banks.list_bank_rows(
            conn=conn, fq_table=fq_store_table, where_clause=where_clause, params=params
        )

    async def bank_page_rows(self, *, conn, fq_table, bank_ids: list[str], sql_owned: list[str]) -> list:
        return await pg_banks.bank_page_rows(conn=conn, fq_table=fq_store_table, bank_ids=bank_ids, sql_owned=sql_owned)

    async def bank_fact_counts(self, *, conn, fq_table, bank_ids: list[str]) -> dict[str, int]:
        return await pg_banks.bank_fact_counts(conn=conn, fq_table=fq_store_table, bank_ids=bank_ids)

    # ------------------------------------------------------------------ retain links and entity resolution
    #
    # Each forwards to :mod:`.pg.links` / :mod:`.pg.entity_resolver` with the caller's connection
    # and arguments unchanged. The functions are looked up on the module at call time, so a test
    # that patches one there still sees its patch.

    def create_entity_resolver(
        self,
        *,
        backend,
        entity_lookup: str,
        entity_resolution_batch_size: int,
        intrabatch_merge_similarity: float,
        entity_resolution_max_candidates: int,
        merge_min_similarity: float,
    ) -> EntityResolver:
        from .pg.entity_resolver import EntityResolver

        return EntityResolver(
            backend,
            entity_lookup=entity_lookup,
            entity_resolution_batch_size=entity_resolution_batch_size,
            intrabatch_merge_similarity=intrabatch_merge_similarity,
            entity_resolution_max_candidates=entity_resolution_max_candidates,
            merge_min_similarity=merge_min_similarity,
        )

    async def resolve_entities(
        self,
        *,
        entity_resolver: EntityResolverHandle,
        conn,
        bank_id: str,
        unit_ids: list[str],
        sentences: list[str],
        context: str,
        fact_dates: list,
        llm_entities: list[list[dict]],
        log_buffer: list[str] | None = None,
        entity_labels: list | None = None,
    ) -> EntityResolutionResult:
        return await pg_links.resolve_entities_only(
            entity_resolver,
            conn,
            bank_id,
            unit_ids,
            sentences,
            context,
            fact_dates,
            llm_entities,
            log_buffer,
            entity_labels=entity_labels,
        )

    async def compute_semantic_links_ann(
        self,
        *,
        conn,
        bank_id: str,
        unit_ids: list[str],
        embeddings: Sequence[EmbeddingLike],
        fact_types: list[str] | None = None,
        top_k: int = 50,
        threshold: float,
        log_buffer: list[str] | None = None,
    ) -> list[tuple]:
        return await pg_links.compute_semantic_links_ann(
            conn,
            bank_id,
            unit_ids,
            embeddings,
            fact_types=fact_types,
            top_k=top_k,
            threshold=threshold,
            log_buffer=log_buffer,
        )

    async def create_temporal_links(self, *, conn, ops, bank_id: str, unit_ids: list[str]) -> int:
        return await pg_links.create_temporal_links_batch_per_fact(conn, bank_id, unit_ids, log_buffer=[], ops=ops)

    async def create_semantic_links(
        self,
        *,
        conn,
        ops,
        bank_id: str,
        unit_ids: list[str],
        embeddings: Sequence[EmbeddingLike],
        threshold: float,
        pre_computed_ann_links: list[tuple] | None = None,
    ) -> int:
        return await pg_links.create_semantic_links_batch(
            conn,
            bank_id,
            unit_ids,
            embeddings,
            threshold=threshold,
            log_buffer=[],
            pre_computed_ann_links=pre_computed_ann_links,
            ops=ops,
        )

    async def create_causal_links(
        self, *, conn, ops, bank_id: str, unit_ids: list[str], causal_relations_per_fact: list[list[CausalRelation]]
    ) -> int:
        return await pg_links.create_causal_links_batch(conn, bank_id, unit_ids, causal_relations_per_fact, ops=ops)

    async def restore_legacy_causal_links(
        self, *, conn, ops, bank_id: str, unit_ids: list[str], causal_relations_per_fact: list[list[CausalRelation]]
    ) -> int:
        return await pg_links.restore_legacy_causal_links_batch(
            conn, bank_id, unit_ids, causal_relations_per_fact, ops=ops
        )

    async def insert_links(self, *, conn, ops, bank_id: str, links: list[tuple]) -> None:
        await pg_links._bulk_insert_links(conn, links, bank_id=bank_id, ops=ops)

    # ------------------------------------------------------------------ consolidation writes and recall expansion

    async def lock_live_memory_ids(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> set[str]:
        return await pg_consolidation.lock_live_memory_ids(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def memories_changed_since(self, *, conn, fq_table, bank_id: str, read_at: dict[str, datetime]) -> list[str]:
        return await pg_consolidation.memories_changed_since(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, read_at=read_at
        )

    async def any_memory_exists(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> bool:
        return await pg_consolidation.any_memory_exists(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids
        )

    async def count_observations_with_tags(self, *, conn, fq_table, bank_id: str, tags: list[str]) -> int:
        return await pg_consolidation.count_observations_with_tags(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, tags=tags
        )

    async def insert_observation(
        self,
        *,
        conn,
        ops,
        fq_table,
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
        created_at: datetime,  # noqa: ARG002 — the row's created_at is the column default.
    ) -> str:
        return await pg_consolidation.insert_observation(
            conn=conn,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            observation_id=observation_id,
            text=text,
            embedding=embedding,
            source_memory_ids=source_memory_ids,
            tags=tags,
            event_date=event_date,
            occurred_start=occurred_start,
            occurred_end=occurred_end,
            mentioned_at=mentioned_at,
        )

    async def rewrite_observation(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        text: str,
        embedding: str | None,
        source_memory_ids: list[uuid.UUID],
        tags: list[str],
        bounds: _TemporalBounds,
        previous: MemoryFact,  # noqa: ARG002 — the UPDATE widens the row it finds; a missing row is a skip.
    ) -> bool:
        return await pg_consolidation.rewrite_observation(
            conn=conn,
            fq_table=fq_store_table,
            observation_id=observation_id,
            text=text,
            embedding=embedding,
            source_memory_ids=source_memory_ids,
            tags=tags,
            bounds=bounds,
        )

    async def fold_sources_into_observation(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        expected_text: str,
        merged_text: str,
        source_memory_ids: list[uuid.UUID],
        bounds: _TemporalBounds,
        embeddings: Embeddings,  # noqa: ARG002 — the fold keeps the stored embedding.
    ) -> bool:
        return await pg_consolidation.fold_sources_into_observation(
            conn=conn,
            fq_table=fq_store_table,
            observation_id=observation_id,
            expected_text=expected_text,
            merged_text=merged_text,
            source_memory_ids=source_memory_ids,
            bounds=bounds,
        )

    async def fold_observation_into_twin(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        observation_text: str,
        twin_id: str,
        twin_text: str,
        merged_text: str,
        embeddings: Embeddings,  # noqa: ARG002 — the fold keeps the twin's stored embedding.
    ) -> bool:
        return await pg_consolidation.fold_observation_into_twin(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            observation_id=observation_id,
            observation_text=observation_text,
            twin_id=twin_id,
            twin_text=twin_text,
            merged_text=merged_text,
        )

    async def delete_observation(self, *, conn, fq_table, bank_id: str, observation_id: str) -> None:
        await pg_consolidation.delete_observation(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, observation_id=observation_id
        )

    async def expand_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> list:
        return await pg_expand.expand_memories(conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_ids=unit_ids)

    async def expand_chunks(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        memories: list,  # noqa: ARG002 — the SQL reads the chunk ids directly.
        chunk_ids: list[str],
    ) -> dict[str, Any]:
        return await pg_expand.expand_chunks(conn=conn, fq_table=fq_store_table, chunk_ids=chunk_ids)

    async def expand_documents(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, Any]:
        return await pg_expand.expand_documents(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_ids=document_ids
        )

    # ------------------------------------------------------------------ transfer, gauges and vector indexes

    async def iter_transfer_documents(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        document_ids: list[str] | None,
        include_lifecycle: bool,
        batch_size: int,
    ) -> AsyncIterator[_LoadedExport]:
        async for loaded in pg_transfer.iter_documents(
            backend=backend,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_ids=document_ids,
            include_lifecycle=include_lifecycle,
            batch_size=batch_size,
        ):
            yield loaded

    async def load_transfer_documents(
        self, *, conn, fq_table, bank_id: str, document_ids: list[str] | None, include_lifecycle: bool
    ) -> _LoadedExport:
        return await pg_transfer.load_documents(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            document_ids=document_ids,
            include_lifecycle=include_lifecycle,
        )

    async def load_transfer_observations(
        self, *, conn, fq_table, bank_id: str, unit_index: dict[Any, _UnitLocation]
    ) -> list[TransferObservation]:
        return await pg_transfer.load_observations(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, unit_index=unit_index
        )

    async def dump_entity_maintenance_queue(self, *, conn, fq_table, bank_id: str) -> list[dict]:
        return await pg_transfer.dump_entity_maintenance_queue(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def dump_archived_memories(self, *, conn, fq_table, bank_id: str) -> list[dict]:
        return await pg_transfer.dump_archived_memories(conn=conn, fq_table=fq_store_table, bank_id=bank_id)

    async def restore_archived_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        rows: list[dict],
        unit_id_map: dict[str, str],
        document_id_map: dict[str, str],
        bank_rows_json_encoding: str,
    ) -> int:
        return await pg_transfer.restore_archived_memories(
            conn=conn,
            fq_table=fq_store_table,
            bank_id=bank_id,
            rows=rows,
            unit_id_map=unit_id_map,
            document_id_map=document_id_map,
            bank_rows_json_encoding=bank_rows_json_encoding,  # ty: ignore[invalid-argument-type]
        )

    async def resolve_entity_ids_by_name(self, *, conn, fq_table, bank_id: str, names: set[str]) -> dict[str, Any]:
        return await pg_transfer.resolve_entity_ids_by_name(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, names=names
        )

    async def transfer_document_exists(self, *, backend, fq_table, bank_id: str, document_id: str) -> bool:
        return await pg_transfer.document_exists(
            backend=backend, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id
        )

    async def restore_document_created_at(
        self, *, conn, fq_table, bank_id: str, document_id: str, created_at: datetime
    ) -> None:
        await pg_transfer.restore_document_created_at(
            conn=conn, fq_table=fq_store_table, bank_id=bank_id, document_id=document_id, created_at=created_at
        )

    async def restore_fact_lifecycle(self, *, conn, fq_table, bank_id: str, rows: list[FactLifecycle]) -> None:
        await pg_transfer.restore_fact_lifecycle(conn=conn, fq_table=fq_store_table, bank_id=bank_id, rows=rows)

    async def import_transfer_observations(
        self,
        *,
        backend,
        ops,
        fq_table,
        bank_id: str,
        resolved: list[tuple[TransferObservation, list[str]]],
        processed: list[ProcessedFact],
        outcome: _ObservationOutcome,
    ) -> _ObservationOutcome:
        return await pg_transfer.import_observations(
            backend=backend,
            ops=ops,
            fq_table=fq_store_table,
            bank_id=bank_id,
            resolved=resolved,
            processed=processed,
            outcome=outcome,
        )

    async def count_consolidation_backlog(
        self, *, conn, schema: str, per_bank: bool, bank_ids: Callable[[], Awaitable[list[str]]]
    ) -> dict[str | None, int]:
        return await pg_admin.count_consolidation_backlog(conn=conn, schema=schema, per_bank=per_bank)

    async def count_consolidation_failed(
        self, *, conn, schema: str, per_bank: bool, bank_ids: Callable[[], Awaitable[list[str]]]
    ) -> dict[str | None, int]:
        return await pg_admin.count_consolidation_failed(conn=conn, schema=schema, per_bank=per_bank)

    async def capped_memory_counts(
        self, *, conn, schema: str, bank_id: str, fact_types: list[str], cap: int
    ) -> dict[str, int]:
        return await pg_admin.capped_memory_counts(
            conn=conn, schema=schema, bank_id=bank_id, fact_types=fact_types, cap=cap
        )

    # The admin CLI's whole-schema operations. Not on the interface: backup, restore and
    # rename-bank act on the Postgres schema itself, whichever store owns a bank's memories, so
    # there is no store-owned answer to give — only the store tables' names to keep private.

    async def admin_count_rows(self, *, conn, schema: str, table: str) -> int:
        """Row count of one table of ``schema`` (backup manifest)."""
        return await pg_admin.count_rows(
            conn=conn, fq_table_explicit=fq_store_table_explicit, schema=schema, table=table
        )

    async def admin_truncate_tables(self, *, conn, schema: str, tables: list[str]) -> None:
        """TRUNCATE ... CASCADE each table of ``schema`` in order (restore)."""
        await pg_admin.truncate_tables(
            conn=conn, fq_table_explicit=fq_store_table_explicit, schema=schema, tables=tables
        )

    async def admin_move_bank_id(
        self, *, conn, schema: str, tables: list[str], old_bank_id: str, new_bank_id: str
    ) -> dict[str, int]:
        """Rewrite ``bank_id`` in each table of ``schema`` (rename-bank); rows moved per table."""
        return await pg_admin.move_bank_id(
            conn=conn,
            fq_table_explicit=fq_store_table_explicit,
            schema=schema,
            tables=tables,
            old_bank_id=old_bank_id,
            new_bank_id=new_bank_id,
        )

    async def admin_bank_file_keys(self, *, conn, schema: str, bank_id: str, prefix: str) -> list:
        """The bank's stored-file keys under ``prefix`` (rename-bank's file move)."""
        return await pg_admin.bank_file_keys(
            conn=conn, fq_table_explicit=fq_store_table_explicit, schema=schema, bank_id=bank_id, prefix=prefix
        )

    async def admin_repoint_file_key(
        self, *, conn, schema: str, table: str, column: str, bank_id: str, old_key: str, new_key: str
    ) -> None:
        """Point one row at a moved stored file (rename-bank's file move)."""
        await pg_admin.repoint_file_key(
            conn=conn,
            fq_table_explicit=fq_store_table_explicit,
            schema=schema,
            table=table,
            column=column,
            bank_id=bank_id,
            old_key=old_key,
            new_key=new_key,
        )


__all__ = ["PostgresMemories"]
