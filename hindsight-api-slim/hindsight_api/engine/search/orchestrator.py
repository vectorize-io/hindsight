"""
Search orchestrator for the recall pipeline.

Coordinates multi-strategy retrieval, fusion, reranking, and hydration stages.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from ...cancellation import OperationCancelledError
from ...config import get_config
from ...metrics import get_metrics_collector
from ...tracing import get_tracer
from ..db_budget import budgeted_operation
from ..db_utils import acquire_with_retry
from ..fact_budget import select_facts_within_budget
from ..response_models import (
    ChunkInfo,
    EntityState,
    MemoryFact,
    MinScores,
    RecallScores,
    TemporalWindow,
)
from ..response_models import RecallResult as RecallResultModel
from ..retain import embedding_utils
from ..schema import fq_table
from ..source_facts import select_source_facts_within_budget
from ..source_scope import tag_filter_is_active, visible_document_ids
from ..token_encoding import count_tokens, truncate_to_tokens
from . import retrieval as retrieval_module
from .fusion import cap_per_source, interleave_fusion, reciprocal_rank_fusion
from .recall_boost import (
    apply_post_rerank_boost,
    stage2_passthrough,
    trim_merged_candidates,
)
from .reranking import ScoredResult, apply_combined_scoring
from .tags import TagGroup, TagsMatch
from .trace import SearchPhaseMetrics
from .tracer import SearchTracer
from .types import RetrievalResult

if TYPE_CHECKING:
    from ...models import RequestContext
    from ..db.base import DatabaseBackend, DatabaseConnection
    from ..embeddings import Embeddings
    from ..memories.base import FullRecallRequest, MemoriesExtension, StoredMemory
    from ..memory_engine import MemoryEngine
    from ..query_analyzer import QueryAnalyzer
    from .reranking import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Recall ranking strategy: how the per-arm (semantic/bm25/graph/temporal) results are
# fused and reranked into the final order.
#   "cross_encoder" — RRF fusion + cross-encoder rerank (default, user-facing recall).
#   "rrf"           — RRF fusion, no cross-encoder (RRF score is the order).
#   "interleave"    — round-robin interleave fusion, no cross-encoder. Guarantees each
#                     arm's top hits a slot (used by consolidation dedup recall, where RRF
#                     buried the near-identical twin below budget). See interleave_fusion.
RecallReranking = Literal["cross_encoder", "rrf", "interleave"]


def utcnow() -> datetime:
    """Get current UTC time with timezone info."""
    return datetime.now(UTC)


def _get_memories_store() -> MemoriesExtension:
    """Lazy load the memories store to prevent circular imports with MemoryEngine."""
    from ..memories import get_memories

    return get_memories()


def _make_full_recall_request(**kwargs: Any) -> FullRecallRequest:
    """Instantiate FullRecallRequest lazily to prevent circular imports with MemoryEngine."""
    from ..memories.base import FullRecallRequest

    return FullRecallRequest(**kwargs)


def _recall_scoring_now(question_date: datetime | None) -> datetime:
    """Return the reference time for recall scoring boosts."""
    if question_date is None:
        return utcnow()
    if question_date.tzinfo is None or question_date.utcoffset() is None:
        return question_date.replace(tzinfo=UTC)
    return question_date.astimezone(UTC)


@dataclass
class EntityReference:
    """Reference to an entity associated with a memory unit."""

    entity_id: str
    canonical_name: str


@dataclass
class HydratedChunks:
    """Result of chunk hydration stage."""

    chunks: dict[str, ChunkInfo] | None
    total_tokens: int


@dataclass
class HydratedSourceFacts:
    """Result of source facts hydration stage."""

    source_facts: dict[str, MemoryFact] | None
    source_fact_ids_by_obs: dict[str, list[str]]
    truncated: bool


@dataclass
class HydratedEntities:
    """Result of entity hydration stage."""

    fact_entity_map: dict[str, list[EntityReference]]
    entities: dict[str, EntityState] | None


@dataclass
class RawSourceFact:
    """Raw database row for source fact before budget selection."""

    id: str
    text: str
    fact_type: str
    context: str | None
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None
    document_id: str | None
    chunk_id: str | None
    tags: list[str] | None
    metadata: dict[str, Any] | None


@dataclass
class ObservationSourceRef:
    """Reference to an observation and its source memory ids."""

    id: str
    source_memory_ids: list[str] | None


@dataclass
class ChunkRow:
    """Internal chunk record for hydration and token counting."""

    chunk_id: str
    chunk_text: str
    chunk_index: int
    document_id: str | None = None


@dataclass
class ChunkOrderItem:
    """Item in chunk hydration processing order."""

    item_type: Literal["chunk", "obs"]
    id: str


@dataclass
class RetrievalTraceEntry:
    """Adapter container for SearchTracer results recording without raw tuples."""

    id: str
    data: dict[str, Any]

    def __iter__(self) -> Iterator[Any]:
        yield self.id
        yield self.data


def to_trace_entries(results: Sequence[RetrievalResult]) -> list[RetrievalTraceEntry]:
    """Convert retrieval results to trace entry containers."""
    return [RetrievalTraceEntry(id=r.id, data=r.__dict__) for r in results]


def _entity_map_from_results(
    ids_by_unit: dict[str, list[str]], names: dict[str, str]
) -> dict[str, list[EntityReference]]:
    """Build the ``{unit_id: [EntityReference]}`` recall shape from the
    entity ids a store carried on its results, given a resolved id->name map.

    Mirrors ``entity_map_for_units`` semantics: an order-preserving per-unit dedupe
    (a unit can carry the same id twice), ids with no resolved name dropped, and —
    crucially — a unit that resolves to no entity is omitted entirely rather than
    mapped to ``[]``, so its fact keeps ``entities=None`` downstream instead of an
    empty list. Pure and connectionless, so it is unit-testable without a store or DB.
    """
    out: dict[str, list[EntityReference]] = {}
    for unit_id, ids in ids_by_unit.items():
        rows: list[EntityReference] = []
        seen: set[str] = set()
        for entity_id in ids:
            if entity_id in seen or entity_id not in names:
                continue
            seen.add(entity_id)
            rows.append(EntityReference(entity_id=entity_id, canonical_name=names[entity_id]))
        if rows:
            out[unit_id] = rows
    return out


@dataclass
class SearchRequest:
    """Parameters for a recall / search operation."""

    bank_id: str
    query: str
    fact_type: list[str]
    thinking_budget: int
    max_tokens: int
    enable_trace: bool = False
    question_date: datetime | None = None
    include_entities: bool = False
    include_chunks: bool = False
    max_chunk_tokens: int = 8192
    request_context: RequestContext | None = None
    semaphore_wait: float = 0.0
    prefer_observations: bool = False
    tags: list[str] | None = None
    tags_match: TagsMatch = "any"
    tag_groups: list[TagGroup] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    min_scores: MinScores | None = None
    temporal_window: TemporalWindow | None = None
    connection_budget: int | None = None
    quiet: bool = False
    include_source_facts: bool = False
    max_source_facts_tokens: int = 4096
    max_source_facts_tokens_per_observation: int = -1
    reranking: RecallReranking = "cross_encoder"
    reranker_max_candidates: int | None = None
    enable_text_search: bool = True
    enable_temporal_retrieval: bool = True
    enable_graph_retrieval: bool = True


class SearchOrchestrator:
    """Orchestrates the multi-strategy recall pipeline."""

    def __init__(
        self,
        engine: MemoryEngine | None = None,
        *,
        backend: DatabaseBackend | None = None,
        get_read_backend: Callable[[], Awaitable[DatabaseBackend]] | None = None,
        store_read_conn: (Callable[[str], AbstractAsyncContextManager[DatabaseConnection | None]] | None) = None,
        embeddings: Embeddings | None = None,
        cross_encoder_reranker: CrossEncoderReranker | None = None,
        query_analyzer: QueryAnalyzer | None = None,
    ) -> None:
        self._engine = engine
        self._backend = backend
        self._get_read_backend_fn = get_read_backend
        self._store_read_conn_fn = store_read_conn
        self._embeddings = embeddings
        self._cross_encoder_reranker = cross_encoder_reranker
        self._query_analyzer = query_analyzer

    async def get_read_backend(self) -> DatabaseBackend:
        if self._get_read_backend_fn is not None:
            return await self._get_read_backend_fn()
        if self._backend is not None:
            return self._backend
        if self._engine is not None:
            return await self._engine._get_read_backend()
        raise RuntimeError("No backend provider configured for SearchOrchestrator")

    @property
    def embeddings(self) -> Embeddings | None:
        if self._embeddings is not None:
            return self._embeddings
        if self._engine is not None:
            return self._engine.embeddings
        return None

    @property
    def cross_encoder_reranker(self) -> CrossEncoderReranker | None:
        if self._cross_encoder_reranker is not None:
            return self._cross_encoder_reranker
        if self._engine is not None:
            return self._engine._cross_encoder_reranker
        return None

    @property
    def query_analyzer(self) -> QueryAnalyzer | None:
        if self._query_analyzer is not None:
            return self._query_analyzer
        if self._engine is not None:
            return self._engine.query_analyzer
        return None

    def store_read_conn(self, bank_id: str) -> AbstractAsyncContextManager[DatabaseConnection | None]:
        if self._store_read_conn_fn is not None:
            return self._store_read_conn_fn(bank_id)
        if self._engine is not None:
            return self._engine._store_read_conn(bank_id)
        raise RuntimeError("No store_read_conn configured for SearchOrchestrator")

    async def search(
        self,
        request: SearchRequest,
    ) -> RecallResultModel:
        """Search implementation with modular retrieval and reranking.

        ``created_after`` / ``created_before`` bound ``updated_at``, not ``created_at`` —
        see the note on :meth:`recall`.

        Architecture:
        1. Retrieval: 4-way parallel (semantic, keyword, graph, temporal graph)
        2. Merge: RRF to combine ranked lists
        3. Reranking: Pluggable strategy (heuristic or cross-encoder)
        4. Combined Scoring: Strategy boosts & recency decay
        5. Chunks: Fetch chunks from top-scored results (BEFORE token filtering)
        6. Token Filter: Limit facts to max_tokens budget
        """
        bank_id = request.bank_id
        query = request.query
        fact_type = request.fact_type
        thinking_budget = request.thinking_budget
        max_tokens = request.max_tokens
        enable_trace = request.enable_trace
        question_date = request.question_date
        include_entities = request.include_entities
        include_chunks = request.include_chunks
        max_chunk_tokens = request.max_chunk_tokens
        request_context = request.request_context
        semaphore_wait = request.semaphore_wait
        prefer_observations = request.prefer_observations
        tags = request.tags
        tags_match = request.tags_match
        tag_groups = request.tag_groups
        created_after = request.created_after
        created_before = request.created_before
        min_scores = request.min_scores
        temporal_window = request.temporal_window
        connection_budget = request.connection_budget
        quiet = request.quiet
        include_source_facts = request.include_source_facts
        max_source_facts_tokens = request.max_source_facts_tokens
        max_source_facts_tokens_per_observation = request.max_source_facts_tokens_per_observation
        reranking = request.reranking
        reranker_max_candidates = request.reranker_max_candidates
        enable_text_search = request.enable_text_search
        enable_temporal_retrieval = request.enable_temporal_retrieval
        enable_graph_retrieval = request.enable_graph_retrieval

        # Initialize tracer if requested
        tracer = SearchTracer(
            query,
            thinking_budget,
            max_tokens,
            tags=tags,
            tags_match=tags_match,
            query_timestamp=_recall_scoring_now(question_date),
        )
        tracer.phases_only = not enable_trace
        tracer.start()

        backend_acquire_start = time.time()
        backend = await self.get_read_backend()
        tracer.add_phase_metric("backend_acquisition", time.time() - backend_acquire_start)
        recall_start = time.time()

        # Buffer logs for clean output in concurrent scenarios.
        recall_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}-{uuid.uuid4().hex[:6]}"
        log_buffer = []
        tags_info = f", tags={tags}, tags_match={tags_match}" if tags else ""
        log_buffer.append(
            f"[RECALL {recall_id}] Query: '{query[:50]}...' (budget={thinking_budget}, max_tokens={max_tokens}{tags_info})"
        )

        tracer_otel = get_tracer()

        try:
            # Step 1: Generate query embedding (for semantic search)
            step_start = time.time()

            embedding_span = tracer_otel.start_span("hindsight.recall_embedding")
            embedding_span.set_attribute("hindsight.bank_id", bank_id)
            embedding_span.set_attribute("hindsight.query", query[:100])

            try:
                get_metrics_collector().record_recall_phase("swr_prelude", time.time() - backend_acquire_start)
                query_embeddings = await embedding_utils.generate_embeddings_batch(
                    cast(Any, self.embeddings),
                    [query],
                    input_type="query",
                )
                query_embedding = query_embeddings[0]
                step_duration = time.time() - step_start
                log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")
            finally:
                embedding_span.end()

            tracer.record_query_embedding(query_embedding)
            tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Cancellation checkpoint: bail before the DB-heavy retrieval stage
            # if the client has gone away (issue #2122).
            if request_context is not None:
                request_context.raise_if_cancelled()

            # Step 1.5: let the store answer the whole recall, if it can and the bank asked it to.
            _full_start = time.time()
            _store_result = await _get_memories_store().full_recall(
                _make_full_recall_request(
                    bank_id=bank_id,
                    fact_types=list(fact_type),
                    query_embedding=str(query_embedding),
                    query_text=query,
                    limit=thinking_budget,
                    temporal_window=(
                        (temporal_window.start, temporal_window.end) if temporal_window is not None else None
                    ),
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    created_after=created_after,
                    created_before=created_before,
                    min_semantic=min_scores.semantic if min_scores else None,
                    min_keyword=min_scores.keyword if min_scores else None,
                    enable_text_search=enable_text_search,
                    enable_graph=enable_graph_retrieval,
                    reranking=reranking,
                    reranker_max_candidates=(
                        reranker_max_candidates
                        if reranker_max_candidates is not None
                        else get_config().reranker_max_candidates
                    ),
                    per_source_cap=get_config().recall_max_candidates_per_source,
                    strategy_boosts=get_config().recall_strategy_boosts,
                    recency_decay_function=get_config().recency_decay_function,
                    recency_decay_linear_window_days=get_config().recency_decay_linear_window_days,
                    recency_decay_halflife_days=get_config().recency_decay_halflife_days,
                    now=_recall_scoring_now(question_date),
                    min_reranker=min_scores.reranker if min_scores else None,
                    min_final=min_scores.final if min_scores else None,
                    truncate_to=thinking_budget * 2,
                    max_tokens=max_tokens,
                    tokenizer_encoding=get_config().tokenizer_encoding,
                    include_entities=include_entities,
                    include_chunks=include_chunks,
                    max_chunk_tokens=max_chunk_tokens,
                    prefer_observations=prefer_observations,
                    include_source_facts=include_source_facts,
                    max_source_facts_tokens=max_source_facts_tokens,
                    max_source_facts_tokens_per_observation=max_source_facts_tokens_per_observation,
                )
            )
            if _store_result is not None:
                _full_elapsed = time.time() - _full_start
                _t0_tail = time.time()
                log_buffer.append(
                    f"  [1.5] Store-answered recall: {len(_store_result.results)} results in {_full_elapsed:.3f}s"
                )
                if not quiet:
                    logger.info("\n" + "\n".join(log_buffer))
                    get_metrics_collector().record_recall_phase("store_branch_tail", time.time() - _t0_tail)

                _store_reported = 0.0
                for _name, _micros in (_store_result.store_stages or {}).items():
                    tracer.add_phase_metric(f"store_{_name}", _micros / 1_000_000)
                    _store_reported += _micros / 1_000_000

                tracer.add_phase_metric("store_hop_overhead", max(0.0, _full_elapsed - _store_reported))
                tracer.add_phase_metric(
                    "full_recall",
                    _full_elapsed,
                    {"results": len(_store_result.results)},
                )
                if enable_trace:
                    _trace = tracer.finalize([r.model_dump() for r in _store_result.results])
                    _store_result.trace = _trace.to_dict() if _trace else None
                return _store_result

            # Step 2: Optimized parallel retrieval using batched queries
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            retrieval_span = tracer_otel.start_span("hindsight.recall_retrieval")
            retrieval_span.set_attribute("hindsight.bank_id", bank_id)
            retrieval_span.set_attribute("hindsight.fact_types", ",".join(fact_type))
            retrieval_span.set_attribute("hindsight.thinking_budget", thinking_budget)

            try:
                config = get_config()
                effective_connection_budget = (
                    connection_budget if connection_budget is not None else config.recall_connection_budget
                )
                async with budgeted_operation(
                    max_connections=effective_connection_budget,
                    operation_id=f"recall-{recall_id}",
                ) as op:
                    budgeted_pool = op.wrap_pool(backend)
                    parallel_start = time.time()
                    multi_result = await retrieval_module.retrieve_all_fact_types_parallel(
                        budgeted_pool,
                        query,
                        query_embedding_str,
                        bank_id,
                        fact_type,
                        thinking_budget,
                        question_date,
                        self.query_analyzer,
                        tags=tags,
                        tags_match=tags_match,
                        tag_groups=tag_groups,
                        created_after=created_after,
                        created_before=created_before,
                        min_semantic=min_scores.semantic if min_scores else None,
                        min_keyword=min_scores.keyword if min_scores else None,
                        temporal_window=temporal_window,
                        enable_text_search=enable_text_search,
                        enable_temporal_retrieval=enable_temporal_retrieval,
                        enable_graph_retrieval=enable_graph_retrieval,
                    )
                    parallel_duration = time.time() - parallel_start
            finally:
                retrieval_span.end()

            semantic_results = []
            bm25_results = []
            graph_results = []
            temporal_results = []
            aggregated_timings = {
                "semantic": 0.0,
                "bm25": 0.0,
                "graph": 0.0,
                "temporal": 0.0,
                "temporal_extraction": 0.0,
            }
            all_graph_timings = []

            detected_temporal_constraint = None
            max_conn_wait = multi_result.max_conn_wait
            for ft in fact_type:
                retrieval_result = multi_result.results_by_fact_type.get(ft)
                if not retrieval_result:
                    continue

                logger.debug(
                    f"[RECALL {recall_id}] Fact type '{ft}': semantic={len(retrieval_result.semantic)}, bm25={len(retrieval_result.bm25)}, graph={len(retrieval_result.graph)}, temporal={len(retrieval_result.temporal) if retrieval_result.temporal else 0}"
                )

                semantic_results.extend(retrieval_result.semantic)
                bm25_results.extend(retrieval_result.bm25)
                graph_results.extend(retrieval_result.graph)
                if retrieval_result.temporal:
                    temporal_results.extend(retrieval_result.temporal)
                for method, duration in retrieval_result.timings.items():
                    aggregated_timings[method] = max(aggregated_timings.get(method, 0.0), duration)
                if retrieval_result.temporal_constraint:
                    detected_temporal_constraint = retrieval_result.temporal_constraint
                if retrieval_result.graph_timings:
                    all_graph_timings.extend(retrieval_result.graph_timings)

            if not temporal_results:
                temporal_results = None

            semantic_results.sort(key=lambda r: r.similarity if hasattr(r, "similarity") else 0, reverse=True)
            bm25_results.sort(key=lambda r: r.bm25_score if hasattr(r, "bm25_score") else 0, reverse=True)
            graph_results.sort(key=lambda r: r.activation if hasattr(r, "activation") else 0, reverse=True)
            if temporal_results:
                temporal_results.sort(key=lambda r: r.temporal_score or 0, reverse=True)

            per_source_cap = get_config().recall_max_candidates_per_source
            if per_source_cap > 0:
                pre_cap_counts = (len(semantic_results), len(bm25_results), len(graph_results))
                semantic_results = cap_per_source(semantic_results, per_source_cap)
                bm25_results = cap_per_source(bm25_results, per_source_cap)
                graph_results = cap_per_source(graph_results, per_source_cap)
                if temporal_results:
                    temporal_results = cap_per_source(temporal_results, per_source_cap)
                if pre_cap_counts != (len(semantic_results), len(bm25_results), len(graph_results)):
                    logger.debug(
                        f"[RECALL {recall_id}] Per-source cap ({per_source_cap}) applied: "
                        f"semantic {pre_cap_counts[0]}->{len(semantic_results)}, "
                        f"bm25 {pre_cap_counts[1]}->{len(bm25_results)}, "
                        f"graph {pre_cap_counts[2]}->{len(graph_results)}"
                    )

            step_duration = time.time() - step_start
            _store_recall = aggregated_timings.get("store_recall", 0.0)
            _store_recall_info = f" | store={_store_recall:.3f}s" if _store_recall else ""
            if _store_recall:
                tracer.add_phase_metric(
                    "store_recall",
                    _store_recall,
                    {"diagnostic": True, "note": "subset of parallel_retrieval"},
                )

            timing_parts = [
                f"semantic={len(semantic_results)}({aggregated_timings['semantic']:.3f}s)",
                f"bm25={len(bm25_results)}({aggregated_timings['bm25']:.3f}s)",
                f"graph={len(graph_results)}({aggregated_timings['graph']:.3f}s)",
                f"temporal_extraction={aggregated_timings['temporal_extraction']:.3f}s",
            ]
            temporal_info = ""
            if detected_temporal_constraint:
                start_dt, end_dt = detected_temporal_constraint
                temporal_count = len(temporal_results) if temporal_results else 0
                timing_parts.append(f"temporal={temporal_count}({aggregated_timings['temporal']:.3f}s)")
                temporal_info = f" | temporal_range={start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
            log_buffer.append(
                f"  [2] Parallel retrieval ({len(fact_type)} fact_types): {', '.join(timing_parts)}"
                f"{_store_recall_info} in {parallel_duration:.3f}s{temporal_info}"
            )

            if all_graph_timings:
                try:
                    retriever_name = retrieval_module.get_default_graph_retriever().name.upper()
                except RuntimeError:
                    # A store that runs its own graph arm may supply no retriever; this is a log line.
                    retriever_name = "STORE"
                graph_total = all_graph_timings[0]
                graph_parts = [
                    f"db_queries={graph_total.db_queries}",
                    f"edge_load={graph_total.edge_load_time:.3f}s",
                    f"edges={graph_total.edge_count}",
                    f"patterns={graph_total.pattern_count}",
                ]
                if graph_total.seeds_time > 0.01:
                    graph_parts.append(f"seeds={graph_total.seeds_time:.3f}s")
                if graph_total.fusion > 0.001:
                    graph_parts.append(f"fusion={graph_total.fusion:.3f}s")
                if graph_total.fetch > 0.001:
                    graph_parts.append(f"fetch={graph_total.fetch:.3f}s")
                log_buffer.append(f"      [{retriever_name}] {', '.join(graph_parts)}")
                if graph_total.hop_details:
                    for hd in graph_total.hop_details:
                        log_buffer.append(
                            f"        hop{hd['hop']}: exec={hd.get('exec_time', 0) * 1000:.0f}ms, "
                            f"uncached={hd.get('uncached_after_filter', 0)}, "
                            f"load={hd.get('load_time', 0) * 1000:.0f}ms, "
                            f"edges={hd.get('edges_loaded', 0)}"
                        )

            if detected_temporal_constraint:
                start_dt, end_dt = detected_temporal_constraint
                tracer.record_temporal_constraint(start_dt, end_dt)

            if enable_trace:
                for ft_name in fact_type:
                    rr = multi_result.results_by_fact_type.get(ft_name)
                    if not rr:
                        continue

                    tracer.add_retrieval_results(
                        method_name="semantic",
                        results=to_trace_entries(rr.semantic),
                        duration_seconds=rr.timings.get("semantic", 0.0),
                        score_field="similarity",
                        metadata={"limit": thinking_budget},
                        fact_type=ft_name,
                    )

                    if enable_text_search:
                        tracer.add_retrieval_results(
                            method_name="bm25",
                            results=to_trace_entries(rr.bm25),
                            duration_seconds=rr.timings.get("bm25", 0.0),
                            score_field="bm25_score",
                            metadata={"limit": thinking_budget},
                            fact_type=ft_name,
                        )

                    if enable_graph_retrieval:
                        tracer.add_retrieval_results(
                            method_name="graph",
                            results=to_trace_entries(rr.graph),
                            duration_seconds=rr.timings.get("graph", 0.0),
                            score_field="activation",
                            metadata={"budget": thinking_budget},
                            fact_type=ft_name,
                        )

                    if rr.temporal is not None or rr.temporal_constraint is not None:
                        temporal_metadata = {"budget": thinking_budget}
                        if rr.temporal_constraint:
                            start_dt, end_dt = rr.temporal_constraint
                            temporal_metadata["constraint"] = {
                                "start": start_dt.isoformat() if start_dt else None,
                                "end": end_dt.isoformat() if end_dt else None,
                            }
                        tracer.add_retrieval_results(
                            method_name="temporal",
                            results=to_trace_entries(rr.temporal or []),
                            duration_seconds=rr.timings.get("temporal", 0.0),
                            score_field="temporal_score",
                            metadata=temporal_metadata,
                            fact_type=ft_name,
                        )

                _entry_points = semantic_results[:10]
                if _entry_points:
                    await _get_memories_store().hydrate_results(bank_id=bank_id, results=_entry_points)
                for rank, entry_pt in enumerate(_entry_points, start=1):
                    tracer.add_entry_point(entry_pt.id, entry_pt.text, entry_pt.similarity or 0.0, rank)

            tracer.add_phase_metric(
                "parallel_retrieval",
                step_duration,
                {
                    "semantic_count": len(semantic_results),
                    "bm25_count": len(bm25_results),
                    "graph_count": len(graph_results),
                    "temporal_count": len(temporal_results) if temporal_results else 0,
                },
            )
            for _method, _dur in aggregated_timings.items():
                if _dur > 0:
                    tracer.add_phase_metric(f"retrieval_{_method}", _dur, {"diagnostic": True})

            # Step 3: Merge ranked lists
            step_start = time.time()

            fusion_span = tracer_otel.start_span("hindsight.recall_fusion")
            fusion_span.set_attribute("hindsight.bank_id", bank_id)
            fusion_span.set_attribute("hindsight.semantic_count", len(semantic_results))
            fusion_span.set_attribute("hindsight.bm25_count", len(bm25_results))
            fusion_span.set_attribute("hindsight.graph_count", len(graph_results))
            fusion_span.set_attribute("hindsight.temporal_count", len(temporal_results) if temporal_results else 0)

            try:
                result_lists = [semantic_results, bm25_results, graph_results]
                if temporal_results:
                    result_lists.append(temporal_results)
                fuse = interleave_fusion if reranking == "interleave" else reciprocal_rank_fusion
                merged_candidates = fuse(result_lists)

                step_duration = time.time() - step_start
                log_buffer.append(
                    f"  [3] {'interleave' if reranking == 'interleave' else 'RRF'} merge: "
                    f"{len(merged_candidates)} unique candidates in {step_duration:.3f}s"
                )
            finally:
                fusion_span.set_attribute("hindsight.merged_count", len(merged_candidates))
                fusion_span.end()

            if enable_trace:
                tracer_merged = [
                    (mc.id, mc.retrieval.__dict__, {"rrf_score": mc.rrf_score, **mc.source_ranks})
                    for mc in merged_candidates
                ]
                tracer.add_rrf_merged(tracer_merged)
            tracer.add_phase_metric("rrf_merge", step_duration, {"candidates_merged": len(merged_candidates)})

            # Step 4: Rerank using cross-encoder
            step_start = time.time()
            reranker_instance = self.cross_encoder_reranker

            rerank_span = tracer_otel.start_span("hindsight.recall_rerank")
            rerank_span.set_attribute("hindsight.bank_id", bank_id)
            rerank_span.set_attribute("hindsight.candidates_count", len(merged_candidates))

            scored_results: list[ScoredResult] = []
            served_provider: str | None = None
            pre_filtered_count = 0
            rerank_kind = "cross-encoder"
            try:
                rerank_config = get_config()
                max_candidates = (
                    reranker_max_candidates
                    if reranker_max_candidates is not None
                    else rerank_config.reranker_max_candidates
                )
                strategy_boosts = rerank_config.recall_strategy_boosts
                trimmed = trim_merged_candidates(merged_candidates, max_candidates, strategy_boosts)
                merged_candidates = trimmed.kept
                pre_filtered_count = trimmed.dropped
                if pre_filtered_count > 0:
                    arm_composition: dict[str, int] = {}
                    for mc in merged_candidates:
                        for key in mc.source_ranks:
                            arm = key.removesuffix("_rank")
                            arm_composition[arm] = arm_composition.get(arm, 0) + 1
                    tracer.add_phase_metric(
                        "rerank_prefilter",
                        0.0,
                        {
                            "kept": len(merged_candidates),
                            "dropped": pre_filtered_count,
                            "max_candidates": max_candidates,
                            "strategy_boosts": dict(strategy_boosts) if strategy_boosts else None,
                            "arm_composition": arm_composition,
                        },
                    )

                _hydrate_start = time.time()
                await _get_memories_store().hydrate_results(
                    bank_id=bank_id, results=[mc.retrieval for mc in merged_candidates]
                )
                tracer.add_phase_metric(
                    "hydrate_results",
                    time.time() - _hydrate_start,
                    {"candidates": len(merged_candidates)},
                )

                if reranking == "cross_encoder":
                    if request_context is not None:
                        request_context.raise_if_cancelled()

                    if reranker_instance is None:
                        raise RuntimeError("No CrossEncoderReranker configured for SearchOrchestrator")

                    await reranker_instance.ensure_initialized()
                    reranked = await reranker_instance.rerank(query, merged_candidates)
                    scored_results = reranked.results
                    served_provider = reranked.provider_name
                else:
                    rerank_kind = f"{reranking}-passthrough"
                    scored_results = [
                        ScoredResult(
                            candidate=mc,
                            cross_encoder_score=0.0,
                            cross_encoder_score_normalized=0.0,
                            weight=0.0,
                        )
                        for mc in sorted(merged_candidates, key=lambda mc: mc.rrf_score, reverse=True)
                    ]

                step_duration = time.time() - step_start
                pre_filter_note = f" (pre-filtered {pre_filtered_count})" if pre_filtered_count > 0 else ""
                log_buffer.append(
                    f"  [4] Reranking [{rerank_kind}]: {len(scored_results)} candidates "
                    f"scored in {step_duration:.3f}s{pre_filter_note}"
                )
            finally:
                rerank_span.set_attribute("hindsight.scored_count", len(scored_results))
                if pre_filtered_count > 0:
                    rerank_span.set_attribute("hindsight.pre_filtered_count", pre_filtered_count)
                rerank_span.end()

            # Step 4.5: Combine cross-encoder score with retrieval signals
            scoring_start = time.time()
            if scored_results and reranking == "interleave":
                for sr in scored_results:
                    sr.weight = sr.candidate.rrf_score
                log_buffer.append("  [4.6] Interleave order preserved (combined scoring skipped)")
            elif scored_results:
                is_passthrough = stage2_passthrough(reranking, served_provider)
                scoring_config = get_config()

                apply_combined_scoring(
                    scored_results,
                    now=_recall_scoring_now(question_date),
                    is_passthrough_reranker=is_passthrough,
                    recency_decay_function=scoring_config.recency_decay_function,
                    recency_decay_linear_window_days=scoring_config.recency_decay_linear_window_days,
                    recency_decay_halflife_days=scoring_config.recency_decay_halflife_days,
                )
                strategy_boosts = scoring_config.recall_strategy_boosts
                stage2: str | None = None
                if strategy_boosts:
                    stage2 = apply_post_rerank_boost(scored_results, strategy_boosts, passthrough=is_passthrough)
                scored_results.sort(key=lambda x: x.weight, reverse=True)
                log_buffer.append("  [4.6] Combined scoring: ce * recency_boost(0.2) * temporal_boost(0.2)")
                if strategy_boosts:
                    log_buffer.append(f"  [4.7] Strategy boosts applied: {strategy_boosts} {stage2}")

            # Step 4.9: Post-query min_scores filters
            min_reranker = min_scores.reranker if min_scores else None
            min_final = min_scores.final if min_scores else None
            if (min_reranker is not None or min_final is not None) and scored_results:
                before_min_score = len(scored_results)
                scored_results = [
                    sr
                    for sr in scored_results
                    if (min_reranker is None or sr.cross_encoder_score_normalized >= min_reranker)
                    and (min_final is None or sr.weight >= min_final)
                ]
                log_buffer.append(
                    f"  [4.9] min_scores(reranker={min_reranker}, final={min_final}): "
                    f"{before_min_score}->{len(scored_results)} results"
                )

            if enable_trace:
                results_dict = [sr.to_dict() for sr in scored_results]
                tracer_merged = [
                    (mc.id, mc.retrieval.__dict__, {"rrf_score": mc.rrf_score, **mc.source_ranks})
                    for mc in merged_candidates
                ]
                tracer.add_reranked(results_dict, tracer_merged)
            tracer.add_phase_metric(
                "reranking",
                step_duration,
                {"reranker_type": rerank_kind, "candidates_reranked": len(scored_results)},
            )
            tracer.add_phase_metric(
                "combined_scoring",
                time.time() - scoring_start,
                {"candidates_scored": len(scored_results)},
            )

            # Cancellation checkpoint: reranking is done
            if request_context is not None:
                request_context.raise_if_cancelled()

            # Step 4.8: prefer-observations dedup
            raw_types_requested = {"world", "experience"} & set(fact_type)
            if prefer_observations and "observation" in fact_type and raw_types_requested:
                observation_srs = [
                    sr for sr in scored_results[: thinking_budget * 2] if sr.retrieval.fact_type == "observation"
                ]
                observation_ids = [uuid.UUID(sr.id) for sr in observation_srs]
                if observation_ids:
                    dedup_start = time.time()
                    superseded_ids: set[str] = set()

                    if not all(sr.retrieval.source_memory_ids is not None for sr in observation_srs):
                        async with self.store_read_conn(bank_id) as dedup_conn:
                            obs_by_id = {
                                m.unit_id: [str(s) for s in (m.source_memory_ids or [])]
                                for m in await _get_memories_store().get_memories(
                                    conn=dedup_conn,
                                    fq_table=fq_table,
                                    bank_id=bank_id,
                                    unit_ids=[str(o) for o in observation_ids],
                                )
                                if m.fact_type == "observation"
                            }
                            for sr in observation_srs:
                                if sr.id in obs_by_id:
                                    sr.retrieval.source_memory_ids = obs_by_id[sr.id]
                    tracer.add_phase_metric(
                        "prefer_observations_dedup",
                        time.time() - dedup_start,
                        {"observations_considered": len(observation_ids)},
                    )
                    for sr in observation_srs:
                        for sid in sr.retrieval.source_memory_ids or []:
                            superseded_ids.add(str(sid))
                    if superseded_ids:
                        before_count = len(scored_results)
                        scored_results = [
                            sr
                            for sr in scored_results
                            if not (sr.retrieval.fact_type in ("world", "experience") and sr.id in superseded_ids)
                        ]
                        log_buffer.append(
                            f"  [4.8] prefer_observations: dropped {before_count - len(scored_results)} "
                            f"raw fact(s) superseded by {len(observation_ids)} observation(s)"
                        )

            # Step 5: Truncate to thinking_budget * 2 for token filtering
            rerank_limit = thinking_budget * 2
            top_scored = scored_results[:rerank_limit]
            log_buffer.append(f"  [5] Truncated to top {len(top_scored)} results")

            # Step 5.5: Fetch chunks from top-scored results (before token filtering)
            chunks_dict = None
            total_chunk_tokens = 0
            if include_chunks and top_scored:
                chunk_fetch_start = time.time()
                hydrated_chunks = await self._hydrate_chunks(
                    bank_id=bank_id,
                    backend=backend,
                    top_scored=top_scored,
                    max_chunk_tokens=max_chunk_tokens,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                )
                chunks_dict = hydrated_chunks.chunks
                total_chunk_tokens = hydrated_chunks.total_tokens
                tracer.add_phase_metric(
                    "chunk_fetch",
                    time.time() - chunk_fetch_start,
                    {"chunks_returned": len(chunks_dict or {}), "chunk_tokens": total_chunk_tokens},
                )

            # Step 6: Token budget filtering
            step_start = time.time()
            selection = select_facts_within_budget(
                fact_ids_ordered=[sr.id for sr in top_scored],
                text_by_id={sr.id: sr.retrieval.text for sr in top_scored},
                max_tokens=max_tokens,
                count_tokens=count_tokens,
            )
            total_tokens = selection.total_tokens
            selected_ids = set(selection.ids)
            top_scored = [sr for sr in top_scored if sr.id in selected_ids]

            step_duration = time.time() - step_start
            truncated_note = " (truncated)" if selection.truncated else ""
            log_buffer.append(
                f"  [6] Token filtering: {len(top_scored)} results, {total_tokens}/{max_tokens} tokens"
                f"{truncated_note} in {step_duration:.3f}s"
            )

            tracer.add_phase_metric(
                "token_filtering",
                step_duration,
                {
                    "results_selected": len(top_scored),
                    "tokens_used": total_tokens,
                    "max_tokens": max_tokens,
                    "truncated": selection.truncated,
                },
            )

            assembly_start = time.time()

            if enable_trace:
                entry_point_ids = {ep.node_id for ep in tracer.entry_points}
                for sr in scored_results:
                    tracer.visit_node(
                        node_id=sr.id,
                        text=sr.retrieval.text,
                        context=sr.retrieval.context or "",
                        event_date=sr.retrieval.occurred_start,
                        is_entry_point=sr.id in entry_point_ids,
                        activation=sr.candidate.rrf_score,
                        semantic_similarity=sr.retrieval.similarity or 0.0,
                        recency=sr.recency,
                        frequency=0.0,
                        final_weight=sr.weight,
                    )

            fact_type_counts = {}
            for sr in top_scored:
                ft = sr.retrieval.fact_type
                fact_type_counts[ft] = fact_type_counts.get(ft, 0) + 1

            fact_type_summary = ", ".join([f"{ft}={count}" for ft, count in sorted(fact_type_counts.items())])

            # Step 7: Fetch source facts for observation-type results
            source_facts_dict = None
            source_fact_ids_by_obs = {}
            source_facts_truncated = False
            if include_source_facts:
                source_fact_start = time.time()
                hydrated_sf = await self._hydrate_source_facts(
                    bank_id=bank_id,
                    backend=backend,
                    top_scored=top_scored,
                    max_source_facts_tokens=max_source_facts_tokens,
                    max_source_facts_tokens_per_observation=max_source_facts_tokens_per_observation,
                )
                source_facts_dict = hydrated_sf.source_facts
                source_fact_ids_by_obs = hydrated_sf.source_fact_ids_by_obs
                source_facts_truncated = hydrated_sf.truncated
                tracer.add_phase_metric(
                    "source_fact_fetch",
                    time.time() - source_fact_start,
                    {"source_facts_returned": len(source_facts_dict or {})},
                )

            # Step 8: Entity hydration
            entities_dict = None
            total_entity_tokens = 0
            fact_entity_map: dict[str, list[EntityReference]] = {}
            if include_entities and top_scored:
                entity_build_start = time.time()
                hydrated_entities = await self._hydrate_entities(
                    bank_id=bank_id,
                    backend=backend,
                    top_scored=top_scored,
                )
                fact_entity_map = hydrated_entities.fact_entity_map
                entities_dict = hydrated_entities.entities
                tracer.add_phase_metric(
                    "entity_build",
                    time.time() - entity_build_start,
                    {"entities_returned": len(entities_dict or {})},
                )

            # Step 9: Convert results to MemoryFact objects
            assembly_start = time.time()
            reranker_passthrough = (reranking != "cross_encoder") or served_provider == "rrf"
            scores_by_id: dict[str, RecallScores] = {
                sr.id: RecallScores(
                    final=sr.weight,
                    reranker=None if reranker_passthrough else sr.cross_encoder_score_normalized,
                    semantic=sr.candidate.arm_scores.semantic,
                    keyword=sr.candidate.arm_scores.keyword,
                )
                for sr in top_scored
            }

            memory_facts = []
            for sr in top_scored:
                result_id = sr.id
                ret = sr.retrieval
                entity_names = None
                if include_entities and result_id in fact_entity_map:
                    entity_names = [e.canonical_name for e in fact_entity_map[result_id]]

                occurred_start = (
                    ret.occurred_start.isoformat() if hasattr(ret.occurred_start, "isoformat") else ret.occurred_start
                )
                occurred_end = (
                    ret.occurred_end.isoformat() if hasattr(ret.occurred_end, "isoformat") else ret.occurred_end
                )
                mentioned_at = (
                    ret.mentioned_at.isoformat() if hasattr(ret.mentioned_at, "isoformat") else ret.mentioned_at
                )

                memory_facts.append(
                    MemoryFact(
                        id=result_id,
                        text=ret.text,
                        fact_type=ret.fact_type or "world",
                        entities=entity_names,
                        context=ret.context,
                        occurred_start=occurred_start,
                        occurred_end=occurred_end,
                        mentioned_at=mentioned_at,
                        document_id=ret.document_id,
                        metadata=ret.metadata,
                        chunk_id=ret.chunk_id,
                        tags=ret.tags,
                        source_fact_ids=source_fact_ids_by_obs.get(result_id) if include_source_facts else None,
                        scores=scores_by_id.get(result_id),
                        attachment_ids=ret.attachment_ids,
                    )
                )

            tracer.add_phase_metric(
                "result_serialization",
                time.time() - assembly_start,
                {"results_serialized": len(memory_facts)},
            )

            if semaphore_wait > 0:
                tracer.add_phase_metric("semaphore_wait", semaphore_wait, {"diagnostic": True})
            if max_conn_wait > 0:
                tracer.add_phase_metric("connection_wait", max_conn_wait, {"diagnostic": True})

            trace_dict = None
            if enable_trace:
                finalize_start = time.time()
                trace = tracer.finalize([{"id": mf.id} for mf in memory_facts])
                trace_dict = trace.to_dict() if trace else None
                if trace_dict is not None:
                    trace_dict["summary"]["phase_metrics"].append(
                        SearchPhaseMetrics(
                            phase_name="trace_finalize",
                            duration_seconds=time.time() - finalize_start,
                            details={"diagnostic": True},
                        ).model_dump()
                    )

            total_time = time.time() - recall_start
            num_chunks = len(chunks_dict) if chunks_dict else 0
            num_entities = len(entities_dict) if entities_dict else 0
            wait_parts = []
            if semaphore_wait > 0.01:
                wait_parts.append(f"sem={semaphore_wait:.3f}s")
            if max_conn_wait > 0.01:
                wait_parts.append(f"conn={max_conn_wait:.3f}s")
            wait_info = f" | waits: {', '.join(wait_parts)}" if wait_parts else ""

            phases = [
                (m.phase_name, m.duration_seconds)
                for m in tracer.phase_metrics
                if not (m.details or {}).get("diagnostic")
            ]
            if phases:
                accounted = sum(d for _, d in phases)
                ordered = sorted(phases, key=lambda kv: -kv[1])
                log_buffer.append(
                    "  [phases] "
                    + ", ".join(f"{n}={d * 1000:.0f}ms" for n, d in ordered)
                    + f" | accounted={accounted * 1000:.0f}ms of {total_time * 1000:.0f}ms"
                )
                diags = [
                    (m.phase_name, m.duration_seconds)
                    for m in tracer.phase_metrics
                    if (m.details or {}).get("diagnostic")
                ]
                if diags:
                    log_buffer.append(
                        "  [phases:subsets] "
                        + ", ".join(f"{n}={d * 1000:.0f}ms" for n, d in sorted(diags, key=lambda kv: -kv[1]))
                    )
            log_buffer.append(
                f"[RECALL {recall_id}] Complete: {len(top_scored)} facts ({total_tokens} tok), {num_chunks} chunks ({total_chunk_tokens} tok), {num_entities} entities ({total_entity_tokens} tok) | {fact_type_summary} | {total_time:.3f}s{wait_info}"
            )
            if not quiet:
                logger.info("\n" + "\n".join(log_buffer))

            return RecallResultModel(
                results=memory_facts,
                trace=trace_dict,
                entities=entities_dict,
                chunks=chunks_dict,
                source_facts=source_facts_dict,
                source_facts_truncated=source_facts_truncated if include_source_facts else None,
            )

        except OperationCancelledError:
            raise
        except Exception as e:
            log_buffer.append(
                f"[RECALL {recall_id}] ERROR after {time.time() - recall_start:.3f}s: {type(e).__name__}: {e!r}"
            )
            if not quiet:
                logger.error("\n" + "\n".join(log_buffer), exc_info=True)
            raise RuntimeError(f"Failed to search memories ({type(e).__name__}): {e!r}") from e

    async def _hydrate_chunks(
        self,
        *,
        bank_id: str,
        backend: DatabaseBackend,
        top_scored: list[ScoredResult],
        max_chunk_tokens: int,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
    ) -> HydratedChunks:
        """Fetch chunks from top-scored results."""
        ordered_items: list[ChunkOrderItem] = []
        seen_chunk_ids: set[str] = set()
        observation_ids_ordered: list[uuid.UUID] = []
        carried_sources: dict[str, list[str] | None] = {}
        for sr in top_scored:
            chunk_id = sr.retrieval.chunk_id
            if chunk_id and chunk_id not in seen_chunk_ids:
                ordered_items.append(ChunkOrderItem(item_type="chunk", id=chunk_id))
                seen_chunk_ids.add(chunk_id)
            elif not chunk_id and sr.retrieval.fact_type == "observation":
                ordered_items.append(ChunkOrderItem(item_type="obs", id=sr.id))
                observation_ids_ordered.append(uuid.UUID(sr.id))
                carried_sources[sr.id] = sr.retrieval.source_memory_ids

        obs_chunk_ids: dict[str, list[str]] = {}
        _chunk_store = _get_memories_store()
        if observation_ids_ordered:
            _obs_chunks = await _chunk_store.recall_observation_chunk_ids(
                backend=backend,
                ops=backend.ops,
                fq_table=fq_table,
                bank_id=bank_id,
                observation_ids=observation_ids_ordered,
                carried_sources=carried_sources,
            )
            # Sources the store had to read to answer go back on the results, so the
            # source-facts step below does not read them a second time.
            _sources_read = _obs_chunks.sources_by_observation
            if _sources_read is not None:
                for sr in top_scored:
                    if sr.retrieval.fact_type == "observation" and sr.id in _sources_read:
                        sr.retrieval.source_memory_ids = _sources_read[sr.id]
            for obs_id, cids in _obs_chunks.chunk_ids_by_observation.items():
                for cid in cids:
                    if cid not in seen_chunk_ids:
                        obs_chunk_ids.setdefault(obs_id, []).append(cid)
                        seen_chunk_ids.add(cid)

        chunk_ids_ordered = []
        for item in ordered_items:
            if item.item_type == "chunk":
                chunk_ids_ordered.append(item.id)
            else:
                chunk_ids_ordered.extend(obs_chunk_ids.get(item.id, []))

        chunks_dict = None
        total_chunk_tokens = 0
        if chunk_ids_ordered:
            chunks_dict = {}
            # Fetch all candidate chunks in a single read. Token-budget accounting
            # happens in Python after the fetch — one round-trip is always faster
            # than multiple batched round-trips when the candidate set is large.
            raw_chunks_lookup = await _chunk_store.recall_chunks(
                backend=backend, fq_table=fq_table, bank_id=bank_id, chunk_ids=chunk_ids_ordered
            )
            chunks_lookup: dict[str, ChunkRow] = {
                cid: ChunkRow(
                    chunk_id=row["chunk_id"],
                    chunk_text=row.get("chunk_text") or "",
                    chunk_index=row["chunk_index"],
                    document_id=row.get("document_id"),
                )
                for cid, row in raw_chunks_lookup.items()
            }
            if chunks_lookup and tag_filter_is_active(tags, tags_match, tag_groups):
                # Source text follows its DOCUMENT's tags, not the fact's (#5030): a fact
                # shared through a tag like ``kind:rule`` must not carry the rest of a
                # document the reader's filter excludes. A chunk with no document fails
                # closed — ``None`` is never among the visible ids.
                async with self.store_read_conn(bank_id) as conn:
                    _visible_docs = await visible_document_ids(
                        conn,
                        fq_table,
                        bank_id,
                        (row.document_id for row in chunks_lookup.values()),
                        tags=tags,
                        tags_match=tags_match,
                        tag_groups=tag_groups,
                    )
                chunks_lookup = {cid: row for cid, row in chunks_lookup.items() if row.document_id in _visible_docs}

            for chunk_id in chunk_ids_ordered:
                if chunk_id not in chunks_lookup:
                    continue

                chunk_row = chunks_lookup[chunk_id]
                chunk_text = chunk_row.chunk_text
                chunk_tokens = count_tokens(chunk_text)

                if total_chunk_tokens + chunk_tokens > max_chunk_tokens:
                    remaining_tokens = max_chunk_tokens - total_chunk_tokens
                    if remaining_tokens > 0:
                        truncated_text = truncate_to_tokens(chunk_text, remaining_tokens).text
                        chunks_dict[chunk_id] = ChunkInfo(
                            chunk_text=truncated_text, chunk_index=chunk_row.chunk_index, truncated=True
                        )
                        total_chunk_tokens = max_chunk_tokens
                    break
                else:
                    chunks_dict[chunk_id] = ChunkInfo(
                        chunk_text=chunk_text, chunk_index=chunk_row.chunk_index, truncated=False
                    )
                    total_chunk_tokens += chunk_tokens

        return HydratedChunks(chunks=chunks_dict, total_tokens=total_chunk_tokens)

    async def _hydrate_source_facts(
        self,
        *,
        bank_id: str,
        backend: DatabaseBackend,
        top_scored: list[ScoredResult],
        max_source_facts_tokens: int,
        max_source_facts_tokens_per_observation: int,
    ) -> HydratedSourceFacts:
        """Fetch source facts for observation-type results."""
        source_fact_ids_by_obs: dict[str, list[str]] = {}
        source_facts_dict: dict[str, MemoryFact] | None = None
        source_facts_truncated = False

        observation_srs = [sr for sr in top_scored if sr.retrieval.fact_type == "observation"]
        observation_ids = [uuid.UUID(sr.id) for sr in observation_srs]
        if not observation_ids:
            return HydratedSourceFacts(
                source_facts=None,
                source_fact_ids_by_obs=source_fact_ids_by_obs,
                truncated=False,
            )

        store = _get_memories_store()

        async with acquire_with_retry(backend) as sf_conn:
            obs_sources: Mapping[str, Any]
            if all(sr.retrieval.source_memory_ids is not None for sr in observation_srs):
                obs_sources = {sr.id: sr.retrieval.source_memory_ids for sr in observation_srs}
            else:
                obs_sources = await store.recall_observation_sources(
                    conn=sf_conn, fq_table=fq_table, bank_id=bank_id, observation_ids=observation_ids
                )

            seen_source_ids: set[str] = set()
            source_ids_ordered: list[str] = []
            for obs_id, obs_source_ids in obs_sources.items():
                sids = [str(s) for s in (obs_source_ids or [])]
                source_fact_ids_by_obs[str(obs_id)] = sids
                for sid in sids:
                    if sid not in seen_source_ids:
                        source_ids_ordered.append(sid)
                        seen_source_ids.add(sid)

            if source_ids_ordered:
                source_row_by_id = await store.recall_source_facts(
                    conn=sf_conn, fq_table=fq_table, bank_id=bank_id, unit_ids=source_ids_ordered
                )

                def _make_source_fact(sid: str, r: StoredMemory) -> MemoryFact:
                    return MemoryFact(
                        id=sid,
                        text=r.text,
                        fact_type=r.fact_type,
                        context=r.context,
                        occurred_start=(
                            r.occurred_start.isoformat() if hasattr(r.occurred_start, "isoformat") else r.occurred_start
                        ),
                        occurred_end=(
                            r.occurred_end.isoformat() if hasattr(r.occurred_end, "isoformat") else r.occurred_end
                        ),
                        mentioned_at=(
                            r.mentioned_at.isoformat() if hasattr(r.mentioned_at, "isoformat") else r.mentioned_at
                        ),
                        document_id=r.document_id,
                        metadata=r.metadata,
                        chunk_id=str(r.chunk_id) if r.chunk_id else None,
                        tags=list(r.tags) if r.tags else None,
                    )

                selection = select_source_facts_within_budget(
                    source_ids_ordered=source_ids_ordered,
                    source_fact_ids_by_obs=source_fact_ids_by_obs,
                    text_by_id={sid: r.text for sid, r in source_row_by_id.items()},
                    max_total_tokens=max_source_facts_tokens,
                    max_tokens_per_observation=max_source_facts_tokens_per_observation,
                    count_tokens=count_tokens,
                )
                source_facts_truncated = selection.truncated
                source_facts_dict = {sid: _make_source_fact(sid, source_row_by_id[sid]) for sid in selection.ids}

        return HydratedSourceFacts(
            source_facts=source_facts_dict,
            source_fact_ids_by_obs=source_fact_ids_by_obs,
            truncated=source_facts_truncated,
        )

    async def _hydrate_entities(
        self,
        *,
        bank_id: str,
        backend: DatabaseBackend,
        top_scored: list[ScoredResult],
    ) -> HydratedEntities:
        """Fetch entities and construct EntityState mapping."""
        fact_entity_map: dict[str, list[EntityReference]] = {}
        entities_dict: dict[str, EntityState] | None = None

        unit_ids = [sr.id for sr in top_scored]
        if not unit_ids:
            return HydratedEntities(fact_entity_map=fact_entity_map, entities=entities_dict)

        if all(sr.retrieval.entity_ids is not None for sr in top_scored):
            ids_by_unit = {sr.id: [str(e) for e in (sr.retrieval.entity_ids or [])] for sr in top_scored}
            union = {e for ids in ids_by_unit.values() for e in ids}
            names: dict[str, str] = {}
            if union:
                async with acquire_with_retry(backend) as entity_conn:
                    names = await _get_memories_store().resolve_entity_names(
                        conn=entity_conn, fq_table=fq_table, bank_id=bank_id, entity_ids=list(union)
                    )
            fact_entity_map = _entity_map_from_results(ids_by_unit, names)
        else:
            async with acquire_with_retry(backend) as entity_conn:
                raw_entity_map = await _get_memories_store().entity_map_for_units(
                    conn=entity_conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids
                )
                fact_entity_map = {
                    uid: [
                        EntityReference(entity_id=e["entity_id"], canonical_name=e["canonical_name"]) for e in entries
                    ]
                    for uid, entries in raw_entity_map.items()
                }

        if fact_entity_map:
            entities_ordered: list[EntityReference] = []
            seen_entity_ids: set[str] = set()

            for sr in top_scored:
                unit_id = sr.id
                if unit_id in fact_entity_map:
                    for entity in fact_entity_map[unit_id]:
                        if entity.entity_id not in seen_entity_ids:
                            entities_ordered.append(entity)
                            seen_entity_ids.add(entity.entity_id)

            entities_dict = {}
            for entity in entities_ordered:
                entities_dict[entity.canonical_name] = EntityState(
                    entity_id=entity.entity_id,
                    canonical_name=entity.canonical_name,
                    observations=[],
                )

        return HydratedEntities(fact_entity_map=fact_entity_map, entities=entities_dict)
