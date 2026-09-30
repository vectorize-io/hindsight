"""
Retrieval module for 4-way parallel search.

Hands every recall arm (semantic, BM25, graph, temporal) to the memories store's
``recall_unified`` and assembles what it returns. The Postgres arms' SQL lives in the
store: :mod:`hindsight_api.engine.memories.pg.recall` and
:mod:`hindsight_api.engine.memories.pg.link_expansion`.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from ...config import get_config
from .graph_retrieval import GraphRetriever
from .tags import TagGroup, TagsMatch
from .types import GraphRetrievalTimings, RetrievalResult

if TYPE_CHECKING:
    from ..query_analyzer import QueryAnalyzer
    from ..response_models import TemporalWindow

logger = logging.getLogger(__name__)


def tokenize_query(query_text: str) -> list[str]:
    """Normalize query text and split into BM25 tokens.

    Strips punctuation, lowercases, and splits on whitespace.
    Returns an empty list when the query contains no word characters.
    """
    return re.sub(r"[^\w\s]", " ", query_text.lower()).split()


@dataclass
class ParallelRetrievalResult:
    """Result from parallel retrieval across all methods."""

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph: list[RetrievalResult]
    temporal: list[RetrievalResult] | None
    timings: dict[str, float] = field(default_factory=dict)
    temporal_constraint: tuple | None = None  # (start_date, end_date)
    graph_timings: list[GraphRetrievalTimings] = field(
        default_factory=list
    )  # Graph retrieval sub-step timings per fact type
    max_conn_wait: float = 0.0  # Maximum connection acquisition wait time across all methods


@dataclass
class MultiFactTypeRetrievalResult:
    """Result from retrieval across all fact types."""

    # Results per fact type
    results_by_fact_type: dict[str, ParallelRetrievalResult]
    # Aggregate timings
    timings: dict[str, float] = field(default_factory=dict)
    # Max connection wait across all operations
    max_conn_wait: float = 0.0


# Default graph retriever instance (can be overridden)
_default_graph_retriever: GraphRetriever | None = None


def get_default_graph_retriever() -> GraphRetriever:
    """Get or create the default graph retriever.

    The memories store supplies it: the graph arm walks the store's links, so only the store
    knows how. Postgres returns the SQL retriever ``config.graph_retriever`` names; a store that
    keeps its links elsewhere returns its own. Resolved once and cached.
    """
    global _default_graph_retriever
    if _default_graph_retriever is None:
        from ..memories import get_memories

        store = get_memories()
        from_store = store.graph_retriever()
        if from_store is None:
            # The SQL retriever used to be the fallback here, which for a store that keeps its
            # links elsewhere walked empty tables and returned nothing, silently.
            raise RuntimeError(f"memories store {store.name!r} supplies no graph retriever")
        _default_graph_retriever = from_store
    return _default_graph_retriever


def set_default_graph_retriever(retriever: GraphRetriever | None) -> None:
    """Set the default graph retriever (for configuration/testing).

    ``None`` clears the cache so the next call re-resolves it — used when the
    memories store changes, since the retriever is chosen from it.
    """
    global _default_graph_retriever
    _default_graph_retriever = retriever


async def retrieve_all_fact_types_parallel(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_types: list[str],
    thinking_budget: int,
    question_date: datetime | None = None,
    query_analyzer: Optional["QueryAnalyzer"] = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    min_semantic: float | None = None,
    min_keyword: float | None = None,
    temporal_window: "TemporalWindow | None" = None,
    enable_text_search: bool = True,
    enable_temporal_retrieval: bool = True,
    enable_graph_retrieval: bool = True,
) -> MultiFactTypeRetrievalResult:
    """
    Retrieve every recall arm for all fact types, through the memories store.

    Extracts the temporal constraint (CPU-only), then hands the whole recall off to the
    store's single ``recall_unified`` method — the one recall interface. How the arms are
    run (a per-arm SQL orchestration for Postgres, a single index query for a store that
    owns its index) is the store's business; this only assembles the per-arm result it
    returns into :class:`MultiFactTypeRetrievalResult`. Fusion/rerank happen downstream.

    Args:
        pool: Database connection pool, handed to the store as its connection handle.
        query_text: Query text
        query_embedding_str: Query embedding as string
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        thinking_budget: Budget for graph traversal and retrieval limits
        question_date: Optional date when question was asked (for temporal filtering)
        query_analyzer: Query analyzer to use (defaults to TransformerQueryAnalyzer)
        temporal_window: Caller-supplied window for the temporal arm. When set, it is used
            verbatim instead of analysing the query text for dates. Gated by
            enable_temporal_retrieval like any other source of a window.
        enable_text_search: Run the keyword (BM25) arm. False leaves the arm out of the
            SQL entirely rather than filtering its rows away, so recall is a pure vector
            query and pays none of the arm's cost.
        enable_temporal_retrieval: Run the temporal arm. False also skips the date-aware
            query analysis that feeds it (no constraint means nothing to filter on).
        enable_graph_retrieval: Run the entity/link graph arm. False skips those queries
            and returns no graph results.

    Returns:
        MultiFactTypeRetrievalResult with results organized by fact type
    """
    import time

    config = get_config()
    start_time = time.time()
    timings: dict[str, float] = {}

    # Step 1: Extract temporal constraint first (CPU work, no DB)
    # Do this before the store call so we know whether the temporal arm is needed at all.
    temporal_extraction_start = time.time()
    temporal_constraint = None
    if enable_temporal_retrieval:
        if temporal_window is not None:
            # The caller already knows the range it means, so there is nothing to
            # infer. Skipping the analysis is also the point: it is pure CPU
            # serialised through a single worker, and costs up to ~1.3s on
            # document-sized query text (see temporal_extraction).
            temporal_constraint = (temporal_window.start, temporal_window.end)
        else:
            from .temporal_extraction import extract_temporal_constraint_async

            # Off the event loop: this is pure CPU and would otherwise stall every
            # other in-flight request in the process, not just this recall.
            temporal_constraint = await extract_temporal_constraint_async(
                query_text, reference_date=question_date, analyzer=query_analyzer
            )
    temporal_extraction_time = time.time() - temporal_extraction_start
    timings["temporal_extraction"] = temporal_extraction_time

    # Step 2: Run every arm for every fact type through the store's single recall method.
    from ..memories import RecallArms, get_memories

    # Time the store call itself. Without it `parallel_retrieval` is a black box: it reported 135ms
    # while a bare store-level query measured 33ms, and there was no way to tell whether the
    # difference was the store doing more work (this is 3 fact types x 4 arms in ONE call, not one
    # query) or the host adding overhead around it. The nine per-arm rows below cannot answer that
    # either -- a store-owned recall returns every arm from a single call, so their durations are
    # literals.
    _unified_start = time.time()
    unified = await get_memories().recall_unified(
        conn=pool,
        bank_id=bank_id,
        fact_types=fact_types,
        query_embedding=query_embedding_str,
        query_text=query_text,
        limit=thinking_budget,
        temporal_window=temporal_constraint,
        temporal_semantic_threshold=config.temporal_semantic_min_similarity,
        tags=tags,
        tags_match=tags_match,
        tag_groups=tag_groups,
        created_after=created_after,
        created_before=created_before,
        min_semantic=min_semantic,
        min_keyword=min_keyword,
        enable_text_search=enable_text_search,
        enable_graph=enable_graph_retrieval,
    )

    _unified_elapsed = time.time() - _unified_start

    results_by_fact_type: dict[str, ParallelRetrievalResult] = {}
    for ft in fact_types:
        arms = unified.get(ft) or RecallArms()
        # An empty temporal list collapses to None — the "no temporal arm" signal downstream.
        temporal_arm = arms.temporal or None
        results_by_fact_type[ft] = ParallelRetrievalResult(
            semantic=arms.semantic,
            bm25=arms.bm25,
            graph=arms.graph,
            temporal=temporal_arm,
            # A store-owned recall returns every arm from ONE call, so there is no per-arm
            # split to report and these stay 0.0 -- they are "not measured", not "instant", and
            # reading them as instant is what sent an investigation looking for the missing time
            # outside the store. `store_recall` carries what IS measurable: the whole call.
            timings={
                "semantic": 0.0,
                "bm25": 0.0,
                "graph": 0.0,
                "temporal": 0.0,
                "temporal_extraction": temporal_extraction_time,
                "store_recall": _unified_elapsed,
            },
            temporal_constraint=temporal_constraint,
            graph_timings=[],
            max_conn_wait=0.0,
        )

    timings["total"] = time.time() - start_time
    return MultiFactTypeRetrievalResult(
        results_by_fact_type=results_by_fact_type,
        timings=timings,
        max_conn_wait=0.0,
    )
