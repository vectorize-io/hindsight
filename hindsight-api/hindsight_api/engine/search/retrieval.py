"""
Retrieval module for 4-way parallel search.

Implements:
1. Semantic retrieval (vector similarity)
2. BM25 retrieval (keyword/full-text search)
3. Graph retrieval (via pluggable GraphRetriever interface)
4. Temporal retrieval (time-aware search with spreading)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

from ...config import get_config
from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from .graph_retrieval import BFSGraphRetriever, GraphRetriever
from .mpfp_retrieval import MPFPGraphRetriever
from .types import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ParallelRetrievalResult:
    """Result from parallel retrieval across all methods."""

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph: list[RetrievalResult]
    temporal: list[RetrievalResult] | None
    timings: dict[str, float] = field(default_factory=dict)
    temporal_constraint: tuple | None = None  # (start_date, end_date)


# Default graph retriever instance (can be overridden)
_default_graph_retriever: GraphRetriever | None = None


def get_default_graph_retriever() -> GraphRetriever:
    """Get or create the default graph retriever based on config."""
    global _default_graph_retriever
    if _default_graph_retriever is None:
        config = get_config()
        retriever_type = config.graph_retriever.lower()
        if retriever_type == "mpfp":
            _default_graph_retriever = MPFPGraphRetriever()
            logger.info("Using MPFP graph retriever")
        elif retriever_type == "bfs":
            _default_graph_retriever = BFSGraphRetriever()
            logger.info("Using BFS graph retriever")
        else:
            logger.warning(f"Unknown graph retriever '{retriever_type}', falling back to MPFP")
            _default_graph_retriever = MPFPGraphRetriever()
    return _default_graph_retriever


def set_default_graph_retriever(retriever: GraphRetriever) -> None:
    """Set the default graph retriever (for configuration/testing)."""
    global _default_graph_retriever
    _default_graph_retriever = retriever


async def retrieve_semantic(
    conn, query_emb_str: str, bank_id: str, fact_type: str, limit: int
) -> list[RetrievalResult]:
    """
    Semantic retrieval via vector similarity.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        agent_id: bank ID
        fact_type: Fact type to filter
        limit: Maximum results to return

    Returns:
        List of RetrievalResult objects
    """
    results = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= 0.3
        ORDER BY embedding <=> $1::vector
        LIMIT $4
        """,
        query_emb_str,
        bank_id,
        fact_type,
        limit,
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_bm25(conn, query_text: str, bank_id: str, fact_type: str, limit: int) -> list[RetrievalResult]:
    """
    BM25 keyword retrieval via full-text search.

    Args:
        conn: Database connection
        query_text: Query text
        agent_id: bank ID
        fact_type: Fact type to filter
        limit: Maximum results to return

    Returns:
        List of RetrievalResult objects
    """
    import re

    # Sanitize query text: remove special characters that have meaning in tsquery
    # Keep only alphanumeric characters and spaces
    sanitized_text = re.sub(r"[^\w\s]", " ", query_text.lower())

    # Split and filter empty strings
    tokens = [token for token in sanitized_text.split() if token]

    if not tokens:
        # If no valid tokens, return empty results
        return []

    # Convert query to tsquery using OR for more flexible matching
    # This prevents empty results when some terms are missing
    query_tsquery = " | ".join(tokens)

    results = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               ts_rank_cd(search_vector, to_tsquery('english', $1)) AS bm25_score
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND fact_type = $3
          AND search_vector @@ to_tsquery('english', $1)
        ORDER BY bm25_score DESC
        LIMIT $4
        """,
        query_tsquery,
        bank_id,
        fact_type,
        limit,
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_temporal(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
) -> list[RetrievalResult]:
    """
    Temporal retrieval with spreading activation.

    Strategy:
    1. Find entry points (facts in date range with semantic relevance)
    2. Spread through temporal links to related facts
    3. Score by temporal proximity + semantic similarity + link weight

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        agent_id: bank ID
        fact_type: Fact type to filter
        start_date: Start of time range
        end_date: End of time range
        budget: Node budget for spreading
        semantic_threshold: Minimum semantic similarity to include

    Returns:
        List of RetrievalResult objects with temporal scores
    """

    # Ensure start_date and end_date are timezone-aware (UTC) to match database datetimes
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    entry_points = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND fact_type = $3
          AND embedding IS NOT NULL
          AND (
              -- Match if occurred range overlaps with query range
              (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
               AND occurred_start <= $5 AND occurred_end >= $4)
              OR
              -- Match if mentioned_at falls within query range
              (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5)
              OR
              -- Match if any occurred date is set and overlaps (even if only start or end is set)
              (occurred_start IS NOT NULL AND occurred_start BETWEEN $4 AND $5)
              OR
              (occurred_end IS NOT NULL AND occurred_end BETWEEN $4 AND $5)
          )
          AND (1 - (embedding <=> $1::vector)) >= $6
        ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC, (embedding <=> $1::vector) ASC
        LIMIT 10
        """,
        query_emb_str,
        bank_id,
        fact_type,
        start_date,
        end_date,
        semantic_threshold,
    )

    if not entry_points:
        return []

    # Calculate temporal scores for entry points
    total_days = (end_date - start_date).total_seconds() / 86400
    mid_date = start_date + (end_date - start_date) / 2  # Calculate once for all comparisons
    results = []
    visited = set()

    for ep in entry_points:
        unit_id = str(ep["id"])
        visited.add(unit_id)

        # Calculate temporal proximity using the most relevant date
        # Priority: occurred_start/end (event time) > mentioned_at (mention time)
        best_date = None
        if ep["occurred_start"] is not None and ep["occurred_end"] is not None:
            # Use midpoint of occurred range
            best_date = ep["occurred_start"] + (ep["occurred_end"] - ep["occurred_start"]) / 2
        elif ep["occurred_start"] is not None:
            best_date = ep["occurred_start"]
        elif ep["occurred_end"] is not None:
            best_date = ep["occurred_end"]
        elif ep["mentioned_at"] is not None:
            best_date = ep["mentioned_at"]

        # Temporal proximity score (closer to range center = higher score)
        if best_date:
            days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
            temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
        else:
            temporal_proximity = 0.5  # Fallback if no dates (shouldn't happen due to WHERE clause)

        # Create RetrievalResult with temporal scores
        ep_result = RetrievalResult.from_db_row(dict(ep))
        ep_result.temporal_score = temporal_proximity
        ep_result.temporal_proximity = temporal_proximity
        results.append(ep_result)

    # Spread through temporal links
    queue = [
        (RetrievalResult.from_db_row(dict(ep)), ep["similarity"], 1.0) for ep in entry_points
    ]  # (unit, semantic_sim, temporal_score)
    budget_remaining = budget - len(entry_points)

    while queue and budget_remaining > 0:
        current, semantic_sim, temporal_score = queue.pop(0)
        current_id = current.id

        # Get neighbors via temporal and causal links
        if budget_remaining > 0:
            neighbors = await conn.fetch(
                f"""
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.access_count, mu.embedding, mu.fact_type, mu.document_id, mu.chunk_id,
                       ml.weight, ml.link_type,
                       1 - (mu.embedding <=> $1::vector) AS similarity
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = $2
                  AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
                  AND ml.weight >= 0.1
                  AND mu.fact_type = $3
                  AND mu.embedding IS NOT NULL
                  AND (1 - (mu.embedding <=> $1::vector)) >= $4
                ORDER BY ml.weight DESC
                LIMIT 10
                """,
                query_emb_str,
                current.id,
                fact_type,
                semantic_threshold,
            )

            for n in neighbors:
                neighbor_id = str(n["id"])
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                budget_remaining -= 1

                # Calculate temporal score for neighbor using best available date
                neighbor_best_date = None
                if n["occurred_start"] is not None and n["occurred_end"] is not None:
                    neighbor_best_date = n["occurred_start"] + (n["occurred_end"] - n["occurred_start"]) / 2
                elif n["occurred_start"] is not None:
                    neighbor_best_date = n["occurred_start"]
                elif n["occurred_end"] is not None:
                    neighbor_best_date = n["occurred_end"]
                elif n["mentioned_at"] is not None:
                    neighbor_best_date = n["mentioned_at"]

                if neighbor_best_date:
                    days_from_mid = abs((neighbor_best_date - mid_date).total_seconds() / 86400)
                    neighbor_temporal_proximity = (
                        1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
                    )
                else:
                    neighbor_temporal_proximity = 0.3  # Lower score if no temporal data

                # Boost causal links (same as graph retrieval)
                link_type = n["link_type"]
                if link_type in ("causes", "caused_by"):
                    causal_boost = 2.0
                elif link_type in ("enables", "prevents"):
                    causal_boost = 1.5
                else:
                    causal_boost = 1.0

                # Propagate temporal score through links (decay, with causal boost)
                propagated_temporal = temporal_score * n["weight"] * causal_boost * 0.7

                # Combined temporal score
                combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)

                # Create RetrievalResult with temporal scores
                neighbor_result = RetrievalResult.from_db_row(dict(n))
                neighbor_result.temporal_score = combined_temporal
                neighbor_result.temporal_proximity = neighbor_temporal_proximity
                results.append(neighbor_result)

                # Add to queue for further spreading
                if budget_remaining > 0 and combined_temporal > 0.2:
                    queue.append((neighbor_result, n["similarity"], combined_temporal))

                if budget_remaining <= 0:
                    break

    return results


async def retrieve_parallel(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    thinking_budget: int,
    question_date: datetime | None = None,
    query_analyzer: Optional["QueryAnalyzer"] = None,
    graph_retriever: GraphRetriever | None = None,
) -> ParallelRetrievalResult:
    """
    Run 3-way or 4-way parallel retrieval (adds temporal if detected).

    Args:
        pool: Database connection pool
        query_text: Query text
        query_embedding_str: Query embedding as string
        bank_id: Bank ID
        fact_type: Fact type to filter
        thinking_budget: Budget for graph traversal and retrieval limits
        question_date: Optional date when question was asked (for temporal filtering)
        query_analyzer: Query analyzer to use (defaults to TransformerQueryAnalyzer)
        graph_retriever: Graph retrieval strategy (defaults to configured retriever)

    Returns:
        ParallelRetrievalResult with semantic, bm25, graph, temporal results and timings
    """
    from .temporal_extraction import extract_temporal_constraint

    temporal_constraint = extract_temporal_constraint(query_text, reference_date=question_date, analyzer=query_analyzer)

    retriever = graph_retriever or get_default_graph_retriever()

    if retriever.name == "mpfp":
        return await _retrieve_parallel_mpfp(
            pool, query_text, query_embedding_str, bank_id, fact_type, thinking_budget, temporal_constraint, retriever
        )
    else:
        return await _retrieve_parallel_bfs(
            pool, query_text, query_embedding_str, bank_id, fact_type, thinking_budget, temporal_constraint, retriever
        )


@dataclass
class _SemanticGraphResult:
    """Internal result from semantic→graph chain."""

    semantic: list[RetrievalResult]
    graph: list[RetrievalResult]
    semantic_time: float
    graph_time: float


@dataclass
class _TimedResult:
    """Internal result with timing."""

    results: list[RetrievalResult]
    time: float


async def _retrieve_parallel_mpfp(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    thinking_budget: int,
    temporal_constraint: tuple | None,
    retriever: GraphRetriever,
) -> ParallelRetrievalResult:
    """
    MPFP retrieval with optimized parallelization.

    Runs 2-3 parallel task chains:
    - Task 1: Semantic → Graph (chained, graph uses semantic seeds)
    - Task 2: BM25 (independent)
    - Task 3: Temporal (if constraint detected)
    """
    import time

    async def run_semantic_then_graph() -> _SemanticGraphResult:
        """Chain: semantic retrieval → graph retrieval (using semantic as seeds)."""
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            semantic = await retrieve_semantic(conn, query_embedding_str, bank_id, fact_type, limit=thinking_budget)
        semantic_time = time.time() - start

        # Get temporal seeds if needed (quick query, part of this chain)
        temporal_seeds = None
        if temporal_constraint:
            tc_start, tc_end = temporal_constraint
            async with acquire_with_retry(pool) as conn:
                temporal_seeds = await _get_temporal_entry_points(
                    conn, query_embedding_str, bank_id, fact_type, tc_start, tc_end, limit=20
                )

        # Run graph with seeds
        start = time.time()
        graph = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=fact_type,
            budget=thinking_budget,
            query_text=query_text,
            semantic_seeds=semantic,
            temporal_seeds=temporal_seeds,
        )
        graph_time = time.time() - start

        return _SemanticGraphResult(semantic, graph, semantic_time, graph_time)

    async def run_bm25() -> _TimedResult:
        """Independent BM25 retrieval."""
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_bm25(conn, query_text, bank_id, fact_type, limit=thinking_budget)
        return _TimedResult(results, time.time() - start)

    async def run_temporal(tc_start, tc_end) -> _TimedResult:
        """Temporal retrieval (uses its own entry point finding)."""
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_temporal(
                conn,
                query_embedding_str,
                bank_id,
                fact_type,
                tc_start,
                tc_end,
                budget=thinking_budget,
                semantic_threshold=0.1,
            )
        return _TimedResult(results, time.time() - start)

    # Run parallel task chains
    if temporal_constraint:
        tc_start, tc_end = temporal_constraint
        sg_result, bm25_result, temporal_result = await asyncio.gather(
            run_semantic_then_graph(),
            run_bm25(),
            run_temporal(tc_start, tc_end),
        )
        return ParallelRetrievalResult(
            semantic=sg_result.semantic,
            bm25=bm25_result.results,
            graph=sg_result.graph,
            temporal=temporal_result.results,
            timings={
                "semantic": sg_result.semantic_time,
                "graph": sg_result.graph_time,
                "bm25": bm25_result.time,
                "temporal": temporal_result.time,
            },
            temporal_constraint=temporal_constraint,
        )
    else:
        sg_result, bm25_result = await asyncio.gather(
            run_semantic_then_graph(),
            run_bm25(),
        )
        return ParallelRetrievalResult(
            semantic=sg_result.semantic,
            bm25=bm25_result.results,
            graph=sg_result.graph,
            temporal=None,
            timings={
                "semantic": sg_result.semantic_time,
                "graph": sg_result.graph_time,
                "bm25": bm25_result.time,
            },
            temporal_constraint=None,
        )


async def _get_temporal_entry_points(
    conn,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    start_date: datetime,
    end_date: datetime,
    limit: int = 20,
    semantic_threshold: float = 0.1,
) -> list[RetrievalResult]:
    """Get temporal entry points (facts in date range with semantic relevance)."""

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    rows = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at,
               access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND fact_type = $3
          AND embedding IS NOT NULL
          AND (
              (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
               AND occurred_start <= $5 AND occurred_end >= $4)
              OR (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5)
              OR (occurred_start IS NOT NULL AND occurred_start BETWEEN $4 AND $5)
              OR (occurred_end IS NOT NULL AND occurred_end BETWEEN $4 AND $5)
          )
          AND (1 - (embedding <=> $1::vector)) >= $6
        ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC,
                 (embedding <=> $1::vector) ASC
        LIMIT $7
        """,
        query_embedding_str,
        bank_id,
        fact_type,
        start_date,
        end_date,
        semantic_threshold,
        limit,
    )

    results = []
    total_days = max((end_date - start_date).total_seconds() / 86400, 1)
    mid_date = start_date + (end_date - start_date) / 2

    for row in rows:
        result = RetrievalResult.from_db_row(dict(row))

        # Calculate temporal proximity score
        best_date = None
        if row["occurred_start"] and row["occurred_end"]:
            best_date = row["occurred_start"] + (row["occurred_end"] - row["occurred_start"]) / 2
        elif row["occurred_start"]:
            best_date = row["occurred_start"]
        elif row["occurred_end"]:
            best_date = row["occurred_end"]
        elif row["mentioned_at"]:
            best_date = row["mentioned_at"]

        if best_date:
            days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
            result.temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0)
        else:
            result.temporal_proximity = 0.5

        result.temporal_score = result.temporal_proximity
        results.append(result)

    return results


async def _retrieve_parallel_bfs(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    thinking_budget: int,
    temporal_constraint: tuple | None,
    retriever: GraphRetriever,
) -> ParallelRetrievalResult:
    """BFS retrieval: all methods run in parallel (original behavior)."""
    import time

    async def run_semantic() -> _TimedResult:
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_semantic(conn, query_embedding_str, bank_id, fact_type, limit=thinking_budget)
        return _TimedResult(results, time.time() - start)

    async def run_bm25() -> _TimedResult:
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_bm25(conn, query_text, bank_id, fact_type, limit=thinking_budget)
        return _TimedResult(results, time.time() - start)

    async def run_graph() -> _TimedResult:
        start = time.time()
        results = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=fact_type,
            budget=thinking_budget,
            query_text=query_text,
        )
        return _TimedResult(results, time.time() - start)

    async def run_temporal(tc_start, tc_end) -> _TimedResult:
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_temporal(
                conn,
                query_embedding_str,
                bank_id,
                fact_type,
                tc_start,
                tc_end,
                budget=thinking_budget,
                semantic_threshold=0.1,
            )
        return _TimedResult(results, time.time() - start)

    if temporal_constraint:
        tc_start, tc_end = temporal_constraint
        semantic_r, bm25_r, graph_r, temporal_r = await asyncio.gather(
            run_semantic(),
            run_bm25(),
            run_graph(),
            run_temporal(tc_start, tc_end),
        )
        return ParallelRetrievalResult(
            semantic=semantic_r.results,
            bm25=bm25_r.results,
            graph=graph_r.results,
            temporal=temporal_r.results,
            timings={
                "semantic": semantic_r.time,
                "bm25": bm25_r.time,
                "graph": graph_r.time,
                "temporal": temporal_r.time,
            },
            temporal_constraint=temporal_constraint,
        )
    else:
        semantic_r, bm25_r, graph_r = await asyncio.gather(
            run_semantic(),
            run_bm25(),
            run_graph(),
        )
        return ParallelRetrievalResult(
            semantic=semantic_r.results,
            bm25=bm25_r.results,
            graph=graph_r.results,
            temporal=None,
            timings={
                "semantic": semantic_r.time,
                "bm25": bm25_r.time,
                "graph": graph_r.time,
            },
            temporal_constraint=None,
        )
