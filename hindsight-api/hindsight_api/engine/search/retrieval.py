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
from .link_expansion_retrieval import LinkExpansionRetriever
from .mpfp_retrieval import MPFPGraphRetriever
from .tags import TagsMatch, build_tags_where_clause_simple
from .types import MPFPTimings, RetrievalResult

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
    mpfp_timings: list[MPFPTimings] = field(default_factory=list)  # MPFP sub-step timings per fact type
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
    """Get or create the default graph retriever based on config."""
    global _default_graph_retriever
    if _default_graph_retriever is None:
        config = get_config()
        retriever_type = config.graph_retriever.lower()
        if retriever_type == "mpfp":
            _default_graph_retriever = MPFPGraphRetriever()
            logger.info(
                f"Using MPFP graph retriever (top_k_neighbors={_default_graph_retriever.config.top_k_neighbors})"
            )
        elif retriever_type == "bfs":
            _default_graph_retriever = BFSGraphRetriever()
            logger.info("Using BFS graph retriever")
        elif retriever_type == "link_expansion":
            _default_graph_retriever = LinkExpansionRetriever()
            logger.info("Using LinkExpansion graph retriever")
        else:
            logger.warning(f"Unknown graph retriever '{retriever_type}', falling back to link_expansion")
            _default_graph_retriever = LinkExpansionRetriever()
    return _default_graph_retriever


def set_default_graph_retriever(retriever: GraphRetriever) -> None:
    """Set the default graph retriever (for configuration/testing)."""
    global _default_graph_retriever
    _default_graph_retriever = retriever


async def retrieve_semantic_bm25_combined(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    pool=None,
) -> dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]]:
    """
    Combined semantic + BM25 retrieval for multiple fact types.

    Uses per-fact_type queries with ORDER BY ... LIMIT to enable the HNSW
    index for semantic search, replacing the previous window-function approach
    that forced full sequential scans of all embeddings.

    Requires partial HNSW indexes per fact_type (idx_mu_emb_world,
    idx_mu_emb_observation, idx_mu_emb_experience), created automatically
    by Alembic migration a3b4c5d6e7f8_add_partial_hnsw_indexes.py.

    HNSW is approximate — we over-fetch by 5x (min 100) and trim in Python
    to compensate.

    If `pool` is provided, semantic queries run in parallel via asyncio.gather()
    with separate connections (one per fact_type). Otherwise, all queries run
    sequentially on `conn`.

    Args:
        conn: Database connection (used for BM25 queries and as fallback for semantic)
        query_emb_str: Query embedding as string
        query_text: Query text for BM25
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        limit: Maximum results per method per fact type
        tags: Optional tags to filter by
        tags_match: Tag matching mode
        pool: Optional connection pool for parallel semantic queries

    Returns:
        Dict mapping fact_type -> (semantic_results, bm25_results)
    """
    import re

    result_dict: dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]] = {ft: ([], []) for ft in fact_types}

    # Over-fetch to compensate for HNSW approximate recall + WHERE post-filtering.
    # pgvector's HNSW scans the index first, then applies WHERE conditions (bank_id,
    # fact_type, similarity >= 0.3) as post-filters. Some index candidates get
    # filtered out, so we request 3x more and trim in Python.
    hnsw_fetch_limit = max(limit * 5, 100)

    # --- SEMANTIC: per-fact_type queries (ORDER BY ... LIMIT enables HNSW index) ---
    # Parameters: $1=embedding, $2=bank_id, $3=fact_type, $4=hnsw_fetch_limit, $5=tags (optional)
    sem_tags_clause = build_tags_where_clause_simple(tags, 5, match=tags_match)

    sem_query = f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end,
               mentioned_at, fact_type, document_id, chunk_id, tags,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= 0.3
          {sem_tags_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $4
    """

    async def _run_semantic_for_ft(ft: str, use_conn=None):
        """Run a single semantic query. Uses provided connection or acquires from pool."""
        sem_params: list = [query_emb_str, bank_id, ft, hnsw_fetch_limit]
        if tags:
            sem_params.append(tags)
        c = use_conn
        acquired = False
        if c is None and pool is not None:
            c = await pool.acquire()
            acquired = True
        try:
            await c.execute("SET hnsw.ef_search = 200")
            rows = await c.fetch(sem_query, *sem_params)
        finally:
            await c.execute("RESET hnsw.ef_search")
            if acquired:
                await pool.release(c)
        return ft, rows

    if pool is not None and len(fact_types) > 1:
        # Parallel: each fact_type gets its own connection from the pool
        sem_tasks = [_run_semantic_for_ft(ft) for ft in fact_types]
        sem_results = await asyncio.gather(*sem_tasks)
    else:
        # Sequential: all queries on the same connection
        sem_results = []
        for ft in fact_types:
            sem_results.append(await _run_semantic_for_ft(ft, use_conn=conn))

    # Collect semantic results, trimming to `limit` per fact_type
    for ft, sem_rows in sem_results:
        for r in sem_rows[:limit]:
            result_dict[ft][0].append(RetrievalResult.from_db_row(dict(r)))

    # --- BM25: per-fact_type queries (sequential on conn — not the bottleneck) ---
    sanitized_text = re.sub(r"[^\w\s]", " ", query_text.lower())
    tokens = [token for token in sanitized_text.split() if token]

    if not tokens:
        return result_dict

    config = get_config()

    # Parameters: $1=bank_id, $2=fact_type, $3=limit, $4=query_text/tsquery, $5=tags (optional)
    bm25_tags_clause = build_tags_where_clause_simple(tags, 5, match=tags_match)

    for ft in fact_types:
        if config.text_search_extension == "vchord":
            bm25_score_expr = (
                "search_vector <&> to_bm25query('idx_memory_units_text_search', tokenize($4, 'llmlingua2'))"
            )
            bm25_order_by = f"{bm25_score_expr} DESC"
            bm25_where_filter = ""
            bm25_params: list = [bank_id, ft, limit, query_text]
        elif config.text_search_extension == "pg_textsearch":
            bm25_score_expr = "-(text <@> to_bm25query($4, 'idx_memory_units_text_search'))"
            bm25_order_by = "text <@> to_bm25query($4, 'idx_memory_units_text_search') ASC"
            bm25_where_filter = ""
            bm25_params = [bank_id, ft, limit, query_text]
        else:  # native
            query_tsquery = " | ".join(tokens)
            bm25_score_expr = "ts_rank_cd(search_vector, to_tsquery('english', $4))"
            bm25_order_by = f"{bm25_score_expr} DESC"
            bm25_where_filter = "AND search_vector @@ to_tsquery('english', $4)"
            bm25_params = [bank_id, ft, limit, query_tsquery]

        if tags:
            bm25_params.append(tags)

        bm25_rows = await conn.fetch(
            f"""
            SELECT id, text, context, event_date, occurred_start, occurred_end,
                   mentioned_at, fact_type, document_id, chunk_id, tags,
                   {bm25_score_expr} AS bm25_score
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND fact_type = $2
              {bm25_where_filter}
              {bm25_tags_clause}
            ORDER BY {bm25_order_by}
            LIMIT $3
            """,
            *bm25_params,
        )
        for r in bm25_rows:
            result_dict[ft][1].append(RetrievalResult.from_db_row(dict(r)))

    return result_dict


async def retrieve_temporal_combined(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_types: list[str],
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
) -> dict[str, list[RetrievalResult]]:
    """
    Temporal retrieval for multiple fact types in a single query.

    Batches the entry point query using window functions to get top-N per fact type,
    then runs spreading for each fact type.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        start_date: Start of time range
        end_date: End of time range
        budget: Node budget for spreading per fact type
        semantic_threshold: Minimum semantic similarity to include

    Returns:
        Dict mapping fact_type -> list of RetrievalResult
    """
    from ..memory_engine import fq_table

    # Ensure dates are timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    # Build tags clause
    tags_clause = build_tags_where_clause_simple(tags, 7, match=tags_match)
    params = [query_emb_str, bank_id, fact_types, start_date, end_date, semantic_threshold]
    if tags:
        params.append(tags)

    # Two-phase entry point query:
    # Phase 1 (date_ranked): rank by date only — no embedding computation — for all units in
    #   the temporal window. This lets the planner use date indexes for filtering.
    # Phase 2 (sim_ranked): join back to memory_units for only the top-50-per-type candidates
    #   and compute embedding similarity for that small set (≤ 50 × len(fact_types) rows).
    # This avoids computing embedding distances for potentially thousands of date-range rows.
    entry_points = await conn.fetch(
        f"""
        WITH date_ranked AS MATERIALIZED (
            SELECT id, fact_type,
                   ROW_NUMBER() OVER (
                       PARTITION BY fact_type
                       ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC NULLS LAST
                   ) AS rn
            FROM {fq_table("memory_units")}
            WHERE bank_id = $2
              AND fact_type = ANY($3)
              AND embedding IS NOT NULL
              AND (
                  (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
                   AND occurred_start <= $5 AND occurred_end >= $4)
                  OR
                  (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5)
                  OR
                  (occurred_start IS NOT NULL AND occurred_start BETWEEN $4 AND $5)
                  OR
                  (occurred_end IS NOT NULL AND occurred_end BETWEEN $4 AND $5)
              )
              {tags_clause}
        ),
        sim_ranked AS (
            SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                   1 - (mu.embedding <=> $1::vector) AS similarity,
                   ROW_NUMBER() OVER (PARTITION BY mu.fact_type ORDER BY mu.embedding <=> $1::vector) AS sim_rn
            FROM date_ranked dr
            JOIN {fq_table("memory_units")} mu ON mu.id = dr.id
            WHERE dr.rn <= 50
              AND (1 - (mu.embedding <=> $1::vector)) >= $6
        )
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, fact_type, document_id, chunk_id, tags, similarity
        FROM sim_ranked
        WHERE sim_rn <= 10
        """,
        *params,
    )

    if not entry_points:
        return {ft: [] for ft in fact_types}

    # Group entry points by fact type
    entries_by_ft: dict[str, list] = {ft: [] for ft in fact_types}
    for ep in entry_points:
        ft = ep["fact_type"]
        if ft in entries_by_ft:
            entries_by_ft[ft].append(ep)

    # Calculate shared temporal parameters
    total_days = (end_date - start_date).total_seconds() / 86400
    mid_date = start_date + (end_date - start_date) / 2

    # Process each fact type (spreading needs to stay per fact type due to link filtering)
    results_by_ft: dict[str, list[RetrievalResult]] = {}

    for ft in fact_types:
        ft_entry_points = entries_by_ft.get(ft, [])
        if not ft_entry_points:
            results_by_ft[ft] = []
            continue

        results = []
        visited = set()
        node_scores = {}

        # Process entry points
        for ep in ft_entry_points:
            unit_id = str(ep["id"])
            visited.add(unit_id)

            # Calculate temporal proximity
            best_date = None
            if ep["occurred_start"] is not None and ep["occurred_end"] is not None:
                best_date = ep["occurred_start"] + (ep["occurred_end"] - ep["occurred_start"]) / 2
            elif ep["occurred_start"] is not None:
                best_date = ep["occurred_start"]
            elif ep["occurred_end"] is not None:
                best_date = ep["occurred_end"]
            elif ep["mentioned_at"] is not None:
                best_date = ep["mentioned_at"]

            if best_date:
                days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
                temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
            else:
                temporal_proximity = 0.5

            ep_result = RetrievalResult.from_db_row(dict(ep))
            ep_result.temporal_score = temporal_proximity
            ep_result.temporal_proximity = temporal_proximity
            results.append(ep_result)
            node_scores[unit_id] = (ep["similarity"], 1.0)

        # Spreading through temporal links (same as single-fact-type version)
        frontier = list(node_scores.keys())
        budget_remaining = budget - len(ft_entry_points)
        batch_size = 20
        # Per-source neighbor limit: lets the planner use the composite index
        # (from_unit_id, link_type, weight DESC) with early termination, avoiding
        # a full scan of all links from all source nodes before sorting.
        per_source_limit = 10
        # Safety cap on BFS iterations to prevent runaway spreading in dense graphs.
        max_iterations = 5
        iteration = 0

        # Build tags clause for spreading (use param 7 since 1-6 are used)
        spreading_tags_clause = build_tags_where_clause_simple(tags, 7, table_alias="mu.", match=tags_match)

        while frontier and budget_remaining > 0 and iteration < max_iterations:
            iteration += 1
            batch_ids = frontier[:batch_size]
            frontier = frontier[batch_size:]

            # $1=query_emb, $2=batch_ids, $3=fact_type, $4=threshold, $5=per_source_limit, $6=bank_id, $7=tags
            spreading_params = [query_emb_str, batch_ids, ft, semantic_threshold, per_source_limit, bank_id]
            if tags:
                spreading_params.append(tags)

            # LATERAL join: for each source node, fetch top-K neighbors by weight using
            # the existing idx_memory_links_from_type_weight index with early-exit semantics.
            # This avoids scanning all temporal links from all source nodes before sorting.
            # bank_id on memory_units lets the planner use idx_memory_units_bank_fact_type.
            neighbors = await conn.fetch(
                f"""
                SELECT src.from_unit_id, mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                       l.weight, l.link_type,
                       1 - (mu.embedding <=> $1::vector) AS similarity
                FROM unnest($2::uuid[]) AS src(from_unit_id)
                CROSS JOIN LATERAL (
                    SELECT ml.to_unit_id, ml.weight, ml.link_type
                    FROM {fq_table("memory_links")} ml
                    WHERE ml.from_unit_id = src.from_unit_id
                      AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
                      AND ml.weight >= 0.1
                    ORDER BY ml.weight DESC
                    LIMIT $5
                ) l
                JOIN {fq_table("memory_units")} mu ON mu.id = l.to_unit_id
                WHERE mu.bank_id = $6
                  AND mu.fact_type = $3
                  AND mu.embedding IS NOT NULL
                  AND (1 - (mu.embedding <=> $1::vector)) >= $4
                  {spreading_tags_clause}
                """,
                *spreading_params,
            )

            for n in neighbors:
                neighbor_id = str(n["id"])
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                budget_remaining -= 1

                parent_id = str(n["from_unit_id"])
                _, parent_temporal_score = node_scores.get(parent_id, (0.5, 0.5))

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
                    neighbor_temporal_proximity = 0.3

                link_type = n["link_type"]
                if link_type in ("causes", "caused_by"):
                    causal_boost = 2.0
                elif link_type in ("enables", "prevents"):
                    causal_boost = 1.5
                else:
                    causal_boost = 1.0

                propagated_temporal = parent_temporal_score * n["weight"] * causal_boost * 0.7
                combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)

                neighbor_result = RetrievalResult.from_db_row(dict(n))
                neighbor_result.temporal_score = combined_temporal
                neighbor_result.temporal_proximity = neighbor_temporal_proximity
                results.append(neighbor_result)

                if budget_remaining > 0 and combined_temporal > 0.2:
                    node_scores[neighbor_id] = (n["similarity"], combined_temporal)
                    frontier.append(neighbor_id)

                if budget_remaining <= 0:
                    break

        results_by_ft[ft] = results

    return results_by_ft


async def retrieve_all_fact_types_parallel(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_types: list[str],
    thinking_budget: int,
    question_date: datetime | None = None,
    query_analyzer: Optional["QueryAnalyzer"] = None,
    graph_retriever: GraphRetriever | None = None,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
) -> MultiFactTypeRetrievalResult:
    """
    Optimized retrieval for multiple fact types using batched queries.

    This reduces database round-trips by:
    1. Combining semantic + BM25 into one CTE query for ALL fact types (1 query instead of 2N)
    2. Running graph retrieval per fact type in parallel (N parallel tasks)
    3. Running temporal retrieval per fact type in parallel (N parallel tasks)

    Args:
        pool: Database connection pool
        query_text: Query text
        query_embedding_str: Query embedding as string
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        thinking_budget: Budget for graph traversal and retrieval limits
        question_date: Optional date when question was asked (for temporal filtering)
        query_analyzer: Query analyzer to use (defaults to TransformerQueryAnalyzer)
        graph_retriever: Graph retrieval strategy (defaults to configured retriever)

    Returns:
        MultiFactTypeRetrievalResult with results organized by fact type
    """
    import time

    retriever = graph_retriever or get_default_graph_retriever()
    start_time = time.time()
    timings: dict[str, float] = {}

    # Step 1: Extract temporal constraint first (CPU work, no DB)
    # Do this before DB queries so we know if we need temporal retrieval
    temporal_extraction_start = time.time()
    from .temporal_extraction import extract_temporal_constraint

    temporal_constraint = extract_temporal_constraint(query_text, reference_date=question_date, analyzer=query_analyzer)
    temporal_extraction_time = time.time() - temporal_extraction_start
    timings["temporal_extraction"] = temporal_extraction_time

    # Step 2: Run semantic + BM25 + temporal combined in ONE connection!
    # This reduces connection usage from 2 to 1 for these operations
    semantic_bm25_start = time.time()
    temporal_results_by_ft: dict[str, list[RetrievalResult]] = {}
    temporal_time = 0.0

    async with acquire_with_retry(pool) as conn:
        conn_wait = time.time() - semantic_bm25_start

        # Semantic + BM25 combined
        semantic_bm25_results = await retrieve_semantic_bm25_combined(
            conn,
            query_embedding_str,
            query_text,
            bank_id,
            fact_types,
            thinking_budget,
            tags=tags,
            tags_match=tags_match,
            pool=pool,
        )
        semantic_bm25_time = time.time() - semantic_bm25_start

        # Temporal combined (if constraint detected) - same connection!
        if temporal_constraint:
            tc_start, tc_end = temporal_constraint
            temporal_start = time.time()
            temporal_results_by_ft = await retrieve_temporal_combined(
                conn,
                query_embedding_str,
                bank_id,
                fact_types,
                tc_start,
                tc_end,
                budget=thinking_budget,
                semantic_threshold=0.1,
                tags=tags,
                tags_match=tags_match,
            )
            temporal_time = time.time() - temporal_start

    timings["semantic_bm25_combined"] = semantic_bm25_time
    timings["temporal_combined"] = temporal_time

    # Step 3: Run graph retrieval for each fact type in parallel
    async def run_graph_for_fact_type(ft: str) -> tuple[str, list[RetrievalResult], float, MPFPTimings | None]:
        graph_start = time.time()
        results, mpfp_timing = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=ft,
            budget=thinking_budget,
            query_text=query_text,
            semantic_seeds=None,
            temporal_seeds=None,
            tags=tags,
            tags_match=tags_match,
        )
        return ft, results, time.time() - graph_start, mpfp_timing

    # Run graph for all fact types in parallel
    graph_tasks = [run_graph_for_fact_type(ft) for ft in fact_types]
    graph_results_list = await asyncio.gather(*graph_tasks)

    # Organize results by fact type
    results_by_fact_type: dict[str, ParallelRetrievalResult] = {}
    max_conn_wait = conn_wait  # Single connection for semantic+bm25+temporal
    all_mpfp_timings: list[MPFPTimings] = []

    for ft in fact_types:
        # Get semantic + bm25 results for this fact type
        semantic_results, bm25_results = semantic_bm25_results.get(ft, ([], []))

        # Find graph results for this fact type
        graph_results = []
        graph_time = 0.0
        mpfp_timing = None
        for gr in graph_results_list:
            if gr[0] == ft:
                graph_results = gr[1]
                graph_time = gr[2]
                mpfp_timing = gr[3]
                if mpfp_timing:
                    all_mpfp_timings.append(mpfp_timing)
                break

        # Get temporal results for this fact type from combined result
        temporal_results = temporal_results_by_ft.get(ft) if temporal_constraint else None
        if temporal_results is not None and len(temporal_results) == 0:
            temporal_results = None

        results_by_fact_type[ft] = ParallelRetrievalResult(
            semantic=semantic_results,
            bm25=bm25_results,
            graph=graph_results,
            temporal=temporal_results,
            timings={
                "semantic": semantic_bm25_time / 2,  # Approximate split
                "bm25": semantic_bm25_time / 2,
                "graph": graph_time,
                "temporal": temporal_time,  # Same for all fact types (single query)
                "temporal_extraction": temporal_extraction_time,
            },
            temporal_constraint=temporal_constraint,
            mpfp_timings=[mpfp_timing] if mpfp_timing else [],
            max_conn_wait=max_conn_wait,
        )

    total_time = time.time() - start_time
    timings["total"] = total_time

    return MultiFactTypeRetrievalResult(
        results_by_fact_type=results_by_fact_type,
        timings=timings,
        max_conn_wait=max_conn_wait,
    )
