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


async def retrieve_semantic(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    limit: int,
    tags: list[str] | None = None,
) -> list[RetrievalResult]:
    """
    Semantic retrieval via vector similarity.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        agent_id: bank ID
        fact_type: Fact type to filter
        limit: Maximum results to return
        tags: Optional list of tags for visibility filtering (OR matching)

    Returns:
        List of RetrievalResult objects
    """
    from .tags import TagsMatch, build_tags_where_clause_simple

    tags_clause = build_tags_where_clause_simple(tags, 5)
    params = [query_emb_str, bank_id, fact_type, limit]
    if tags:
        params.append(tags)

    results = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= 0.3
          {tags_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $4
        """,
        *params,
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_bm25(
    conn,
    query_text: str,
    bank_id: str,
    fact_type: str,
    limit: int,
    tags: list[str] | None = None,
) -> list[RetrievalResult]:
    """
    BM25 keyword retrieval via full-text search.

    Args:
        conn: Database connection
        query_text: Query text
        agent_id: bank ID
        fact_type: Fact type to filter
        limit: Maximum results to return
        tags: Optional list of tags for visibility filtering (OR matching)

    Returns:
        List of RetrievalResult objects
    """
    import re

    from .tags import TagsMatch, build_tags_where_clause_simple

    # Sanitize query text for native backend: remove special characters that have meaning in tsquery
    # Keep only alphanumeric characters and spaces
    sanitized_text = re.sub(r"[^\w\s]", " ", query_text.lower())

    # Split and filter empty strings
    tokens = [token for token in sanitized_text.split() if token]

    if not tokens:
        # If no valid tokens, return empty results
        return []

    # Build query based on text search backend
    config = get_config()
    tags_clause = build_tags_where_clause_simple(tags, 5)

    if config.text_search_extension == "vchord":
        # VectorChord BM25: use <&> operator with to_bm25query and tokenize
        params = [bank_id, fact_type, limit, query_text]  # Use raw query_text for tokenization
        if tags:
            params.append(query_text)  # VectorChord doesn't need sanitization

        query = f"""
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   search_vector <&> to_bm25query('idx_memory_units_text_search', tokenize($4, 'llmlingua2')) AS bm25_score
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND fact_type = $2
              {tags_clause}
            ORDER BY bm25_score DESC
            LIMIT $3
        """
    else:  # native
        # Native PostgreSQL: use ts_rank_cd with to_tsquery
        query_tsquery = " | ".join(tokens)
        params = [query_tsquery, bank_id, fact_type, limit]
        if tags:
            params.append(tags)

        query = f"""
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   ts_rank_cd(search_vector, to_tsquery('english', $1)) AS bm25_score
            FROM {fq_table("memory_units")}
            WHERE bank_id = $2
              AND fact_type = $3
              AND search_vector @@ to_tsquery('english', $1)
              {tags_clause}
            ORDER BY bm25_score DESC
            LIMIT $4
        """

    results = await conn.fetch(query, *params)
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_semantic_bm25_combined(
    conn,
    query_emb_str: str,
    query_text: str,
    bank_id: str,
    fact_types: list[str],
    limit: int,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
) -> dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]]:
    """
    Combined semantic + BM25 retrieval for multiple fact types in a single query.

    Uses CTEs with window functions to get top-N results per fact type per method,
    all in one database round-trip.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        query_text: Query text for BM25
        bank_id: Bank ID
        fact_types: List of fact types to retrieve
        limit: Maximum results per method per fact type

    Returns:
        Dict mapping fact_type -> (semantic_results, bm25_results)
    """
    import re

    # Sanitize query text for BM25 (same as retrieve_bm25)
    sanitized_text = re.sub(r"[^\w\s]", " ", query_text.lower())
    tokens = [token for token in sanitized_text.split() if token]

    # If no valid tokens for BM25, just run semantic
    if not tokens:
        tags_clause = build_tags_where_clause_simple(tags, 5, match=tags_match)
        params = [query_emb_str, bank_id, fact_types, limit]
        if tags:
            params.append(tags)
        results = await conn.fetch(
            f"""
            WITH semantic_ranked AS (
                SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                       1 - (embedding <=> $1::vector) AS similarity,
                       NULL::float AS bm25_score,
                       'semantic' AS source,
                       ROW_NUMBER() OVER (PARTITION BY fact_type ORDER BY embedding <=> $1::vector) AS rn
                FROM {fq_table("memory_units")}
                WHERE bank_id = $2
                  AND embedding IS NOT NULL
                  AND fact_type = ANY($3)
                  AND (1 - (embedding <=> $1::vector)) >= 0.3
                  {tags_clause}
            )
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   similarity, bm25_score, source
            FROM semantic_ranked
            WHERE rn <= $4
            """,
            *params,
        )
        # Group by fact_type
        result_dict: dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]] = {
            ft: ([], []) for ft in fact_types
        }
        for r in results:
            row = dict(r)
            ft = row.get("fact_type")
            row.pop("source", None)
            if ft in result_dict:
                result_dict[ft][0].append(RetrievalResult.from_db_row(row))
        return result_dict

    # Build BM25 query based on text search backend
    config = get_config()

    # Build tags clause - param 6 if tags provided
    tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)

    # Build backend-specific BM25 parts
    if config.text_search_extension == "vchord":
        # VectorChord BM25: use <&> operator with to_bm25query and tokenize
        # Note: VectorChord scores are negative (higher = better, so -1 > -10)
        bm25_score_expr = "search_vector <&> to_bm25query('idx_memory_units_text_search', tokenize($5, 'llmlingua2'))"
        bm25_order_by = f"{bm25_score_expr} DESC"
        bm25_where_filter = ""  # No additional WHERE filter for vchord
        params = [query_emb_str, bank_id, fact_types, limit, query_text]  # Pass raw query_text for tokenization
    elif config.text_search_extension == "pg_textsearch":
        # Timescale pg_textsearch: use <@> operator with to_bm25query
        # Note: pg_textsearch scores are negative (lower/more negative = better, so -10 > -1)
        # We negate the score to maintain API consistency (higher = better)
        bm25_score_expr = "-(text <@> to_bm25query($5, 'idx_memory_units_text_search'))"
        bm25_order_by = "text <@> to_bm25query($5, 'idx_memory_units_text_search') ASC"
        bm25_where_filter = ""  # No additional WHERE filter for pg_textsearch
        params = [query_emb_str, bank_id, fact_types, limit, query_text]
    else:  # native
        # Native PostgreSQL: use ts_rank_cd with to_tsquery
        query_tsquery = " | ".join(tokens)
        bm25_score_expr = "ts_rank_cd(search_vector, to_tsquery('english', $5))"
        bm25_order_by = f"{bm25_score_expr} DESC"
        bm25_where_filter = "AND search_vector @@ to_tsquery('english', $5)"
        params = [query_emb_str, bank_id, fact_types, limit, query_tsquery]

    if tags:
        params.append(tags)

    # Single query template with backend-specific parts injected
    query = f"""
        WITH semantic_ranked AS (
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   1 - (embedding <=> $1::vector) AS similarity,
                   NULL::float AS bm25_score,
                   'semantic' AS source,
                   ROW_NUMBER() OVER (PARTITION BY fact_type ORDER BY embedding <=> $1::vector) AS rn
            FROM {fq_table("memory_units")}
            WHERE bank_id = $2
              AND embedding IS NOT NULL
              AND fact_type = ANY($3)
              AND (1 - (embedding <=> $1::vector)) >= 0.3
              {tags_clause}
        ),
        bm25_ranked AS (
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   NULL::float AS similarity,
                   {bm25_score_expr} AS bm25_score,
                   'bm25' AS source,
                   ROW_NUMBER() OVER (PARTITION BY fact_type ORDER BY {bm25_order_by}) AS rn
            FROM {fq_table("memory_units")}
            WHERE bank_id = $2
              AND fact_type = ANY($3)
              {bm25_where_filter}
              {tags_clause}
        ),
        semantic AS (
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   similarity, bm25_score, source
            FROM semantic_ranked WHERE rn <= $4
        ),
        bm25 AS (
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   similarity, bm25_score, source
            FROM bm25_ranked WHERE rn <= $4
        )
        SELECT * FROM semantic
        UNION ALL
        SELECT * FROM bm25
    """

    # Combined CTE query for both semantic and BM25 across all fact types
    # Uses window functions to limit per fact_type per method
    results = await conn.fetch(query, *params)

    # Group results by fact_type and source
    result_dict: dict[str, tuple[list[RetrievalResult], list[RetrievalResult]]] = {ft: ([], []) for ft in fact_types}
    for r in results:
        row = dict(r)
        source = row.pop("source", None)
        ft = row.get("fact_type")
        if ft in result_dict:
            if source == "semantic":
                result_dict[ft][0].append(RetrievalResult.from_db_row(row))
            else:
                result_dict[ft][1].append(RetrievalResult.from_db_row(row))

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

    # Batch query: Get entry points for ALL fact types at once with window function
    entry_points = await conn.fetch(
        f"""
        WITH ranked_entries AS (
            SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
                   1 - (embedding <=> $1::vector) AS similarity,
                   ROW_NUMBER() OVER (PARTITION BY fact_type ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC, embedding <=> $1::vector) AS rn
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
              AND (1 - (embedding <=> $1::vector)) >= $6
              {tags_clause}
        )
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags, similarity
        FROM ranked_entries
        WHERE rn <= 10
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

        # Build tags clause for spreading (use param 6 since 1-5 are used)
        spreading_tags_clause = build_tags_where_clause_simple(tags, 6, table_alias="mu.", match=tags_match)

        while frontier and budget_remaining > 0:
            batch_ids = frontier[:batch_size]
            frontier = frontier[batch_size:]

            spreading_params = [query_emb_str, batch_ids, ft, semantic_threshold, batch_size * 10]
            if tags:
                spreading_params.append(tags)

            neighbors = await conn.fetch(
                f"""
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.embedding, mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                       ml.weight, ml.link_type, ml.from_unit_id,
                       1 - (mu.embedding <=> $1::vector) AS similarity
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($2::uuid[])
                  AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
                  AND ml.weight >= 0.1
                  AND mu.fact_type = $3
                  AND mu.embedding IS NOT NULL
                  AND (1 - (mu.embedding <=> $1::vector)) >= $4
                  {spreading_tags_clause}
                ORDER BY ml.weight DESC
                LIMIT $5
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


async def retrieve_temporal(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1,
    tags: list[str] | None = None,
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
        tags: Optional list of tags for visibility filtering (OR matching)

    Returns:
        List of RetrievalResult objects with temporal scores
    """

    # Ensure start_date and end_date are timezone-aware (UTC) to match database datetimes
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    from .tags import TagsMatch, build_tags_where_clause_simple

    tags_clause = build_tags_where_clause_simple(tags, 7)
    params = [query_emb_str, bank_id, fact_type, start_date, end_date, semantic_threshold]
    if tags:
        params.append(tags)

    entry_points = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
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
          {tags_clause}
        ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC, (embedding <=> $1::vector) ASC
        LIMIT 10
        """,
        *params,
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

    # Spread through temporal links using BATCHED neighbor fetching
    # Map node_id -> (semantic_sim, temporal_score) for propagation
    node_scores = {str(ep["id"]): (ep["similarity"], 1.0) for ep in entry_points}
    frontier = list(node_scores.keys())  # Current batch of nodes to expand
    budget_remaining = budget - len(entry_points)
    batch_size = 20  # Process this many nodes per DB query

    while frontier and budget_remaining > 0:
        # Take a batch from frontier
        batch_ids = frontier[:batch_size]
        frontier = frontier[batch_size:]

        # Batch fetch all neighbors for this batch of nodes
        neighbors = await conn.fetch(
            f"""
            SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.embedding, mu.fact_type, mu.document_id, mu.chunk_id,
                   ml.weight, ml.link_type, ml.from_unit_id,
                   1 - (mu.embedding <=> $1::vector) AS similarity
            FROM {fq_table("memory_links")} ml
            JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
            WHERE ml.from_unit_id = ANY($2::uuid[])
              AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
              AND ml.weight >= 0.1
              AND mu.fact_type = $3
              AND mu.embedding IS NOT NULL
              AND (1 - (mu.embedding <=> $1::vector)) >= $4
            ORDER BY ml.weight DESC
            LIMIT $5
            """,
            query_emb_str,
            batch_ids,
            fact_type,
            semantic_threshold,
            batch_size * 10,  # Allow up to 10 neighbors per node in batch
        )

        for n in neighbors:
            neighbor_id = str(n["id"])
            if neighbor_id in visited:
                continue

            visited.add(neighbor_id)
            budget_remaining -= 1

            # Get parent's scores for propagation
            parent_id = str(n["from_unit_id"])
            _, parent_temporal_score = node_scores.get(parent_id, (0.5, 0.5))

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
            propagated_temporal = parent_temporal_score * n["weight"] * causal_boost * 0.7

            # Combined temporal score
            combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)

            # Create RetrievalResult with temporal scores
            neighbor_result = RetrievalResult.from_db_row(dict(n))
            neighbor_result.temporal_score = combined_temporal
            neighbor_result.temporal_proximity = neighbor_temporal_proximity
            results.append(neighbor_result)

            # Track scores for propagation and add to frontier
            if budget_remaining > 0 and combined_temporal > 0.2:
                node_scores[neighbor_id] = (n["similarity"], combined_temporal)
                frontier.append(neighbor_id)

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
    temporal_constraint: tuple | None = None,  # Pre-extracted temporal constraint
    tags: list[str] | None = None,  # Visibility scope tags for filtering
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
        temporal_constraint: Pre-extracted temporal constraint (optional)
        tags: Optional list of tags for visibility filtering (OR matching)

    Returns:
        ParallelRetrievalResult with semantic, bm25, graph, temporal results and timings
    """
    retriever = graph_retriever or get_default_graph_retriever()

    # Use optimized parallel path for MPFP and LinkExpansion (runs all methods truly in parallel)
    # BFS uses legacy path that extracts temporal constraint upfront
    if retriever.name in ("mpfp", "link_expansion"):
        return await _retrieve_parallel_mpfp(
            pool,
            query_text,
            query_embedding_str,
            bank_id,
            fact_type,
            thinking_budget,
            temporal_constraint,
            retriever,
            question_date,
            query_analyzer,
            tags=tags,
        )
    else:
        # For BFS, extract temporal constraint upfront (legacy path)
        if temporal_constraint is None:
            from .temporal_extraction import extract_temporal_constraint

            temporal_constraint = extract_temporal_constraint(
                query_text, reference_date=question_date, analyzer=query_analyzer
            )
        return await _retrieve_parallel_bfs(
            pool,
            query_text,
            query_embedding_str,
            bank_id,
            fact_type,
            thinking_budget,
            temporal_constraint,
            retriever,
            tags=tags,
        )


@dataclass
class _TimedResult:
    """Internal result with timing."""

    results: list[RetrievalResult]
    time: float
    conn_wait: float = 0.0  # Connection acquisition wait time


async def _retrieve_parallel_mpfp(
    pool,
    query_text: str,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    thinking_budget: int,
    temporal_constraint: tuple | None,
    retriever: GraphRetriever,
    question_date: datetime | None = None,
    query_analyzer=None,
    tags: list[str] | None = None,
) -> ParallelRetrievalResult:
    """
    MPFP retrieval with true parallelization.

    All methods run independently in parallel:
    - Semantic: vector similarity search
    - BM25: keyword search
    - Graph: MPFP traversal (does its own semantic seeds internally)
    - Temporal: date extraction (if needed) + date-range search

    Temporal extraction runs IN PARALLEL with other retrievals, so even if
    dateparser is slow, it doesn't block semantic/BM25/graph.
    """
    import time

    async def run_semantic() -> _TimedResult:
        """Independent semantic retrieval."""
        start = time.time()
        acquire_start = time.time()
        async with acquire_with_retry(pool) as conn:
            conn_wait = time.time() - acquire_start
            results = await retrieve_semantic(
                conn, query_embedding_str, bank_id, fact_type, limit=thinking_budget, tags=tags
            )
        return _TimedResult(results, time.time() - start, conn_wait)

    async def run_bm25() -> _TimedResult:
        """Independent BM25 retrieval."""
        start = time.time()
        acquire_start = time.time()
        async with acquire_with_retry(pool) as conn:
            conn_wait = time.time() - acquire_start
            results = await retrieve_bm25(conn, query_text, bank_id, fact_type, limit=thinking_budget, tags=tags)
        return _TimedResult(results, time.time() - start, conn_wait)

    async def run_graph() -> tuple[list[RetrievalResult], float, MPFPTimings | None]:
        """Independent graph retrieval - does its own semantic seeds."""
        start = time.time()

        # MPFP does its own semantic seeds via _find_semantic_seeds
        # Note: temporal_seeds not used here to avoid dependency on temporal extraction
        results, mpfp_timing = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=fact_type,
            budget=thinking_budget,
            query_text=query_text,
            semantic_seeds=None,  # Let MPFP find its own seeds
            temporal_seeds=None,  # Don't wait for temporal extraction
            tags=tags,
        )
        return results, time.time() - start, mpfp_timing

    @dataclass
    class _TemporalWithConstraint:
        """Temporal results with the extracted constraint."""

        results: list[RetrievalResult]
        time: float
        constraint: tuple | None
        extraction_time: float  # Time spent in query analyzer (dateparser)
        conn_wait: float = 0.0  # Connection acquisition wait time

    async def run_temporal_with_extraction() -> _TemporalWithConstraint:
        """
        Extract temporal constraint AND run temporal retrieval.

        This runs in parallel with semantic/BM25/graph, so dateparser
        latency doesn't block other retrievals.
        """
        start = time.time()

        # Use pre-provided constraint if available
        tc = temporal_constraint
        extraction_time = 0.0

        # Otherwise extract from query (this is the potentially slow dateparser call)
        if tc is None:
            from .temporal_extraction import extract_temporal_constraint

            extraction_start = time.time()
            tc = extract_temporal_constraint(query_text, reference_date=question_date, analyzer=query_analyzer)
            extraction_time = time.time() - extraction_start

        # If no temporal constraint found, return empty (but still report extraction time)
        if tc is None:
            return _TemporalWithConstraint([], time.time() - start, None, extraction_time, 0.0)

        # Run temporal retrieval with the extracted constraint
        tc_start, tc_end = tc
        acquire_start = time.time()
        async with acquire_with_retry(pool) as conn:
            conn_wait = time.time() - acquire_start
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
        return _TemporalWithConstraint(results, time.time() - start, tc, extraction_time, conn_wait)

    # Run ALL methods in parallel (including temporal extraction!)
    semantic_result, bm25_result, graph_result, temporal_result = await asyncio.gather(
        run_semantic(),
        run_bm25(),
        run_graph(),
        run_temporal_with_extraction(),
    )
    graph_results, graph_time, mpfp_timing = graph_result

    # Compute max connection wait across all methods (graph handles its own connections)
    max_conn_wait = max(semantic_result.conn_wait, bm25_result.conn_wait, temporal_result.conn_wait)

    return ParallelRetrievalResult(
        semantic=semantic_result.results,
        bm25=bm25_result.results,
        graph=graph_results,
        temporal=temporal_result.results if temporal_result.results else None,
        timings={
            "semantic": semantic_result.time,
            "bm25": bm25_result.time,
            "graph": graph_time,
            "temporal": temporal_result.time,
            "temporal_extraction": temporal_result.extraction_time,
        },
        temporal_constraint=temporal_result.constraint,
        mpfp_timings=[mpfp_timing] if mpfp_timing else [],
        max_conn_wait=max_conn_wait,
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
               embedding, fact_type, document_id, chunk_id,
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
    tags: list[str] | None = None,
) -> ParallelRetrievalResult:
    """BFS retrieval: all methods run in parallel (original behavior)."""
    import time

    async def run_semantic() -> _TimedResult:
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_semantic(
                conn, query_embedding_str, bank_id, fact_type, limit=thinking_budget, tags=tags
            )
        return _TimedResult(results, time.time() - start)

    async def run_bm25() -> _TimedResult:
        start = time.time()
        async with acquire_with_retry(pool) as conn:
            results = await retrieve_bm25(conn, query_text, bank_id, fact_type, limit=thinking_budget, tags=tags)
        return _TimedResult(results, time.time() - start)

    async def run_graph() -> _TimedResult:
        start = time.time()
        results, _ = await retriever.retrieve(
            pool=pool,
            query_embedding_str=query_embedding_str,
            bank_id=bank_id,
            fact_type=fact_type,
            budget=thinking_budget,
            query_text=query_text,
            tags=tags,
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
                tags=tags,
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
