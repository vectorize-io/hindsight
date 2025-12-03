"""
Retrieval module for 4-way parallel search.

Implements:
1. Semantic retrieval (vector similarity)
2. BM25 retrieval (keyword/full-text search)
3. Graph retrieval (spreading activation)
4. Temporal retrieval (time-aware search with spreading)
"""

from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import asyncio
from ..db_utils import acquire_with_retry
from .types import RetrievalResult


async def retrieve_semantic(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    limit: int
) -> List[RetrievalResult]:
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
        """
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memory_units
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= 0.3
        ORDER BY embedding <=> $1::vector
        LIMIT $4
        """,
        query_emb_str, bank_id, fact_type, limit
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_bm25(
    conn,
    query_text: str,
    bank_id: str,
    fact_type: str,
    limit: int
) -> List[RetrievalResult]:
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
    sanitized_text = re.sub(r'[^\w\s]', ' ', query_text.lower())

    # Split and filter empty strings
    tokens = [token for token in sanitized_text.split() if token]

    if not tokens:
        # If no valid tokens, return empty results
        return []

    # Convert query to tsquery using OR for more flexible matching
    # This prevents empty results when some terms are missing
    query_tsquery = " | ".join(tokens)

    results = await conn.fetch(
        """
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               ts_rank_cd(search_vector, to_tsquery('english', $1)) AS bm25_score
        FROM memory_units
        WHERE bank_id = $2
          AND fact_type = $3
          AND search_vector @@ to_tsquery('english', $1)
        ORDER BY bm25_score DESC
        LIMIT $4
        """,
        query_tsquery, bank_id, fact_type, limit
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in results]


async def retrieve_graph(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    budget: int
) -> List[RetrievalResult]:
    """
    Graph retrieval via spreading activation.

    Args:
        conn: Database connection
        query_emb_str: Query embedding as string
        agent_id: bank ID
        fact_type: Fact type to filter
        budget: Node budget for graph traversal

    Returns:
        List of RetrievalResult objects
    """
    # Find entry points
    entry_points = await conn.fetch(
        """
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memory_units
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= 0.5
        ORDER BY embedding <=> $1::vector
        LIMIT 5
        """,
        query_emb_str, bank_id, fact_type
    )

    if not entry_points:
        return []

    # BFS-style spreading activation with batched neighbor fetching
    visited = set()
    results = []
    queue = [(RetrievalResult.from_db_row(dict(r)), r["similarity"]) for r in entry_points]
    budget_remaining = budget

    # Process nodes in batches to reduce DB roundtrips
    batch_size = 20  # Fetch neighbors for up to 20 nodes at once

    while queue and budget_remaining > 0:
        # Collect a batch of nodes to process
        batch_nodes = []
        batch_activations = {}

        while queue and len(batch_nodes) < batch_size and budget_remaining > 0:
            current, activation = queue.pop(0)
            unit_id = current.id

            if unit_id not in visited:
                visited.add(unit_id)
                budget_remaining -= 1
                results.append(current)
                batch_nodes.append(current.id)
                batch_activations[unit_id] = activation

        # Batch fetch neighbors for all nodes in this batch
        # Fetch top weighted neighbors (batch_size * 10 = ~200 for good distribution)
        if batch_nodes and budget_remaining > 0:
            max_neighbors = len(batch_nodes) * 10
            neighbors = await conn.fetch(
                """
                SELECT mu.id, mu.text, mu.context, mu.occurred_start, mu.occurred_end, mu.mentioned_at,
                       mu.access_count, mu.embedding, mu.fact_type, mu.document_id, mu.chunk_id,
                       ml.weight, ml.link_type, ml.from_unit_id
                FROM memory_links ml
                JOIN memory_units mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.weight >= 0.1
                  AND mu.fact_type = $2
                ORDER BY ml.weight DESC
                LIMIT $3
                """,
                batch_nodes, fact_type, max_neighbors
            )

            for n in neighbors:
                neighbor_id = str(n["id"])
                if neighbor_id not in visited:
                    # Get parent activation
                    parent_id = str(n["from_unit_id"])
                    activation = batch_activations.get(parent_id, 0.5)

                    # Boost activation for causal links (they're high-value relationships)
                    link_type = n["link_type"]
                    base_weight = n["weight"]

                    # Causal links get 1.5-2.0x boost depending on type
                    if link_type in ("causes", "caused_by"):
                        # Direct causation - very strong relationship
                        causal_boost = 2.0
                    elif link_type in ("enables", "prevents"):
                        # Conditional causation - strong but not as direct
                        causal_boost = 1.5
                    else:
                        # Temporal, semantic, entity links - standard weight
                        causal_boost = 1.0

                    effective_weight = base_weight * causal_boost
                    new_activation = activation * effective_weight * 0.8
                    if new_activation > 0.1:
                        neighbor_result = RetrievalResult.from_db_row(dict(n))
                        queue.append((neighbor_result, new_activation))

    return results


async def retrieve_temporal(
    conn,
    query_emb_str: str,
    bank_id: str,
    fact_type: str,
    start_date: datetime,
    end_date: datetime,
    budget: int,
    semantic_threshold: float = 0.1
) -> List[RetrievalResult]:
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
    from datetime import timezone

    # Ensure start_date and end_date are timezone-aware (UTC) to match database datetimes
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    entry_points = await conn.fetch(
        """
        SELECT id, text, context, event_date, occurred_start, occurred_end, mentioned_at, access_count, embedding, fact_type, document_id, chunk_id,
               1 - (embedding <=> $1::vector) AS similarity
        FROM memory_units
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
        query_emb_str, bank_id, fact_type, start_date, end_date, semantic_threshold
    )

    import logging
    logger = logging.getLogger(__name__)

    if not entry_points:
        # Check if there are ANY memories with temporal metadata for this bank
        total_with_dates = await conn.fetchval(
            """SELECT COUNT(*) FROM memory_units
               WHERE bank_id = $1 AND fact_type = $2
               AND (occurred_start IS NOT NULL OR occurred_end IS NOT NULL OR mentioned_at IS NOT NULL)""",
            bank_id, fact_type
        )
        # Check how many have mentioned_at in the range
        in_range = await conn.fetchval(
            """SELECT COUNT(*) FROM memory_units
               WHERE bank_id = $1 AND fact_type = $2
               AND mentioned_at IS NOT NULL AND mentioned_at BETWEEN $3 AND $4""",
            bank_id, fact_type, start_date, end_date
        )
        # Check semantic similarity of those in range
        sample = await conn.fetch(
            """SELECT id, text, mentioned_at, 1 - (embedding <=> $1::vector) AS similarity
               FROM memory_units
               WHERE bank_id = $2 AND fact_type = $3
               AND mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5
               AND embedding IS NOT NULL
               ORDER BY mentioned_at DESC
               LIMIT 5""",
            query_emb_str, bank_id, fact_type, start_date, end_date
        )
        logger.info(f"[TEMPORAL] No entry points for {bank_id}/{fact_type} in {start_date} to {end_date}.")
        logger.info(f"[TEMPORAL] Total with dates: {total_with_dates}, In date range: {in_range}")
        for row in sample:
            logger.info(f"[TEMPORAL] Sample: {row['text'][:60]}... mentioned_at={row['mentioned_at']} sim={row['similarity']:.3f}")
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
    queue = [(RetrievalResult.from_db_row(dict(ep)), ep["similarity"], 1.0) for ep in entry_points]  # (unit, semantic_sim, temporal_score)
    budget_remaining = budget - len(entry_points)

    while queue and budget_remaining > 0:
        current, semantic_sim, temporal_score = queue.pop(0)
        current_id = current.id

        # Get neighbors via temporal and causal links
        if budget_remaining > 0:
            neighbors = await conn.fetch(
                """
                SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start, mu.occurred_end, mu.mentioned_at, mu.access_count, mu.embedding, mu.fact_type, mu.document_id, mu.chunk_id,
                       ml.weight, ml.link_type,
                       1 - (mu.embedding <=> $1::vector) AS similarity
                FROM memory_links ml
                JOIN memory_units mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = $2
                  AND ml.link_type IN ('temporal', 'causes', 'caused_by', 'enables', 'prevents')
                  AND ml.weight >= 0.1
                  AND mu.fact_type = $3
                  AND mu.embedding IS NOT NULL
                  AND (1 - (mu.embedding <=> $1::vector)) >= $4
                ORDER BY ml.weight DESC
                LIMIT 10
                """,
                query_emb_str, current.id, fact_type, semantic_threshold
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
                    neighbor_temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0) if total_days > 0 else 1.0
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
    question_date: Optional[datetime] = None,
    query_analyzer: Optional["QueryAnalyzer"] = None
) -> Tuple[List[RetrievalResult], List[RetrievalResult], List[RetrievalResult], Optional[List[RetrievalResult]], Dict[str, float], Optional[Tuple[datetime, datetime]]]:
    """
    Run 3-way or 4-way parallel retrieval (adds temporal if detected).

    Args:
        pool: Database connection pool
        query_text: Query text
        query_embedding_str: Query embedding as string
        agent_id: bank ID
        fact_type: Fact type to filter
        thinking_budget: Budget for graph traversal and retrieval limits
        question_date: Optional date when question was asked (for temporal filtering)
        query_analyzer: Query analyzer to use (defaults to TransformerQueryAnalyzer)

    Returns:
        Tuple of (semantic_results, bm25_results, graph_results, temporal_results, timings, temporal_constraint)
        Each results list contains RetrievalResult objects
        temporal_results is None if no temporal constraint detected
        timings is a dict with per-method latencies in seconds
        temporal_constraint is the (start_date, end_date) tuple if detected, else None
    """
    # Detect temporal constraint
    from .temporal_extraction import extract_temporal_constraint
    import logging
    import time
    logger = logging.getLogger(__name__)

    temporal_constraint = extract_temporal_constraint(
        query_text, reference_date=question_date, analyzer=query_analyzer
    )

    # Wrapper to track timing for each retrieval method
    async def timed_retrieval(name: str, coro):
        start = time.time()
        result = await coro
        duration = time.time() - start
        return result, name, duration

    async def run_semantic():
        async with acquire_with_retry(pool) as conn:
            return await retrieve_semantic(conn, query_embedding_str, bank_id, fact_type, limit=thinking_budget)

    async def run_bm25():
        async with acquire_with_retry(pool) as conn:
            return await retrieve_bm25(conn, query_text, bank_id, fact_type, limit=thinking_budget)

    async def run_graph():
        async with acquire_with_retry(pool) as conn:
            return await retrieve_graph(conn, query_embedding_str, bank_id, fact_type, budget=thinking_budget)

    async def run_temporal(start_date, end_date):
        async with acquire_with_retry(pool) as conn:
            return await retrieve_temporal(
                conn, query_embedding_str, bank_id, fact_type,
                start_date, end_date, budget=thinking_budget, semantic_threshold=0.1
            )

    # Run retrievals in parallel with timing
    timings = {}
    if temporal_constraint:
        start_date, end_date = temporal_constraint
        results = await asyncio.gather(
            timed_retrieval("semantic", run_semantic()),
            timed_retrieval("bm25", run_bm25()),
            timed_retrieval("graph", run_graph()),
            timed_retrieval("temporal", run_temporal(start_date, end_date))
        )
        semantic_results, _, timings["semantic"] = results[0]
        bm25_results, _, timings["bm25"] = results[1]
        graph_results, _, timings["graph"] = results[2]
        temporal_results, _, timings["temporal"] = results[3]
    else:
        results = await asyncio.gather(
            timed_retrieval("semantic", run_semantic()),
            timed_retrieval("bm25", run_bm25()),
            timed_retrieval("graph", run_graph())
        )
        semantic_results, _, timings["semantic"] = results[0]
        bm25_results, _, timings["bm25"] = results[1]
        graph_results, _, timings["graph"] = results[2]
        temporal_results = None

    return semantic_results, bm25_results, graph_results, temporal_results, timings, temporal_constraint
