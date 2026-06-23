"""
Link Expansion graph retrieval.

Expands from semantic/temporal seeds through three parallel, first-class signals
stored in memory_links:

1. Entity links  — query-time self-join through unit_entities. Score = number of distinct
                   shared entities between the seed set and each candidate, computed via
                   COUNT(DISTINCT entity_id). Uses a LATERAL per-entity cap
                   (graph_per_entity_limit, default 200) to prevent high-fanout entities
                   from exploding the self-join intermediate rows.
2. Semantic links — precomputed kNN graph (each new fact linked to its top-5 most
                    similar existing facts at insert time, similarity >= 0.7). Checked
                    in both directions since the graph is not symmetric. Score = weight.
3. Causal links  — explicit causal chains (causes/caused_by/enables/prevents).
                   Score = weight + 1.0 (boosted as highest-quality signal).

Entity expansion is bounded by graph_per_entity_limit (LATERAL cap per entity).
A timeout fallback (graph_expansion_timeout) drops entity expansion entirely if the
query still exceeds the budget.

For non-observation fact types the three expansions are issued as a single CTE query
(one roundtrip, one connection) with a `source` discriminator column so the Python
merge step can apply per-signal score transformations.
"""

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Any

from ...config import get_config
from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from .graph_retrieval import GraphRetriever
from .tags import TagGroup, TagsMatch, filter_results_by_tag_groups, filter_results_by_tags
from .types import GraphRetrievalTimings, RetrievalResult

logger = logging.getLogger(__name__)


async def _find_semantic_seeds(
    conn,
    query_embedding_str: str,
    bank_id: str,
    fact_type: str,
    limit: int = 20,
    threshold: float = 0.3,
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
    tag_groups: list[TagGroup] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
) -> list[RetrievalResult]:
    """Find semantic seeds via embedding search."""
    from .tags import build_tag_groups_where_clause, build_tags_where_clause_simple

    tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)
    tag_groups_param_start = 6 + (1 if tags else 0)
    groups_clause, groups_params, _ = build_tag_groups_where_clause(tag_groups, tag_groups_param_start)

    _next_idx = tag_groups_param_start + len(groups_params)
    created_range_clause = ""
    created_range_params: list[Any] = []
    if created_after is not None:
        created_range_params.append(created_after)
        created_range_clause += f" AND updated_at > ${_next_idx}"
        _next_idx += 1
    if created_before is not None:
        created_range_params.append(created_before)
        created_range_clause += f" AND updated_at < ${_next_idx}"
        _next_idx += 1

    params = [query_embedding_str, bank_id, fact_type, threshold, limit]
    if tags:
        params.append(tags)
    params.extend(groups_params)
    params.extend(created_range_params)

    rows = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end,
               mentioned_at, fact_type, document_id, chunk_id, tags, proof_count,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= $4
          {tags_clause}
          {groups_clause}
          {created_range_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $5
        """,
        *params,
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in rows]


class LinkExpansionRetriever(GraphRetriever):
    """
    Graph retrieval via direct link expansion from seeds.

    Runs three expansions through precomputed memory_links: entity co-occurrence,
    semantic kNN, and causal chains, all bounded at retain time.

    For non-observation fact types the three expansions are issued as a single CTE
    query (one roundtrip, one connection slot) with a `source` discriminator column.
    The Python merge step applies per-signal score transformations.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "link_expansion"

    async def retrieve(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        budget: int,
        query_text: str | None = None,
        semantic_seeds: list[RetrievalResult] | None = None,
        temporal_seeds: list[RetrievalResult] | None = None,
        adjacency=None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        created_after: "datetime | None" = None,
        created_before: "datetime | None" = None,
    ) -> tuple[list[RetrievalResult], GraphRetrievalTimings | None]:
        """
        Retrieve facts by expanding links from seeds.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding as string
            bank_id: Memory bank ID
            fact_type: Fact type to filter
            budget: Maximum results to return
            query_text: Original query text (unused)
            semantic_seeds: Pre-computed semantic entry points
            temporal_seeds: Pre-computed temporal entry points
            adjacency: Unused, kept for interface compatibility
            tags: Optional list of tags for visibility filtering

        Returns:
            Tuple of (results, timings)
        """
        start_time = time.time()
        timings = GraphRetrievalTimings(fact_type=fact_type)

        async with acquire_with_retry(pool) as conn:
            # Find seeds if not provided
            if semantic_seeds:
                all_seeds = list(semantic_seeds)
            else:
                seeds_start = time.time()
                all_seeds = await _find_semantic_seeds(
                    conn,
                    query_embedding_str,
                    bank_id,
                    fact_type,
                    limit=20,
                    threshold=0.3,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    created_after=created_after,
                    created_before=created_before,
                )
                timings.seeds_time = time.time() - seeds_start
                logger.debug(
                    f"[LinkExpansion] Found {len(all_seeds)} semantic seeds for fact_type={fact_type} "
                    f"(tags={tags}, tags_match={tags_match})"
                )

            if temporal_seeds:
                all_seeds.extend(temporal_seeds)

            if not all_seeds:
                return [], timings

            seed_ids = list({s.id for s in all_seeds})
            timings.pattern_count = len(seed_ids)

            query_start = time.time()

            ops = pool.ops
            if fact_type == "observation":
                entity_rows, semantic_rows, causal_rows = await self._expand_observations(
                    conn, seed_ids, budget, ops=ops
                )
            else:
                entity_rows, semantic_rows, causal_rows = await self._expand_combined(
                    conn, seed_ids, fact_type, budget, ops=ops
                )

            timings.edge_load_time = time.time() - query_start
            timings.db_queries = 1
            timings.edge_count = len(entity_rows) + len(semantic_rows) + len(causal_rows)

        # Merge results with additive intra-score: entity + semantic + causal ∈ [0, 3].
        #
        # Entity score: tanh(count × 0.5) maps shared-entity count to [0, 1]:
        #   1 entity → 0.46,  2 → 0.76,  3 → 0.91,  4 → 0.96  (saturates naturally)
        # Semantic score: similarity weight, already ∈ [0.7, 1.0].
        # Causal score:   link weight, already ∈ [0, 1].
        #
        # Facts appearing in multiple signals accumulate higher scores, rewarding
        # convergent evidence. The outer RRF uses rank position from this sorted list.
        entity_scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        causal_scores: dict[str, float] = {}
        row_map: dict[str, dict] = {}

        for row in entity_rows:
            fact_id = str(row["id"])
            entity_scores[fact_id] = math.tanh(row["score"] * 0.5)
            row_map[fact_id] = dict(row)

        for row in semantic_rows:
            fact_id = str(row["id"])
            semantic_scores[fact_id] = max(semantic_scores.get(fact_id, 0.0), row["score"])
            row_map.setdefault(fact_id, dict(row))

        for row in causal_rows:
            fact_id = str(row["id"])
            causal_scores[fact_id] = max(causal_scores.get(fact_id, 0.0), row["score"])
            row_map.setdefault(fact_id, dict(row))

        all_ids = set(entity_scores) | set(semantic_scores) | set(causal_scores)
        score_map = {
            fid: entity_scores.get(fid, 0.0) + semantic_scores.get(fid, 0.0) + causal_scores.get(fid, 0.0)
            for fid in all_ids
        }

        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:budget]
        rows = [row_map[fact_id] for fact_id in sorted_ids]

        results = []
        for row in rows:
            result = RetrievalResult.from_db_row(dict(row))
            result.activation = row["score"]
            results.append(result)

        # filter_results_by_tags is a no-op when no filter applies (tags falsy and not
        # the exact-empty/global scope), so call it unconditionally — gating on `if tags:`
        # would skip the untagged-only filter for tags=[] + tags_match="exact".
        results = filter_results_by_tags(results, tags, match=tags_match)

        if tag_groups:
            results = filter_results_by_tag_groups(results, tag_groups)

        timings.result_count = len(results)
        timings.traverse = time.time() - start_time

        logger.debug(
            f"LinkExpansion: {len(results)} results from {len(seed_ids)} seeds "
            f"in {timings.traverse * 1000:.1f}ms (query: {timings.edge_load_time * 1000:.1f}ms)"
        )

        return results, timings

    async def _expand_combined(
        self,
        conn,
        seed_ids: list,
        fact_type: str,
        budget: int,
        *,
        ops,
    ) -> tuple[list, list, list]:
        """
        Single-roundtrip CTE query combining entity, semantic, and causal expansions.

        Uses a `source` discriminator column so the caller can apply per-signal
        score transformations.  The three CTEs share one connection slot — important
        for asyncpg which does not allow concurrent queries on the same connection.

        Index coverage:
          entity:   idx_unit_entities_entity_unit (entity_id, unit_id) — entity
                    expansion traverses unit_entities, not memory_links.
          semantic: idx_memory_links_from_type_weight / _to_type_weight
                    (from_unit_id|to_unit_id, link_type, weight DESC) serve both
                    outgoing and incoming sides as single composite index scans.
        """
        config = get_config()
        ml = fq_table("memory_links")
        mu = fq_table("memory_units")
        ue = fq_table("unit_entities")

        per_entity_limit = config.link_expansion_per_entity_limit

        entity_cte = ops.build_entity_expansion_cte(mu, ue, per_entity_limit)
        semantic_causal_cte = ops.build_semantic_causal_cte(ml, mu)

        full_query = f"""
            WITH {entity_cte},
            {semantic_causal_cte}
            SELECT * FROM entity_expanded
            UNION ALL
            SELECT * FROM semantic_expanded
            UNION ALL
            SELECT * FROM causal_expanded
            """

        params = [seed_ids, fact_type, budget]

        try:
            all_rows = await asyncio.wait_for(
                conn.fetch(full_query, *params),
                timeout=config.link_expansion_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[LinkExpansion] Entity expansion timed out after {config.link_expansion_timeout}s "
                f"for fact_type={fact_type}, falling back to semantic+causal only"
            )
            fallback_query = f"""
                WITH {semantic_causal_cte}
                SELECT * FROM semantic_expanded
                UNION ALL
                SELECT * FROM causal_expanded
                LIMIT $3
                """
            all_rows = await conn.fetch(fallback_query, *params)

        entity_rows = [r for r in all_rows if r["source"] == "entity"]
        semantic_rows = [r for r in all_rows if r["source"] == "semantic"]
        causal_rows = [r for r in all_rows if r["source"] == "causal"]
        return entity_rows, semantic_rows, causal_rows

    async def _expand_observations(
        self,
        conn,
        seed_ids: list,
        budget: int,
        *,
        ops,
    ) -> tuple[list, list, list]:
        """
        Observation-specific expansion.

        Observations don't have direct entity links in memory_links (they're created
        by consolidation, not retain).  Instead, traverse source_memory_ids → world
        facts → entities → other world facts → their observations.

        Semantic and causal expansions run as a second combined CTE query.
        """
        source_ids_found: list = []
        if logger.isEnabledFor(logging.DEBUG):
            debug_rows = await conn.fetch(
                f"""
                SELECT id, source_memory_ids
                FROM {fq_table("memory_units")}
                WHERE id = ANY($1::uuid[])
                """,
                seed_ids,
            )
            for row in debug_rows:
                if row["source_memory_ids"]:
                    source_ids_found.extend(row["source_memory_ids"])
            logger.debug(
                f"[LinkExpansion] observation graph: {len(seed_ids)} seeds, "
                f"{len(source_ids_found)} source_memory_ids found"
            )

        config = get_config()
        ue = fq_table("unit_entities")
        ml = fq_table("memory_links")
        mu = fq_table("memory_units")
        per_entity_limit = config.link_expansion_per_entity_limit

        # Delegate to DataAccessOps. Both backends now use the observation_sources
        # junction table with standard SQL joins (previously PG used native array
        # ops and Oracle used JSON_TABLE).
        return await ops.expand_observations(
            conn,
            mu,
            ue,
            ml,
            seed_ids,
            budget,
            per_entity_limit,
        )
