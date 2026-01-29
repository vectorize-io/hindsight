"""
Link Expansion graph retrieval.

A simple, fast graph retrieval that expands from seeds via:
1. Entity links: Find facts sharing entities with seeds (filtered by entity frequency)
2. Causal links: Find facts causally linked to seeds (top-k by weight)

Characteristics:
- 2-3 DB queries (seed finding + parallel entity/causal expansion)
- Sublinear: only touches connected facts via indexes
- No iteration, no propagation, no normalization
- Target: <100ms
"""

import logging
import time

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from .graph_retrieval import GraphRetriever
from .tags import TagsMatch, filter_results_by_tags
from .types import MPFPTimings, RetrievalResult

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
) -> list[RetrievalResult]:
    """Find semantic seeds via embedding search."""
    from .tags import build_tags_where_clause_simple

    tags_clause = build_tags_where_clause_simple(tags, 6, match=tags_match)
    params = [query_embedding_str, bank_id, fact_type, threshold, limit]
    if tags:
        params.append(tags)

    rows = await conn.fetch(
        f"""
        SELECT id, text, context, event_date, occurred_start, occurred_end,
               mentioned_at, embedding, fact_type, document_id, chunk_id, tags,
               1 - (embedding <=> $1::vector) AS similarity
        FROM {fq_table("memory_units")}
        WHERE bank_id = $2
          AND embedding IS NOT NULL
          AND fact_type = $3
          AND (1 - (embedding <=> $1::vector)) >= $4
          {tags_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $5
        """,
        *params,
    )
    return [RetrievalResult.from_db_row(dict(r)) for r in rows]


class LinkExpansionRetriever(GraphRetriever):
    """
    Graph retrieval via direct link expansion from seeds.

    Expands through entity co-occurrence and causal links in a single query.
    Fast and simple alternative to MPFP.
    """

    def __init__(
        self,
        max_entity_frequency: int = 500,
        causal_weight_threshold: float = 0.3,
        causal_limit_per_seed: int = 10,
    ):
        """
        Initialize link expansion retriever.

        Args:
            max_entity_frequency: Skip entities appearing in more than this many facts
            causal_weight_threshold: Minimum weight for causal links
            causal_limit_per_seed: Max causal links to follow per seed
        """
        self.max_entity_frequency = max_entity_frequency
        self.causal_weight_threshold = causal_weight_threshold
        self.causal_limit_per_seed = causal_limit_per_seed

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
    ) -> tuple[list[RetrievalResult], MPFPTimings | None]:
        """
        Retrieve facts by expanding links from seeds.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding (unused, kept for interface)
            bank_id: Memory bank ID
            fact_type: Fact type to filter
            budget: Maximum results to return
            query_text: Original query text (unused)
            semantic_seeds: Pre-computed semantic entry points
            temporal_seeds: Pre-computed temporal entry points
            adjacency: Unused, kept for interface compatibility
            tags: Optional list of tags for visibility filtering (OR matching)

        Returns:
            Tuple of (results, timings)
        """
        start_time = time.time()
        timings = MPFPTimings(fact_type=fact_type)

        # Use single connection for all queries to reduce pool pressure
        # (queries are fast ~50ms each, connection acquisition is the bottleneck)
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
                )
                timings.seeds_time = time.time() - seeds_start
                logger.debug(
                    f"[LinkExpansion] Found {len(all_seeds)} semantic seeds for fact_type={fact_type} "
                    f"(tags={tags}, tags_match={tags_match})"
                )

            # Add temporal seeds if provided
            if temporal_seeds:
                all_seeds.extend(temporal_seeds)

            if not all_seeds:
                return [], timings

            seed_ids = list({s.id for s in all_seeds})
            timings.pattern_count = len(seed_ids)

            # Run entity and causal expansion sequentially on same connection
            query_start = time.time()

            # For observations, traverse through source_memory_ids to find entity connections.
            # Observations don't have direct unit_entities - they inherit entities via their
            # source world/experience facts.
            #
            # Path: observation → source_memory_ids → world fact → entities →
            #       ALL world facts with those entities → their observations (excluding seeds)
            if fact_type == "observation":
                # Debug: Check what source_memory_ids exist on seed observations
                debug_sources = await conn.fetch(
                    f"""
                    SELECT id, source_memory_ids
                    FROM {fq_table("memory_units")}
                    WHERE id = ANY($1::uuid[])
                    """,
                    seed_ids,
                )
                source_ids_found = []
                for row in debug_sources:
                    if row["source_memory_ids"]:
                        source_ids_found.extend(row["source_memory_ids"])
                logger.debug(
                    f"[LinkExpansion] observation graph: {len(seed_ids)} seeds, "
                    f"{len(source_ids_found)} source_memory_ids found"
                )

                entity_rows = await conn.fetch(
                    f"""
                    WITH seed_sources AS (
                        -- Get source memory IDs from seed observations
                        SELECT DISTINCT unnest(source_memory_ids) AS source_id
                        FROM {fq_table("memory_units")}
                        WHERE id = ANY($1::uuid[])
                          AND source_memory_ids IS NOT NULL
                    ),
                    source_entities AS (
                        -- Get entities from those source memories (filtered by frequency)
                        SELECT DISTINCT ue.entity_id
                        FROM seed_sources ss
                        JOIN {fq_table("unit_entities")} ue ON ss.source_id = ue.unit_id
                        JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                        WHERE e.mention_count < $2
                    ),
                    all_connected_sources AS (
                        -- Find ALL world facts sharing those entities (don't exclude seed sources)
                        -- The exclusion happens at the observation level, not the source level
                        SELECT DISTINCT other_ue.unit_id AS source_id
                        FROM source_entities se
                        JOIN {fq_table("unit_entities")} other_ue ON se.entity_id = other_ue.entity_id
                    )
                    -- Find observations derived from connected source memories
                    -- Only exclude the actual seed observations
                    SELECT
                        mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                        mu.occurred_end, mu.mentioned_at, mu.embedding,
                        mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                        COUNT(DISTINCT cs.source_id)::float AS score
                    FROM all_connected_sources cs
                    JOIN {fq_table("memory_units")} mu
                        ON mu.source_memory_ids @> ARRAY[cs.source_id]
                    WHERE mu.fact_type = 'observation'
                      AND mu.id != ALL($1::uuid[])
                    GROUP BY mu.id
                    ORDER BY score DESC
                    LIMIT $3
                    """,
                    seed_ids,
                    self.max_entity_frequency,
                    budget,
                )
                logger.debug(f"[LinkExpansion] observation graph: found {len(entity_rows)} connected observations")
            else:
                # For world/experience facts, use direct entity lookup
                entity_rows = await conn.fetch(
                    f"""
                    SELECT
                        mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                        mu.occurred_end, mu.mentioned_at, mu.embedding,
                        mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                        COUNT(*)::float AS score
                    FROM {fq_table("unit_entities")} seed_ue
                    JOIN {fq_table("entities")} e ON seed_ue.entity_id = e.id
                    JOIN {fq_table("unit_entities")} other_ue ON seed_ue.entity_id = other_ue.entity_id
                    JOIN {fq_table("memory_units")} mu ON other_ue.unit_id = mu.id
                    WHERE seed_ue.unit_id = ANY($1::uuid[])
                      AND e.mention_count < $2
                      AND mu.id != ALL($1::uuid[])
                      AND mu.fact_type = $3
                    GROUP BY mu.id
                    ORDER BY score DESC
                    LIMIT $4
                    """,
                    seed_ids,
                    self.max_entity_frequency,
                    fact_type,
                    budget,
                )

            causal_rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (mu.id)
                    mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                    mu.occurred_end, mu.mentioned_at, mu.embedding,
                    mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                    ml.weight + 1.0 AS score
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
                WHERE ml.from_unit_id = ANY($1::uuid[])
                  AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
                  AND ml.weight >= $2
                  AND mu.fact_type = $3
                ORDER BY mu.id, ml.weight DESC
                LIMIT $4
                """,
                seed_ids,
                self.causal_weight_threshold,
                fact_type,
                budget,
            )

            # Fallback: semantic/temporal/entity links from memory_links table
            # These are secondary to entity links (via unit_entities) and causal links
            # Weight is halved (0.5x) to prioritize primary link types
            # Check both directions: seeds -> others AND others -> seeds
            fallback_rows = await conn.fetch(
                f"""
                WITH outgoing AS (
                    -- Links FROM seeds TO other facts
                    SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                           mu.occurred_end, mu.mentioned_at, mu.embedding,
                           mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                           ml.weight
                    FROM {fq_table("memory_links")} ml
                    JOIN {fq_table("memory_units")} mu ON ml.to_unit_id = mu.id
                    WHERE ml.from_unit_id = ANY($1::uuid[])
                      AND ml.link_type IN ('semantic', 'temporal', 'entity')
                      AND ml.weight >= $2
                      AND mu.fact_type = $3
                      AND mu.id != ALL($1::uuid[])
                ),
                incoming AS (
                    -- Links FROM other facts TO seeds (reverse direction)
                    SELECT mu.id, mu.text, mu.context, mu.event_date, mu.occurred_start,
                           mu.occurred_end, mu.mentioned_at, mu.embedding,
                           mu.fact_type, mu.document_id, mu.chunk_id, mu.tags,
                           ml.weight
                    FROM {fq_table("memory_links")} ml
                    JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
                    WHERE ml.to_unit_id = ANY($1::uuid[])
                      AND ml.link_type IN ('semantic', 'temporal', 'entity')
                      AND ml.weight >= $2
                      AND mu.fact_type = $3
                      AND mu.id != ALL($1::uuid[])
                ),
                combined AS (
                    SELECT * FROM outgoing
                    UNION ALL
                    SELECT * FROM incoming
                )
                SELECT DISTINCT ON (id)
                    id, text, context, event_date, occurred_start,
                    occurred_end, mentioned_at, embedding,
                    fact_type, document_id, chunk_id, tags,
                    (MAX(weight) * 0.5) AS score
                FROM combined
                GROUP BY id, text, context, event_date, occurred_start,
                         occurred_end, mentioned_at, embedding,
                         fact_type, document_id, chunk_id, tags
                ORDER BY id, score DESC
                LIMIT $4
                """,
                seed_ids,
                self.causal_weight_threshold,
                fact_type,
                budget,
            )

            timings.edge_load_time = time.time() - query_start
            timings.db_queries = 3
            timings.edge_count = len(entity_rows) + len(causal_rows) + len(fallback_rows)

        # Merge results, taking max score per fact
        # Priority: entity links (unit_entities) > causal links > fallback links
        score_map: dict[str, float] = {}
        row_map: dict[str, dict] = {}

        for row in entity_rows:
            fact_id = str(row["id"])
            score_map[fact_id] = max(score_map.get(fact_id, 0), row["score"])
            row_map[fact_id] = dict(row)

        for row in causal_rows:
            fact_id = str(row["id"])
            score_map[fact_id] = max(score_map.get(fact_id, 0), row["score"])
            if fact_id not in row_map:
                row_map[fact_id] = dict(row)

        for row in fallback_rows:
            fact_id = str(row["id"])
            score_map[fact_id] = max(score_map.get(fact_id, 0), row["score"])
            if fact_id not in row_map:
                row_map[fact_id] = dict(row)

        # Sort by score and limit
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:budget]
        rows = [row_map[fact_id] for fact_id in sorted_ids]

        # Convert to results
        results = []
        for row in rows:
            result = RetrievalResult.from_db_row(dict(row))
            result.activation = row["score"]
            results.append(result)

        # Apply tags filtering (graph expansion may reach untagged memories)
        if tags:
            results = filter_results_by_tags(results, tags, match=tags_match)

        timings.result_count = len(results)
        timings.traverse = time.time() - start_time

        logger.debug(
            f"LinkExpansion: {len(results)} results from {len(seed_ids)} seeds "
            f"in {timings.traverse * 1000:.1f}ms (query: {timings.edge_load_time * 1000:.1f}ms)"
        )

        return results, timings
