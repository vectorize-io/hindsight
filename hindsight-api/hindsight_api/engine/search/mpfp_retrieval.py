"""
Meta-Path Forward Push (MPFP) graph retrieval.

A sublinear graph traversal algorithm for memory retrieval over heterogeneous
graphs with multiple edge types (semantic, temporal, causal, entity).

Combines meta-path patterns from HIN literature with Forward Push local
propagation from Approximate PPR.

Key properties:
- Sublinear in graph size (threshold pruning bounds active nodes)
- Lazy edge loading: only loads edges for frontier nodes, not entire graph
- Predefined patterns capture different retrieval intents
- All patterns run in parallel, results fused via RRF
- No LLM in the loop during traversal
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table
from .graph_retrieval import GraphRetriever
from .types import MPFPTimings, RetrievalResult

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class EdgeTarget:
    """A neighbor node with its edge weight."""

    node_id: str
    weight: float


@dataclass
class EdgeCache:
    """
    Cache for lazily-loaded edges.

    Grows per-hop as edges are loaded for frontier nodes.
    Shared across patterns to avoid redundant loads.
    Loads ALL edge types at once to minimize DB queries.
    Thread-safe via asyncio lock to prevent redundant concurrent loads.
    """

    # edge_type -> from_node_id -> list of EdgeTarget
    graphs: dict[str, dict[str, list[EdgeTarget]]] = field(default_factory=dict)
    # Track which nodes have been fully loaded (all edge types)
    _fully_loaded: set[str] = field(default_factory=set)
    # Timing stats
    db_queries: int = 0
    edge_load_time: float = 0.0
    # Detailed hop timing for debugging
    hop_details: list[dict] = field(default_factory=list)
    # Lock to prevent redundant concurrent loads
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def get_neighbors(self, edge_type: str, node_id: str) -> list[EdgeTarget]:
        """Get neighbors for a node via a specific edge type."""
        return self.graphs.get(edge_type, {}).get(node_id, [])

    def get_normalized_neighbors(self, edge_type: str, node_id: str, top_k: int) -> list[EdgeTarget]:
        """Get top-k neighbors with weights normalized to sum to 1."""
        neighbors = self.get_neighbors(edge_type, node_id)[:top_k]
        if not neighbors:
            return []

        total = sum(n.weight for n in neighbors)
        if total == 0:
            return []

        return [EdgeTarget(node_id=n.node_id, weight=n.weight / total) for n in neighbors]

    def is_fully_loaded(self, node_id: str) -> bool:
        """Check if all edges for this node have been loaded."""
        return node_id in self._fully_loaded

    def get_uncached(self, node_ids: list[str]) -> list[str]:
        """Get node IDs that haven't been fully loaded yet."""
        return [n for n in node_ids if not self.is_fully_loaded(n)]

    def add_all_edges(self, edges_by_type: dict[str, dict[str, list[EdgeTarget]]], all_queried: list[str]):
        """
        Add loaded edges to the cache (all edge types at once).

        Args:
            edges_by_type: Dict mapping edge_type -> from_node_id -> list of EdgeTarget
            all_queried: All node IDs that were queried (marks them as fully loaded)
        """
        for edge_type, edges in edges_by_type.items():
            if edge_type not in self.graphs:
                self.graphs[edge_type] = {}
            for node_id, neighbors in edges.items():
                self.graphs[edge_type][node_id] = neighbors

        # Mark all queried nodes as fully loaded (even if they have no edges)
        self._fully_loaded.update(all_queried)


@dataclass
class PatternResult:
    """Result from a single pattern traversal."""

    pattern: list[str]
    scores: dict[str, float]  # node_id -> accumulated mass


@dataclass
class MPFPConfig:
    """Configuration for MPFP algorithm."""

    alpha: float = 0.15  # teleport/keep probability
    threshold: float = 1e-6  # mass pruning threshold (lower = explore more)
    top_k_neighbors: int = 20  # fan-out limit per node

    # Patterns from semantic seeds
    patterns_semantic: list[list[str]] = field(
        default_factory=lambda: [
            ["semantic", "semantic"],  # topic expansion
            ["entity", "temporal"],  # entity timeline
            ["semantic", "causes"],  # reasoning chains (forward)
            ["semantic", "caused_by"],  # reasoning chains (backward)
            ["entity", "semantic"],  # entity context
        ]
    )

    # Patterns from temporal seeds
    patterns_temporal: list[list[str]] = field(
        default_factory=lambda: [
            ["temporal", "semantic"],  # what was happening then
            ["temporal", "entity"],  # who was involved then
        ]
    )


@dataclass
class SeedNode:
    """An entry point node with its initial score."""

    node_id: str
    score: float  # initial mass (e.g., similarity score)


# -----------------------------------------------------------------------------
# Lazy Edge Loading
# -----------------------------------------------------------------------------


async def load_all_edges_for_frontier(
    pool,
    node_ids: list[str],
    top_k_per_type: int = 20,
) -> dict[str, dict[str, list[EdgeTarget]]]:
    """
    Load top-k edges per (node, edge_type) for frontier nodes.

    Uses a LATERAL join to efficiently fetch only the top-k edges per type,
    avoiding loading hundreds of entity edges when only 20 are needed.

    Requires composite index: (from_unit_id, link_type, weight DESC)

    Args:
        pool: Database connection pool
        node_ids: Frontier node IDs to load edges for
        top_k_per_type: Max edges to load per (node, link_type) pair

    Returns:
        Dict mapping edge_type -> from_node_id -> list of EdgeTarget
    """
    if not node_ids:
        return {}

    async with acquire_with_retry(pool) as conn:
        # Use LATERAL join to get top-k per (from_node, link_type)
        # This leverages the composite index for efficient early termination
        rows = await conn.fetch(
            f"""
            WITH frontier(node_id) AS (SELECT unnest($1::uuid[]))
            SELECT f.node_id as from_unit_id, lt.link_type, edges.to_unit_id, edges.weight
            FROM frontier f
            CROSS JOIN (VALUES ('semantic'), ('temporal'), ('entity'), ('causes'), ('caused_by')) AS lt(link_type)
            CROSS JOIN LATERAL (
                SELECT ml.to_unit_id, ml.weight
                FROM {fq_table("memory_links")} ml
                WHERE ml.from_unit_id = f.node_id
                  AND ml.link_type = lt.link_type
                  AND ml.weight >= 0.1
                ORDER BY ml.weight DESC
                LIMIT $2
            ) edges
            """,
            node_ids,
            top_k_per_type,
        )

    # Group by edge_type -> from_node -> neighbors
    result: dict[str, dict[str, list[EdgeTarget]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        edge_type = row["link_type"]
        from_id = str(row["from_unit_id"])
        to_id = str(row["to_unit_id"])
        weight = row["weight"]
        result[edge_type][from_id].append(EdgeTarget(node_id=to_id, weight=weight))

    # Convert nested defaultdicts to regular dicts
    return {edge_type: dict(edges) for edge_type, edges in result.items()}


# -----------------------------------------------------------------------------
# Core Algorithm (Async with Lazy Loading)
# -----------------------------------------------------------------------------


@dataclass
class PatternState:
    """State for a pattern traversal between hops."""

    pattern: list[str]
    hop_index: int
    scores: dict[str, float]
    frontier: dict[str, float]


def _init_pattern_state(seeds: list[SeedNode], pattern: list[str]) -> PatternState:
    """Initialize pattern state from seeds."""
    if not seeds:
        return PatternState(pattern=pattern, hop_index=0, scores={}, frontier={})

    total_seed_score = sum(s.score for s in seeds)
    if total_seed_score == 0:
        total_seed_score = len(seeds)

    frontier = {s.node_id: s.score / total_seed_score for s in seeds}
    return PatternState(pattern=pattern, hop_index=0, scores={}, frontier=frontier)


def _execute_hop(state: PatternState, cache: EdgeCache, config: MPFPConfig) -> set[str]:
    """
    Execute ONE hop of traversal, return frontier nodes for next hop.

    This is a pure function that uses cached edges (no DB access).
    Returns set of uncached nodes needed for next hop.
    """
    if state.hop_index >= len(state.pattern):
        return set()

    edge_type = state.pattern[state.hop_index]

    # Collect active nodes above threshold
    active_nodes = [node_id for node_id, mass in state.frontier.items() if mass >= config.threshold]
    if not active_nodes:
        state.frontier = {}
        return set()

    # Propagate mass using cached edges
    next_frontier: dict[str, float] = {}
    uncached_for_next: set[str] = set()

    for node_id, mass in state.frontier.items():
        if mass < config.threshold:
            continue

        # Keep α portion for this node
        state.scores[node_id] = state.scores.get(node_id, 0) + config.alpha * mass

        # Push (1-α) to neighbors
        push_mass = (1 - config.alpha) * mass
        neighbors = cache.get_normalized_neighbors(edge_type, node_id, config.top_k_neighbors)

        for neighbor in neighbors:
            next_frontier[neighbor.node_id] = next_frontier.get(neighbor.node_id, 0) + push_mass * neighbor.weight
            # Track if we'll need edges for this node in the next hop
            if not cache.is_fully_loaded(neighbor.node_id):
                uncached_for_next.add(neighbor.node_id)

    state.frontier = next_frontier
    state.hop_index += 1

    return uncached_for_next


def _finalize_pattern(state: PatternState, config: MPFPConfig) -> PatternResult:
    """Finalize pattern by adding remaining frontier mass to scores."""
    for node_id, mass in state.frontier.items():
        if mass >= config.threshold:
            state.scores[node_id] = state.scores.get(node_id, 0) + mass

    return PatternResult(pattern=state.pattern, scores=state.scores)


async def mpfp_traverse_hop_synchronized(
    pool,
    pattern_jobs: list[tuple[list[SeedNode], list[str]]],
    config: MPFPConfig,
    cache: EdgeCache,
) -> list[PatternResult]:
    """
    Execute ALL patterns with hop-synchronized edge loading.

    Instead of running each pattern independently (causing multiple DB queries),
    this function:
    1. Runs hop 1 for ALL patterns (using pre-warmed seed edges)
    2. Collects ALL unique hop-2 frontier nodes across patterns
    3. Pre-warms hop-2 edges in ONE query
    4. Runs hop 2 for ALL patterns

    This reduces DB queries from O(patterns * hops) to O(hops).

    Args:
        pool: Database connection pool
        pattern_jobs: List of (seeds, pattern) tuples
        config: Algorithm parameters
        cache: Shared edge cache (should be pre-warmed with seed edges)

    Returns:
        List of PatternResult for each pattern
    """
    import time

    # Initialize all pattern states
    states = [_init_pattern_state(seeds, pattern) for seeds, pattern in pattern_jobs]

    # Determine max hops (all patterns should be same length, but be safe)
    max_hops = max((len(p) for _, p in pattern_jobs), default=0)

    # Detailed timing for debugging
    hop_times: list[dict] = []

    # Execute hop-by-hop across ALL patterns
    for hop in range(max_hops):
        hop_start = time.time()
        hop_timing = {"hop": hop, "patterns_executed": 0, "uncached_count": 0, "load_time": 0.0}

        # Execute this hop for all patterns, collect uncached nodes for next hop
        all_uncached: set[str] = set()
        exec_start = time.time()
        for state in states:
            if state.hop_index < len(state.pattern):
                uncached = _execute_hop(state, cache, config)
                all_uncached.update(uncached)
                hop_timing["patterns_executed"] += 1
        hop_timing["exec_time"] = time.time() - exec_start

        # Pre-warm edges for ALL uncached nodes before next hop
        hop_timing["uncached_count"] = len(all_uncached)
        if all_uncached:
            uncached_list = list(all_uncached - cache._fully_loaded)
            hop_timing["uncached_after_filter"] = len(uncached_list)
            if uncached_list:
                load_start = time.time()
                edges_by_type = await load_all_edges_for_frontier(pool, uncached_list, config.top_k_neighbors)
                hop_timing["load_time"] = time.time() - load_start
                cache.edge_load_time += hop_timing["load_time"]
                cache.db_queries += 1
                cache.add_all_edges(edges_by_type, uncached_list)
                hop_timing["edges_loaded"] = sum(
                    len(neighbors) for edges in edges_by_type.values() for neighbors in edges.values()
                )

        hop_timing["total_time"] = time.time() - hop_start
        hop_times.append(hop_timing)

    # Store hop timing details in cache for logging
    cache.hop_details = hop_times

    # Finalize all patterns
    return [_finalize_pattern(state, config) for state in states]


async def mpfp_traverse_async(
    pool,
    seeds: list[SeedNode],
    pattern: list[str],
    config: MPFPConfig,
    cache: EdgeCache,
) -> PatternResult:
    """
    Async Forward Push traversal with lazy edge loading.

    NOTE: For better performance with multiple patterns, use mpfp_traverse_hop_synchronized().
    This function is kept for single-pattern use cases.
    """
    if not seeds:
        return PatternResult(pattern=pattern, scores={})

    results = await mpfp_traverse_hop_synchronized(pool, [(seeds, pattern)], config, cache)
    return results[0] if results else PatternResult(pattern=pattern, scores={})


def rrf_fusion(
    results: list[PatternResult],
    k: int = 60,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion to combine pattern results.

    Args:
        results: List of pattern results
        k: RRF constant (higher = more uniform weighting)
        top_k: Number of results to return

    Returns:
        List of (node_id, fused_score) tuples, sorted by score descending
    """
    fused: dict[str, float] = {}

    for result in results:
        if not result.scores:
            continue

        # Rank nodes by their score in this pattern
        ranked = sorted(result.scores.keys(), key=lambda n: result.scores[n], reverse=True)

        for rank, node_id in enumerate(ranked):
            fused[node_id] = fused.get(node_id, 0) + 1.0 / (k + rank + 1)

    # Sort by fused score and return top-k
    sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_k]


# -----------------------------------------------------------------------------
# Database Loading
# -----------------------------------------------------------------------------


async def fetch_memory_units_by_ids(
    pool,
    node_ids: list[str],
    fact_type: str,
) -> list[RetrievalResult]:
    """Fetch full memory unit details for a list of node IDs."""
    if not node_ids:
        return []

    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, text, context, event_date, occurred_start, occurred_end,
                   mentioned_at, access_count, embedding, fact_type, document_id, chunk_id
            FROM {fq_table("memory_units")}
            WHERE id = ANY($1::uuid[])
              AND fact_type = $2
            """,
            node_ids,
            fact_type,
        )

    return [RetrievalResult.from_db_row(dict(r)) for r in rows]


# -----------------------------------------------------------------------------
# Graph Retriever Implementation
# -----------------------------------------------------------------------------


class MPFPGraphRetriever(GraphRetriever):
    """
    Graph retrieval using Meta-Path Forward Push with lazy edge loading.

    Runs predefined patterns in parallel from semantic and temporal seeds,
    loading edges on-demand per hop instead of loading entire graph upfront.
    """

    def __init__(self, config: MPFPConfig | None = None):
        """
        Initialize MPFP retriever.

        Args:
            config: Algorithm configuration (uses defaults if None)
        """
        if config is None:
            # Read top_k_neighbors from global config
            from ...config import get_config

            global_config = get_config()
            config = MPFPConfig(top_k_neighbors=global_config.mpfp_top_k_neighbors)
        self.config = config

    @property
    def name(self) -> str:
        return "mpfp"

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
        adjacency=None,  # Ignored - kept for interface compatibility
    ) -> tuple[list[RetrievalResult], MPFPTimings | None]:
        """
        Retrieve facts using MPFP algorithm with lazy edge loading.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding (used for fallback seed finding)
            bank_id: Memory bank ID
            fact_type: Fact type to filter
            budget: Maximum results to return
            query_text: Original query text (optional)
            semantic_seeds: Pre-computed semantic entry points
            temporal_seeds: Pre-computed temporal entry points
            adjacency: Ignored (kept for interface compatibility)

        Returns:
            Tuple of (List of RetrievalResult with activation scores, MPFPTimings)
        """
        import time

        timings = MPFPTimings(fact_type=fact_type)

        # Convert seeds to SeedNode format
        semantic_seed_nodes = self._convert_seeds(semantic_seeds, "similarity")
        temporal_seed_nodes = self._convert_seeds(temporal_seeds, "temporal_score")

        # If no semantic seeds provided, fall back to finding our own
        if not semantic_seed_nodes:
            seeds_start = time.time()
            semantic_seed_nodes = await self._find_semantic_seeds(pool, query_embedding_str, bank_id, fact_type)
            timings.seeds_time = time.time() - seeds_start

        # Collect all pattern jobs
        pattern_jobs = []

        # Patterns from semantic seeds
        for pattern in self.config.patterns_semantic:
            if semantic_seed_nodes:
                pattern_jobs.append((semantic_seed_nodes, pattern))

        # Patterns from temporal seeds
        for pattern in self.config.patterns_temporal:
            if temporal_seed_nodes:
                pattern_jobs.append((temporal_seed_nodes, pattern))

        if not pattern_jobs:
            return [], timings

        timings.pattern_count = len(pattern_jobs)

        # Shared edge cache across all patterns
        cache = EdgeCache()

        # Pre-warm cache with ALL seed node edges BEFORE running patterns
        # This prevents redundant DB queries at hop 1
        all_seed_ids = list({s.node_id for seeds, _ in pattern_jobs for s in seeds})
        if all_seed_ids:
            import time as time_module

            prewarm_start = time_module.time()
            edges_by_type = await load_all_edges_for_frontier(pool, all_seed_ids, self.config.top_k_neighbors)
            cache.edge_load_time += time_module.time() - prewarm_start
            cache.db_queries += 1
            cache.add_all_edges(edges_by_type, all_seed_ids)

        # Run all patterns with HOP-SYNCHRONIZED edge loading
        # This batches hop-2 edge loads across ALL patterns into ONE query
        # Reduces DB queries from O(patterns * hops) to O(hops)
        step_start = time.time()
        pattern_results = await mpfp_traverse_hop_synchronized(pool, pattern_jobs, self.config, cache)
        timings.traverse = time.time() - step_start

        # Record edge loading stats from cache
        timings.edge_count = sum(len(neighbors) for g in cache.graphs.values() for neighbors in g.values())
        timings.db_queries = cache.db_queries
        timings.edge_load_time = cache.edge_load_time
        timings.hop_details = cache.hop_details

        # Fuse results
        step_start = time.time()
        fused = rrf_fusion(pattern_results, top_k=budget)
        timings.fusion = time.time() - step_start

        if not fused:
            return [], timings

        # Get top result IDs
        result_ids = [node_id for node_id, score in fused][:budget]

        # Fetch full details
        step_start = time.time()
        results = await fetch_memory_units_by_ids(pool, result_ids, fact_type)
        timings.fetch = time.time() - step_start
        timings.result_count = len(results)

        # Add activation scores from fusion
        score_map = {node_id: score for node_id, score in fused}
        for result in results:
            result.activation = score_map.get(result.id, 0.0)

        # Sort by activation
        results.sort(key=lambda r: r.activation or 0, reverse=True)

        return results, timings

    def _convert_seeds(
        self,
        seeds: list[RetrievalResult] | None,
        score_attr: str,
    ) -> list[SeedNode]:
        """Convert RetrievalResult seeds to SeedNode format."""
        if not seeds:
            return []

        result = []
        for seed in seeds:
            score = getattr(seed, score_attr, None)
            if score is None:
                score = seed.activation or seed.similarity or 1.0
            result.append(SeedNode(node_id=seed.id, score=score))

        return result

    async def _find_semantic_seeds(
        self,
        pool,
        query_embedding_str: str,
        bank_id: str,
        fact_type: str,
        limit: int = 20,
        threshold: float = 0.3,
    ) -> list[SeedNode]:
        """Fallback: find semantic seeds via embedding search."""
        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, 1 - (embedding <=> $1::vector) AS similarity
                FROM {fq_table("memory_units")}
                WHERE bank_id = $2
                  AND embedding IS NOT NULL
                  AND fact_type = $3
                  AND (1 - (embedding <=> $1::vector)) >= $4
                ORDER BY embedding <=> $1::vector
                LIMIT $5
                """,
                query_embedding_str,
                bank_id,
                fact_type,
                threshold,
                limit,
            )

        return [SeedNode(node_id=str(r["id"]), score=r["similarity"]) for r in rows]
