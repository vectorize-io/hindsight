"""
Meta-Path Forward Push (MPFP) graph retrieval.

A sublinear graph traversal algorithm for memory retrieval over heterogeneous
graphs with multiple edge types (semantic, temporal, causal, entity).

Combines meta-path patterns from HIN literature with Forward Push local
propagation from Approximate PPR.

Key properties:
- Sublinear in graph size (threshold pruning bounds active nodes)
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
from .types import RetrievalResult

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
class TypedAdjacency:
    """Adjacency lists split by edge type."""

    # edge_type -> from_node_id -> list of (to_node_id, weight)
    graphs: dict[str, dict[str, list[EdgeTarget]]] = field(default_factory=dict)

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
# Core Algorithm
# -----------------------------------------------------------------------------


def mpfp_traverse(
    seeds: list[SeedNode],
    pattern: list[str],
    adjacency: TypedAdjacency,
    config: MPFPConfig,
) -> PatternResult:
    """
    Forward Push traversal following a meta-path pattern.

    Args:
        seeds: Entry point nodes with initial scores
        pattern: Sequence of edge types to follow
        adjacency: Typed adjacency structure
        config: Algorithm parameters

    Returns:
        PatternResult with accumulated scores per node
    """
    if not seeds:
        return PatternResult(pattern=pattern, scores={})

    scores: dict[str, float] = {}

    # Initialize frontier with seed masses (normalized)
    total_seed_score = sum(s.score for s in seeds)
    if total_seed_score == 0:
        total_seed_score = len(seeds)  # fallback to uniform

    frontier: dict[str, float] = {s.node_id: s.score / total_seed_score for s in seeds}

    # Follow pattern hop by hop
    for edge_type in pattern:
        next_frontier: dict[str, float] = {}

        for node_id, mass in frontier.items():
            if mass < config.threshold:
                continue

            # Keep α portion for this node
            scores[node_id] = scores.get(node_id, 0) + config.alpha * mass

            # Push (1-α) to neighbors
            push_mass = (1 - config.alpha) * mass
            neighbors = adjacency.get_normalized_neighbors(edge_type, node_id, config.top_k_neighbors)

            for neighbor in neighbors:
                next_frontier[neighbor.node_id] = next_frontier.get(neighbor.node_id, 0) + push_mass * neighbor.weight

        frontier = next_frontier

    # Final frontier nodes get their remaining mass
    for node_id, mass in frontier.items():
        if mass >= config.threshold:
            scores[node_id] = scores.get(node_id, 0) + mass

    return PatternResult(pattern=pattern, scores=scores)


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


async def load_typed_adjacency(pool, bank_id: str) -> TypedAdjacency:
    """
    Load all edges for a bank, split by edge type.

    Single query, then organize in-memory for fast traversal.
    """
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT ml.from_unit_id, ml.to_unit_id, ml.link_type, ml.weight
            FROM {fq_table("memory_links")} ml
            JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
            WHERE mu.bank_id = $1
              AND ml.weight >= 0.1
            ORDER BY ml.from_unit_id, ml.weight DESC
            """,
            bank_id,
        )

    graphs: dict[str, dict[str, list[EdgeTarget]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        from_id = str(row["from_unit_id"])
        to_id = str(row["to_unit_id"])
        link_type = row["link_type"]
        weight = row["weight"]

        graphs[link_type][from_id].append(EdgeTarget(node_id=to_id, weight=weight))

    return TypedAdjacency(graphs=dict(graphs))


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
    Graph retrieval using Meta-Path Forward Push.

    Runs predefined patterns in parallel from semantic and temporal seeds,
    then fuses results via RRF.
    """

    def __init__(self, config: MPFPConfig | None = None):
        """
        Initialize MPFP retriever.

        Args:
            config: Algorithm configuration (uses defaults if None)
        """
        self.config = config or MPFPConfig()
        self._adjacency_cache: dict[str, TypedAdjacency] = {}

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
    ) -> list[RetrievalResult]:
        """
        Retrieve facts using MPFP algorithm.

        Args:
            pool: Database connection pool
            query_embedding_str: Query embedding (used for fallback seed finding)
            bank_id: Memory bank ID
            fact_type: Fact type to filter
            budget: Maximum results to return
            query_text: Original query text (optional)
            semantic_seeds: Pre-computed semantic entry points
            temporal_seeds: Pre-computed temporal entry points

        Returns:
            List of RetrievalResult with activation scores
        """
        # Load typed adjacency (could cache per bank_id with TTL)
        adjacency = await load_typed_adjacency(pool, bank_id)

        # Convert seeds to SeedNode format
        semantic_seed_nodes = self._convert_seeds(semantic_seeds, "similarity")
        temporal_seed_nodes = self._convert_seeds(temporal_seeds, "temporal_score")

        # If no semantic seeds provided, fall back to finding our own
        if not semantic_seed_nodes:
            semantic_seed_nodes = await self._find_semantic_seeds(pool, query_embedding_str, bank_id, fact_type)

        # Run all patterns in parallel
        tasks = []

        # Patterns from semantic seeds
        for pattern in self.config.patterns_semantic:
            if semantic_seed_nodes:
                tasks.append(
                    asyncio.to_thread(
                        mpfp_traverse,
                        semantic_seed_nodes,
                        pattern,
                        adjacency,
                        self.config,
                    )
                )

        # Patterns from temporal seeds
        for pattern in self.config.patterns_temporal:
            if temporal_seed_nodes:
                tasks.append(
                    asyncio.to_thread(
                        mpfp_traverse,
                        temporal_seed_nodes,
                        pattern,
                        adjacency,
                        self.config,
                    )
                )

        if not tasks:
            return []

        # Gather pattern results
        pattern_results = await asyncio.gather(*tasks)

        # Fuse results
        fused = rrf_fusion(pattern_results, top_k=budget)

        if not fused:
            return []

        # Get top result IDs (don't exclude seeds - they may be highly relevant)
        result_ids = [node_id for node_id, score in fused][:budget]

        # Fetch full details
        results = await fetch_memory_units_by_ids(pool, result_ids, fact_type)

        # Add activation scores from fusion
        score_map = {node_id: score for node_id, score in fused}
        for result in results:
            result.activation = score_map.get(result.id, 0.0)

        # Sort by activation
        results.sort(key=lambda r: r.activation or 0, reverse=True)

        return results

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
