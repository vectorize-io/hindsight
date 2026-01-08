"""
Entity summary generation using a 4-step pipeline.

Pipeline:
1. FACET DISCOVERY - LLM generates diverse probe queries based on entity name
2. PARALLEL RETRIEVAL - Execute probes against recall in parallel
3. DIVERSITY SELECTION - Clustering + MMR to select representative facts
4. SYNTHESIS - LLM generates coherent summary from selected facts

Total LLM calls: 2 (Step 1 and Step 4 only)
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from ..response_models import MemoryFact

logger = logging.getLogger(__name__)


# ============================================================================
# Summary Result
# ============================================================================


class SummaryResult:
    """Result from entity summary generation."""

    def __init__(self, entity_name: str, summary: str | None, log_buffer: list[str], duration: float):
        self.entity_name = entity_name
        self.summary = summary
        self.log_buffer = log_buffer
        self.duration = duration

    def log(self):
        """Log the buffered messages."""
        if self.log_buffer:
            logger.info("\n" + "\n".join(self.log_buffer))


# ============================================================================
# Constants
# ============================================================================

TOTAL_TOKEN_BUDGET = 200_000  # Total tokens for fact retrieval
MAX_PROBES = 5  # Maximum number of probe queries to generate
MAX_FACTS_PER_PROBE = 100  # Maximum facts to retrieve per probe
MAX_CLUSTERS = 50  # Maximum clusters for diversity selection
MIN_CLUSTERS = 5  # Minimum clusters
TARGET_SUMMARY_TOKENS = 2000  # Target summary length in tokens


# ============================================================================
# Response Models
# ============================================================================


class FacetProbesResponse(BaseModel):
    """Response from facet discovery LLM call."""

    probes: list[str] = Field(description="List of diverse probe queries to explore the entity")


class EntitySummaryResponse(BaseModel):
    """Response from synthesis LLM call."""

    summary: str = Field(default="", description="A cohesive summary about the entity")
    # Some LLMs return 'description' instead of 'summary'
    description: str | None = Field(default=None, exclude=True)

    def get_summary(self) -> str:
        """Get the summary, falling back to description if summary is empty."""
        return self.summary or self.description or ""


class Facet(BaseModel):
    """A facet with its probe query and retrieved memories."""

    probe: str
    memories: list[MemoryFact] = Field(default_factory=list)
    embeddings: list[list[float]] = Field(default_factory=list)


# ============================================================================
# Step 1: Facet Discovery (LLM)
# ============================================================================


def build_facet_discovery_prompt(entity_name: str, custom_directions: str | None = None) -> str:
    """Build the prompt for facet discovery.

    Args:
        entity_name: Name of the entity/topic to explore
        custom_directions: Optional custom directions for mental models
    """
    if custom_directions:
        # For mental models with custom directions, use those to guide facet discovery
        return f"""Generate 3-5 diverse search queries to explore what a personal knowledge base might contain about "{entity_name}".

IMPORTANT FOCUS AREA: {custom_directions}

Generate queries that specifically help gather information about:
- The main topic: "{entity_name}"
- Following the directions above

Keep queries short (2-5 words each) and make them dissimilar from each other to cover different aspects of this focus area."""

    return f"""Generate 3-5 diverse search queries to explore what a personal knowledge base might contain about "{entity_name}".

Consider that:
- The entity type is unknown (could be person, company, concept, place, event, etc.)
- Include factual, experiential, relational, and opinion-based angles
- Keep queries short (2-5 words each)
- Make queries dissimilar from each other to cover different aspects
- Think about: identity, roles, characteristics, relationships, activities, preferences, history, opinions

Examples of diverse queries for a person named "John":
- "John's job role"
- "John family members"
- "John hobbies interests"
- "John personality traits"
- "opinions about John"
- "John achievements"
- "John challenges problems"

Generate queries specifically for "{entity_name}"."""


def get_facet_discovery_system_message() -> str:
    """System message for facet discovery."""
    return "You are a search query generator. Generate diverse, short queries that would help discover different aspects of an entity in a personal knowledge base. Output only the probe queries as a JSON array."


async def discover_facets(llm_config, entity_name: str, custom_directions: str | None = None) -> list[str]:
    """
    Step 1: Use LLM to generate diverse probe queries for the entity.

    Args:
        llm_config: LLM configuration
        entity_name: Name of the entity to explore
        custom_directions: Optional custom directions for mental models

    Returns:
        List of probe query strings

    Raises:
        Exception: If LLM call fails (no fallback)
    """
    prompt = build_facet_discovery_prompt(entity_name, custom_directions)

    result = await llm_config.call(
        messages=[
            {"role": "system", "content": get_facet_discovery_system_message()},
            {"role": "user", "content": prompt},
        ],
        response_format=FacetProbesResponse,
        scope="memory_facet_discovery",
    )

    probes = result.probes[:MAX_PROBES]  # Limit to max probes
    logger.debug(f"[FACET_DISCOVERY] Generated {len(probes)} probes for {entity_name}")

    return probes


# ============================================================================
# Step 2: Parallel Retrieval
# ============================================================================


async def retrieve_facets_parallel(
    probes: list[str],
    recall_fn: Callable[[str, int], Awaitable[tuple[list[MemoryFact], list[list[float]]]]],
    total_token_budget: int = TOTAL_TOKEN_BUDGET,
) -> list[Facet]:
    """
    Step 2: Execute all probe queries against recall in parallel.

    Args:
        probes: List of probe queries from facet discovery
        recall_fn: Async function (query, max_tokens) -> (facts, embeddings)
        total_token_budget: Total token budget to distribute across probes

    Returns:
        List of Facets with memories and embeddings
    """
    # Distribute budget equally among probes
    tokens_per_probe = total_token_budget // len(probes)

    async def fetch_facet(probe: str) -> Facet:
        try:
            facts, embeddings = await recall_fn(probe, tokens_per_probe)
            return Facet(probe=probe, memories=facts, embeddings=embeddings)
        except Exception as e:
            logger.warning(f"[RETRIEVAL] Failed to retrieve for probe '{probe}': {e}")
            return Facet(probe=probe, memories=[], embeddings=[])

    # Execute all probes in parallel
    facets = await asyncio.gather(*[fetch_facet(probe) for probe in probes])

    # Filter out empty facets
    non_empty_facets = [f for f in facets if f.memories]
    logger.debug(
        f"[RETRIEVAL] Retrieved {sum(len(f.memories) for f in non_empty_facets)} "
        f"memories across {len(non_empty_facets)} facets"
    )

    return non_empty_facets


def deduplicate_facets(facets: list[Facet], overlap_threshold: float = 0.7) -> list[Facet]:
    """
    Merge facets with high memory overlap.

    Args:
        facets: List of facets to deduplicate
        overlap_threshold: Merge facets with overlap >= this threshold

    Returns:
        Deduplicated list of facets
    """
    if not facets:
        return facets

    # Build memory ID sets for each facet
    facet_memory_ids = [{m.id for m in f.memories} for f in facets]

    merged_indices: set[int] = set()
    result: list[Facet] = []

    for i, facet in enumerate(facets):
        if i in merged_indices:
            continue

        merged_memories = list(facet.memories)
        merged_embeddings = list(facet.embeddings)
        merged_ids = set(facet_memory_ids[i])

        # Check for overlapping facets
        for j in range(i + 1, len(facets)):
            if j in merged_indices:
                continue

            # Calculate overlap
            intersection = len(facet_memory_ids[i] & facet_memory_ids[j])
            union = len(facet_memory_ids[i] | facet_memory_ids[j])
            overlap = intersection / union if union > 0 else 0

            if overlap >= overlap_threshold:
                # Merge facet j into facet i
                for k, mem in enumerate(facets[j].memories):
                    if mem.id not in merged_ids:
                        merged_memories.append(mem)
                        if k < len(facets[j].embeddings):
                            merged_embeddings.append(facets[j].embeddings[k])
                        merged_ids.add(mem.id)
                merged_indices.add(j)

        result.append(Facet(probe=facet.probe, memories=merged_memories, embeddings=merged_embeddings))

    logger.debug(f"[DEDUP] Merged {len(facets)} facets into {len(result)} after deduplication")
    return result


# ============================================================================
# Step 3: Diversity Selection (Clustering + MMR)
# ============================================================================


def cluster_memories(
    memories: list[MemoryFact],
    embeddings: list[list[float]],
    max_clusters: int = MAX_CLUSTERS,
    min_clusters: int = MIN_CLUSTERS,
) -> list[tuple[MemoryFact, list[float]]]:
    """
    Cluster memories and select representatives from each cluster.

    Args:
        memories: List of memories
        embeddings: Corresponding embeddings
        max_clusters: Maximum number of clusters
        min_clusters: Minimum number of clusters

    Returns:
        List of (memory, embedding) tuples - one representative per cluster
    """
    if not memories or not embeddings:
        return []

    if len(memories) != len(embeddings):
        logger.warning(f"[CLUSTER] Mismatch: {len(memories)} memories vs {len(embeddings)} embeddings")
        # Use min length
        n = min(len(memories), len(embeddings))
        memories = memories[:n]
        embeddings = embeddings[:n]

    if len(memories) <= min_clusters:
        return list(zip(memories, embeddings))

    # Calculate number of clusters (proportional to memory count)
    n_clusters = min(max_clusters, max(min_clusters, len(memories) // 5))

    # Convert to numpy array
    embedding_matrix = np.array(embeddings)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding_matrix)

    # Select representative from each cluster (closest to centroid)
    representatives: list[tuple[MemoryFact, list[float]]] = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # Find memory closest to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_embeddings = embedding_matrix[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]

        representatives.append((memories[closest_idx], embeddings[closest_idx]))

    logger.debug(f"[CLUSTER] Clustered {len(memories)} memories into {len(representatives)} representatives")
    return representatives


def mmr_selection(
    candidates: list[tuple[MemoryFact, list[float]]],
    relevance_scores: list[float],
    target_count: int,
    lambda_param: float = 0.5,
) -> list[MemoryFact]:
    """
    Maximal Marginal Relevance selection for diversity.

    Args:
        candidates: List of (memory, embedding) tuples
        relevance_scores: Relevance score for each candidate
        target_count: Number of memories to select
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        Selected memories
    """
    if not candidates:
        return []

    if len(candidates) <= target_count:
        return [mem for mem, _ in candidates]

    selected: list[tuple[MemoryFact, list[float]]] = []
    selected_indices: set[int] = set()
    embeddings = np.array([emb for _, emb in candidates])

    for _ in range(target_count):
        best_idx = -1
        best_score = float("-inf")

        for i, (mem, emb) in enumerate(candidates):
            if i in selected_indices:
                continue

            relevance = relevance_scores[i] if i < len(relevance_scores) else 0.5

            # Calculate max similarity to already selected
            if selected:
                selected_embs = np.array([e for _, e in selected])
                similarities = np.dot(selected_embs, emb) / (
                    np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(emb) + 1e-8
                )
                max_similarity = np.max(similarities)
            else:
                max_similarity = 0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates[best_idx])
            selected_indices.add(best_idx)

    return [mem for mem, _ in selected]


def select_diverse_memories(
    facets: list[Facet],
    target_token_budget: int,
    tokens_per_memory: int = 200,  # Rough estimate
) -> dict[str, list[MemoryFact]]:
    """
    Step 3: Select diverse, representative memories from each facet.

    Args:
        facets: List of facets with memories and embeddings
        target_token_budget: Total token budget for selected memories
        tokens_per_memory: Estimated tokens per memory

    Returns:
        Dict mapping facet probe to selected memories
    """
    if not facets:
        return {}

    # Calculate target memories per facet (proportional allocation)
    total_memories = sum(len(f.memories) for f in facets)
    target_total = target_token_budget // tokens_per_memory

    result: dict[str, list[MemoryFact]] = {}

    for facet in facets:
        if not facet.memories:
            continue

        # Proportional allocation with min/max bounds
        proportion = len(facet.memories) / total_memories if total_memories > 0 else 0
        facet_target = max(3, min(50, int(target_total * proportion)))

        # Step 3a: Cluster to get representatives
        representatives = cluster_memories(facet.memories, facet.embeddings)

        if not representatives:
            result[facet.probe] = []
            continue

        # Step 3b: MMR selection from representatives
        # Use uniform relevance scores (all representatives are equally relevant)
        relevance_scores = [1.0] * len(representatives)
        selected = mmr_selection(representatives, relevance_scores, facet_target)

        result[facet.probe] = selected

    total_selected = sum(len(mems) for mems in result.values())
    logger.debug(f"[DIVERSITY] Selected {total_selected} diverse memories across {len(result)} facets")

    return result


# ============================================================================
# Step 4: Synthesis (LLM)
# ============================================================================


def format_memories_for_synthesis(
    entity_name: str,
    facet_memories: dict[str, list[MemoryFact]],
) -> str:
    """Format selected memories organized by facet for synthesis prompt."""
    sections = [f"Entity: {entity_name}\n"]

    for probe, memories in facet_memories.items():
        if not memories:
            continue

        sections.append(f"\n## Facet: {probe}")
        for mem in memories:
            # Include temporal info if available
            temporal = ""
            if mem.occurred_start:
                temporal = f" (when: {mem.occurred_start})"
            sections.append(f"- {mem.text}{temporal}")

    return "\n".join(sections)


def build_synthesis_prompt(
    entity_name: str, formatted_memories: str, target_tokens: int, custom_directions: str | None = None
) -> str:
    """Build the synthesis prompt.

    Args:
        entity_name: Name of the entity/topic
        formatted_memories: Formatted memories text
        target_tokens: Target summary length
        custom_directions: Optional custom directions for mental models
    """
    if custom_directions:
        # For mental models with custom directions
        return f"""Given these memories about "{entity_name}" organized by facet, create a focused summary.

IMPORTANT FOCUS AREA: {custom_directions}

{formatted_memories}

Create a summary that:
1. Focuses on the directions above for "{entity_name}"
2. Uses markdown headers (## Section Name) to organize by topic
3. Integrates information into a coherent narrative
4. Resolves contradictions (prefer more recent information)
5. Notes uncertainty where information is incomplete
6. Uses plain prose under each header, no bullet points or bold/italic styling
7. Is approximately {target_tokens} tokens long

Write a well-organized summary with clear sections that addresses the focus area."""

    return f"""Given these memories about "{entity_name}" organized by facet, create a comprehensive summary.

{formatted_memories}

Create a summary that:
1. Integrates information across facets into a coherent narrative
2. Uses markdown headers (## Section Name) to organize by topic (e.g., ## Background, ## Work, ## Relationships)
3. Resolves contradictions (prefer more recent information)
4. Notes uncertainty where information is incomplete
5. Is written in third person ("{entity_name} is..." not "I think...")
6. Uses plain prose under each header, no bullet points or bold/italic styling
7. Is approximately {target_tokens} tokens long

Write a well-organized summary with clear sections that captures everything known about {entity_name}."""


def get_synthesis_system_message() -> str:
    """System message for synthesis."""
    return "You are synthesizing fragmented memories into a coherent, readable summary. Use markdown headers (## Section) to organize by topic. Write clear, factual prose under each section. No bullet points, bold, or italic - just headers and plain text."


async def synthesize_summary(
    llm_config,
    entity_name: str,
    facet_memories: dict[str, list[MemoryFact]],
    target_tokens: int = TARGET_SUMMARY_TOKENS,
    custom_directions: str | None = None,
) -> str:
    """
    Step 4: Generate coherent summary from selected memories.

    Args:
        llm_config: LLM configuration
        entity_name: Name of the entity
        facet_memories: Dict mapping facet probes to selected memories
        target_tokens: Target summary length in tokens
        custom_directions: Optional custom directions for mental models

    Returns:
        Summary string

    Raises:
        Exception: If LLM call fails
    """
    formatted = format_memories_for_synthesis(entity_name, facet_memories)
    prompt = build_synthesis_prompt(entity_name, formatted, target_tokens, custom_directions)

    result = await llm_config.call(
        messages=[
            {"role": "system", "content": get_synthesis_system_message()},
            {"role": "user", "content": prompt},
        ],
        response_format=EntitySummaryResponse,
        scope="memory_generate_summary",
    )

    return result.get_summary()


# ============================================================================
# Main Pipeline
# ============================================================================


async def generate_entity_summary(
    llm_config,
    entity_name: str,
    recall_fn: Callable[[str, int], Awaitable[tuple[list[MemoryFact], list[list[float]]]]],
    *,
    bank_id: str | None = None,
    total_token_budget: int = TOTAL_TOKEN_BUDGET,
    target_summary_tokens: int = TARGET_SUMMARY_TOKENS,
    custom_directions: str | None = None,
) -> SummaryResult:
    """
    Generate a cohesive summary for an entity using the 4-step pipeline.

    Pipeline:
    1. FACET DISCOVERY - LLM generates diverse probe queries
    2. PARALLEL RETRIEVAL - Execute probes against recall
    3. DIVERSITY SELECTION - Clustering + MMR for representative facts
    4. SYNTHESIS - LLM generates coherent summary

    Args:
        llm_config: LLM configuration
        entity_name: Name of the entity to summarize
        recall_fn: Async function (query, max_tokens) -> (facts, embeddings)
        bank_id: Bank identifier for logging
        total_token_budget: Total tokens for fact retrieval (default 200k)
        target_summary_tokens: Target summary length (default 2k tokens)
        custom_directions: Optional custom directions for mental models

    Returns:
        SummaryResult containing summary, log buffer, and timing info
    """
    start_time = time.time()
    log_buffer: list[str] = []
    short_name = entity_name[:30] + "..." if len(entity_name) > 30 else entity_name
    bank_short = bank_id[:8] if bank_id else "unknown"
    summary_id = f"{bank_short}-{int(time.time() * 1000) % 100000}"

    log_buffer.append(f"[SUMMARY {summary_id}] Entity: '{short_name}' (budget={total_token_budget})")

    try:
        # Step 1: Facet Discovery
        step1_start = time.time()
        probes = await discover_facets(llm_config, entity_name, custom_directions)
        step1_time = time.time() - step1_start

        if not probes:
            log_buffer.append(f"  [1] Facet discovery: 0 probes (failed) in {step1_time:.3f}s")
            total_time = time.time() - start_time
            log_buffer.append(f"  [SUMMARY {summary_id}] FAILED: No probes generated | {total_time:.3f}s")
            return SummaryResult(entity_name, None, log_buffer, total_time)

        log_buffer.append(f"  [1] Facet discovery: {len(probes)} probes in {step1_time:.3f}s")

        # Step 2: Parallel Retrieval
        step2_start = time.time()
        facets = await retrieve_facets_parallel(probes, recall_fn, total_token_budget)
        step2_time = time.time() - step2_start

        if not facets:
            log_buffer.append(f"  [2] Parallel retrieval: 0 memories in {step2_time:.3f}s")
            total_time = time.time() - start_time
            log_buffer.append(f"  [SUMMARY {summary_id}] FAILED: No memories found | {total_time:.3f}s")
            return SummaryResult(entity_name, None, log_buffer, total_time)

        total_memories = sum(len(f.memories) for f in facets)
        log_buffer.append(
            f"  [2] Parallel retrieval: {total_memories} memories from {len(facets)} facets in {step2_time:.3f}s"
        )

        # Deduplicate overlapping facets
        facets = deduplicate_facets(facets)
        facets_after_dedup = len(facets)
        log_buffer.append(f"  [2.5] Deduplication: {facets_after_dedup} unique facets")

        # Step 3: Diversity Selection
        step3_start = time.time()
        selection_budget = total_token_budget - (target_summary_tokens * 2)
        facet_memories = select_diverse_memories(facets, selection_budget)
        step3_time = time.time() - step3_start

        total_selected = sum(len(mems) for mems in facet_memories.values())
        if total_selected == 0:
            log_buffer.append(f"  [3] Diversity selection: 0 memories selected in {step3_time:.3f}s")
            total_time = time.time() - start_time
            log_buffer.append(f"  [SUMMARY {summary_id}] FAILED: No memories selected | {total_time:.3f}s")
            return SummaryResult(entity_name, None, log_buffer, total_time)

        log_buffer.append(f"  [3] Diversity selection: {total_selected} memories selected in {step3_time:.3f}s")

        # Step 4: Synthesis
        step4_start = time.time()
        summary = await synthesize_summary(
            llm_config, entity_name, facet_memories, target_summary_tokens, custom_directions
        )
        step4_time = time.time() - step4_start

        log_buffer.append(f"  [4] Synthesis: {len(summary)} chars in {step4_time:.3f}s")

        total_time = time.time() - start_time
        log_buffer.append(f"  [SUMMARY {summary_id}] Complete: {len(summary)} chars | {total_time:.3f}s")

        return SummaryResult(entity_name, summary, log_buffer, total_time)

    except Exception as e:
        total_time = time.time() - start_time
        log_buffer.append(f"  [SUMMARY {summary_id}] ERROR: {str(e)} | {total_time:.3f}s")
        return SummaryResult(entity_name, None, log_buffer, total_time)
