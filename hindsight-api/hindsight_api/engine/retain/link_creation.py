"""
Link creation for retain pipeline.

Handles creation of temporal, semantic, and causal links between facts.
"""
import logging
from typing import List

from .types import ProcessedFact, CausalRelation
from . import link_utils

logger = logging.getLogger(__name__)


async def create_temporal_links_batch(
    conn,
    bank_id: str,
    unit_ids: List[str]
) -> int:
    """
    Create temporal links between facts.

    Links facts that occurred close in time to each other.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: List of unit IDs to create links for

    Returns:
        Number of temporal links created
    """
    if not unit_ids:
        return 0

    return await link_utils.create_temporal_links_batch_per_fact(
        conn,
        bank_id,
        unit_ids,
        log_buffer=[]
    )


async def create_semantic_links_batch(
    conn,
    bank_id: str,
    unit_ids: List[str],
    embeddings: List[List[float]]
) -> int:
    """
    Create semantic links between facts.

    Links facts that are semantically similar based on embeddings.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: List of unit IDs to create links for
        embeddings: List of embedding vectors (same length as unit_ids)

    Returns:
        Number of semantic links created
    """
    if not unit_ids or not embeddings:
        return 0

    if len(unit_ids) != len(embeddings):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and embeddings ({len(embeddings)})")

    return await link_utils.create_semantic_links_batch(
        conn,
        bank_id,
        unit_ids,
        embeddings,
        log_buffer=[]
    )


async def create_causal_links_batch(
    conn,
    unit_ids: List[str],
    facts: List[ProcessedFact]
) -> int:
    """
    Create causal links between facts.

    Links facts that have causal relationships (causes, enables, prevents).

    Args:
        conn: Database connection
        unit_ids: List of unit IDs (same length as facts)
        facts: List of ProcessedFact objects with causal_relations

    Returns:
        Number of causal links created
    """
    if not unit_ids or not facts:
        return 0

    if len(unit_ids) != len(facts):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and facts ({len(facts)})")

    # Extract causal relations in the format expected by link_utils
    # Format: List of lists, where each inner list is the causal relations for that fact
    causal_relations_per_fact = []
    for fact in facts:
        if fact.causal_relations:
            # Convert CausalRelation objects to dicts
            relations_dicts = [
                {
                    'relation_type': rel.relation_type,
                    'target_fact_index': rel.target_fact_index,
                    'strength': rel.strength
                }
                for rel in fact.causal_relations
            ]
            causal_relations_per_fact.append(relations_dicts)
        else:
            causal_relations_per_fact.append([])

    link_count = await link_utils.create_causal_links_batch(
        conn,
        unit_ids,
        causal_relations_per_fact
    )

    return link_count
