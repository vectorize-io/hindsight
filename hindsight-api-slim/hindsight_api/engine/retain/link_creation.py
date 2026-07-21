"""
Link creation for retain pipeline.

Handles creation of temporal, semantic, and causal links between facts.
"""

import logging

from . import link_utils
from .types import ProcessedFact

logger = logging.getLogger(__name__)


async def create_temporal_links_batch(conn, bank_id: str, unit_ids: list[str], ops=None) -> int:
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

    return await link_utils.create_temporal_links_batch_per_fact(conn, bank_id, unit_ids, log_buffer=[], ops=ops)


async def create_semantic_links_batch(
    conn,
    bank_id: str,
    unit_ids: list[str],
    embeddings: list[list[float]],
    threshold: float,
    pre_computed_ann_links: list[tuple] | None = None,
    ops=None,
) -> int:
    """
    Create semantic links between facts.

    Links facts that are semantically similar based on embeddings.
    When pre_computed_ann_links are provided (from Phase 1), they are used
    instead of running ANN queries inside the transaction.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: List of unit IDs to create links for
        embeddings: List of embedding vectors (same length as unit_ids)
        threshold: Minimum cosine similarity for semantic links
        pre_computed_ann_links: Pre-computed ANN results from Phase 1

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
        threshold=threshold,
        log_buffer=[],
        pre_computed_ann_links=pre_computed_ann_links,
        ops=ops,
    )


async def create_causal_links_batch(
    conn, bank_id: str, unit_ids: list[str], facts: list[ProcessedFact], ops=None
) -> int:
    """
    Create causal links between facts.

    Retain writes the canonical ``caused_by`` relationship only. The database and
    retrieval paths also recognize historical causal types so imported and
    pre-existing memories remain traversable.

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

    causal_relations_per_fact = [fact.causal_relations or [] for fact in facts]

    link_count = await link_utils.create_causal_links_batch(conn, bank_id, unit_ids, causal_relations_per_fact, ops=ops)

    return link_count
