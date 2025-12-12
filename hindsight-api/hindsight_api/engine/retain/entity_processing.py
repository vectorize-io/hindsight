"""
Entity processing for retain pipeline.

Handles entity extraction, resolution, and link creation for stored facts.
"""
import logging
from typing import List, Tuple, Dict, Any
from uuid import UUID

from .types import ProcessedFact, EntityRef, EntityLink
from . import link_utils

logger = logging.getLogger(__name__)


async def process_entities_batch(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: List[str],
    facts: List[ProcessedFact],
    log_buffer: List[str] = None
) -> List[EntityLink]:
    """
    Process entities for all facts and create entity links.

    This function:
    1. Extracts entity mentions from fact texts
    2. Resolves entity names to canonical entities
    3. Creates entity records in the database
    4. Returns entity links ready for insertion

    Args:
        entity_resolver: EntityResolver instance for entity resolution
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: List of unit IDs (same length as facts)
        facts: List of ProcessedFact objects
        log_buffer: Optional buffer for detailed logging

    Returns:
        List of EntityLink objects for batch insertion
    """
    if not unit_ids or not facts:
        return []

    if len(unit_ids) != len(facts):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and facts ({len(facts)})")

    # Extract data for link_utils function
    fact_texts = [fact.fact_text for fact in facts]
    # Use occurred_start if available, otherwise use mentioned_at for entity timestamps
    fact_dates = [fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at for fact in facts]
    # Convert EntityRef objects to dict format expected by link_utils
    entities_per_fact = [
        [{'text': entity.name, 'type': 'CONCEPT'} for entity in (fact.entities or [])]
        for fact in facts
    ]

    # Use existing link_utils function for entity processing
    entity_links = await link_utils.extract_entities_batch_optimized(
        entity_resolver,
        conn,
        bank_id,
        unit_ids,
        fact_texts,
        "",  # context (not used in current implementation)
        fact_dates,
        entities_per_fact,
        log_buffer  # Pass log_buffer for detailed logging
    )

    return entity_links


async def insert_entity_links_batch(
    conn,
    entity_links: List[EntityLink]
) -> None:
    """
    Insert entity links in batch.

    Args:
        conn: Database connection
        entity_links: List of EntityLink objects
    """
    if not entity_links:
        return

    await link_utils.insert_entity_links_batch(conn, entity_links)
