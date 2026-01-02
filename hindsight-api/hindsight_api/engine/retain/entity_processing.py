"""
Entity processing for retain pipeline.

Handles entity extraction, resolution, and link creation for stored facts.
"""

import logging

from . import link_utils
from .types import EntityLink, ProcessedFact

logger = logging.getLogger(__name__)


async def process_entities_batch(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    log_buffer: list[str] = None,
    user_entities_per_content: dict[int, list[dict]] = None,
) -> list[EntityLink]:
    """
    Process entities for all facts and create entity links.

    This function:
    1. Extracts entity mentions from fact texts
    2. Merges user-provided entities with LLM-extracted entities
    3. Resolves entity names to canonical entities
    4. Creates entity records in the database
    5. Returns entity links ready for insertion

    Args:
        entity_resolver: EntityResolver instance for entity resolution
        conn: Database connection
        bank_id: Bank identifier
        unit_ids: List of unit IDs (same length as facts)
        facts: List of ProcessedFact objects
        log_buffer: Optional buffer for detailed logging
        user_entities_per_content: Dict mapping content_index to list of user-provided entities

    Returns:
        List of EntityLink objects for batch insertion
    """
    if not unit_ids or not facts:
        return []

    if len(unit_ids) != len(facts):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and facts ({len(facts)})")

    user_entities_per_content = user_entities_per_content or {}

    # Extract data for link_utils function
    fact_texts = [fact.fact_text for fact in facts]
    # Use occurred_start if available, otherwise use mentioned_at for entity timestamps
    fact_dates = [fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at for fact in facts]

    # Convert EntityRef objects to dict format and merge with user-provided entities
    entities_per_fact = []
    for fact in facts:
        # Start with LLM-extracted entities
        llm_entities = [{"text": entity.name, "type": "CONCEPT"} for entity in (fact.entities or [])]

        # Get user entities for this content (use content_index from fact)
        user_entities = user_entities_per_content.get(fact.content_index, [])

        # Merge with case-insensitive deduplication
        seen_texts = {e["text"].lower() for e in llm_entities}
        for user_entity in user_entities:
            if user_entity["text"].lower() not in seen_texts:
                llm_entities.append({
                    "text": user_entity["text"],
                    "type": user_entity.get("type", "CONCEPT"),
                })
                seen_texts.add(user_entity["text"].lower())

        entities_per_fact.append(llm_entities)

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
        log_buffer,  # Pass log_buffer for detailed logging
    )

    return entity_links


async def insert_entity_links_batch(conn, entity_links: list[EntityLink]) -> None:
    """
    Insert entity links in batch.

    Args:
        conn: Database connection
        entity_links: List of EntityLink objects
    """
    if not entity_links:
        return

    await link_utils.insert_entity_links_batch(conn, entity_links)
