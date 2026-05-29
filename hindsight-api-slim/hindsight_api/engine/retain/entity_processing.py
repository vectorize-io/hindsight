"""
Entity processing for retain pipeline.

Handles entity extraction and resolution for stored facts.
"""

import logging

from . import link_utils
from .types import ProcessedFact

logger = logging.getLogger(__name__)


def _prepare_facts_for_entity_processing(
    facts: list[ProcessedFact],
    user_entities_per_content: dict[int, list[dict]] | None = None,
) -> tuple[list[str], list, list[list[dict]]]:
    """
    Extract fact texts, dates, and merged entity lists from ProcessedFact objects.

    Returns:
        Tuple of (fact_texts, fact_dates, entities_per_fact)
    """
    user_entities_per_content = user_entities_per_content or {}

    fact_texts = [fact.fact_text for fact in facts]
    fact_dates = [fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at for fact in facts]

    entities_per_fact = []
    for fact in facts:
        llm_entities = [{"text": entity.name, "type": "CONCEPT"} for entity in (fact.entities or [])]

        user_entities = user_entities_per_content.get(fact.content_index, [])

        seen_texts = {e["text"].lower() for e in llm_entities}
        for user_entity in user_entities:
            if user_entity["text"].lower() not in seen_texts:
                llm_entities.append(
                    {
                        "text": user_entity["text"],
                        "type": user_entity.get("type", "CONCEPT"),
                    }
                )
                seen_texts.add(user_entity["text"].lower())

        entities_per_fact.append(llm_entities)

    return fact_texts, fact_dates, entities_per_fact


async def resolve_entities(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    log_buffer: list[str] = None,
    user_entities_per_content: dict[int, list[dict]] = None,
    entity_labels: list | None = None,
) -> tuple[list[str], list[tuple], dict[str, list[str]]]:
    """
    Phase 1: Resolve entity names to canonical IDs (read-heavy).

    Should be called on a SEPARATE connection OUTSIDE the main write transaction
    to avoid holding the transaction open during expensive trigram scans.

    Args:
        entity_resolver: EntityResolver instance
        conn: Database connection (separate from the main write transaction)
        bank_id: Bank identifier
        unit_ids: Placeholder unit IDs (used only for grouping)
        facts: List of ProcessedFact objects
        log_buffer: Optional buffer for detailed logging
        user_entities_per_content: Dict mapping content_index to user-provided entities
        entity_labels: Optional entity label taxonomy

    Returns:
        Tuple of (resolved_entity_ids, entity_to_unit, unit_to_entity_ids).
    """
    if not unit_ids or not facts:
        return [], [], {}

    if len(unit_ids) != len(facts):
        raise ValueError(f"Mismatch between unit_ids ({len(unit_ids)}) and facts ({len(facts)})")

    fact_texts, fact_dates, entities_per_fact = _prepare_facts_for_entity_processing(facts, user_entities_per_content)

    return await link_utils.resolve_entities_only(
        entity_resolver,
        conn,
        bank_id,
        unit_ids,
        fact_texts,
        "",  # context (not used in current implementation)
        fact_dates,
        entities_per_fact,
        log_buffer,
        entity_labels=entity_labels,
    )
