"""
Entity processing for retain pipeline.

Handles entity extraction and resolution for stored facts.
"""

import logging

from . import link_utils
from .types import EntityResolutionResult, ProcessedFact, UserEntities

logger = logging.getLogger(__name__)


def _prepare_facts_for_entity_processing(
    facts: list[ProcessedFact],
    user_entities_per_content: dict[int, UserEntities] | None = None,
) -> tuple[list[str], list, list[list[dict]]]:
    """
    Extract fact texts, dates, and merged entity lists from ProcessedFact objects.

    Extracted names always carry ``resolve=True`` — they are the extractor's guess at a name, so
    matching them onto the bank's existing entities is the point. Caller-supplied names carry the
    content item's ``resolve_entities`` flag, so a caller can have their own names taken literally
    without turning off resolution for the extractor's (#3479).

    Returns:
        Tuple of (fact_texts, fact_dates, entities_per_fact)
    """
    user_entities_per_content = user_entities_per_content or {}

    fact_texts = [fact.fact_text for fact in facts]
    fact_dates = [fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at for fact in facts]

    entities_per_fact = []
    for fact in facts:
        llm_entities = [{"text": entity.name, "type": "CONCEPT", "resolve": True} for entity in (fact.entities or [])]

        supplied = user_entities_per_content.get(fact.content_index)
        user_entities = supplied.entities if supplied else []
        user_resolve = supplied.resolve if supplied else True

        by_text = {e["text"].lower(): e for e in llm_entities}
        for user_entity in user_entities:
            text_lower = user_entity["text"].lower()
            existing = by_text.get(text_lower)
            if existing is None:
                entity = {
                    "text": user_entity["text"],
                    "type": user_entity.get("type", "CONCEPT"),
                    "resolve": user_resolve,
                }
                llm_entities.append(entity)
                by_text[text_lower] = entity
            else:
                # The extractor produced this name too. The caller still authored it, so their
                # intent wins: a literal name must not become resolvable just because extraction
                # happened to agree on the spelling.
                existing["resolve"] = existing["resolve"] and user_resolve

        entities_per_fact.append(llm_entities)

    return fact_texts, fact_dates, entities_per_fact


async def resolve_entities(
    entity_resolver,
    conn,
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    log_buffer: list[str] = None,
    user_entities_per_content: dict[int, UserEntities] | None = None,
    entity_labels: list | None = None,
) -> EntityResolutionResult:
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
        user_entities_per_content: Dict mapping content_index to the caller-supplied
            entities for that content item and whether to resolve them
        entity_labels: Optional entity label taxonomy

    Returns:
        EntityResolutionResult with the resolved identities and unit mappings.
    """
    if not unit_ids or not facts:
        return EntityResolutionResult(resolved_entities=[], entity_to_unit=[], unit_to_entity_ids={})

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
