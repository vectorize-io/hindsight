"""Entity-relation extraction for the Graphiti federation forwarder (prototype).

Extends fact extraction with entity-to-entity relations so a single LLM
extraction pass can feed both the local retain pipeline and a shared Graphiti
world graph via ``add_triplet`` (design:
graphify-out/HINDSIGHT_GRAPHITI_DEEPDIVE_B1_C2.md, Part 2).

Key decisions encoded here:

* Relation endpoints are **indexes into the fact's own entities array**, not
  entity names. Validation is an integer bounds check (reuses the
  ``causal_relations.target_index`` anti-hallucination precedent) instead of
  Graphiti's runtime name matching.
* Relation-level temporal fields (``rel_valid_at``/``rel_invalid_at``) are
  decoupled from fact-level ``occurred_*``. Hindsight's "conversation facts
  carry no dates" rule is unchanged; Graphiti's "ongoing relations get the
  reference time" rule applies to relations only.
* ``predicate`` is always English SCREAMING_SNAKE_CASE (a schema vocabulary,
  mergeable across languages); fact text follows the input language.
* Defensive parsing mirrors Graphiti ``edge_operations.py`` — malformed
  relations are dropped with a warning, the fact itself survives.
"""

import logging
import re

from pydantic import BaseModel, Field

from .fact_extraction import ExtractedFact

logger = logging.getLogger(__name__)

# SCREAMING_SNAKE_CASE: starts with a letter, letters/digits/underscores.
_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Inserted into the base extraction prompt (after the ENTITIES section) when
# the bank participates in federation; rendered as "" otherwise — mirroring
# the {retain_mission_section} placeholder mechanism.
RELATIONS_PROMPT_SECTION = """
══════════════════════════════════════════════════════════════════════════
RELATIONS (between entities within ONE fact)
══════════════════════════════════════════════════════════════════════════

For each fact, if two entities in its "entities" array have a clearly stated
relationship, add it to "relations".

R1. source_entity_index / target_entity_index are positions in THIS fact's
    "entities" array (0-based). They MUST be valid indices and MUST differ.
R2. "predicate" is a SCREAMING_SNAKE_CASE English verb phrase: WORKS_AT,
    LIVES_IN, PREFERS, LEADS, MARRIED_TO. ALWAYS English — even when the
    fact text is in another language. The fact text itself stays in the
    input language as required above.
R3. Preserve specificity in entity choice: relate to the concrete entity
    ("Gamecube"), never a generalization ("gaming console").
R4. Temporal bounds — for relations ONLY (fact-level occurred_* rules are
    unchanged):
    - ongoing relationship in present tense → rel_valid_at = Event Date
    - explicit start ("since March 2024")   → rel_valid_at = that date
    - explicit end ("no longer", "quit")    → rel_invalid_at = that date
    - otherwise leave both null. NEVER infer dates from unrelated events.
R5. Only relate entities that BOTH appear in this fact's "entities" array.
    If the natural second endpoint is missing from the array, add it to
    "entities" first (it must satisfy the ENTITIES rules above).
R6. A fact with zero relations is normal. Do not force relations.
R7. CRITICAL: relations must NOT change WHICH facts you extract or how
    "what" is worded. Apply the selectivity rules first, exactly as before;
    only then look for relations inside the facts you already chose.

Example (Event Date: June 10, 2024):
Input: "Alice has been leading the infra team since March. She's married
        to Bob, who works at Google."
Fact 1: what="Alice leads infrastructure team since March 2024",
        entities=["Alice", "infrastructure team"],
        relations=[{source_entity_index: 0, target_entity_index: 1,
                    predicate: "LEADS", rel_valid_at: "2024-03-01T00:00:00Z"}]
Fact 2: what="Alice is married to Bob; Bob works at Google",
        entities=["Alice", "Bob", "Google"],
        relations=[{source_entity_index: 0, target_entity_index: 1,
                    predicate: "MARRIED_TO"},
                   {source_entity_index: 1, target_entity_index: 2,
                    predicate: "WORKS_AT",
                    rel_valid_at: "2024-06-10T00:00:00Z"}]
"""


class ExtractedRelation(BaseModel):
    """A relationship between two entities of the same fact (index references)."""

    source_entity_index: int = Field(
        description="Index into THIS fact's entities array (0-based). "
        "MUST be a valid index and MUST differ from target_entity_index."
    )
    target_entity_index: int = Field(description="Index into THIS fact's entities array (0-based).")
    predicate: str = Field(
        description="Relationship type in SCREAMING_SNAKE_CASE English "
        "(e.g., WORKS_AT, LIVES_IN, PREFERS, LEADS). Always English, "
        "even when the fact text is in another language."
    )
    rel_valid_at: str | None = Field(
        default=None,
        description="ISO 8601: when this relationship became true. "
        "For ongoing relationships stated in present tense, use the Event Date.",
    )
    rel_invalid_at: str | None = Field(
        default=None,
        description="ISO 8601: when this relationship stopped being true (only if an end/change is explicitly stated).",
    )


class ExtractedFactUnified(ExtractedFact):
    """ExtractedFact plus entity relations. Selected when the bank federates
    to a Graphiti world graph (``graphiti_group_id`` set); non-federated banks
    keep using the plain schema unchanged."""

    relations: list[ExtractedRelation] | None = Field(
        default=None,
        description="Relationships between entities listed in THIS fact. "
        "Empty/omitted is normal — only emit when clearly stated.",
    )


class FactExtractionResponseUnified(BaseModel):
    """Response schema for the unified (federation) extraction mode."""

    facts: list[ExtractedFactUnified] = Field(description="List of extracted factual statements")


def normalize_predicate(predicate: str) -> str:
    """Best-effort normalization to SCREAMING_SNAKE_CASE before validation.

    Tolerates the common LLM slips (lowercase, spaces, hyphens) without
    accepting arbitrary garbage — anything that doesn't normalize to the
    strict form is rejected by :func:`validate_relations`.
    """
    return re.sub(r"[\s\-]+", "_", predicate.strip()).upper()


def validate_relations(fact: ExtractedFactUnified) -> list[ExtractedRelation]:
    """Return the fact's structurally valid relations, dropping the rest.

    Defensive parsing (mirrors Graphiti's index-bounds guard in
    ``edge_operations.py``): a malformed relation is logged and dropped, the
    fact itself is never rejected. Checks, per relation:

    * both endpoint indexes are within the fact's entities array,
    * endpoints differ (no self-relations),
    * the predicate normalizes to SCREAMING_SNAKE_CASE.

    Returns relations with their predicate normalized in place.
    """
    if not fact.relations:
        return []

    entity_count = len(fact.entities or [])
    valid: list[ExtractedRelation] = []
    for rel in fact.relations:
        if not (0 <= rel.source_entity_index < entity_count) or not (0 <= rel.target_entity_index < entity_count):
            logger.warning(
                "Dropping relation with out-of-range entity index "
                f"({rel.source_entity_index} -> {rel.target_entity_index}, "
                f"entity_count={entity_count}, predicate={rel.predicate!r})"
            )
            continue
        if rel.source_entity_index == rel.target_entity_index:
            logger.warning(f"Dropping self-relation (index {rel.source_entity_index}, predicate={rel.predicate!r})")
            continue
        predicate = normalize_predicate(rel.predicate)
        if not _PREDICATE_RE.match(predicate):
            logger.warning(f"Dropping relation with invalid predicate {rel.predicate!r}")
            continue
        valid.append(rel.model_copy(update={"predicate": predicate}))
    return valid
