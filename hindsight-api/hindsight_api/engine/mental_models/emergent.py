"""
Emergent mental model detection and promotion.

Emergent models are discovered from data patterns:
- Named entity extraction (people, projects, systems)
- Temporal clustering (events with multiple references)
- Causal patterns ("Because X, we do Y")
- Behavioral anchors ("After X, we started Y")
- Reference frequency (anything mentioned repeatedly)

When a pattern is detected, it goes through a mission filter to check relevance,
and if relevant, is promoted to a mental model.
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .models import EmergentCandidate

if TYPE_CHECKING:
    from ..llm_wrapper import LLMConfig

logger = logging.getLogger(__name__)


class MissionFilterCandidate(BaseModel):
    """Result of mission filtering for a single candidate."""

    name: str
    promote: bool = Field(description="True if this is a specific named entity worth tracking")
    reason: str = Field(description="Brief explanation for the decision")


class MissionFilterResponse(BaseModel):
    """Response from LLM for mission filtering."""

    candidates: list[MissionFilterCandidate] = Field(description="Filtering decision for each candidate")


def build_mission_filter_prompt(mission: str, candidates: list[EmergentCandidate]) -> str:
    """Build the prompt for filtering candidates by mission relevance."""
    candidate_list = "\n".join(
        [f"- {c.name} (mentions: {c.mention_count}, method: {c.detection_method})" for c in candidates]
    )

    return f"""Filter these detected entities. For each one, decide: promote=true or promote=false.

MISSION: {mission}

DETECTED ENTITIES:
{candidate_list}

=== DECISION RULES ===

Set promote=true ONLY for specific, named entities:
- Person names: "John", "Maria", "Alice Chen", "Dr. Smith"
- Named organizations: "Google", "Acme Corp", "Frontend Team"
- Named places: "Central Park Zoo", "NYC Office", "Building A"
- Named projects: "Project Phoenix", "Auth Service v2"

Set promote=false for EVERYTHING ELSE, including:
- Common English words: user, support, help, family, kids, parents, friends, people, team, photo, nature, park, office, home, work, school, joy, love, hope, fear, anger, gratitude, kindness, passion, motivation, inspiration, encouragement, positivity, energy, community, connection, commitment, collaboration, growth, impact, difference, success, progress, change, education, volunteering, veterans, homeless, shelter, meeting, project, system, process, event
- Generic categories (even capitalized): Users, Customers, Team, Family, Kids, Veterans, Community
- Abstract concepts: motivation, inspiration, gratitude, commitment, resilience

THE TEST: Is this a specific name you'd find in a contact list or org chart?
- "John" → YES (promote=true)
- "kids" → NO (promote=false)
- "community" → NO (promote=false)
- "Maria" → YES (promote=true)
- "park" → NO (promote=false)

When in doubt, set promote=false."""


def get_mission_filter_system_message() -> str:
    """System message for mission filtering."""
    return """You filter entities for promotion. Output JSON with 'candidates' array.

Rules:
- promote=true ONLY for specific names (people, organizations, named places/projects)
- promote=false for common words, generic categories, abstract concepts

Examples:
- "John" → promote=true (person name)
- "kids" → promote=false (generic category)
- "community" → promote=false (abstract concept)
- "Google" → promote=true (organization name)
- "motivation" → promote=false (abstract concept)

When in doubt, promote=false. Most entities should be rejected."""


async def filter_candidates_by_mission(
    llm_config: "LLMConfig",
    mission: str,
    candidates: list[EmergentCandidate],
) -> list[EmergentCandidate]:
    """
    Filter emergent candidates to keep only specific, named entities.

    Args:
        llm_config: LLM configuration
        mission: The bank's mission (used for context)
        candidates: List of detected candidates

    Returns:
        Filtered list of candidates that are specific named entities
    """
    if not candidates:
        return []

    if not mission:
        # No mission = no filtering, keep all candidates
        logger.debug("[EMERGENT] No mission set, skipping filter")
        return candidates

    prompt = build_mission_filter_prompt(mission, candidates)

    try:
        result = await llm_config.call(
            messages=[
                {"role": "system", "content": get_mission_filter_system_message()},
                {"role": "user", "content": prompt},
            ],
            response_format=MissionFilterResponse,
            scope="mental_model_mission_filter",
        )

        # Build name -> promote map
        promote_map = {c.name: c.promote for c in result.candidates}

        # Filter candidates
        filtered = []
        for candidate in candidates:
            if candidate.name in promote_map:
                if promote_map[candidate.name]:
                    filtered.append(candidate)
                    logger.debug(f"[EMERGENT] Promoting '{candidate.name}'")
                else:
                    logger.debug(f"[EMERGENT] Rejecting '{candidate.name}'")
            else:
                # Candidate not in response - reject by default
                logger.debug(f"[EMERGENT] '{candidate.name}' not in response, rejecting")

        logger.info(f"[EMERGENT] Mission filter: {len(filtered)}/{len(candidates)} candidates promoted")
        return filtered

    except Exception as e:
        logger.warning(f"[EMERGENT] Mission filter failed, rejecting all candidates: {e}")
        return []


async def evaluate_emergent_models(
    llm_config: "LLMConfig",
    models: list[dict],
) -> list[str]:
    """
    Evaluate existing emergent models to check if they should be kept.

    This re-evaluates emergent models using the same filtering criteria
    as new candidates. Models that are generic/abstract will be removed.

    Args:
        llm_config: LLM configuration
        models: List of existing emergent model dicts with 'name', 'id'

    Returns:
        List of model IDs that should be REMOVED (no longer valid)
    """
    if not models:
        return []

    # Convert existing models to candidates for evaluation
    candidates = [
        EmergentCandidate(
            name=m["name"],
            detection_method="existing_emergent_model",
            mention_count=0,
        )
        for m in models
    ]

    # Build a simple prompt for re-evaluation
    names_list = "\n".join([f"- {m['name']}" for m in models])
    prompt = f"""Re-evaluate these existing mental models. For each one, decide: promote=true (keep) or promote=false (remove).

EXISTING MODELS:
{names_list}

=== DECISION RULES ===

Set promote=true ONLY for specific, named entities:
- Person names: "John", "Maria", "Alice Chen", "Dr. Smith"
- Named organizations: "Google", "Acme Corp", "Frontend Team"
- Named places: "Central Park Zoo", "NYC Office", "Building A"
- Named projects: "Project Phoenix", "Auth Service v2"

Set promote=false for EVERYTHING ELSE, including:
- Common English words: user, support, help, family, kids, parents, friends, people, team, photo, nature, park, office, home, work, school, joy, love, hope, fear, anger, gratitude, kindness, passion, motivation, inspiration, encouragement, positivity, energy, community, connection, commitment, collaboration, growth, impact, difference, success, progress, change, education, volunteering, veterans, homeless, shelter, meeting, project, system, process, event
- Generic categories (even capitalized): Users, Customers, Team, Family, Kids, Veterans, Community
- Abstract concepts: motivation, inspiration, gratitude, commitment, resilience

THE TEST: Is this a specific name you'd find in a contact list or org chart?
- "John" → YES (promote=true)
- "kids" → NO (promote=false)
- "community" → NO (promote=false)

When in doubt, set promote=false."""

    try:
        result = await llm_config.call(
            messages=[
                {"role": "system", "content": get_mission_filter_system_message()},
                {"role": "user", "content": prompt},
            ],
            response_format=MissionFilterResponse,
            scope="mental_model_emergent_evaluation",
        )

        # Build name -> promote map
        promote_map = {c.name: c.promote for c in result.candidates}

        # Find models to remove
        models_to_remove = []
        for model in models:
            name = model["name"]
            if name in promote_map:
                if not promote_map[name]:
                    models_to_remove.append(model["id"])
                else:
                    logger.debug(f"[EMERGENT] Keeping '{name}'")
            else:
                # Model not in response - remove to be safe
                logger.info(f"[EMERGENT] '{name}' not in evaluation response, marking for removal")
                models_to_remove.append(model["id"])

        logger.info(f"[EMERGENT] Evaluation: {len(models_to_remove)}/{len(models)} emergent models marked for removal")
        return models_to_remove

    except Exception as e:
        logger.warning(f"[EMERGENT] Evaluation failed, keeping all models: {e}")
        return []


async def detect_entity_candidates(
    pool,
    bank_id: str,
    min_mentions: int = 5,
    top_percent: int = 20,
) -> list[EmergentCandidate]:
    """
    Detect entities that are candidates for promotion to mental models.

    Args:
        pool: Database connection pool
        bank_id: Bank identifier
        min_mentions: Minimum mention count to consider
        top_percent: Only consider top X% by mention count

    Returns:
        List of entity candidates
    """
    from ..db_utils import acquire_with_retry
    from ..memory_engine import fq_table

    candidates = []

    async with acquire_with_retry(pool) as conn:
        # Get entities that meet criteria and don't already have mental models
        rows = await conn.fetch(
            f"""
            WITH ranked AS (
                SELECT
                    e.id,
                    e.canonical_name,
                    e.mention_count,
                    PERCENT_RANK() OVER (ORDER BY e.mention_count DESC) as rank_pct
                FROM {fq_table("entities")} e
                LEFT JOIN {fq_table("mental_models")} mm
                    ON mm.entity_id = e.id AND mm.bank_id = e.bank_id
                WHERE e.bank_id = $1
                  AND e.mention_count >= $2
                  AND mm.id IS NULL  -- Not already a mental model
            )
            SELECT id, canonical_name, mention_count
            FROM ranked
            WHERE rank_pct <= $3
            ORDER BY mention_count DESC
            LIMIT 50
            """,
            bank_id,
            min_mentions,
            top_percent / 100.0,
        )

        for row in rows:
            candidates.append(
                EmergentCandidate(
                    name=row["canonical_name"],
                    detection_method="named_entity_extraction",
                    mention_count=row["mention_count"],
                    entity_id=str(row["id"]),
                    relevance_score=0.0,
                )
            )

    logger.debug(f"[EMERGENT] Detected {len(candidates)} entity candidates")
    return candidates
