"""
Structural mental model derivation from bank mission.

Structural models are derived from the bank's mission - they represent what
any agent with this role would need to track. For example:

Mission: "Be a PM for engineering team"
Structural models:
  - Team Structure (who's on the team, roles)
  - Project Overview (current projects, status)
  - Processes (how releases work, how decisions are made)
  - Key Systems (what we own, dependencies)
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .models import StructuralModelTemplate

if TYPE_CHECKING:
    from ..llm_wrapper import LLMConfig

logger = logging.getLogger(__name__)


class StructuralDerivationResponse(BaseModel):
    """Response from LLM for structural model derivation."""

    templates: list[StructuralModelTemplate] = Field(description="Structural model templates derived from the mission")


class StructuralRelevanceResult(BaseModel):
    """Result of evaluating a structural model's relevance to the mission."""

    name: str
    relevant: bool
    reason: str


class StructuralRelevanceResponse(BaseModel):
    """Response from LLM for structural model relevance evaluation."""

    models: list[StructuralRelevanceResult] = Field(description="Relevance evaluation for each model")


def build_structural_derivation_prompt(mission: str, existing_models: list[dict] | None = None) -> str:
    """Build the prompt for deriving structural models from a mission."""
    existing_section = ""
    if existing_models:
        model_list = "\n".join([f"- id='{m['id']}' name='{m['name']}': {m['description']}" for m in existing_models])
        existing_section = f"""
EXISTING STRUCTURAL MODELS:
{model_list}

IMPORTANT: If keeping an existing model, you MUST return its EXACT 'id' value.
Models not included in your output will be REMOVED.
"""

    return f"""Given this agent mission, identify the KEY THINGS to track to achieve it.

MISSION: {mission}
{existing_section}
IMPORTANT CONSTRAINTS:
- Return 0-3 structural models MAXIMUM (less is better!)
- Only include models for SPECIFIC, CONCRETE things the agent needs to track
- Each model must be DIRECTLY tied to achieving the mission
- If the mission is simple, return 0 models (empty array is fine)
- If existing models are provided and you want to keep one, use its EXACT id
- Do NOT create near-duplicates (e.g., don't create "topic-map" if "topic-connections" exists)

GOOD examples (specific, actionable):
- Mission: "Be a PM for engineering team" → "Team Members" (track who's on the team)
- Mission: "Track customer feedback" → "Customer Issues" (track specific complaints/requests)
- Mission: "Manage project X" → "Project X Milestones" (track progress)

BAD examples (too generic, don't create these):
- "Processes", "Workflows", "Key Systems", "Important Events"
- "Communication", "Collaboration", "Progress", "Status"
- Generic role-based models not tied to the specific mission

For each model:
1. id: Use EXACT existing id if keeping a model, or leave empty for new models
2. name: Short, specific name (e.g., "Team Members", "Sprint Goals")
3. description: One line describing what to track
4. initial_probes: 2-3 search queries to find relevant information

Return ONLY the models that should exist. Existing models not in your output will be deleted."""


def get_structural_derivation_system_message() -> str:
    """System message for structural model derivation."""
    return """You identify the key things to track for a mission. Be VERY selective.

Rules:
- Maximum 3 models (prefer fewer)
- Only SPECIFIC, CONCRETE things - not generic categories
- Each must DIRECTLY help achieve the mission
- Empty array is valid if no models are truly needed
- If existing models are shown and you want to keep one, return its EXACT id
- Never create duplicates - if a similar model exists, keep the existing one

Output JSON with 'templates' array (can be empty)."""


def _normalize_id(text: str) -> str:
    """Normalize a string to a canonical form for comparison.

    Removes common suffixes, pluralization, and normalizes separators.
    """
    # Lowercase and normalize separators
    normalized = text.lower().replace(" ", "-").replace("_", "-")

    # Remove common suffixes that indicate the same concept
    suffixes_to_remove = ["-map", "-list", "-overview", "-tracker", "-s"]
    for suffix in suffixes_to_remove:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]

    return normalized


def _find_similar_existing_id(new_id: str, existing_models: list[dict]) -> str | None:
    """Find an existing model ID that is similar to the new ID.

    Returns the existing ID if a similar one is found, None otherwise.
    """
    if not existing_models:
        return None

    new_normalized = _normalize_id(new_id)

    for model in existing_models:
        existing_id = model.get("id", "")
        existing_normalized = _normalize_id(existing_id)

        # Check if one is a prefix of the other (normalized)
        if new_normalized.startswith(existing_normalized) or existing_normalized.startswith(new_normalized):
            return existing_id

        # Check if they're the same when normalized
        if new_normalized == existing_normalized:
            return existing_id

    return None


async def derive_structural_models(
    llm_config: "LLMConfig",
    mission: str,
    existing_models: list[dict] | None = None,
) -> tuple[list[StructuralModelTemplate], list[str]]:
    """
    Derive structural model templates from a bank's mission.

    This combines derivation and evaluation in one call. The LLM sees existing
    models and decides which to keep. Any existing model not in the output
    will be marked for removal.

    Args:
        llm_config: LLM configuration for calling the model
        mission: The bank's mission (e.g., "Be a PM for engineering team")
        existing_models: Optional list of existing model dicts with 'name', 'description', 'id'

    Returns:
        Tuple of (templates to create/keep, IDs of existing models to remove)

    Raises:
        Exception: If LLM call fails
    """
    prompt = build_structural_derivation_prompt(mission, existing_models)

    result = await llm_config.call(
        messages=[
            {"role": "system", "content": get_structural_derivation_system_message()},
            {"role": "user", "content": prompt},
        ],
        response_format=StructuralDerivationResponse,
        scope="mental_model_structural_derivation",
    )

    templates = result.templates
    logger.info(f"[STRUCTURAL] LLM returned {len(templates)} structural models")

    # Build set of existing IDs for quick lookup
    existing_ids = {m["id"] for m in existing_models} if existing_models else set()

    # Process templates: validate IDs, deduplicate, assign stable IDs
    processed_templates: list[StructuralModelTemplate] = []
    kept_existing_ids: set[str] = set()

    for template in templates:
        # If LLM returned an ID, check if it's a valid existing ID
        if template.id and template.id in existing_ids:
            # LLM is keeping an existing model
            kept_existing_ids.add(template.id)
            processed_templates.append(template)
            logger.info(f"[STRUCTURAL] Keeping existing model: {template.id}")
        else:
            # New model or LLM didn't return a valid ID
            # Generate ID from name
            generated_id = template.name.lower().replace(" ", "-").replace("_", "-")

            # Check for similar existing models to prevent near-duplicates
            similar_id = _find_similar_existing_id(generated_id, existing_models)
            if similar_id and similar_id not in kept_existing_ids:
                # Use the existing similar model instead of creating a new one
                logger.info(f"[STRUCTURAL] Detected near-duplicate: '{generated_id}' matches existing '{similar_id}'")
                template.id = similar_id
                kept_existing_ids.add(similar_id)
            else:
                template.id = generated_id

            processed_templates.append(template)

    # Find existing models to remove (not kept in LLM output)
    models_to_remove = []
    if existing_models:
        for model in existing_models:
            if model["id"] not in kept_existing_ids:
                logger.info(f"[STRUCTURAL] Marking '{model['name']}' (id={model['id']}) for removal")
                models_to_remove.append(model["id"])

    if models_to_remove:
        logger.info(f"[STRUCTURAL] {len(models_to_remove)} existing models will be removed")

    return processed_templates, models_to_remove
