"""
Structural mental model derivation from bank goal.

Structural models are derived from the bank's goal - they represent what
any agent with this role would need to track. For example:

Goal: "Be a PM for engineering team"
Structural models:
  - Team Structure (who's on the team, roles)
  - Project Overview (current projects, status)
  - Processes (how releases work, how decisions are made)
  - Key Systems (what we own, dependencies)
"""

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .models import MentalModelType, StructuralModelTemplate

if TYPE_CHECKING:
    from ..llm_wrapper import LLMConfig

logger = logging.getLogger(__name__)


class StructuralDerivationResponse(BaseModel):
    """Response from LLM for structural model derivation."""

    templates: list[StructuralModelTemplate] = Field(description="Structural model templates derived from the goal")


class StructuralRelevanceResult(BaseModel):
    """Result of evaluating a structural model's relevance to the goal."""

    name: str
    relevant: bool
    reason: str


class StructuralRelevanceResponse(BaseModel):
    """Response from LLM for structural model relevance evaluation."""

    models: list[StructuralRelevanceResult] = Field(description="Relevance evaluation for each model")


def build_structural_derivation_prompt(goal: str, existing_models: list[dict] | None = None) -> str:
    """Build the prompt for deriving structural models from a goal."""
    existing_section = ""
    if existing_models:
        model_list = "\n".join([f"- {m['name']}: {m['description']}" for m in existing_models])
        existing_section = f"""
EXISTING STRUCTURAL MODELS:
{model_list}

Review these existing models. Include them in your output ONLY if they are still relevant.
Models not included in your output will be REMOVED.
"""

    return f"""Given this agent goal, identify the KEY THINGS to track to achieve it.

GOAL: {goal}
{existing_section}
IMPORTANT CONSTRAINTS:
- Return 0-3 structural models MAXIMUM (less is better!)
- Only include models for SPECIFIC, CONCRETE things the agent needs to track
- Each model must be DIRECTLY tied to achieving the goal
- If the goal is simple, return 0 models (empty array is fine)
- If existing models are provided, only include ones that are still relevant

GOOD examples (specific, actionable):
- Goal: "Be a PM for engineering team" → "Team Members" (track who's on the team)
- Goal: "Track customer feedback" → "Customer Issues" (track specific complaints/requests)
- Goal: "Manage project X" → "Project X Milestones" (track progress)

BAD examples (too generic, don't create these):
- "Processes", "Workflows", "Key Systems", "Important Events"
- "Communication", "Collaboration", "Progress", "Status"
- Generic role-based models not tied to the specific goal

For each model:
1. type: "entity" (people/orgs), "concept" (processes), or "event" (occurrences)
2. name: Short, specific name (e.g., "Team Members", "Sprint Goals")
3. description: One line describing what to track
4. initial_probes: 2-3 search queries to find relevant information

Return ONLY the models that should exist. Existing models not in your output will be deleted."""


def get_structural_derivation_system_message() -> str:
    """System message for structural model derivation."""
    return """You identify the key things to track for a goal. Be VERY selective.

Rules:
- Maximum 3 models (prefer fewer)
- Only SPECIFIC, CONCRETE things - not generic categories
- Each must DIRECTLY help achieve the goal
- Empty array is valid if no models are truly needed
- If existing models are shown, only include ones worth keeping

Output JSON with 'templates' array (can be empty)."""


async def derive_structural_models(
    llm_config: "LLMConfig",
    goal: str,
    existing_models: list[dict] | None = None,
) -> tuple[list[StructuralModelTemplate], list[str]]:
    """
    Derive structural model templates from a bank's goal.

    This combines derivation and evaluation in one call. The LLM sees existing
    models and decides which to keep. Any existing model not in the output
    will be marked for removal.

    Args:
        llm_config: LLM configuration for calling the model
        goal: The bank's goal (e.g., "Be a PM for engineering team")
        existing_models: Optional list of existing model dicts with 'name', 'description', 'id'

    Returns:
        Tuple of (templates to create/keep, IDs of existing models to remove)

    Raises:
        Exception: If LLM call fails
    """
    prompt = build_structural_derivation_prompt(goal, existing_models)

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

    # Generate stable IDs for templates
    for template in templates:
        if not template.id:
            # Generate ID from name (lowercase, hyphenated)
            template.id = template.name.lower().replace(" ", "-").replace("_", "-")

    # Find existing models to remove (not in LLM output)
    models_to_remove = []
    if existing_models:
        returned_names = {t.name for t in templates}
        for model in existing_models:
            if model["name"] not in returned_names:
                logger.info(f"[STRUCTURAL] Marking '{model['name']}' for removal (not in LLM output)")
                models_to_remove.append(model["id"])

    if models_to_remove:
        logger.info(f"[STRUCTURAL] {len(models_to_remove)} existing models will be removed")

    return templates, models_to_remove
