"""
Observation utilities for generating entity observations from facts.

Observations are objective facts synthesized from multiple memory facts
about an entity, without personality influence.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .response_models import MemoryFact

logger = logging.getLogger(__name__)


class Observation(BaseModel):
    """An observation about an entity."""
    observation: str = Field(description="The observation text - a factual statement about the entity")


class ObservationExtractionResponse(BaseModel):
    """Response containing extracted observations."""
    observations: List[Observation] = Field(
        default_factory=list,
        description="List of observations about the entity"
    )


def format_facts_for_observation_prompt(facts: List[MemoryFact]) -> str:
    """Format facts as text for observation extraction prompt."""
    import json

    if not facts:
        return "[]"
    formatted = []
    for fact in facts:
        fact_obj = {
            "text": fact.text
        }

        # Add context if available
        if fact.context:
            fact_obj["context"] = fact.context

        # Add occurred_start if available
        if fact.occurred_start:
            fact_obj["occurred_at"] = fact.occurred_start

        formatted.append(fact_obj)

    return json.dumps(formatted, indent=2)


def build_observation_prompt(
    entity_name: str,
    facts_text: str,
) -> str:
    """Build the observation extraction prompt for the LLM."""
    return f"""Based on the following facts about "{entity_name}", generate a list of key observations.

FACTS ABOUT {entity_name.upper()}:
{facts_text}

Your task: Synthesize the facts into clear, objective observations about {entity_name}.

GUIDELINES:
1. Each observation should be a factual statement about {entity_name}
2. Combine related facts into single observations where appropriate
3. Be objective - do not add opinions, judgments, or interpretations
4. Focus on what we KNOW about {entity_name}, not what we assume
5. Include observations about: identity, characteristics, roles, relationships, activities
6. Write in third person (e.g., "John is..." not "I think John is...")
7. If there are conflicting facts, note the most recent or most supported one

EXAMPLES of good observations:
- "John works at Google as a software engineer"
- "John is detail-oriented and methodical in his approach"
- "John collaborates frequently with Sarah on the AI project"
- "John joined the company in 2023"

EXAMPLES of bad observations (avoid these):
- "John seems like a good person" (opinion/judgment)
- "John probably likes his job" (assumption)
- "I believe John is reliable" (first-person opinion)

Generate 3-7 observations based on the available facts. If there are very few facts, generate fewer observations."""


def get_observation_system_message() -> str:
    """Get the system message for observation extraction."""
    return "You are an objective observer synthesizing facts about an entity. Generate clear, factual observations without opinions or personality influence. Be concise and accurate."


async def extract_observations_from_facts(
    llm_config,
    entity_name: str,
    facts: List[MemoryFact]
) -> List[str]:
    """
    Extract observations from facts about an entity using LLM.

    Args:
        llm_config: LLM configuration to use
        entity_name: Name of the entity to generate observations about
        facts: List of facts mentioning the entity

    Returns:
        List of observation strings
    """
    if not facts:
        return []

    facts_text = format_facts_for_observation_prompt(facts)
    prompt = build_observation_prompt(entity_name, facts_text)

    try:
        result = await llm_config.call(
            messages=[
                {"role": "system", "content": get_observation_system_message()},
                {"role": "user", "content": prompt}
            ],
            response_format=ObservationExtractionResponse,
            scope="memory_extract_observation"
        )

        observations = [op.observation for op in result.observations]
        logger.debug(f"Extracted {len(observations)} observations for entity {entity_name}")
        return observations

    except Exception as e:
        logger.warning(f"Failed to extract observations for {entity_name}: {str(e)}")
        return []
