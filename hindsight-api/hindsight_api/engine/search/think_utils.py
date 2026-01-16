"""
Think operation utilities for formulating answers based on agent and world facts.
"""

import logging
from datetime import datetime

from ..response_models import DispositionTraits, MemoryFact

logger = logging.getLogger(__name__)


def describe_trait_level(value: int) -> str:
    """Convert trait value (1-5) to descriptive text."""
    levels = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
    return levels.get(value, "moderate")


def build_disposition_description(disposition: DispositionTraits) -> str:
    """Build a disposition description string from disposition traits."""
    skepticism_desc = {
        1: "You are very trusting and tend to take information at face value.",
        2: "You tend to trust information but may question obvious inconsistencies.",
        3: "You have a balanced approach to information, neither too trusting nor too skeptical.",
        4: "You are somewhat skeptical and often question the reliability of information.",
        5: "You are highly skeptical and critically examine all information for accuracy and hidden motives.",
    }

    literalism_desc = {
        1: "You interpret information very flexibly, reading between the lines and inferring intent.",
        2: "You tend to consider context and implied meaning alongside literal statements.",
        3: "You balance literal interpretation with contextual understanding.",
        4: "You prefer to interpret information more literally and precisely.",
        5: "You interpret information very literally and focus on exact wording and commitments.",
    }

    empathy_desc = {
        1: "You focus primarily on facts and data, setting aside emotional context.",
        2: "You consider facts first but acknowledge emotional factors exist.",
        3: "You balance factual analysis with emotional understanding.",
        4: "You give significant weight to emotional context and human factors.",
        5: "You strongly consider the emotional state and circumstances of others when forming memories.",
    }

    return f"""Your disposition traits:
- Skepticism ({describe_trait_level(disposition.skepticism)}): {skepticism_desc.get(disposition.skepticism, skepticism_desc[3])}
- Literalism ({describe_trait_level(disposition.literalism)}): {literalism_desc.get(disposition.literalism, literalism_desc[3])}
- Empathy ({describe_trait_level(disposition.empathy)}): {empathy_desc.get(disposition.empathy, empathy_desc[3])}"""


def format_facts_for_prompt(facts: list[MemoryFact]) -> str:
    """Format facts as JSON for LLM prompt."""
    import json

    if not facts:
        return "[]"
    formatted = []
    for fact in facts:
        fact_obj = {"text": fact.text}

        # Add context if available
        if fact.context:
            fact_obj["context"] = fact.context

        # Add occurred_start if available (when the fact occurred)
        if fact.occurred_start:
            occurred_start = fact.occurred_start
            if isinstance(occurred_start, str):
                fact_obj["occurred_start"] = occurred_start
            elif isinstance(occurred_start, datetime):
                fact_obj["occurred_start"] = occurred_start.strftime("%Y-%m-%d %H:%M:%S")

        formatted.append(fact_obj)

    return json.dumps(formatted, indent=2)


def format_entity_summaries_for_prompt(entities: dict) -> str:
    """Format entity summaries for inclusion in the reflect prompt.

    Args:
        entities: Dict mapping entity name to EntityState objects

    Returns:
        Formatted string with entity summaries, or empty string if no summaries
    """
    if not entities:
        return ""

    summaries = []
    for name, state in entities.items():
        # Get summary from observations (summary is stored as single observation)
        if state.observations:
            summary_text = state.observations[0].text
            summaries.append(f"## {name}\n{summary_text}")

    if not summaries:
        return ""

    return "\n\n".join(summaries)


def build_think_prompt(
    agent_facts_text: str,
    world_facts_text: str,
    query: str,
    name: str,
    disposition: DispositionTraits,
    background: str,
    context: str | None = None,
    entity_summaries_text: str | None = None,
) -> str:
    """Build the think prompt for the LLM.

    Note: opinion_facts_text parameter removed - opinions are now stored as mental models
    and included via entity_summaries_text.
    """
    disposition_desc = build_disposition_description(disposition)

    name_section = f"""

Your name: {name}
"""

    background_section = ""
    if background:
        background_section = f"""

Your background:
{background}
"""

    context_section = ""
    if context:
        context_section = f"""
ADDITIONAL CONTEXT:
{context}

"""

    entity_section = ""
    if entity_summaries_text:
        entity_section = f"""
KEY PEOPLE, PLACES & THINGS I KNOW ABOUT:
{entity_summaries_text}

"""

    return f"""Here's what I know and have experienced:

MY IDENTITY & EXPERIENCES:
{agent_facts_text}

WHAT I KNOW ABOUT THE WORLD:
{world_facts_text}

{entity_section}{context_section}{name_section}{disposition_desc}{background_section}

QUESTION: {query}

Based on everything I know, believe, and who I am (including my name, disposition and background), here's what I genuinely think about this question. I'll draw on my experiences, knowledge, and personal traits to give you my honest perspective."""


def get_system_message(disposition: DispositionTraits) -> str:
    """Get the system message for the think LLM call."""
    # Build disposition-specific instructions based on trait values
    instructions = []

    # Skepticism influences how much to question/doubt information
    if disposition.skepticism >= 4:
        instructions.append("Be skeptical of claims and look for potential issues or inconsistencies.")
    elif disposition.skepticism <= 2:
        instructions.append("Trust the information provided and take statements at face value.")

    # Literalism influences interpretation style
    if disposition.literalism >= 4:
        instructions.append("Interpret information literally and focus on exact commitments and wording.")
    elif disposition.literalism <= 2:
        instructions.append("Read between the lines and consider implied meaning and context.")

    # Empathy influences consideration of emotional factors
    if disposition.empathy >= 4:
        instructions.append("Consider the emotional state and circumstances behind the information.")
    elif disposition.empathy <= 2:
        instructions.append("Focus on facts and outcomes rather than emotional context.")

    disposition_instruction = (
        " ".join(instructions) if instructions else "Balance your disposition traits when interpreting information."
    )

    return f"You are a person with your own thoughts, experiences, opinions, and disposition. When asked a question, respond naturally like a human would - share what you genuinely think based on what you know and have experienced. {disposition_instruction} Be direct, express your views confidently, and use 'I think', 'I believe', 'in my view', etc. Respond in plain text without markdown formatting. IMPORTANT: Detect the language of the question and respond in the SAME language. Do not translate to English if the question is in another language."


async def reflect(
    llm_config,
    query: str,
    experience_facts: list[str] = None,
    world_facts: list[str] = None,
    name: str = "Assistant",
    disposition: DispositionTraits = None,
    background: str = "",
    context: str = None,
) -> str:
    """
    Standalone reflect function for generating answers based on facts.

    This is a static version of the reflect operation that can be called
    without a MemoryEngine instance, useful for testing.

    Args:
        llm_config: LLM provider instance
        query: Question to answer
        experience_facts: List of experience/agent fact strings
        world_facts: List of world fact strings
        name: Name of the agent/persona
        disposition: Disposition traits (defaults to neutral)
        background: Background information
        context: Additional context for the prompt

    Returns:
        Generated answer text
    """
    # Default disposition if not provided
    if disposition is None:
        disposition = DispositionTraits(skepticism=3, literalism=3, empathy=3)

    # Convert string lists to MemoryFact format for formatting
    def to_memory_facts(facts: list[str], fact_type: str) -> list[MemoryFact]:
        if not facts:
            return []
        return [MemoryFact(id=f"test-{i}", text=f, fact_type=fact_type) for i, f in enumerate(facts)]

    agent_results = to_memory_facts(experience_facts or [], "experience")
    world_results = to_memory_facts(world_facts or [], "world")

    # Format facts for prompt
    agent_facts_text = format_facts_for_prompt(agent_results)
    world_facts_text = format_facts_for_prompt(world_results)

    # Build prompt
    prompt = build_think_prompt(
        agent_facts_text=agent_facts_text,
        world_facts_text=world_facts_text,
        query=query,
        name=name,
        disposition=disposition,
        background=background,
        context=context,
    )

    system_message = get_system_message(disposition)

    # Call LLM
    answer_text = await llm_config.call(
        messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
        scope="memory_think",
        temperature=0.9,
        max_completion_tokens=1000,
    )

    return answer_text.strip()
