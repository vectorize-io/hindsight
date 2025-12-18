"""
Think operation utilities for formulating answers based on agent and world facts.
"""

import logging
import re
from datetime import datetime

from pydantic import BaseModel, Field

from ..response_models import DispositionTraits, MemoryFact

logger = logging.getLogger(__name__)


class Opinion(BaseModel):
    """An opinion formed by the bank."""

    opinion: str = Field(description="The opinion or perspective with reasoning included")
    confidence: float = Field(description="Confidence score for this opinion (0.0 to 1.0, where 1.0 is very confident)")


class OpinionExtractionResponse(BaseModel):
    """Response containing extracted opinions."""

    opinions: list[Opinion] = Field(
        default_factory=list, description="List of opinions formed with their supporting reasons and confidence scores"
    )


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


def build_think_prompt(
    agent_facts_text: str,
    world_facts_text: str,
    opinion_facts_text: str,
    query: str,
    name: str,
    disposition: DispositionTraits,
    background: str,
    context: str | None = None,
) -> str:
    """Build the think prompt for the LLM."""
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

    return f"""Here's what I know and have experienced:

MY IDENTITY & EXPERIENCES:
{agent_facts_text}

WHAT I KNOW ABOUT THE WORLD:
{world_facts_text}

MY EXISTING OPINIONS & BELIEFS:
{opinion_facts_text}

{context_section}{name_section}{disposition_desc}{background_section}

QUESTION: {query}

Based on everything I know, believe, and who I am (including my name, disposition and background), here's what I genuinely think about this question. I'll draw on my experiences, knowledge, opinions, and personal traits to give you my honest perspective."""


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

    return f"You are a person with your own thoughts, experiences, opinions, and disposition. When asked a question, respond naturally like a human would - share what you genuinely think based on what you know and have experienced. {disposition_instruction} Be direct, express your views confidently, and use 'I think', 'I believe', 'in my view', etc. Respond in plain text without markdown formatting."


async def extract_opinions_from_text(llm_config, text: str, query: str) -> list[Opinion]:
    """
    Extract opinions with reasons and confidence from text using LLM.

    Args:
        llm_config: LLM configuration to use
        text: Text to extract opinions from
        query: The original query that prompted this response

    Returns:
        List of Opinion objects with text and confidence
    """
    extraction_prompt = f"""Extract any NEW opinions or perspectives from the answer below and rewrite them in FIRST-PERSON as if YOU are stating the opinion directly.

ORIGINAL QUESTION:
{query}

ANSWER PROVIDED:
{text}

Your task: Find opinions in the answer and rewrite them AS IF YOU ARE THE ONE SAYING THEM.

An opinion is a judgment, viewpoint, or conclusion that goes beyond just stating facts.

IMPORTANT: Do NOT extract statements like:
- "I don't have enough information"
- "The facts don't contain information about X"
- "I cannot answer because..."

ONLY extract actual opinions about substantive topics.

CRITICAL FORMAT REQUIREMENTS:
1. **ALWAYS start with first-person phrases**: "I think...", "I believe...", "In my view...", "I've come to believe...", "Previously I thought... but now..."
2. **NEVER use third-person**: Do NOT say "The speaker thinks..." or "They believe..." - always use "I"
3. Include the reasoning naturally within the statement
4. Provide a confidence score (0.0 to 1.0)

CORRECT Examples (✓ FIRST-PERSON):
- "I think Alice is more reliable because she consistently delivers on time and writes clean code"
- "Previously I thought all engineers were equal, but now I feel that experience and track record really matter"
- "I believe reliability is best measured by consistent output over time"
- "I've come to believe that track records are more important than potential"

WRONG Examples (✗ THIRD-PERSON - DO NOT USE):
- "The speaker thinks Alice is more reliable"
- "They believe reliability matters"
- "It is believed that Alice is better"

If no genuine opinions are expressed (e.g., the response just says "I don't know"), return an empty list."""

    try:
        result = await llm_config.call(
            messages=[
                {
                    "role": "system",
                    "content": "You are converting opinions from text into first-person statements. Always use 'I think', 'I believe', 'I feel', etc. NEVER use third-person like 'The speaker' or 'They'.",
                },
                {"role": "user", "content": extraction_prompt},
            ],
            response_format=OpinionExtractionResponse,
            scope="memory_extract_opinion",
        )

        # Format opinions with confidence score and convert to first-person
        formatted_opinions = []
        for op in result.opinions:
            # Convert third-person to first-person if needed
            opinion_text = op.opinion

            # Replace common third-person patterns with first-person
            def singularize_verb(verb):
                if verb.endswith("es"):
                    return verb[:-1]  # believes -> believe
                elif verb.endswith("s"):
                    return verb[:-1]  # thinks -> think
                return verb

            # Pattern: "The speaker/user [verb]..." -> "I [verb]..."
            match = re.match(
                r"^(The speaker|The user|They|It is believed) (believes?|thinks?|feels?|says|asserts?|considers?)(\s+that)?(.*)$",
                opinion_text,
                re.IGNORECASE,
            )
            if match:
                verb = singularize_verb(match.group(2))
                that_part = match.group(3) or ""  # Keep " that" if present
                rest = match.group(4)
                opinion_text = f"I {verb}{that_part}{rest}"

            # If still doesn't start with first-person, prepend "I believe that "
            first_person_starters = [
                "I think",
                "I believe",
                "I feel",
                "In my view",
                "I've come to believe",
                "Previously I",
            ]
            if not any(opinion_text.startswith(starter) for starter in first_person_starters):
                opinion_text = "I believe that " + opinion_text[0].lower() + opinion_text[1:]

            formatted_opinions.append(Opinion(opinion=opinion_text, confidence=op.confidence))

        return formatted_opinions

    except Exception as e:
        logger.warning(f"Failed to extract opinions: {str(e)}")
        return []


async def reflect(
    llm_config,
    query: str,
    experience_facts: list[str] = None,
    world_facts: list[str] = None,
    opinion_facts: list[str] = None,
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
        opinion_facts: List of opinion fact strings
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
    opinion_results = to_memory_facts(opinion_facts or [], "opinion")

    # Format facts for prompt
    agent_facts_text = format_facts_for_prompt(agent_results)
    world_facts_text = format_facts_for_prompt(world_results)
    opinion_facts_text = format_facts_for_prompt(opinion_results)

    # Build prompt
    prompt = build_think_prompt(
        agent_facts_text=agent_facts_text,
        world_facts_text=world_facts_text,
        opinion_facts_text=opinion_facts_text,
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
