"""
Think operation utilities for formulating answers based on agent and world facts.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from .response_models import ThinkResult, MemoryFact

logger = logging.getLogger(__name__)


class Opinion(BaseModel):
    """An opinion formed by the agent."""
    opinion: str = Field(description="The opinion or perspective with reasoning included")
    confidence: float = Field(description="Confidence score for this opinion (0.0 to 1.0, where 1.0 is very confident)")


class OpinionExtractionResponse(BaseModel):
    """Response containing extracted opinions."""
    opinions: List[Opinion] = Field(
        default_factory=list,
        description="List of opinions formed with their supporting reasons and confidence scores"
    )


def describe_trait(name: str, value: float) -> str:
    """Convert trait value to descriptive text."""
    if value >= 0.8:
        return f"very high {name}"
    elif value >= 0.6:
        return f"high {name}"
    elif value >= 0.4:
        return f"moderate {name}"
    elif value >= 0.2:
        return f"low {name}"
    else:
        return f"very low {name}"


def build_personality_description(personality: Dict) -> str:
    """Build a personality description string from personality traits."""
    return f"""Your personality traits:
- {describe_trait('openness to new ideas', personality['openness'])}
- {describe_trait('conscientiousness and organization', personality['conscientiousness'])}
- {describe_trait('extraversion and sociability', personality['extraversion'])}
- {describe_trait('agreeableness and cooperation', personality['agreeableness'])}
- {describe_trait('emotional sensitivity', personality['neuroticism'])}

Personality influence strength: {int(personality['bias_strength'] * 100)}% (how much your personality shapes your opinions)"""


def format_facts_for_prompt(facts: List[MemoryFact]) -> str:
    """Format facts as JSON for LLM prompt."""
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

        # Add occurred_start if available (when the fact occurred)
        if fact.occurred_start:
            occurred_start = fact.occurred_start
            if isinstance(occurred_start, str):
                fact_obj["occurred_start"] = occurred_start
            elif isinstance(occurred_start, datetime):
                fact_obj["occurred_start"] = occurred_start.strftime('%Y-%m-%d %H:%M:%S')

        # Add activation if available
        if fact.activation is not None:
            fact_obj["score"] = fact.activation

        formatted.append(fact_obj)

    return json.dumps(formatted, indent=2)


def build_think_prompt(
    agent_facts_text: str,
    world_facts_text: str,
    opinion_facts_text: str,
    query: str,
    name: str,
    personality: Dict,
    background: str,
    context: str = None,
) -> str:
    """Build the think prompt for the LLM."""
    personality_desc = build_personality_description(personality)

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

{context_section}{name_section}{personality_desc}{background_section}

QUESTION: {query}

Based on everything I know, believe, and who I am (including my name, personality and background), here's what I genuinely think about this question. I'll draw on my experiences, knowledge, opinions, and personal traits to give you my honest perspective."""


def get_system_message(personality: Dict) -> str:
    """Get the system message for the think LLM call."""
    bias_strength = personality['bias_strength']
    if bias_strength >= 0.7:
        personality_instruction = "Your personality strongly influences your thinking. Let your traits guide how you interpret facts and form opinions. Don't be afraid to be biased based on your personality."
    elif bias_strength >= 0.4:
        personality_instruction = "Your personality moderately influences your thinking. Balance your personal traits with objective analysis."
    else:
        personality_instruction = "Your personality has minimal influence on your thinking. Focus primarily on facts while keeping your traits in mind."

    return f"You are a person with your own thoughts, experiences, opinions, and personality. When asked a question, respond naturally like a human would - share what you genuinely think based on what you know and have experienced. {personality_instruction} Be direct, express your views confidently, and use 'I think', 'I believe', 'in my view', etc. Respond in plain text without markdown formatting."


async def extract_opinions_from_text(
    llm_config,
    text: str,
    query: str
) -> List[Dict[str, Any]]:
    """
    Extract opinions with reasons and confidence from text using LLM.

    Args:
        llm_config: LLM configuration to use
        text: Text to extract opinions from
        query: The original query that prompted this response

    Returns:
        List of dicts with keys: 'text' (opinion with reasons), 'confidence' (score 0-1)
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
                {"role": "system", "content": "You are converting opinions from text into first-person statements. Always use 'I think', 'I believe', 'I feel', etc. NEVER use third-person like 'The speaker' or 'They'."},
                {"role": "user", "content": extraction_prompt}
            ],
            response_format=OpinionExtractionResponse,
            scope="memory_extract_opinion"
        )

        # Format opinions with confidence score and convert to first-person
        formatted_opinions = []
        for op in result.opinions:
            # Convert third-person to first-person if needed
            opinion_text = op.opinion

            # Replace common third-person patterns with first-person
            def singularize_verb(verb):
                if verb.endswith('es'):
                    return verb[:-1]  # believes -> believe
                elif verb.endswith('s'):
                    return verb[:-1]  # thinks -> think
                return verb

            # Pattern: "The speaker/user [verb]..." -> "I [verb]..."
            match = re.match(r'^(The speaker|The user|They|It is believed) (believes?|thinks?|feels?|says|asserts?|considers?)(\s+that)?(.*)$', opinion_text, re.IGNORECASE)
            if match:
                verb = singularize_verb(match.group(2))
                that_part = match.group(3) or ""  # Keep " that" if present
                rest = match.group(4)
                opinion_text = f"I {verb}{that_part}{rest}"

            # If still doesn't start with first-person, prepend "I believe that "
            first_person_starters = ["I think", "I believe", "I feel", "In my view", "I've come to believe", "Previously I"]
            if not any(opinion_text.startswith(starter) for starter in first_person_starters):
                opinion_text = "I believe that " + opinion_text[0].lower() + opinion_text[1:]

            formatted_opinions.append({
                "text": opinion_text,
                "confidence": op.confidence
            })

        return formatted_opinions

    except Exception as e:
        logger.warning(f"Failed to extract opinions: {str(e)}")
        return []
