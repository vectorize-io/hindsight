"""
Fact extraction from text using LLM.

Extracts semantic facts, entities, and temporal information from text.
Uses the LLMConfig wrapper for all LLM calls.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...config import get_config
from ..llm_wrapper import LLMConfig, OutputTooLongError
from ..response_models import TokenUsage


def _infer_temporal_date(fact_text: str, event_date: datetime) -> str | None:
    """
    Infer a temporal date from fact text when LLM didn't provide occurred_start.

    This is a fallback for when the LLM fails to extract temporal information
    from relative time expressions like "last night", "yesterday", etc.
    """
    import re

    fact_lower = fact_text.lower()

    # Map relative time expressions to day offsets
    temporal_patterns = {
        r"\blast night\b": -1,
        r"\byesterday\b": -1,
        r"\btoday\b": 0,
        r"\bthis morning\b": 0,
        r"\bthis afternoon\b": 0,
        r"\bthis evening\b": 0,
        r"\btonigh?t\b": 0,
        r"\btomorrow\b": 1,
        r"\blast week\b": -7,
        r"\bthis week\b": 0,
        r"\bnext week\b": 7,
        r"\blast month\b": -30,
        r"\bthis month\b": 0,
        r"\bnext month\b": 30,
    }

    for pattern, offset_days in temporal_patterns.items():
        if re.search(pattern, fact_lower):
            target_date = event_date + timedelta(days=offset_days)
            return target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    # If no relative time expression found, return None
    return None


def _sanitize_text(text: str) -> str:
    """
    Sanitize text by removing invalid Unicode surrogate characters.

    Surrogate characters (U+D800 to U+DFFF) are used in UTF-16 encoding
    but cannot be encoded in UTF-8. They can appear in Python strings
    from improperly decoded data (e.g., from JavaScript or broken files).

    This function removes unpaired surrogates to prevent UnicodeEncodeError
    when the text is sent to the LLM API.
    """
    if not text:
        return text
    # Remove surrogate characters (U+D800 to U+DFFF) using regex
    # These are invalid in UTF-8 and cause encoding errors
    return re.sub(r"[\ud800-\udfff]", "", text)


class Entity(BaseModel):
    """An entity extracted from text."""

    text: str = Field(
        description="The specific, named entity as it appears in the fact. Must be a proper noun or specific identifier."
    )


class Fact(BaseModel):
    """
    Final fact model for storage - built from lenient parsing of LLM response.

    This is what fact_extraction returns and what the rest of the pipeline expects.
    Combined fact text format: "what | when | where | who | why"
    """

    # Required fields
    fact: str = Field(description="Combined fact text: what | when | where | who | why")
    fact_type: Literal["world", "experience", "opinion"] = Field(description="Perspective: world/experience/opinion")

    # Optional temporal fields
    occurred_start: str | None = None
    occurred_end: str | None = None
    mentioned_at: str | None = None

    # Optional location field
    where: str | None = Field(
        None, description="WHERE the fact occurred or is about (specific location, place, or area)"
    )

    # Optional structured data
    entities: list[Entity] | None = None
    causal_relations: list["CausalRelation"] | None = None


class CausalRelation(BaseModel):
    """Causal relationship from this fact to a previous fact (stored format)."""

    target_fact_index: int = Field(description="Index of the related fact in the facts array (0-based).")
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target: 'caused_by' = this fact was caused by the target"
    )
    strength: float = Field(
        description="Strength of relationship (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class FactCausalRelation(BaseModel):
    """
    Causal relationship from this fact to a PREVIOUS fact (embedded in each fact).

    Uses index-based references but ONLY allows referencing facts that appear
    BEFORE this fact in the list. This prevents hallucination of invalid indices.
    """

    target_index: int = Field(
        description="Index of the PREVIOUS fact this relates to (0-based). "
        "MUST be less than this fact's position in the list. "
        "Example: if this is fact #5, target_index can only be 0, 1, 2, 3, or 4."
    )
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target fact: 'caused_by' = this fact was caused by the target fact"
    )
    strength: float = Field(
        description="Strength of relationship (0.0 to 1.0). 1.0 = strong, 0.5 = moderate",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class ExtractedFact(BaseModel):
    """A single extracted fact."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(description="Core fact - concise but complete (1-2 sentences)")
    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")
    why: str = Field(description="Context/significance if important. 'N/A' if obvious.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(description="'world' or 'assistant'")
    entities: list[Entity] | None = Field(default=None, description="People, places, concepts")
    causal_relations: list[FactCausalRelation] | None = Field(
        default=None, description="Links to previous facts (target_index < this fact's index)"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        """Ensure entities is always a list (convert None to empty list)."""
        if v is None:
            return []
        return v

    def build_fact_text(self) -> str:
        """Combine all dimensions into a single comprehensive fact string."""
        parts = [self.what]

        # Add 'who' if not N/A
        if self.who and self.who.upper() != "N/A":
            parts.append(f"Involving: {self.who}")

        # Add 'why' if not N/A
        if self.why and self.why.upper() != "N/A":
            parts.append(self.why)

        if len(parts) == 1:
            return parts[0]

        return " | ".join(parts)


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts (causal relations are embedded in each fact)."""

    facts: list[ExtractedFact] = Field(description="List of extracted factual statements")


class ExtractedFactVerbose(BaseModel):
    """A single extracted fact with verbose field descriptions for detailed extraction."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(
        description="WHAT happened - COMPLETE, DETAILED description with ALL specifics. "
        "NEVER summarize or omit details. Include: exact actions, objects, quantities, specifics. "
        "BE VERBOSE - capture every detail that was mentioned. "
        "Example: 'Emily got married to Sarah at a rooftop garden ceremony with 50 guests attending and a live jazz band playing' "
        "NOT: 'A wedding happened' or 'Emily got married'"
    )

    when: str = Field(
        description="WHEN it happened - ALWAYS include temporal information if mentioned. "
        "Include: specific dates, times, durations, relative time references. "
        "Examples: 'on June 15th, 2024 at 3pm', 'last weekend', 'for the past 3 years', 'every morning at 6am'. "
        "Write 'N/A' ONLY if absolutely no temporal context exists. Prefer converting to absolute dates when possible."
    )

    where: str = Field(
        description="WHERE it happened or is about - SPECIFIC locations, places, areas, regions if applicable. "
        "Include: cities, neighborhoods, venues, buildings, countries, specific addresses when mentioned. "
        "Examples: 'downtown San Francisco at a rooftop garden venue', 'at the user's home in Brooklyn', 'online via Zoom', 'Paris, France'. "
        "Write 'N/A' ONLY if absolutely no location context exists or if the fact is completely location-agnostic."
    )

    who: str = Field(
        description="WHO is involved - ALL people/entities with FULL context and relationships. "
        "Include: names, roles, relationships to user, background details. "
        "Resolve coreferences (if 'my roommate' is later named 'Emily', write 'Emily, the user's college roommate'). "
        "BE DETAILED about relationships and roles. "
        "Example: 'Emily (user's college roommate from Stanford, now works at Google), Sarah (Emily's partner of 5 years, software engineer)' "
        "NOT: 'my friend' or 'Emily and Sarah'"
    )

    why: str = Field(
        description="WHY it matters - ALL emotional, contextual, and motivational details. "
        "Include EVERYTHING: feelings, preferences, motivations, observations, context, background, significance. "
        "BE VERBOSE - capture all the nuance and meaning. "
        "FOR ASSISTANT FACTS: MUST include what the user asked/requested that led to this interaction! "
        "Example (world): 'The user felt thrilled and inspired, has always dreamed of an outdoor ceremony, mentioned wanting a similar garden venue, was particularly moved by the intimate atmosphere and personal vows' "
        "Example (assistant): 'User asked how to fix slow API performance with 1000+ concurrent users, expected 70-80% reduction in database load' "
        "NOT: 'User liked it' or 'To help user'"
    )

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence (set occurred dates), 'conversation' = general info (no occurred dates)",
    )

    occurred_start: str | None = Field(
        default=None,
        description="WHEN the event happened (ISO timestamp). Only for fact_kind='event'. Leave null for conversations.",
    )
    occurred_end: str | None = Field(
        default=None,
        description="WHEN the event ended (ISO timestamp). Only for events with duration. Leave null for conversations.",
    )

    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = about the user/others (background, experiences). 'assistant' = experience with the assistant."
    )

    entities: list[Entity] | None = Field(
        default=None,
        description="Named entities, objects, AND abstract concepts from the fact. Include: people names, organizations, places, significant objects (e.g., 'coffee maker', 'car'), AND abstract concepts/themes (e.g., 'friendship', 'career growth', 'loss', 'celebration'). Extract anything that could help link related facts together.",
    )

    causal_relations: list[FactCausalRelation] | None = Field(
        default=None,
        description="Causal links to PREVIOUS facts only. target_index MUST be less than this fact's position. "
        "Example: fact #3 can only reference facts 0, 1, or 2. Max 2 relations per fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class FactExtractionResponseVerbose(BaseModel):
    """Response for verbose fact extraction."""

    facts: list[ExtractedFactVerbose] = Field(description="List of extracted factual statements")


class ExtractedFactNoCausal(BaseModel):
    """A single extracted fact WITHOUT causal relations (for when causal extraction is disabled)."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    # Same fields as ExtractedFact but without causal_relations
    what: str = Field(description="WHAT happened - COMPLETE, DETAILED description with ALL specifics.")
    when: str = Field(description="WHEN it happened - include temporal information if mentioned.")
    where: str = Field(description="WHERE it happened - SPECIFIC locations if applicable.")
    who: str = Field(description="WHO is involved - ALL people/entities with relationships.")
    why: str = Field(description="WHY it matters - emotional, contextual, and motivational details.")

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence, 'conversation' = general info",
    )
    occurred_start: str | None = Field(default=None, description="WHEN the event happened (ISO timestamp).")
    occurred_end: str | None = Field(default=None, description="WHEN the event ended (ISO timestamp).")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = about the user/others. 'assistant' = experience with assistant."
    )
    entities: list[Entity] | None = Field(
        default=None,
        description="Named entities, objects, and concepts from the fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class FactExtractionResponseNoCausal(BaseModel):
    """Response for fact extraction without causal relations."""

    facts: list[ExtractedFactNoCausal] = Field(description="List of extracted factual statements")


def chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns), splits at turn boundaries
    while preserving speaker context. For plain text, uses sentence-aware splitting.

    Args:
        text: Input text to chunk (plain text or JSON conversation)
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)

    Returns:
        List of text chunks, roughly under max_chars
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    # Try to parse as JSON conversation array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
            # This looks like a conversation - chunk at turn boundaries
            return _chunk_conversation(parsed, max_chars)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to sentence-aware text splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            "! ",  # Exclamations
            "? ",  # Questions
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters (last resort)
        ],
    )

    return splitter.split_text(text)


def _chunk_conversation(turns: list[dict], max_chars: int) -> list[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk

    Returns:
        List of JSON-serialized chunks, each containing complete turns
    """

    chunks = []
    current_chunk = []
    current_size = 2  # Account for "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_size = len(turn_json) + 1  # +1 for comma

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2  # Reset to "[]"

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    if current_chunk:
        chunks.append(json.dumps(current_chunk, ensure_ascii=False))

    return chunks if chunks else [json.dumps(turns, ensure_ascii=False)]


# =============================================================================
# FACT EXTRACTION PROMPTS
# =============================================================================

# Base prompt template (shared by concise and custom modes)
# Uses {extraction_guidelines} placeholder for mode-specific instructions
_BASE_FACT_EXTRACTION_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE REQUIREMENT: Detect the language of the input text. All extracted facts, entity names, descriptions, and other output MUST be in the SAME language as the input. Do not translate to another language.

{fact_types_instruction}

{extraction_guidelines}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. **what**: Core fact - concise but complete (1-2 sentences max)
2. **when**: Temporal info if mentioned. "N/A" if none. Use day name when known.
3. **where**: Location if relevant. "N/A" if none.
4. **who**: People involved with relationships. "N/A" if just general info.
5. **why**: Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": About user's life, other people, external events
- "assistant": Interactions with assistant (requests, recommendations)

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- "yesterday" relative to Event Date, not today
- For events: set occurred_start AND occurred_end (same for point events)
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user.{examples}"""

# Concise mode guidelines
_CONCISE_GUIDELINES = """══════════════════════════════════════════════════════════════════════════
SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)
══════════════════════════════════════════════════════════════════════════

ONLY extract facts that are:
✅ Personal info: names, relationships, roles, background
✅ Preferences: likes, dislikes, habits, interests (e.g., "Alice likes coffee")
✅ Significant events: milestones, decisions, achievements, changes
✅ Plans/goals: future intentions, deadlines, commitments
✅ Expertise: skills, knowledge, certifications, experience
✅ Important context: projects, problems, constraints
✅ Sensory/emotional details: feelings, sensations, perceptions that provide context
✅ Observations: descriptions of people, places, things with specific details

DO NOT extract:
❌ Generic greetings: "how are you", "hello", pleasantries without substance
❌ Pure filler: "thanks", "sounds good", "ok", "got it", "sure"
❌ Process chatter: "let me check", "one moment", "I'll look into it"
❌ Repeated info: if already stated, don't extract again

CONSOLIDATE related statements into ONE fact when possible."""

# Concise mode examples
_CONCISE_EXAMPLES = """

══════════════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════════════

Example 1 - Selective extraction (Event Date: June 10, 2024):
Input: "Hey! How's it going? Good morning! So I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee on the way."

Output: ONLY 2 facts (skip greetings, weather, coffee):
1. what="User planning wedding, wants small outdoor ceremony", who="user", why="N/A", entities=["user", "wedding"]
2. what="Emily married Sarah at rooftop garden", who="Emily (user's friend), Sarah", occurred_start="2024-06-09", entities=["Emily", "Sarah", "wedding"]

Example 2 - Professional context:
Input: "Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output: ONLY 2 facts (skip coffee preference - too trivial):
1. what="Alice has 5 years Kubernetes experience, CKA certified", who="Alice", entities=["Alice", "Kubernetes", "CKA"]
2. what="Alice leads infrastructure team since March", who="Alice", entities=["Alice", "infrastructure"]

══════════════════════════════════════════════════════════════════════════
QUALITY OVER QUANTITY
══════════════════════════════════════════════════════════════════════════

Ask: "Would this be useful to recall in 6 months?" If no, skip it."""

# Assembled concise prompt (backward compatible - exact same output as before)
CONCISE_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    fact_types_instruction="{fact_types_instruction}",
    extraction_guidelines=_CONCISE_GUIDELINES,
    examples=_CONCISE_EXAMPLES,
)

# Custom prompt uses same base but without examples
CUSTOM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    fact_types_instruction="{fact_types_instruction}",
    extraction_guidelines="{custom_instructions}",
    examples="",  # No examples for custom mode
)


# Verbose extraction prompt - detailed, comprehensive facts (legacy mode)
VERBOSE_FACT_EXTRACTION_PROMPT = """Extract facts from text into structured format with FIVE required dimensions - BE EXTREMELY DETAILED.

LANGUAGE REQUIREMENT: Detect the language of the input text. All extracted facts, entity names, descriptions,
and other output MUST be in the SAME language as the input. Do not translate to English if the input is in another language.

{fact_types_instruction}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - ALL FIVE DIMENSIONS REQUIRED - MAXIMUM VERBOSITY
══════════════════════════════════════════════════════════════════════════

For EACH fact, CAPTURE ALL DETAILS - NEVER SUMMARIZE OR OMIT:

1. **what**: WHAT happened - COMPLETE description with ALL specifics (objects, actions, quantities, details)
2. **when**: WHEN it happened - ALWAYS include temporal info with DAY OF WEEK (e.g., "Monday, June 10, 2024")
   - Always include the day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
   - Format: "day_name, month day, year" (e.g., "Saturday, June 9, 2024")
3. **where**: WHERE it happened or is about - SPECIFIC locations, places, areas, regions (if applicable)
4. **who**: WHO is involved - ALL people/entities with FULL relationships and background
5. **why**: WHY it matters - ALL emotions, preferences, motivations, significance, nuance
   - For assistant facts: MUST include what the user asked/requested that triggered this!

Plus: fact_type, fact_kind, entities, occurred_start/end (for structured dates), where (structured location)

VERBOSITY REQUIREMENT: Include EVERY detail mentioned. More detail is ALWAYS better than less.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION (CRITICAL)
══════════════════════════════════════════════════════════════════════════

When text uses BOTH a generic relation AND a name for the same person → LINK THEM!

Example input: "I went to my college roommate's wedding last June. Emily finally married Sarah after 5 years together."

CORRECT output:
- what: "Emily got married to Sarah at a rooftop garden ceremony"
- when: "Saturday, June 8, 2024, after dating for 5 years"
- where: "downtown San Francisco, at a rooftop garden venue"
- who: "Emily (user's college roommate), Sarah (Emily's partner of 5 years)"
- why: "User found it romantic and beautiful, dreams of similar outdoor ceremony"
- where (structured): "San Francisco"

WRONG output:
- what: "User's roommate got married" ← LOSES THE NAME!
- who: "the roommate" ← WRONG - use the actual name!
- where: (missing) ← WRONG - include the location!

══════════════════════════════════════════════════════════════════════════
FACT_KIND CLASSIFICATION (CRITICAL FOR TEMPORAL HANDLING)
══════════════════════════════════════════════════════════════════════════

⚠️ MUST set fact_kind correctly - this determines whether occurred_start/end are set!

fact_kind="event" - USE FOR:
- Actions that happened at a specific time: "went to", "attended", "visited", "bought", "made"
- Past events: "yesterday I...", "last week...", "in March 2020..."
- Future plans with dates: "will go to", "scheduled for"
- Examples: "I went to a pottery workshop" → event
           "Alice visited Paris in February" → event
           "I bought a new car yesterday" → event
           "The user graduated from MIT in March 2020" → event

fact_kind="conversation" - USE FOR:
- Ongoing states: "works as", "lives in", "is married to"
- Preferences: "loves", "prefers", "enjoys"
- Traits/abilities: "speaks fluent French", "knows Python"
- Examples: "I love Italian food" → conversation
           "Alice works at Google" → conversation
           "I prefer outdoor dining" → conversation

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING (CRITICAL - USE EVENT DATE AS REFERENCE)
══════════════════════════════════════════════════════════════════════════

⚠️ IMPORTANT: Use the "Event Date" provided in the input as your reference point!
All relative dates ("yesterday", "last week", "recently") must be resolved relative to the Event Date, NOT today's date.

For EVENTS (fact_kind="event") - MUST SET BOTH occurred_start AND occurred_end:
- Convert relative dates → absolute using Event Date as reference
- If Event Date is "Saturday, March 15, 2020", then "yesterday" = Friday, March 14, 2020
- Dates mentioned in text (e.g., "in March 2020") should use THAT year, not current year
- Always include the day name (Monday, Tuesday, etc.) in the 'when' field
- Set occurred_start AND occurred_end to WHEN IT HAPPENED (not when mentioned)
- For single-day/point events: set occurred_end = occurred_start (same timestamp)

For CONVERSATIONS (fact_kind="conversation"):
- General info, preferences, ongoing states → NO occurred dates
- Examples: "loves coffee", "works as engineer"

══════════════════════════════════════════════════════════════════════════
FACT TYPE
══════════════════════════════════════════════════════════════════════════

- **world**: User's life, other people, events (would exist without this conversation)
- **assistant**: Interactions with assistant (requests, recommendations, help)
  ⚠️ CRITICAL for assistant facts: ALWAYS capture the user's request/question in the fact!
  Include: what the user asked, what problem they wanted solved, what context they provided

══════════════════════════════════════════════════════════════════════════
ENTITIES - EXTRACT EVERYTHING
══════════════════════════════════════════════════════════════════════════

Extract ALL of the following from the fact:
- People names (Emily, Alice, Dr. Smith)
- Organizations (Google, MIT, local coffee shop)
- Places (San Francisco, Brooklyn, Paris)
- Significant objects mentioned (coffee maker, new car, wedding dress)
- Abstract concepts/themes (friendship, career growth, loss, celebration)

ALWAYS include "user" when fact is about the user.
Extract anything that could help link related facts together."""


# Causal relationships section - appended when causal extraction is enabled
CAUSAL_RELATIONSHIPS_SECTION = """

══════════════════════════════════════════════════════════════════════════
CAUSAL RELATIONSHIPS
══════════════════════════════════════════════════════════════════════════

Link facts with causal_relations (max 2 per fact). target_index must be < this fact's index.
Type: "caused_by" (this fact was caused by the target fact)

Example: "Lost job → couldn't pay rent → moved apartment"
- Fact 0: Lost job, causal_relations: null
- Fact 1: Couldn't pay rent, causal_relations: [{target_index: 0, relation_type: "caused_by"}]
- Fact 2: Moved apartment, causal_relations: [{target_index: 1, relation_type: "caused_by"}]"""


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: "LLMConfig",
    agent_name: str = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    import logging

    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Determine which fact types to extract
    # Note: We use "assistant" in the prompt but convert to "bank" for storage
    fact_types_instruction = "Extract ONLY 'world' and 'assistant' type facts."

    # Check config for extraction mode and causal link extraction
    config = get_config()
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # Select base prompt based on extraction mode
    if extraction_mode == "custom":
        # Custom mode: inject user-provided guidelines
        if not config.retain_custom_instructions:
            logger.warning(
                "extraction_mode='custom' but HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS not set. "
                "Falling back to 'concise' mode."
            )
            base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(fact_types_instruction=fact_types_instruction)
        else:
            base_prompt = CUSTOM_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                fact_types_instruction=fact_types_instruction,
                custom_instructions=config.retain_custom_instructions,
            )
    elif extraction_mode == "verbose":
        base_prompt = VERBOSE_FACT_EXTRACTION_PROMPT
        prompt = base_prompt.format(fact_types_instruction=fact_types_instruction)
    else:
        base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
        prompt = base_prompt.format(fact_types_instruction=fact_types_instruction)

    # Build the full prompt with or without causal relationships section
    # Select appropriate response schema based on extraction mode and causal links
    if extract_causal_links:
        prompt = prompt + CAUSAL_RELATIONSHIPS_SECTION
        if extraction_mode == "verbose":
            response_schema = FactExtractionResponseVerbose
        else:
            response_schema = FactExtractionResponse
    else:
        response_schema = FactExtractionResponseNoCausal

    # Retry logic for JSON validation errors
    max_retries = 2
    last_error = None

    # Sanitize input text to prevent Unicode encoding errors (e.g., unpaired surrogates)
    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    # Build user message with metadata and chunk content in a clear format
    # Format event_date with day of week for better temporal reasoning
    # Handle both datetime objects and ISO string formats (from deserialized async tasks)
    from .orchestrator import parse_datetime_flexible

    event_date = parse_datetime_flexible(event_date)
    event_date_formatted = event_date.strftime("%A, %B %d, %Y")  # e.g., "Monday, June 10, 2024"
    user_message = f"""Extract facts from the following text chunk.

Chunk: {chunk_index + 1}/{total_chunks}
Event Date: {event_date_formatted} ({event_date.isoformat()})
Context: {sanitized_context}

Text:
{sanitized_chunk}"""

    usage = TokenUsage()  # Track cumulative usage across retries
    for attempt in range(max_retries):
        try:
            # Use retain-specific overrides if set, otherwise fall back to global LLM config
            max_retries = (
                config.retain_llm_max_retries if config.retain_llm_max_retries is not None else config.llm_max_retries
            )
            initial_backoff = (
                config.retain_llm_initial_backoff
                if config.retain_llm_initial_backoff is not None
                else config.llm_initial_backoff
            )
            max_backoff = (
                config.retain_llm_max_backoff if config.retain_llm_max_backoff is not None else config.llm_max_backoff
            )

            extraction_response_json, call_usage = await llm_config.call(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=response_schema,
                scope="memory_extract_facts",
                temperature=0.1,
                max_completion_tokens=config.retain_max_completion_tokens,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=True,  # Get raw JSON, we'll validate leniently
                return_usage=True,
            )
            usage = usage + call_usage  # Aggregate usage across retries

            # Lenient parsing of facts from raw JSON
            chunk_facts = []
            has_malformed_facts = False

            # Handle malformed LLM responses
            if not isinstance(extraction_response_json, dict):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"LLM returned non-dict JSON on attempt {attempt + 1}/{max_retries}: {type(extraction_response_json).__name__}. Retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"LLM returned non-dict JSON after {max_retries} attempts: {type(extraction_response_json).__name__}. "
                        f"Raw: {str(extraction_response_json)[:500]}"
                    )
                    return [], usage

            raw_facts = extraction_response_json.get("facts", [])

            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat()}, "
                    f"context: {context if context else 'none'}, "
                    f"text: {chunk}"
                )

            for i, llm_fact in enumerate(raw_facts):
                # Skip non-dict entries but track them for retry
                if not isinstance(llm_fact, dict):
                    logger.warning(f"Skipping non-dict fact at index {i}")
                    has_malformed_facts = True
                    continue

                # Helper to get non-empty value
                def get_value(field_name):
                    value = llm_fact.get(field_name)
                    if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                        return value
                    return None

                # NEW FORMAT: what, when, who, why (all required)
                what = get_value("what")
                when = get_value("when")
                who = get_value("who")
                why = get_value("why")

                # Fallback to old format if new fields not present
                if not what:
                    what = get_value("factual_core")
                if not what:
                    logger.warning(f"Skipping fact {i}: missing 'what' field")
                    continue

                # Critical field: fact_type
                # LLM uses "assistant" but we convert to "experience" for storage
                original_fact_type = llm_fact.get("fact_type")
                fact_type = original_fact_type

                # Convert "assistant" → "experience" for storage
                if fact_type == "assistant":
                    fact_type = "experience"

                # Validate fact_type (after conversion)
                if fact_type not in ["world", "experience", "opinion"]:
                    # Try to fix common mistakes - check if they swapped fact_type and fact_kind
                    fact_kind = llm_fact.get("fact_kind")
                    if fact_kind == "assistant":
                        fact_type = "experience"
                    elif fact_kind in ["world", "experience", "opinion"]:
                        fact_type = fact_kind
                    else:
                        # Default to 'world' if we can't determine
                        fact_type = "world"
                        logger.warning(
                            f"Fact {i}: defaulting to fact_type='world' "
                            f"(original fact_type={original_fact_type!r}, fact_kind={fact_kind!r})"
                        )

                # Get fact_kind for temporal handling (but don't store it)
                fact_kind = llm_fact.get("fact_kind", "conversation")
                if fact_kind not in ["conversation", "event", "other"]:
                    fact_kind = "conversation"

                # Build combined fact text from the 4 dimensions: what | when | who | why
                fact_data = {}
                combined_parts = [what]

                if when:
                    combined_parts.append(f"When: {when}")

                if who:
                    combined_parts.append(f"Involving: {who}")

                if why:
                    combined_parts.append(why)

                combined_text = " | ".join(combined_parts)

                # Add temporal fields
                # For events: occurred_start/occurred_end (when the event happened)
                if fact_kind == "event":
                    occurred_start = get_value("occurred_start")
                    occurred_end = get_value("occurred_end")

                    # If LLM didn't set temporal fields, try to extract them from the fact text
                    if not occurred_start:
                        fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                    else:
                        fact_data["occurred_start"] = occurred_start

                    # For point events: if occurred_end not set, default to occurred_start
                    if occurred_end:
                        fact_data["occurred_end"] = occurred_end
                    elif fact_data.get("occurred_start"):
                        fact_data["occurred_end"] = fact_data["occurred_start"]

                # Add entities if present (validate as Entity objects)
                # LLM sometimes returns strings instead of {"text": "..."} format
                entities = get_value("entities")
                if entities:
                    # Validate and normalize each entity
                    validated_entities = []
                    for ent in entities:
                        if isinstance(ent, str):
                            # Normalize string to Entity object
                            validated_entities.append(Entity(text=ent))
                        elif isinstance(ent, dict) and "text" in ent:
                            try:
                                validated_entities.append(Entity.model_validate(ent))
                            except Exception as e:
                                logger.warning(f"Invalid entity {ent}: {e}")
                    if validated_entities:
                        fact_data["entities"] = validated_entities

                # Add per-fact causal relations (only if enabled in config)
                if extract_causal_links:
                    validated_relations = []
                    causal_relations_raw = get_value("causal_relations")
                    if causal_relations_raw:
                        for rel in causal_relations_raw:
                            if not isinstance(rel, dict):
                                continue
                            # New schema uses target_index
                            target_idx = rel.get("target_index")
                            relation_type = rel.get("relation_type")
                            strength = rel.get("strength", 1.0)

                            if target_idx is None or relation_type is None:
                                continue

                            # Validate: target_index must be < current fact index
                            if target_idx < 0 or target_idx >= i:
                                logger.debug(
                                    f"Invalid target_index {target_idx} for fact {i} (must be 0 to {i - 1}). Skipping."
                                )
                                continue

                            try:
                                validated_relations.append(
                                    CausalRelation(
                                        target_fact_index=target_idx,
                                        relation_type=relation_type,
                                        strength=strength,
                                    )
                                )
                            except Exception as e:
                                logger.debug(f"Invalid causal relation {rel}: {e}")

                    if validated_relations:
                        fact_data["causal_relations"] = validated_relations

                # Always set mentioned_at to the event_date (when the conversation/document occurred)
                fact_data["mentioned_at"] = event_date.isoformat()

                # Build Fact model instance
                try:
                    fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                    chunk_facts.append(fact)
                except Exception as e:
                    logger.error(f"Failed to create Fact model for fact {i}: {e}")
                    has_malformed_facts = True
                    continue

            # If we got malformed facts and haven't exhausted retries, try again
            if has_malformed_facts and len(chunk_facts) < len(raw_facts) * 0.8 and attempt < max_retries - 1:
                logger.warning(
                    f"Got {len(raw_facts) - len(chunk_facts)} malformed facts out of {len(raw_facts)} on attempt {attempt + 1}/{max_retries}. Retrying..."
                )
                continue

            return chunk_facts, usage

        except BadRequestError as e:
            last_error = e
            if "json_validate_failed" in str(e):
                logger.warning(
                    f"          [1.3.{chunk_index + 1}] Attempt {attempt + 1}/{max_retries} failed with JSON validation error: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"          [1.3.{chunk_index + 1}] Retrying...")
                    continue
            # If it's not a JSON validation error or we're out of retries, re-raise
            raise

    # If we exhausted all retries, raise the last error
    raise last_error


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: LLMConfig,
    agent_name: str = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a chunk with automatic splitting if output exceeds token limits.

    If the LLM output is too long (OutputTooLongError), this function automatically
    splits the chunk in half and processes each half recursively.

    Args:
        chunk: Text chunk to process
        chunk_index: Index of this chunk in the original list
        total_chunks: Total number of original chunks
        event_date: Reference date for temporal information
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        agent_name: Optional agent name (memory owner)

    Returns:
        Tuple of (facts list, token usage) extracted from the chunk (possibly from sub-chunks)
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Try to extract facts from the full chunk
        return await _extract_facts_from_chunk(
            chunk=chunk,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
        )
    except OutputTooLongError:
        # Output exceeded token limits - split the chunk in half and retry
        logger.warning(
            f"Output too long for chunk {chunk_index + 1}/{total_chunks} "
            f"({len(chunk)} chars). Splitting in half and retrying..."
        )

        # Split at the midpoint, preferring sentence boundaries
        mid_point = len(chunk) // 2

        # Try to find a sentence boundary near the midpoint
        # Look for ". ", "! ", "? " within 20% of midpoint
        search_range = int(len(chunk) * 0.2)
        search_start = max(0, mid_point - search_range)
        search_end = min(len(chunk), mid_point + search_range)

        sentence_endings = [". ", "! ", "? ", "\n\n"]
        best_split = mid_point

        for ending in sentence_endings:
            pos = chunk.rfind(ending, search_start, search_end)
            if pos != -1:
                best_split = pos + len(ending)
                break

        # Split the chunk
        first_half = chunk[:best_split].strip()
        second_half = chunk[best_split:].strip()

        logger.info(
            f"Split chunk {chunk_index + 1} into two sub-chunks: {len(first_half)} chars and {len(second_half)} chars"
        )

        # Process both halves recursively (in parallel)
        sub_tasks = [
            _extract_facts_with_auto_split(
                chunk=first_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
            ),
        ]

        sub_results = await asyncio.gather(*sub_tasks)

        # Combine results from both halves
        all_facts = []
        total_usage = TokenUsage()
        for sub_facts, sub_usage in sub_results:
            all_facts.extend(sub_facts)
            total_usage = total_usage + sub_usage

        logger.info(f"Successfully extracted {len(all_facts)} facts from split chunk {chunk_index + 1}")

        return all_facts, total_usage


async def extract_facts_from_text(
    text: str,
    event_date: datetime,
    llm_config: LLMConfig,
    agent_name: str,
    context: str = "",
) -> tuple[list[Fact], list[tuple[str, int]], TokenUsage]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>3000 chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    If a chunk produces output that exceeds token limits (OutputTooLongError), it is
    automatically split in half and retried recursively until successful.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        llm_config: LLM configuration to use
        agent_name: Agent name (memory owner)

    Returns:
        Tuple of (facts, chunks, usage) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
        - usage: Aggregated token usage across all LLM calls
    """
    config = get_config()
    chunks = chunk_text(text, max_chars=config.retain_chunk_size)

    # Log chunk count before starting LLM requests
    total_chars = sum(len(c) for c in chunks)
    if len(chunks) > 1:
        logger.debug(
            f"[FACT_EXTRACTION] Text chunked into {len(chunks)} chunks ({total_chars:,} chars total, "
            f"chunk_size={config.retain_chunk_size:,}) - starting parallel LLM extraction"
        )

    tasks = [
        _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
        )
        for i, chunk in enumerate(chunks)
    ]
    chunk_results = await asyncio.gather(*tasks)
    all_facts = []
    chunk_metadata = []  # [(chunk_text, fact_count), ...]
    total_usage = TokenUsage()
    for chunk, (chunk_facts, chunk_usage) in zip(chunks, chunk_results):
        all_facts.extend(chunk_facts)
        chunk_metadata.append((chunk, len(chunk_facts)))
        total_usage = total_usage + chunk_usage
    return all_facts, chunk_metadata, total_usage


# ============================================================================
# ORCHESTRATION LAYER
# ============================================================================

# Import types for the orchestration layer (note: ExtractedFact here is different from the Pydantic model above)

from .types import CausalRelation as CausalRelationType
from .types import ChunkMetadata, RetainContent
from .types import ExtractedFact as ExtractedFactType

logger = logging.getLogger(__name__)

# Each fact gets 10 seconds offset to preserve ordering within a document
SECONDS_PER_FACT = 10


async def extract_facts_from_contents(
    contents: list[RetainContent], llm_config, agent_name: str
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts from multiple content items in parallel.

    This function:
    1. Extracts facts from all contents in parallel using the LLM
    2. Tracks which facts came from which chunks
    3. Adds time offsets to preserve fact ordering within each content
    4. Returns typed ExtractedFact and ChunkMetadata objects

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration for fact extraction
        agent_name: Name of the agent (for agent-related fact detection)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)
    """
    if not contents:
        return [], [], TokenUsage()

    # Step 1: Create parallel fact extraction tasks
    fact_extraction_tasks = []
    for item in contents:
        # Call extract_facts_from_text directly (defined earlier in this file)
        # to avoid circular import with utils.extract_facts
        task = extract_facts_from_text(
            text=item.content,
            event_date=item.event_date,
            context=item.context,
            llm_config=llm_config,
            agent_name=agent_name,
        )
        fact_extraction_tasks.append(task)

    # Step 2: Wait for all fact extractions to complete
    all_fact_results = await asyncio.gather(*fact_extraction_tasks)

    # Step 3: Flatten and convert to typed objects
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    total_usage = TokenUsage()

    global_chunk_idx = 0
    global_fact_idx = 0

    for content_index, (content, (facts_from_llm, chunks_from_llm, content_usage)) in enumerate(
        zip(contents, all_fact_results)
    ):
        total_usage = total_usage + content_usage
        chunk_start_idx = global_chunk_idx

        # Convert chunk tuples to ChunkMetadata objects
        for chunk_index_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_metadata = ChunkMetadata(
                chunk_text=chunk_text,
                fact_count=chunk_fact_count,
                content_index=content_index,
                chunk_index=global_chunk_idx,
            )
            chunks_metadata.append(chunk_metadata)
            global_chunk_idx += 1

        # Convert facts to ExtractedFact objects with proper indexing
        fact_idx_in_content = 0
        for chunk_idx_in_content, (chunk_text, chunk_fact_count) in enumerate(chunks_from_llm):
            chunk_global_idx = chunk_start_idx + chunk_idx_in_content

            for _ in range(chunk_fact_count):
                if fact_idx_in_content < len(facts_from_llm):
                    fact_from_llm = facts_from_llm[fact_idx_in_content]

                    # Convert Fact model from LLM to ExtractedFactType dataclass
                    # mentioned_at is always the event_date (when the conversation/document occurred)
                    extracted_fact = ExtractedFactType(
                        fact_text=fact_from_llm.fact,
                        fact_type=fact_from_llm.fact_type,
                        entities=[e.text for e in (fact_from_llm.entities or [])],
                        # occurred_start/end: from LLM only, leave None if not provided
                        occurred_start=_parse_datetime(fact_from_llm.occurred_start)
                        if fact_from_llm.occurred_start
                        else None,
                        occurred_end=_parse_datetime(fact_from_llm.occurred_end)
                        if fact_from_llm.occurred_end
                        else None,
                        causal_relations=_convert_causal_relations(
                            fact_from_llm.causal_relations or [], global_fact_idx
                        ),
                        content_index=content_index,
                        chunk_index=chunk_global_idx,
                        context=content.context,
                        # mentioned_at: always the event_date (when the conversation/document occurred)
                        mentioned_at=content.event_date,
                        metadata=content.metadata,
                        tags=content.tags,
                    )

                    extracted_facts.append(extracted_fact)
                    global_fact_idx += 1
                    fact_idx_in_content += 1

    # Step 4: Add time offsets to preserve ordering within each content
    _add_temporal_offsets(extracted_facts, contents)

    return extracted_facts, chunks_metadata, total_usage


def _parse_datetime(date_str: str):
    """Parse ISO datetime string."""
    from dateutil import parser as date_parser

    try:
        return date_parser.isoparse(date_str)
    except Exception:
        return None


def _convert_causal_relations(relations_from_llm, fact_start_idx: int) -> list[CausalRelationType]:
    """
    Convert causal relations from LLM format to ExtractedFact format.

    Adjusts target_fact_index from content-relative to global indices.
    """
    causal_relations = []
    for rel in relations_from_llm:
        causal_relation = CausalRelationType(
            relation_type=rel.relation_type,
            target_fact_index=fact_start_idx + rel.target_fact_index,
            strength=rel.strength,
        )
        causal_relations.append(causal_relation)
    return causal_relations


def _add_temporal_offsets(facts: list[ExtractedFactType], contents: list[RetainContent]) -> None:
    """
    Add time offsets to preserve fact ordering within each content.

    This allows retrieval to distinguish between facts that happened earlier vs later
    in the same conversation, even when the base event_date is the same.

    Modifies facts in place.
    """
    from .orchestrator import parse_datetime_flexible

    # Group facts by content_index
    current_content_idx = 0
    content_fact_start = 0

    for i, fact in enumerate(facts):
        if fact.content_index != current_content_idx:
            # Moved to next content
            current_content_idx = fact.content_index
            content_fact_start = i

        # Calculate position within this content
        fact_position = i - content_fact_start
        offset = timedelta(seconds=fact_position * SECONDS_PER_FACT)

        # Apply offset to all temporal fields (handle both datetime objects and ISO strings)
        if fact.occurred_start:
            fact.occurred_start = parse_datetime_flexible(fact.occurred_start) + offset
        if fact.occurred_end:
            fact.occurred_end = parse_datetime_flexible(fact.occurred_end) + offset
        if fact.mentioned_at:
            fact.mentioned_at = parse_datetime_flexible(fact.mentioned_at) + offset
