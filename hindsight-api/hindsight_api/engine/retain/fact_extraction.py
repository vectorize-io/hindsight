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
    """Causal relationship between facts (legacy - embedded in each fact)."""

    target_fact_index: int = Field(
        description="Index of the related fact in the facts array (0-based). "
        "This creates a directed causal link to another fact in the extraction."
    )
    relation_type: Literal["causes", "caused_by", "enables", "prevents"] = Field(
        description="Type of causal relationship: "
        "'causes' = this fact directly causes the target fact, "
        "'caused_by' = this fact was caused by the target fact, "
        "'enables' = this fact enables/allows the target fact, "
        "'prevents' = this fact prevents/blocks the target fact"
    )
    strength: float = Field(
        description="Strength of causal relationship (0.0 to 1.0). "
        "1.0 = direct/strong causation, 0.5 = moderate, 0.3 = weak/indirect",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class TopLevelCausalRelation(BaseModel):
    """
    Causal relationship between two facts (top-level schema).

    This is the preferred format - defined AFTER all facts are extracted,
    allowing the LLM to see the full list of facts before specifying relationships.
    """

    from_fact_index: int = Field(
        description="Index of the source fact (0-based). The fact that causes/enables/prevents."
    )
    to_fact_index: int = Field(
        description="Index of the target fact (0-based). The fact that is caused/enabled/prevented."
    )
    relation_type: Literal["causes", "caused_by", "enables", "prevents"] = Field(
        description="Type of causal relationship: "
        "'causes' = source fact directly causes the target fact, "
        "'caused_by' = source fact was caused by the target fact, "
        "'enables' = source fact enables/allows the target fact, "
        "'prevents' = source fact prevents/blocks the target fact"
    )
    strength: float = Field(
        description="Strength of causal relationship (0.0 to 1.0). "
        "1.0 = direct/strong causation, 0.5 = moderate, 0.3 = weak/indirect",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class ExtractedFact(BaseModel):
    """A single extracted fact with 5 required dimensions for comprehensive capture."""

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    # ==========================================================================
    # FIVE REQUIRED DIMENSIONS - LLM must think about each one
    # ==========================================================================

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

    # ==========================================================================
    # CLASSIFICATION
    # ==========================================================================

    fact_kind: str = Field(
        default="conversation",
        description="'event' = specific datable occurrence (set occurred dates), 'conversation' = general info (no occurred dates)",
    )

    # Temporal fields - optional
    occurred_start: str | None = Field(
        default=None,
        description="WHEN the event happened (ISO timestamp). Only for fact_kind='event'. Leave null for conversations.",
    )
    occurred_end: str | None = Field(
        default=None,
        description="WHEN the event ended (ISO timestamp). Only for events with duration. Leave null for conversations.",
    )

    # Classification (CRITICAL - required)
    # Note: LLM uses "assistant" but we convert to "bank" for storage
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = about the user/others (background, experiences). 'assistant' = experience with the assistant."
    )

    # Entities - extracted from fact content
    entities: list[Entity] | None = Field(
        default=None,
        description="Named entities, objects, AND abstract concepts from the fact. Include: people names, organizations, places, significant objects (e.g., 'coffee maker', 'car'), AND abstract concepts/themes (e.g., 'friendship', 'career growth', 'loss', 'celebration'). Extract anything that could help link related facts together.",
    )
    causal_relations: list[CausalRelation] | None = Field(
        default=None, description="Causal links to other facts. Can be null."
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        """Ensure entities is always a list (convert None to empty list)."""
        if v is None:
            return []
        return v

    @field_validator("causal_relations", mode="before")
    @classmethod
    def ensure_causal_relations_list(cls, v):
        """Ensure causal_relations is always a list (convert None to empty list)."""
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
    """Response containing all extracted facts and their causal relationships."""

    facts: list[ExtractedFact] = Field(description="List of extracted factual statements")
    causal_relationships: list[TopLevelCausalRelation] | None = Field(
        default=None,
        description="Causal relationships between facts. Define these AFTER listing all facts. "
        "Each relationship specifies from_fact_index -> to_fact_index with a relation type. "
        "Indices must be valid (0 to N-1 where N is the number of facts).",
    )


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


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: "LLMConfig",
    agent_name: str = None,
    extract_opinions: bool = False,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    memory_bank_context = f"\n- Your name: {agent_name}" if agent_name and extract_opinions else ""

    # Determine which fact types to extract based on the flag
    # Note: We use "assistant" in the prompt but convert to "bank" for storage
    if extract_opinions:
        # Opinion extraction uses a separate prompt (not this one)
        fact_types_instruction = "Extract ONLY 'opinion' type facts (formed opinions, beliefs, and perspectives). DO NOT extract 'world' or 'assistant' facts."
    else:
        fact_types_instruction = (
            "Extract ONLY 'world' and 'assistant' type facts. DO NOT extract opinions - those are extracted separately."
        )

    prompt = f"""Extract facts from text into structured format with FOUR required dimensions - BE EXTREMELY DETAILED.

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
USER PREFERENCES (CRITICAL)
══════════════════════════════════════════════════════════════════════════

ALWAYS extract user preferences as separate facts! Watch for these keywords:
- "enjoy", "like", "love", "prefer", "hate", "dislike", "favorite", "ideal", "dream", "want"

Example: "I love Italian food and prefer outdoor dining"
→ Fact 1: what="User loves Italian food", who="user", why="This is a food preference", entities=["user"]
→ Fact 2: what="User prefers outdoor dining", who="user", why="This is a dining preference", entities=["user"]

══════════════════════════════════════════════════════════════════════════
ENTITIES - INCLUDE PEOPLE, PLACES, OBJECTS, AND CONCEPTS (CRITICAL)
══════════════════════════════════════════════════════════════════════════

Extract entities that help link related facts together. Include:
1. "user" - when the fact is about the user
2. People names - Emily, Dr. Smith, etc.
3. Organizations/Places - IKEA, Goodwill, New York, etc.
4. Specific objects - coffee maker, toaster, car, laptop, kitchen, etc.
5. Abstract concepts - themes, values, emotions, or ideas that capture the essence of the fact:
   - "friendship" for facts about friends helping each other, bonding, loyalty
   - "career growth" for facts about promotions, learning new skills, job changes
   - "loss" or "grief" for facts about death, endings, saying goodbye
   - "celebration" for facts about parties, achievements, milestones
   - "trust" or "betrayal" for facts involving those themes

✅ CORRECT: entities=["user", "coffee maker", "Goodwill", "kitchen"] for "User donated their coffee maker to Goodwill"
✅ CORRECT: entities=["user", "Emily", "friendship"] for "Emily helped user move to a new apartment"
✅ CORRECT: entities=["user", "promotion", "career growth"] for "User got promoted to senior engineer"
✅ CORRECT: entities=["user", "grandmother", "loss", "grief"] for "User's grandmother passed away last week"
❌ WRONG: entities=["user", "Emily"] only - missing the "friendship" concept that links to other friendship facts!

══════════════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════════════

Example 1 - World Facts (Event Date: Tuesday, June 10, 2024):
Input: "I'm planning my wedding and want a small outdoor ceremony. I just got back from my college roommate Emily's wedding - she married Sarah at a rooftop garden, it was so romantic!"

Output facts:

1. User's wedding preference
   - what: "User wants a small outdoor ceremony for their wedding"
   - who: "user"
   - why: "User prefers intimate outdoor settings"
   - fact_type: "world", fact_kind: "conversation"
   - entities: ["user", "wedding", "outdoor ceremony"]

2. User planning wedding
   - what: "User is planning their own wedding"
   - who: "user"
   - why: "Inspired by Emily's ceremony"
   - fact_type: "world", fact_kind: "conversation"
   - entities: ["user", "wedding"]

3. Emily's wedding (THE EVENT - note occurred_start AND occurred_end both set)
   - what: "Emily got married to Sarah at a rooftop garden ceremony in the city"
   - who: "Emily (user's college roommate), Sarah (Emily's partner)"
   - why: "User found it romantic and beautiful"
   - fact_type: "world", fact_kind: "event"
   - occurred_start: "2024-06-09T00:00:00Z" (recently, user "just got back" - relative to Event Date June 10, 2024)
   - occurred_end: "2024-06-09T23:59:59Z" (same day - point event)
   - entities: ["user", "Emily", "Sarah", "wedding", "rooftop garden"]

Example 2 - Assistant Facts (Context: March 5, 2024):
Input: "User: My API is really slow when we have 1000+ concurrent users. What can I do?
Assistant: I'd recommend implementing Redis for caching frequently-accessed data, which should reduce your database load by 70-80%."

Output fact:
   - what: "Assistant recommended implementing Redis for caching frequently-accessed data to improve API performance"
   - when: "March 5, 2024 during conversation"
   - who: "user, assistant"
   - why: "User asked how to fix slow API performance with 1000+ concurrent users, expected 70-80% reduction in database load"
   - fact_type: "assistant", fact_kind: "conversation"
   - entities: ["user", "API", "Redis"]

Example 3 - Kitchen Items with Concept Inference (Event Date: Thursday, May 30, 2024):
Input: "I finally donated my old coffee maker to Goodwill. I upgraded to that new espresso machine last month and the old one was just taking up counter space."

Output fact:
   - what: "User donated their old coffee maker to Goodwill after upgrading to a new espresso machine"
   - when: "Thursday, May 30, 2024"
   - who: "user"
   - why: "The old coffee maker was taking up counter space after the upgrade"
   - fact_type: "world", fact_kind: "event"
   - occurred_start: "2024-05-30T00:00:00Z" (uses Event Date year)
   - occurred_end: "2024-05-30T23:59:59Z" (same day - point event)
   - entities: ["user", "coffee maker", "Goodwill", "espresso machine", "kitchen"]

Note: "kitchen" is inferred as a concept because coffee makers and espresso machines are kitchen appliances.
This links the fact to other kitchen-related facts (toaster, faucet, kitchen mat, etc.) via the shared "kitchen" entity.

Note how the "why" field captures the FULL STORY: what the user asked AND what outcome was expected!

══════════════════════════════════════════════════════════════════════════
WHAT TO EXTRACT vs SKIP
══════════════════════════════════════════════════════════════════════════

✅ EXTRACT: User preferences (ALWAYS as separate facts!), feelings, plans, events, relationships, achievements
❌ SKIP: Greetings, filler ("thanks", "cool"), purely structural statements

══════════════════════════════════════════════════════════════════════════
CAUSAL RELATIONSHIPS (CRITICAL - DEFINE AFTER ALL FACTS)
══════════════════════════════════════════════════════════════════════════

⚠️ IMPORTANT: Causal relationships are defined at the TOP LEVEL, AFTER listing all facts!

The `causal_relationships` array goes at the root of your response (NOT inside each fact).
This allows you to see all facts first before defining how they relate.

Format:
```json
{{
  "facts": [...all your extracted facts...],
  "causal_relationships": [
    {{"from_fact_index": 0, "to_fact_index": 1, "relation_type": "causes", "strength": 0.9}},
    {{"from_fact_index": 1, "to_fact_index": 2, "relation_type": "enables", "strength": 0.7}}
  ]
}}
```

Relationship types:
- "causes": Fact A directly causes Fact B (A → B)
- "caused_by": Fact A was caused by Fact B (A ← B)
- "enables": Fact A enables/allows Fact B to happen
- "prevents": Fact A prevents/blocks Fact B from happening

⚠️ INDEX VALIDATION: If you extract N facts (indices 0 to N-1), both from_fact_index and to_fact_index MUST be in range [0, N-1].

Example (Event Date: March 15, 2024):
Input: "I lost my job in January. Because of that, I couldn't pay rent. So I had to move to a cheaper apartment."

Facts extracted:
- Fact 0: "User lost their job in January due to layoffs"
- Fact 1: "User couldn't pay rent because of job loss"
- Fact 2: "User moved to a cheaper apartment"

Causal relationships (at root level):
```json
"causal_relationships": [
  {{"from_fact_index": 0, "to_fact_index": 1, "relation_type": "causes", "strength": 1.0}},
  {{"from_fact_index": 1, "to_fact_index": 2, "relation_type": "causes", "strength": 0.9}}
]
```

This creates a chain: Job loss (0) → Can't pay rent (1) → Moved to cheaper apartment (2)"""

    import logging

    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Retry logic for JSON validation errors
    max_retries = 2
    last_error = None
    config = get_config()

    # Sanitize input text to prevent Unicode encoding errors (e.g., unpaired surrogates)
    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    # Build user message with metadata and chunk content in a clear format
    # Format event_date with day of week for better temporal reasoning
    event_date_formatted = event_date.strftime("%A, %B %d, %Y")  # e.g., "Monday, June 10, 2024"
    user_message = f"""Extract facts from the following text chunk.
{memory_bank_context}

Chunk: {chunk_index + 1}/{total_chunks}
Event Date: {event_date_formatted} ({event_date.isoformat()})
Context: {sanitized_context}

Text:
{sanitized_chunk}"""

    usage = TokenUsage()  # Track cumulative usage across retries
    for attempt in range(max_retries):
        try:
            extraction_response_json, call_usage = await llm_config.call(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=FactExtractionResponse,
                scope="memory_extract_facts",
                temperature=0.1,
                max_completion_tokens=config.retain_max_completion_tokens,
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
            # Get top-level causal relationships (new schema)
            top_level_causal_relations = extraction_response_json.get("causal_relationships", [])

            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat()}, "
                    f"context: {context if context else 'none'}, "
                    f"text: {chunk}"
                )

            # Build a map from fact index to causal relations (from top-level field)
            # This converts from_fact_index -> [{target_fact_index, relation_type, strength}]
            causal_relations_by_fact: dict[int, list[dict]] = {}
            if top_level_causal_relations:
                num_facts = len(raw_facts)
                for rel in top_level_causal_relations:
                    if not isinstance(rel, dict):
                        continue
                    from_idx = rel.get("from_fact_index")
                    to_idx = rel.get("to_fact_index")
                    relation_type = rel.get("relation_type")
                    strength = rel.get("strength", 1.0)

                    # Validate indices
                    if from_idx is None or to_idx is None or relation_type is None:
                        logger.warning(f"Skipping malformed top-level causal relation: {rel}")
                        continue
                    if from_idx < 0 or from_idx >= num_facts:
                        logger.warning(
                            f"Invalid from_fact_index {from_idx} in top-level causal relation "
                            f"(valid range: 0-{num_facts - 1}). Skipping."
                        )
                        continue
                    if to_idx < 0 or to_idx >= num_facts:
                        logger.warning(
                            f"Invalid to_fact_index {to_idx} in top-level causal relation "
                            f"(valid range: 0-{num_facts - 1}). Skipping."
                        )
                        continue

                    # Add to the map for the from_fact_index
                    if from_idx not in causal_relations_by_fact:
                        causal_relations_by_fact[from_idx] = []
                    causal_relations_by_fact[from_idx].append(
                        {
                            "target_fact_index": to_idx,
                            "relation_type": relation_type,
                            "strength": strength,
                        }
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
                fact_type = llm_fact.get("fact_type")

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
                        logger.warning(f"Fact {i}: defaulting to fact_type='world'")

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

                # Add causal relations from both sources:
                # 1. Top-level causal_relationships (preferred, new schema)
                # 2. Per-fact causal_relations (legacy, for backward compatibility)
                validated_relations = []

                # First, add relations from top-level (already validated above)
                if i in causal_relations_by_fact:
                    for rel in causal_relations_by_fact[i]:
                        try:
                            validated_relations.append(CausalRelation.model_validate(rel))
                        except Exception as e:
                            logger.warning(f"Invalid top-level causal relation for fact {i}: {rel}: {e}")

                # Then, add any legacy per-fact relations (with index validation)
                legacy_causal_relations = get_value("causal_relations")
                if legacy_causal_relations:
                    num_facts = len(raw_facts)
                    for rel in legacy_causal_relations:
                        if isinstance(rel, dict) and "target_fact_index" in rel and "relation_type" in rel:
                            target_idx = rel.get("target_fact_index")
                            # Validate target index for legacy format too
                            if target_idx is not None and 0 <= target_idx < num_facts:
                                try:
                                    validated_relations.append(CausalRelation.model_validate(rel))
                                except Exception as e:
                                    logger.warning(f"Invalid causal relation {rel}: {e}")
                            else:
                                logger.warning(
                                    f"Invalid target_fact_index {target_idx} in per-fact causal relation "
                                    f"from fact {i} (valid range: 0-{num_facts - 1}). Skipping."
                                )

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
    extract_opinions: bool = False,
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
        extract_opinions: If True, extract ONLY opinions. If False, extract world and agent facts (no opinions)

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
            extract_opinions=extract_opinions,
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
                extract_opinions=extract_opinions,
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
                extract_opinions=extract_opinions,
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
    extract_opinions: bool = False,
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
        extract_opinions: If True, extract ONLY opinions. If False, extract world and bank facts (no opinions)

    Returns:
        Tuple of (facts, chunks, usage) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
        - usage: Aggregated token usage across all LLM calls
    """
    chunks = chunk_text(text, max_chars=3000)
    tasks = [
        _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            agent_name=agent_name,
            extract_opinions=extract_opinions,
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
    contents: list[RetainContent], llm_config, agent_name: str, extract_opinions: bool = False
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
        extract_opinions: If True, extract only opinions; otherwise world/bank facts

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
            extract_opinions=extract_opinions,
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

        # Apply offset to all temporal fields
        if fact.occurred_start:
            fact.occurred_start = fact.occurred_start + offset
        if fact.occurred_end:
            fact.occurred_end = fact.occurred_end + offset
        if fact.mentioned_at:
            fact.mentioned_at = fact.mentioned_at + offset
