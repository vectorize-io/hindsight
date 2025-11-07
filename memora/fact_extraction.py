"""
Fact extraction from text using LLM.

Extracts semantic facts, entities, and temporal information from text.
Uses the LLMConfig wrapper for all LLM calls.
"""
import os
import json
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An entity extracted from text."""
    text: str = Field(
        description="The entity name as it appears in the fact"
    )
    type: Literal["PERSON", "ORG", "PLACE", "PRODUCT", "CONCEPT", "OTHER"] = Field(
        description="Entity type: PERSON, ORG, PLACE, PRODUCT, CONCEPT, or OTHER for entities that don't fit other categories"
    )


class ExtractedFact(BaseModel):
    """A single extracted fact from text."""
    fact: str = Field(
        description="Self-contained factual statement with subject + action + context"
    )
    date: str = Field(
        description="Absolute date/time when this fact occurred in ISO format (YYYY-MM-DDTHH:MM:SSZ). If text mentions relative time (yesterday, last week, this morning), calculate absolute date from the provided context date."
    )
    fact_type: Literal["world", "agent", "opinion"] = Field(
        description="Type of fact: 'world' for general facts about the world (events, people, things that happen), 'agent' for facts about what the AI agent specifically did or actions the agent took (conversations with the user, tasks performed by the agent), 'opinion' for the agent's formed opinions and perspectives"
    )
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of important entities mentioned in this fact with their types"
    )


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""
    facts: List[ExtractedFact] = Field(
        description="List of extracted factual statements"
    )


def chunk_text(text: str, max_chars: int = 120000) -> List[str]:
    """
    Split text into chunks at sentence boundaries using LangChain's text splitter.

    Uses RecursiveCharacterTextSplitter which intelligently splits at sentence boundaries
    and allows chunks to slightly exceed max_chars to finish sentences naturally.

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 120k ≈ 30k tokens)
                   Note: chunks may slightly exceed this to complete sentences

    Returns:
        List of text chunks, roughly under max_chars
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    # Configure splitter to split at sentence boundaries first
    # Separators in order of preference: paragraphs, newlines, sentences, words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Words
            "",      # Characters (last resort)
        ],
    )

    return splitter.split_text(text)


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: 'LLMConfig'
) -> List[Dict[str, str]]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).
    """
    # Format event_date for the prompt
    event_date_str = event_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    prompt = f"""You are extracting facts from text for an AI memory system. Each fact will be stored and retrieved later.

## CONTEXT INFORMATION
- Current reference date/time: {event_date_str}
- Context: {context if context else 'no context provided'}

## CRITICAL: Facts must be DETAILED, COMPREHENSIVE, and CONTEXT-RICH

Each fact should:
1. Be SELF-CONTAINED - readable without the original context
2. Include ALL relevant details: WHO, WHAT, WHERE, WHEN, WHY, HOW
3. **CRITICAL: ALWAYS include the SUBJECT (who is doing/saying/experiencing)**
4. **Preserve ALL context**: photos/images, "new" things, visual elements, medium of communication
5. Preserve specific names, dates, numbers, locations, relationships, modifiers (new, old, first, etc.)
6. Resolve pronouns to actual names/entities (I → speaker name, their → possessor name)
7. **CRITICAL: Preserve possessive relationships** (their kids → whose kids, his car → whose car)
8. Capture nuances, reasons, causes, implications, and surrounding context

**COMMON MISTAKES TO AVOID:**
- ❌ "The kids were excited" → Missing WHO the kids belong to
- ✅ "Melanie's kids were excited" or "Melanie took her kids who were excited"
- ❌ "Nate chose his hair color because it's bright and bold" → Missing that it's NEW and in a PHOTO
- ✅ "Nate shared a photo of his new hair color, which he chose because it's bright and bold"
- ❌ "Alice started a job at Google" → Missing that it's NEW
- ✅ "Alice started a new job at Google"

## TEMPORAL INFORMATION (VERY IMPORTANT)
For each fact, extract the ABSOLUTE date/time when it occurred:
- If text mentions ABSOLUTE dates ("on March 15, 2024", "last Tuesday"), use that date
- If text mentions RELATIVE times ("yesterday", "last week", "last month", "last year", "this morning", "3 days ago", "next year"), calculate the absolute date using the reference date above
- **CRITICAL**: Transform relative temporal expressions in the FACT TEXT to absolute context:
  - "last year" → "in [calculated year]" (e.g., if reference is 2023, "last year" becomes "in 2022")
  - "last month" → "in [month name] [year]" (e.g., if reference is March 2024, "last month" becomes "in February 2024")
  - "last week" → "week of [date]" or keep as "last week" with absolute date field
  - "yesterday" → can stay as "yesterday" with absolute date field
- If NO specific time is mentioned, use the reference date
- Always output dates in ISO format: YYYY-MM-DDTHH:MM:SSZ

Examples of date extraction and fact text transformation:
- Reference: 2024-03-20T10:00:00Z
- "Yesterday I went hiking" → fact: "Yesterday I went hiking", date: 2024-03-19T10:00:00Z
- "Last week I joined Google" → fact: "Last week I joined Google", date: 2024-03-13T10:00:00Z (approximately)
- "Last year we visited Paris" → fact: "In 2023 we visited Paris", date: 2023-03-20T10:00:00Z
- "Last month I started a new job" → fact: "In February 2024 I started a new job", date: 2024-02-20T10:00:00Z
- "This morning I had coffee" → fact: "This morning I had coffee", date: 2024-03-20T08:00:00Z
- "I work at Google" (no time mentioned) → date: 2024-03-20T10:00:00Z (use reference)

## What to EXTRACT (BE EXHAUSTIVE - DO NOT SKIP ANYTHING):
- **Biographical information (CRITICAL - NEVER MISS)**:
  - Origins: home country, birthplace, where someone is from ("my home country Sweden" = Caroline is from Sweden)
  - Current location: where they live now
  - Jobs, roles, backgrounds, experiences, skills
  - Family background, heritage, cultural identity
  - Education, training, certifications
- **Events (NEVER MISS THESE)**:
  - ANY action that happened (went, did, attended, joined, started, finished, etc.)
  - Photos, images, videos shared or taken ("here's a photo", "took a picture", "captured")
  - Social activities (meetups, gatherings, meals, conversations)
  - Achievements, milestones, accomplishments
  - Travels, visits, locations visited
  - Purchases, acquisitions, creations
- **Identity and personal details**:
  - Origins, nationality, home country, roots
  - Cultural background, heritage
  - Family connections (grandmother from X, parents in Y)
- **Opinions and beliefs**: who believes what and why
- **Recommendations and advice**: specific suggestions with reasoning
- **Descriptions**: detailed explanations of how things work
- **Social relationships and nicknames (CRITICAL - ALWAYS EXTRACT)**:
  - Nicknames: how different people refer to someone ("Andrey calls Joanne 'Jo'", "Everyone calls him Bobby")
  - Terms of address: how people address each other (formal names, nicknames, titles)
  - Relationship indicators: how people describe their relationships ("considers X as a mentor", "refers to Y as their best friend")
  - Social dynamics: who knows whom, who interacts with whom
  - Even if not an "event", these are FACTS about social relationships
  - Extract BOTH the person using the name AND the person being referred to
- **Relationships**: connections between people, organizations, concepts
- **States and conditions**: current status, ongoing situations

## CRITICAL: Extract EVERY event with FULL CONTEXT
- "here's a photo of my new car" = shared a photo of their NEW car (preserve "new")
- "I was with friends last week" = meetup/gathering with friends last week
- "sent you that link" = action of sending a link
- "got a new job" = preserve "new" - it's important context
- DO NOT skip events just because they seem minor or casual
- DO NOT drop modifiers like "new", "first", "old", "favorite" - they're critical context

## What to SKIP (ONLY these):
- Greetings, thank yous, acknowledgments (unless they reveal information)
- Filler words ("um", "uh", "like")
- Pure reactions without content ("wow", "cool", "nice")
- Incomplete thoughts or sentence fragments with no meaning

## FACT TYPE CLASSIFICATION (CRITICAL):
For EACH fact, classify it as either 'world' or 'agent':

- **'world'**: General facts about the world, events, people, things that happen
  - Examples: "Alice works at Google", "Bob went hiking in Yosemite", "The meeting is scheduled for Monday"
  - Most facts will be 'world' type

- **'agent'**: Facts specifically about what the AI agent did or actions the agent took
  - Examples: "The AI agent helped the user debug their code", "The agent answered a question about Python", "The agent created a new file"
  - ONLY use 'agent' if the fact is explicitly about the AI agent's actions
  - Conversations with the user where the agent participated are 'agent' type
  - Tasks performed BY the agent are 'agent' type

When in doubt, classify as 'world'.

## ENTITY EXTRACTION (CRITICAL):
For EACH fact, extract ALL important entities mentioned with their types:
- **PERSON**: Names of individuals (Alice, Bob, Dr. Smith)
- **ORG**: Companies, institutions, teams (Google, MIT, AI Team)
- **PLACE**: Cities, countries, locations, venues (Mountain View, Yosemite, The Coffee Shop)
- **PRODUCT**: Specific products, tools, technologies (iPhone, Python, TensorFlow)
- **CONCEPT**: Important topics, projects, subjects (AI, machine learning, Project Phoenix)
- **OTHER**: Entities that don't fit the above categories (events, time periods, etc.)

Entity extraction rules:
- Use the EXACT form as it appears in the fact (preserve capitalization)
- Assign the correct type to distinguish ambiguous entities (Apple the company = ORG, apple the fruit = PRODUCT/CONCEPT)
- Use OTHER only for entities that truly don't fit the other categories
- Include both full names and commonly used short forms if both appear
- Extract proper nouns and key identifying terms
- Skip generic terms (the, a, some) and pronouns (he, she, they)
- Each entity must have both text and type

## EXAMPLES of GOOD facts (detailed, comprehensive):

Input: "Alice mentioned she works at Google in Mountain View. She joined the AI team last year."
GOOD fact: "Alice works at Google in Mountain View on the AI team, which she joined in 2023"
GOOD fact_type: "world"
GOOD date: 2023-03-20T10:00:00Z (if reference is 2024-03-20, "last year" = 2023)
GOOD entities: [
  {{"text": "Alice", "type": "PERSON"}},
  {{"text": "Google", "type": "ORG"}},
  {{"text": "Mountain View", "type": "PLACE"}},
  {{"text": "AI team", "type": "ORG"}}
]
NOTE: "last year" was transformed to "in 2023" in the fact text

Input: "Yesterday Bob went hiking in Yosemite because it helps him clear his mind."
GOOD fact: "Bob went hiking in Yosemite because it helps him clear his mind"
GOOD fact_type: "world"
GOOD date: Reference date minus 1 day
GOOD entities: [
  {{"text": "Bob", "type": "PERSON"}},
  {{"text": "Yosemite", "type": "PLACE"}}
]

Input: "Here's a photo of me with my friends taken last week at the beach."
GOOD fact: "Someone shared a photo taken last week showing them with their friends at the beach"
GOOD date: Reference date minus 7 days (last week)
GOOD entities: []
NOTE: Include that it's a PHOTO being shared, when it was taken, and who/what/where is in it

Input: "Nate: Here's a photo of my new hair! Friend: Why that color? Nate: I picked this color because it's bright and bold"
BAD fact: "Nate chose his hair color because it's bright and bold"
PROBLEM: Missing that it's NEW hair and he SHARED A PHOTO of it!
GOOD fact: "Nate shared a photo of his new hair color, which he chose because it's bright and bold"
GOOD entities: [{{"text": "Nate", "type": "PERSON"}}]
NOTE: Preserve "new" and "photo" - critical context about what happened

Input: "I sent you that article about AI last Tuesday."
GOOD fact: "Someone sent an article about AI"
GOOD fact_type: "world"
GOOD date: Calculate last Tuesday from reference date
GOOD entities: [
  {{"text": "AI", "type": "CONCEPT"}}
]

Input: "The AI agent helped me write a Python script to analyze my data."
GOOD fact: "The AI agent helped someone write a Python script to analyze their data"
GOOD fact_type: "agent"
GOOD date: Reference date (no specific time mentioned)
GOOD entities: [
  {{"text": "Python", "type": "PRODUCT"}},
  {{"text": "data analysis", "type": "CONCEPT"}}
]

Input: "I bought an Apple laptop and some apples from the store."
GOOD fact: "Someone bought an Apple laptop and some apples from the store"
GOOD entities: [
  {{"text": "Apple", "type": "ORG"}},
  {{"text": "apples", "type": "PRODUCT"}}
]
NOTE: Use type to distinguish "Apple" the company from "apples" the fruit

Input: "The conference starts on Monday at the convention center."
GOOD fact: "The conference starts on Monday at the convention center"
GOOD entities: [
  {{"text": "conference", "type": "OTHER"}},
  {{"text": "Monday", "type": "OTHER"}},
  {{"text": "convention center", "type": "PLACE"}}
]
NOTE: Use OTHER for entities like events (conference) or time references (Monday) that don't fit other categories

Input: "Melanie said 'Yesterday I took the kids to the museum - it was so cool seeing their eyes light up!'"
BAD fact: "The kids were excited about the museum"
BAD entities: [{{"text": "museum", "type": "PLACE"}}]
PROBLEM: Missing WHO (Melanie) and whose kids!

GOOD fact: "Melanie took her kids to the museum yesterday and they were excited, with their eyes lighting up"
GOOD date: Reference date minus 1 day
GOOD entities: [
  {{"text": "Melanie", "type": "PERSON"}},
  {{"text": "museum", "type": "PLACE"}}
]
NOTE: Preserved the subject (Melanie) and possessive relationship (her kids)

Input: "Caroline said 'This necklace is from my grandma in my home country, Sweden. She gave it to me when I was young.'"
BAD fact: "Caroline received a necklace as a gift from her grandmother when she was young"
BAD entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "necklace", "type": "PRODUCT"}}]
PROBLEM: Missing the CRITICAL biographical info that Caroline is from Sweden!

GOOD facts (extract MULTIPLE facts):
1. "Caroline is from Sweden, which is her home country"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}]
2. "Caroline's grandmother is from Sweden"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}]
3. "Caroline received a necklace as a gift from her grandmother in Sweden when she was young"
   entities: [{{"text": "Caroline", "type": "PERSON"}}, {{"text": "Sweden", "type": "PLACE"}}, {{"text": "necklace", "type": "PRODUCT"}}]
NOTE: Extract SEPARATE facts for biographical details (home country) AND events (gift received)

## EXAMPLES of SOCIAL RELATIONSHIPS and NICKNAMES (CRITICAL):

Input: "Joanne was referred to as 'Jo' by Andrey during the meeting."
GOOD fact: "Andrey calls Joanne 'Jo'"
GOOD fact_type: "world"
GOOD date: Reference date (no specific time mentioned)
GOOD entities: [
  {{"text": "Andrey", "type": "PERSON"}},
  {{"text": "Joanne", "type": "PERSON"}}
]
NOTE: This is a FACT about their social relationship, even if it's not an "event"

Input: "Everyone calls him Bobby, but his real name is Robert."
GOOD facts (extract MULTIPLE facts):
1. "People call Robert by the nickname 'Bobby'"
   entities: [{{"text": "Robert", "type": "PERSON"}}]
2. "Robert's real name is Robert (goes by Bobby)"
   entities: [{{"text": "Robert", "type": "PERSON"}}]
NOTE: Extract the social fact about how people refer to him

Input: "Sarah introduced me to Dr. Chen, but she told me to just call him Michael."
GOOD facts (extract MULTIPLE facts):
1. "Sarah introduced someone to Dr. Chen (Michael)"
   entities: [{{"text": "Sarah", "type": "PERSON"}}, {{"text": "Dr. Chen", "type": "PERSON"}}, {{"text": "Michael", "type": "PERSON"}}]
2. "Sarah told someone to call Dr. Chen by his first name Michael"
   entities: [{{"text": "Sarah", "type": "PERSON"}}, {{"text": "Dr. Chen", "type": "PERSON"}}, {{"text": "Michael", "type": "PERSON"}}]
NOTE: Extract both the event (introduction) AND the social relationship fact (how to address him)

Input: "Alex considers Maria his mentor and always refers to her as 'the expert'."
GOOD facts (extract MULTIPLE facts):
1. "Alex considers Maria his mentor"
   entities: [{{"text": "Alex", "type": "PERSON"}}, {{"text": "Maria", "type": "PERSON"}}]
2. "Alex refers to Maria as 'the expert'"
   entities: [{{"text": "Alex", "type": "PERSON"}}, {{"text": "Maria", "type": "PERSON"}}]
NOTE: Capture both the relationship and how Alex refers to Maria

Input: "My grandmother - we call her Nana - lives in Boston."
GOOD facts (extract MULTIPLE facts):
1. "Someone's grandmother lives in Boston"
   entities: [{{"text": "Boston", "type": "PLACE"}}]
2. "Someone and their family call their grandmother 'Nana'"
   entities: []
NOTE: Extract both the biographical fact AND the nickname/term of address

## TEXT TO EXTRACT FROM:
{chunk}

Remember:
1. BE EXHAUSTIVE - Extract EVERY event, action, and fact with FULL CONTEXT
2. **PRESERVE ALL CONTEXT** - photos, visual elements, "new" things, modifiers (new/old/first/favorite)
3. **ALWAYS include the SUBJECT** - never say "the kids" without saying whose kids
4. **Preserve possessive relationships** - "their kids" must become "Person's kids"
5. **Extract biographical details as SEPARATE facts** - "my home country Sweden" → "Person is from Sweden"
6. **Extract SOCIAL RELATIONSHIPS and NICKNAMES** - even if not events, these are facts
7. DO NOT drop modifiers or context - "new hair" stays "new hair", "photo of X" stays "photo of X"
8. Extract absolute dates by calculating relative times from the reference date
9. **CLASSIFY EACH FACT**: 'world' for general facts, 'agent' for AI agent actions
10. Extract ALL entities with types (PERSON, ORG, PLACE, PRODUCT, CONCEPT, OTHER)
11. When in doubt, EXTRACT IT with MORE CONTEXT rather than less"""

    import time
    import logging
    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Retry logic for JSON validation errors
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            llm_call_start = time.time()
            extraction_response = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an EXHAUSTIVE fact and entity extractor. CRITICAL RULES: 1) ALWAYS include the SUBJECT (never 'the kids' without whose kids), 2) **PRESERVE ALL CONTEXT** - photos, 'new' things, modifiers (new/old/first/favorite), visual elements - DO NOT drop these details, 3) Extract biographical details as SEPARATE facts ('my home country Sweden' → 'Person is from Sweden'), 4) **Extract SOCIAL RELATIONSHIPS and NICKNAMES** as facts even if not events ('Andrey calls Joanne Jo'), 5) Extract EVERY event with FULL CONTEXT - 'photo of my new hair' must preserve 'photo' AND 'new', 6) **TRANSFORM RELATIVE DATES IN FACT TEXT**: 'last year' → 'in [year]', 'last month' → 'in [month year]'. Extract ALL entities with types: PERSON, ORG, PLACE, PRODUCT, CONCEPT, OTHER. Preserve possessive relationships (their→whose). When in doubt, include MORE context rather than less - missing context loses critical information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=FactExtractionResponse,
                scope="memory_extract_facts",
                temperature=0.1,
                max_tokens=65000,
                extra_body={"service_tier": "auto"}
            )
            llm_call_time = time.time() - llm_call_start

            # Convert to dict format
            chunk_facts = [fact.model_dump() for fact in extraction_response.facts]

            return chunk_facts

        except BadRequestError as e:
            last_error = e
            if "json_validate_failed" in str(e):
                logger.warning(f"          [1.3.{chunk_index + 1}] Attempt {attempt + 1}/{max_retries} failed with JSON validation error: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"          [1.3.{chunk_index + 1}] Retrying...")
                    continue
            # If it's not a JSON validation error or we're out of retries, re-raise
            raise

    # If we exhausted all retries, raise the last error
    raise last_error


async def extract_facts_from_text(
    text: str,
    event_date: datetime,
    context: str = "",
    llm_config: Optional['LLMConfig'] = None,
    chunk_size: int = 5000
) -> List[Dict[str, str]]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>chunk_size chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        llm_config: LLM configuration to use (if None, uses default from environment)
        chunk_size: Maximum characters per chunk

    Returns:
        List of fact dictionaries with 'fact' and 'date' keys
    """
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .llm_wrapper import LLMConfig

    if llm_config is None:
        from .llm_wrapper import LLMConfig
        llm_config = LLMConfig.for_memory()

    chunks = chunk_text(text, max_chars=chunk_size)
    tasks = [
        _extract_facts_from_chunk(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config
        )
        for i, chunk in enumerate(chunks)
    ]
    chunk_results = await asyncio.gather(*tasks)
    all_facts = []
    for chunk_facts in chunk_results:
        all_facts.extend(chunk_facts)
    return all_facts
