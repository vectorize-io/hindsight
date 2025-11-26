"""
Fact extraction from text using LLM.

Extracts semantic facts, entities, and temporal information from text.
Uses the LLMConfig wrapper for all LLM calls.
"""
import logging
import os
import json
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from .llm_wrapper import OutputTooLongError, LLMConfig


class Entity(BaseModel):
    """An entity extracted from text."""
    text: str = Field(
        description="The specific, named entity as it appears in the fact. Must be a proper noun or specific identifier."
    )


class CausalRelation(BaseModel):
    """Causal relationship between facts."""
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
        default=1.0
    )


class ExtractedFact(BaseModel):
    """A single extracted fact with structured dimensions for comprehensive capture."""

    # Core factual dimension (required)
    factual_core: str = Field(
        description="ACTUAL FACTS - what literally happened/was said. Capture WHAT was said, not just THAT something was said! 'Gina said Jon is the perfect mentor with positivity and determination' NOT 'Jon received encouragement'. Preserve: compliments, assessments, descriptions, key phrases. Be specific!"
    )

    # Optional dimensions - only include if present in the text
    emotional_significance: Optional[str] = Field(
        default=None,
        description="Emotions, feelings, personal meaning, AND qualitative descriptors if present. Include ALL experiential/evaluative terms like 'magical', 'wonderful', 'amazing', 'thrilling'. Examples: 'felt thrilled', 'was her favorite memory', 'it was magical', 'devastating experience', 'proudest moment'"
    )
    reasoning_motivation: Optional[str] = Field(
        default=None,
        description="WHY it happened, intentions, goals, causes if present. Examples: 'because she wanted to celebrate', 'in order to cope with grief', 'motivated by curiosity'"
    )
    preferences_opinions: Optional[str] = Field(
        default=None,
        description="Likes, dislikes, beliefs, values if present. Examples: 'loves coffee', 'thinks AI is transformative', 'prefers working remotely'"
    )
    sensory_details: Optional[str] = Field(
        default=None,
        description="Visual, auditory, physical descriptions AND all descriptive adjectives - USE EXACT WORDS from the text! Don't paraphrase adjectives. If they said 'awesome' write 'awesome' not 'amazing'. Examples: 'bright orange hair', 'so graceful', 'awesome beach', 'epic visuals', 'freezing cold'."
    )
    observations: Optional[str] = Field(
        default=None,
        description="Observations and inferences from the conversation - things that can be deduced but weren't explicitly stated. Includes: travel (if someone is 'shooting in Miami' → they went/will go to Miami), possession implies achievement ('my trophy' → won it), actions imply location/travel ('doing the shoot in Miami' → traveled to Miami), capabilities ('she coded it' → knows programming). Examples: 'Calvin traveled to Miami', 'Gina won dance trophies', 'knows programming'"
    )

    # Fact kind - determines temporal handling (used for prompt engineering, not stored in DB)
    fact_kind: Literal["conversation", "event", "other"] = Field(
        description="Determines if occurred dates should be set. 'conversation' = general info, activities, preferences (NO occurred dates). 'event' = specific datable occurrence like competition, wedding, meeting (HAS occurred_start/end). 'other' = anything else (NO occurred dates). Only 'event' gets occurred dates!"
    )

    # Temporal fields - ONLY for fact_kind='event'
    occurred_start: Optional[str] = Field(
        default=None,
        description="ONLY set when fact_kind='event'. ISO format. Leave null for fact_kind='conversation'."
    )
    occurred_end: Optional[str] = Field(
        default=None,
        description="ONLY set when fact_kind='event'. ISO format. Leave null for fact_kind='conversation'."
    )

    # Classification
    fact_type: Literal["world", "agent", "opinion"] = Field(
        description="'world' = facts about others (third person), 'agent' = facts about YOU the memory owner (FIRST PERSON: 'I did...'), 'opinion' = your beliefs (first person)"
    )

    # Entities and relations
    entities: List[Entity] = Field(
        default_factory=list,
        description="ONLY specific, named entities worth tracking: people's names (e.g., 'Sarah', 'Dr. Smith'), organizations (e.g., 'Google', 'MIT'), specific places (e.g., 'Paris', 'Central Park'). DO NOT include: generic relations (mom, friend, boss, colleague), common nouns (apple, car, house), pronouns (he, she), or vague references (someone, a guy)."
    )
    causal_relations: Optional[List[CausalRelation]] = Field(
        default=None,
        description="Causal links to other facts in this batch. Example: fact about rain causes fact about cancelled game."
    )

    def build_fact_text(self) -> str:
        """Combine all dimensions into a single comprehensive fact string."""
        parts = [self.factual_core]

        if self.emotional_significance:
            parts.append(self.emotional_significance)
        if self.reasoning_motivation:
            parts.append(self.reasoning_motivation)
        if self.preferences_opinions:
            parts.append(self.preferences_opinions)
        if self.sensory_details:
            parts.append(self.sensory_details)
        if self.observations:
            parts.append(self.observations)

        # Join with appropriate connectors
        if len(parts) == 1:
            return parts[0]

        # Combine: "Core fact - emotional/significance context"
        return f"{parts[0]} - {' - '.join(parts[1:])}"


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""
    facts: List[ExtractedFact] = Field(
        description="List of extracted factual statements"
    )


def chunk_text(text: str, max_chars: int) -> List[str]:
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
    llm_config: 'LLMConfig',
    agent_name: str = None,
    extract_opinions: bool = False
) -> List[Dict[str, str]]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).
    """
    # Format event_date for the prompt
    event_date_str = event_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    agent_context = f"\n- Your name: {agent_name}" if agent_name else ""

    # Determine which fact types to extract based on the flag
    if extract_opinions:
        fact_types_instruction = "Extract ONLY 'opinion' type facts (the agent's formed opinions, beliefs, and perspectives). DO NOT extract 'world' or 'agent' facts."
    else:
        fact_types_instruction = "Extract ONLY 'world' and 'agent' type facts. DO NOT extract 'opinion' type facts - opinions should never be created during normal memory storage."

    prompt = f"""You are extracting comprehensive, narrative facts from conversations/document for an AI memory system.

{fact_types_instruction}

## CONTEXT INFORMATION
- Today time: {datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}
- Current document date/time: {event_date_str}
- Context: {context if context else 'no additional context provided'}{agent_context}

## CORE PRINCIPLE: Extract ALL Meaningful Information Efficiently

**GOAL**: Capture ALL meaningful information, but combine related exchanges efficiently. Don't create separate facts for questions - merge Q&A into single facts.

Each fact should:
1. **CAPTURE ALL MEANINGFUL CONTENT** - Activities, projects, preferences, recommendations, encouragement WITH specific content
2. **BE SELF-CONTAINED** - Readable without the original text
3. **PRESERVE SPECIFIC CONTENT** - Capture WHAT was said, not just THAT something was said
4. **COMBINE Q&A** - A question and its answer = ONE fact, not two separate facts

## COMBINE Q&A - CRITICAL!

**❌ BAD (2 separate facts):**
- "James asks what projects John is working on"
- "John is working on a website for a local small business"

**✅ GOOD (1 combined fact):**
- "John is working on a website for a local small business; it's his first professional project outside of class"

**❌ BAD (question as standalone fact):**
- "James asks John what challenges he has encountered"

**✅ GOOD (merged with answer):**
- "John says payment integration was challenging; he used resources to understand the process and is getting closer to a solution"

## WHAT TO SKIP (only these!)

- **Standalone questions** - merge with answers instead
- **Pure filler with no content** - "Always happy to help", "Sounds good", "Thanks!"
- **Greetings** - "Hey!", "What's up?"

## WHAT TO ALWAYS EXTRACT

- Specific encouragement WITH content: "James says hiccups are normal, use them to learn and grow, push through"
- Reactions that reveal preferences: "John says the art is awesome, takes him back to reading fantasy books"
- Recommendations: "John recommends 'The Name of the Wind' - great novel with awesome writing"
- Plans/intentions: "James will check out 'The Name of the Wind'"
- All activities, projects, purchases, events with details

## ESSENTIAL DETAILS TO PRESERVE - NEVER LOSE THESE

When extracting facts, you MUST preserve:

1. **ALL PARTICIPANTS** - Who said/did what
2. **INDIVIDUAL PREFERENCES** - Each person's specific likes/favorites! "Jon's favorite is contemporary because it's expressive" - DO NOT LOSE THIS!
3. **FULL REASONING** - Why decisions were made, motivations, explanations
4. **TEMPORAL CONTEXT - CRITICAL** - ALWAYS convert relative time references to SPECIFIC ABSOLUTE dates in the fact text!
   - "last week" (doc date Aug 23) → "around August 16, 2023" (NOT just "in August 2023"!)
   - "last month" (doc date Aug 2023) → "in July 2023"
   - "yesterday" (doc date Aug 19) → "on August 18, 2023"
   - "next week" (doc date Aug 19) → "around August 26, 2023"
   - "three days ago" (doc date Aug 19) → "on August 16, 2023"
   - "last year" → "in 2022"
   - BE SPECIFIC! "last week" is NOT "in August" - calculate the actual week!
5. **VISUAL/MEDIA ELEMENTS** - Photos, images, videos shared
6. **MODIFIERS** - "new", "first", "old", "favorite" (critical context)
7. **POSSESSIVE RELATIONSHIPS** - "their kids" → "Person's kids"
8. **BIOGRAPHICAL DETAILS** - Origins, locations, jobs, family background
9. **SOCIAL DYNAMICS** - Nicknames, how people address each other, relationships

## STRUCTURED FACT DIMENSIONS - CRITICAL ⚠️

Each fact MUST be extracted into structured dimensions. This ensures no important context is lost.

### Required field:
- **factual_core**: ACTUAL FACTS - capture WHAT was said, not just THAT something was said!
  - ❌ BAD: "Jon received encouragement from Gina" (loses what Gina actually said)
  - ✅ GOOD: "Gina said Jon is the perfect mentor with positivity and determination; his studio will be a hit"
  - ❌ BAD: "Jon supports Gina" (generic)
  - ✅ GOOD: "Gina found the perfect spot for her store; Jon says her hard work is paying off"
  - Preserve: compliments, assessments, descriptions, predictions, key phrases

### Optional fields (include when present in text):
- **emotional_significance**: Emotions, feelings, personal meaning, AND qualitative descriptors
  - Examples: "felt thrilled", "was her favorite memory", "it's magical", "devastating experience", "proudest moment"
  - Captures: emotions, intensity, personal significance, AND experiential descriptors ("magical", "wonderful", "amazing", "thrilling", "beautiful")

- **reasoning_motivation**: WHY it happened, intentions, goals, causes
  - Examples: "because she wanted to celebrate", "in order to cope with grief", "motivated by curiosity"
  - Captures: reasons, intentions, goals, causal explanations

- **preferences_opinions**: Likes, dislikes, beliefs, values, ideals - CAPTURE EACH PERSON'S SPECIFIC PREFERENCES
  - Examples: "Jon's ideal dance studio is by the water", "Jon's favorite dance is contemporary", "loves coffee", "prefers remote work"
  - Captures: preferences, opinions, beliefs, judgments, ideals, dreams
  - PREFERENCE INDICATORS: "ideal", "favorite", "dream", "perfect", "love", "hate", "prefer" → MUST capture in this dimension!
  - CRITICAL: Never lose individual preferences! "Jon's ideal studio is by the water" must be captured!

- **sensory_details**: Visual, auditory, physical descriptions AND all descriptive adjectives - USE EXACT WORDS!
  - Examples: "bright orange hair", "loud music", "freezing cold", "so graceful", "awesome beach", "epic visuals"
  - Captures: colors, sounds, textures, temperatures, appearances, AND adjectives describing people/things/performances
  - CRITICAL: Use the EXACT adjectives from the text! If they said "awesome" don't write "amazing". If they said "epic" don't write "perfect"!

- **observations**: Things that can be inferred/deduced from the conversation - not explicitly stated but clearly implied
  - TRAVEL: "doing the shoot in Miami" → "Calvin traveled/will travel to Miami"
  - POSSESSION: "my trophy" → "won the trophy"
  - CAPABILITIES: "she coded it" → "knows programming"
  - Examples: "Calvin traveled to Miami for the shoot", "Gina won dance trophies", "knows programming"

### Example extraction:

**Input**: "I used to compete in dance competitions - my fav memory was when my team won first place at regionals at age fifteen. It was an awesome feeling of accomplishment!"

**Output**:
```
factual_core: "Gina's team won first place at a regional dance competition when she was 15"
emotional_significance: "this was her favorite memory; felt an awesome sense of accomplishment"
reasoning_motivation: null
preferences_opinions: null
sensory_details: null
```

### CRITICAL: Never strip away dimensions!
- ❌ BAD: Only extracting factual_core and ignoring emotional context
- ✅ GOOD: Capturing ALL dimensions present in the text

## FACT KIND AND TEMPORAL RULES

### fact_kind determines if occurred dates are set:

**`conversation`** - General info, activities, preferences, ongoing things
- NO occurred_start/end (leave null)
- Examples: "Jon is expanding his studio", "Jon loves dance", "Gina's ideal studio is by water"

**`event`** - Specific datable occurrence (competition, wedding, meeting, trip, loss, start/end of something)
- MUST set occurred_start/end
- Ask: "Is this a SPECIFIC EVENT with a DATE?"
- Examples: "Dance competition on May 15", "Lost job in January 2023", "Wedding next Saturday"

**`other`** - Anything else that doesn't fit above
- NO occurred_start/end (leave null)
- Catch-all to not lose information

### Rules:
1. **ALWAYS include dates in fact text** - "in January 2023", "on May 15, 2024"
2. **Only 'event' gets occurred dates** - conversation and other = null
3. **SPLIT events from conversation facts** - "Jon is expanding his studio (conversation) and hosting a competition next month (event)" → 2 separate facts!

## CAUSAL RELATIONSHIPS

When splitting related facts, link them with causal_relations:
- **causes**: This fact causes the target
- **caused_by**: This fact was caused by target
- **enables/prevents**: This fact enables/prevents the target

Only link when there's explicit or clear implicit causation ("because", "so", "therefore").

## FACT TYPE CLASSIFICATION

- **'world'**: Facts about others (third person)
- **'agent'**: Facts about YOU the memory owner (FIRST PERSON: "I did...", "I said...")
- **'opinion'**: Your beliefs/perspectives (first person: "I believe...")

**Speaker attribution**: If context says "Your name: Marcus", only extract 'agent' facts from "Marcus:" lines.

## WHAT TO SKIP
- Greetings, filler words, pure reactions ("wow", "cool")
- Structural statements ("let's get started", "see you next time")
- Calls to action ("subscribe", "follow")

## EXAMPLE: SPLITTING CONVERSATION VS EVENT FACTS

**Input (conversation date: April 3, 2023):**
"I'm expanding my dance studio's social media presence and offering workshops to local schools. I'm also hosting a dance competition next month to showcase local talent. The dancers are so excited!"

**Output (2 facts - conversation + event):**

**Fact 1 (kind=conversation - ongoing activities, no occurred dates):**
```
fact_kind: "conversation"
factual_core: "Jon is expanding his dance studio's social media presence in April 2023; offering workshops and classes to local schools and centers; seeing progress and dancers are excited"
emotional_significance: "excited and proud of progress"
preferences_opinions: "Jon loves giving dancers a place to express themselves"
observations: "Jon owns/runs a dance studio"
occurred_start: null  ← conversation kind = no occurred dates
occurred_end: null
```

**Fact 2 (kind=event - specific datable occurrence):**
```
fact_kind: "event"
factual_core: "Jon will host a dance competition in May 2023 to showcase local talent and bring attention to his studio"
emotional_significance: "excited about the event"
occurred_start: "2023-05-01T00:00:00Z"  ← event kind = HAS occurred dates
occurred_end: "2023-05-31T23:59:59Z"
```

**❌ BAD:** Combining both into one fact with occurred=May (makes ongoing activities look like they happened in May!)

## TEXT TO EXTRACT FROM:
{chunk}

## CRITICAL REMINDERS:
1. **COMBINE Q&A** - Never create standalone question facts! Merge questions with their answers into single facts.
2. **CAPTURE ALL MEANINGFUL CONTENT** - Activities, encouragement (with specific words!), recommendations, reactions, preferences
3. **CONVERT RELATIVE DATES TO SPECIFIC DATES** - "last week" → "around August 16" (NOT "in August"!), "yesterday" → "on August 18". Be precise!
4. **CAPTURE WHAT WAS SAID** - "Gina said Jon is perfect mentor with determination" NOT "Jon received encouragement". Preserve the actual content!
5. **FACT_KIND DETERMINES OCCURRED DATES** - Only 'event' gets occurred_start/end. 'conversation' and 'other' = null
6. **CAPTURE PREFERENCES** - "ideal", "favorite", "love" → preferences_opinions
7. **CAPTURE EXACT ADJECTIVES** - Use the EXACT words! "awesome" not "amazing", "epic" not "perfect" → sensory_details
8. **CAPTURE OBSERVATIONS** - "shooting in Miami" → observations: "traveled to Miami". Infer travel, achievements, capabilities!"""

    import logging
    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Retry logic for JSON validation errors
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            extraction_response = await llm_config.call(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract ALL meaningful content. COMBINE Q&A into single facts (no standalone questions!). Skip only greetings and pure filler. CONVERT RELATIVE DATES TO SPECIFIC DATES ('last week' → 'around Aug 16' NOT 'in August'!). factual_core = WHAT was said, not THAT something was said! fact_kind: 'conversation'/'event'/'other'. Only 'event' gets occurred dates."
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
            )
            # Build combined fact text from dimensions and include in output
            chunk_facts = []
            for fact in extraction_response.facts:
                fact_dict = fact.model_dump()
                # Add combined 'fact' field from structured dimensions
                fact_dict['fact'] = fact.build_fact_text()

                # Safety net: strip occurred dates if fact_kind is not 'event'
                # (in case LLM doesn't follow the rules)
                if fact_dict.get('fact_kind') != 'event':
                    fact_dict['occurred_start'] = None
                    fact_dict['occurred_end'] = None

                # Remove fact_kind from output (only used for prompt engineering, not stored)
                fact_dict.pop('fact_kind', None)

                chunk_facts.append(fact_dict)
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


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime,
    context: str,
    llm_config: LLMConfig,
    agent_name: str = None,
    extract_opinions: bool = False
) -> List[Dict[str, str]]:
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
        List of fact dictionaries extracted from the chunk (possibly from sub-chunks)
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
            extract_opinions=extract_opinions
        )
    except OutputTooLongError as e:
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

        sentence_endings = ['. ', '! ', '? ', '\n\n']
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
            f"Split chunk {chunk_index + 1} into two sub-chunks: "
            f"{len(first_half)} chars and {len(second_half)} chars"
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
                extract_opinions=extract_opinions
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                agent_name=agent_name,
                extract_opinions=extract_opinions
            )
        ]

        sub_results = await asyncio.gather(*sub_tasks)

        # Combine results from both halves
        all_facts = []
        for sub_result in sub_results:
            all_facts.extend(sub_result)

        logger.info(
            f"Successfully extracted {len(all_facts)} facts from split chunk {chunk_index + 1}"
        )

        return all_facts


async def extract_facts_from_text(
    text: str,
    event_date: datetime,
    llm_config: LLMConfig,
    agent_name: str,
    context: str = "",
    extract_opinions: bool = False,
) -> List[Dict[str, str]]:
    """
    Extract semantic facts from conversational or narrative text using LLM.

    For large texts (>chunk_size chars), automatically chunks at sentence boundaries
    to avoid hitting output token limits. Processes ALL chunks in PARALLEL for speed.

    If a chunk produces output that exceeds token limits (OutputTooLongError), it is
    automatically split in half and retried recursively until successful.

    Args:
        text: Input text (conversation, article, etc.)
        event_date: Reference date for resolving relative times
        context: Context about the conversation/document
        llm_config: LLM configuration to use (if None, uses default from environment)
        chunk_size: Maximum characters per chunk
        agent_name: Optional agent name (memory owner)
        extract_opinions: If True, extract ONLY opinions. If False, extract world and agent facts (no opinions)

    Returns:
        List of fact dictionaries with 'fact' and 'date' keys
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
            extract_opinions=extract_opinions
        )
        for i, chunk in enumerate(chunks)
    ]
    chunk_results = await asyncio.gather(*tasks)
    all_facts = []
    for chunk_facts in chunk_results:
        all_facts.extend(chunk_facts)
    return all_facts
