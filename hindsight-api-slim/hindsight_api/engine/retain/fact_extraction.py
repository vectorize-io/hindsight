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
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from ..llm_interface import ProviderRateLimitResetError
from ..llm_wrapper import LLMConfig, OutputTooLongError, parse_llm_json, sanitize_llm_output
from ..operation_metadata import RetainExtractionErrors
from ..response_models import TokenUsage
from ..structured_output import strict_json_schema
from .entity_labels import (
    EntityLabelsConfig,
    MapField,
    build_labels_lookup,
    build_labels_model,
    is_label_entity,
    parse_entity_labels,
)


def _extract_map_entities(
    entity_obj: dict,
    fields: dict[str, MapField],
    prefix: str,
    validated_entities: list[str],
    existing_texts_lower: set[str],
) -> None:
    """Recursively extract key:field:value entity strings from a map entity dict."""
    for field_name, map_field in fields.items():
        field_val = entity_obj.get(field_name)
        if field_val is None or field_val == "":
            continue
        if map_field.type == "map" and map_field.fields:
            # Nested map: recurse into each sub-entity
            sub_list = field_val if isinstance(field_val, list) else [field_val]
            for sub_obj in sub_list:
                if isinstance(sub_obj, dict):
                    _extract_map_entities(
                        sub_obj,
                        map_field.fields,
                        f"{prefix}{field_name}:",
                        validated_entities,
                        existing_texts_lower,
                    )
        elif map_field.type == "multi-values":
            vals = field_val if isinstance(field_val, list) else [field_val]
            for v in vals:
                if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                    continue
                label_str = f"{prefix}{field_name}:{v.strip()}"
                if label_str.lower() not in existing_texts_lower:
                    validated_entities.append(label_str)
                    existing_texts_lower.add(label_str.lower())
        else:
            # text or value — single string
            if not isinstance(field_val, str) or not field_val.strip() or field_val.lower() in ("none", "null", "n/a"):
                continue
            label_str = f"{prefix}{field_name}:{field_val.strip()}"
            if label_str.lower() not in existing_texts_lower:
                validated_entities.append(label_str)
                existing_texts_lower.add(label_str.lower())


def _infer_temporal_date(fact_text: str, event_date: datetime | None) -> str | None:
    """
    Infer a temporal date from fact text when LLM didn't provide occurred_start.

    This is a fallback for when the LLM fails to extract temporal information
    from relative time expressions like "last night", "yesterday", etc.
    """
    if event_date is None:
        return None

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


def _sanitize_text(text: str | None) -> str | None:
    return sanitize_llm_output(text)


def _coerce_entity_strings(v: Any) -> Any:
    """
    Normalize the LLM's `entities` field to a plain list of strings.

    The schema previously asked for `Entity` objects ({"text": "..."}) while the
    prompt's few-shot examples taught a flat string array. Models that followed
    the examples literally returned strings, and the entities were silently
    dropped — none were ever persisted (#2749). The `Entity` wrapper carried no
    information beyond the string, so it was removed rather than taught to the
    prompt; the object form is still unwrapped here for models that learned it
    and for in-flight batch jobs.

    Returns non-list input untouched so pydantic reports the type error itself.
    """
    if v is None:
        return []
    if not isinstance(v, list):
        return v
    coerced = []
    for item in v:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                coerced.append(text)
        else:
            coerced.append(item)
    return coerced


class Fact(BaseModel):
    """
    Final fact model for storage - built from lenient parsing of LLM response.

    This is what fact_extraction returns and what the rest of the pipeline expects.
    Combined fact text format: "what | when | where | who | why"
    """

    # Required fields
    fact: str = Field(description="Combined fact text: what | when | where | who | why")
    fact_type: Literal["world", "experience"] = Field(description="Perspective: world/experience")

    # Optional temporal fields
    occurred_start: str | None = None
    occurred_end: str | None = None

    # Optional location field
    where: str | None = Field(
        None, description="WHERE the fact occurred or is about (specific location, place, or area)"
    )

    # Optional structured data
    entities: list[str] | None = None
    causal_relations: list["CausalRelation"] | None = None


class CausalRelation(BaseModel):
    """Causal relationship from this fact to a previous fact (stored format)."""

    target_fact_index: int = Field(description="Index of the related fact in the facts array (0-based).")
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target: 'caused_by' = this fact was caused by the target"
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
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts, including user preferences, rules, corrections, and constraints even when stated during a conversation. 'assistant' = actions, experiences, or observations the assistant/agent actually performed."
    )
    entities: list[str] = Field(
        default_factory=list, description='People, places, concepts - plain strings, e.g. ["Alice", "Kubernetes"]'
    )
    causal_relations: list[FactCausalRelation] | None = Field(
        default=None, description="Links to previous facts (target_index < this fact's index)"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)

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


def _split_chunk_for_output_retry(chunk: str) -> tuple[str, str] | None:
    """Split an oversized extraction chunk without corrupting structured input."""
    stripped = chunk.strip()
    if len(stripped) <= 1:
        return None

    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, list):
        if len(parsed) >= 2:
            mid = len(parsed) // 2
            return json.dumps(parsed[:mid]), json.dumps(parsed[mid:])

        if len(parsed) == 1 and isinstance(parsed[0], dict):
            turn = parsed[0]
            content = turn.get("content")
            if isinstance(content, str) and len(content) > 1:
                cut = len(content) // 2
                first_turn = dict(turn)
                second_turn = dict(turn)
                first_turn["content"] = content[:cut]
                second_turn["content"] = content[cut:]
                return json.dumps([first_turn]), json.dumps([second_turn])

        return None

    # Split plain text at the midpoint, preferring sentence boundaries nearby.
    mid_point = len(stripped) // 2
    search_range = int(len(stripped) * 0.2)
    search_start = max(0, mid_point - search_range)
    search_end = min(len(stripped), mid_point + search_range)

    best_split = mid_point
    for ending in [". ", "! ", "? ", "\n\n"]:
        pos = stripped.rfind(ending, search_start, search_end)
        if pos != -1:
            best_split = pos + len(ending)
            break

    first_half = stripped[:best_split].strip()
    second_half = stripped[best_split:].strip()
    if not first_half or not second_half or first_half == stripped or second_half == stripped:
        return None
    return first_half, second_half


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
        description="'world' = objective/external facts about the user, other people, events, general knowledge, preferences, rules, corrections, or constraints. 'assistant' = actions, experiences, or observations the assistant/agent actually performed (e.g., 'I changed X', 'I discovered Y')."
    )

    entities: list[str] = Field(
        default_factory=list,
        description="Named entities, objects, AND abstract concepts from the fact, as plain strings (e.g. [\"Alice\", \"friendship\"]). Include: people names, organizations, places, significant objects (e.g., 'coffee maker', 'car'), AND abstract concepts/themes (e.g., 'friendship', 'career growth', 'loss', 'celebration'). Extract anything that could help link related facts together.",
    )

    causal_relations: list[FactCausalRelation] | None = Field(
        default=None,
        description="Causal links to PREVIOUS facts only. target_index MUST be less than this fact's position. "
        "Example: fact #3 can only reference facts 0, 1, or 2. Max 2 relations per fact.",
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


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
        description="'world' = about the user/others, including user preferences, rules, corrections, and constraints. 'assistant' = actions or experiences the assistant/agent actually performed."
    )
    entities: list[str] = Field(
        default_factory=list,
        description='Named entities, objects, and concepts from the fact, as plain strings (e.g. ["Alice", "Kubernetes"]).',
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class FactExtractionResponseNoCausal(BaseModel):
    """Response for fact extraction without causal relations."""

    facts: list[ExtractedFactNoCausal] = Field(description="List of extracted factual statements")


class VerbatimExtractedFact(BaseModel):
    """
    Schema for verbatim extraction mode.

    Omits 'what' entirely — the original chunk text is used as fact_text in code.
    The LLM only extracts metadata: entities, temporal info, location, people.
    """

    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["when", "where", "who", "fact_type"]},
    )

    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(
        description="'world' = objective/external facts. 'assistant' = first-person actions, experiences, or observations by the speaker."
    )
    entities: list[str] = Field(
        default_factory=list, description='People, places, concepts - plain strings, e.g. ["Alice", "Kubernetes"]'
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        return _coerce_entity_strings(v)


class VerbatimFactExtractionResponse(BaseModel):
    """Response for verbatim extraction mode (one entry per chunk, no fact text)."""

    facts: list[VerbatimExtractedFact] = Field(description="List of metadata entries (one per chunk)")


# Separators for sentence-aware recursive text splitting, ordered most- to
# least-preferred. The final "" lets the splitter break mid-word as a last
# resort so a chunk can never exceed the size budget.
_RECURSIVE_TEXT_SEPARATORS = [
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    ". ",  # Sentence endings
    "! ",  # Exclamations
    "? ",  # Questions
    "; ",  # Semicolons
    ", ",  # Commas
    " ",  # Words
    "",  # Characters (last resort)
]


def _split_oversized_unit(text: str, max_chars: int) -> list[str]:
    """Sentence-aware split of a single unit that overflowed the budget.

    Used when one JSONL line / conversation turn is so large it can't be kept
    whole within the configured structured-chunk limit. The resulting fragments
    are no longer valid JSON, but the fact extractor treats every chunk as plain
    text.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=_RECURSIVE_TEXT_SEPARATORS,
    )
    return splitter.split_text(text)


def chunk_text(text: str, max_chars: int, structured_chunk_size: int | None = None) -> list[str]:
    """
    Split text into chunks, preserving conversation structure when possible.

    For JSON conversation arrays (user/assistant turns) and JSONL (newline-delimited
    JSON objects), splits at turn/line boundaries so no object is split across chunks.
    A single turn/line that overflows ``max_chars`` is kept whole only up to
    ``structured_chunk_size``. When unset, that limit defaults to ``max_chars``.
    For plain text, uses sentence-aware splitting.

    The result is idempotent: re-chunking any chunk this returns yields that chunk
    unchanged. The streaming retain pipeline pre-chunks each document once and then
    re-chunks every piece during extraction; if a piece re-split, its sub-chunks
    would inherit one chunk_index and collide on ``chunk_id`` (issue #2301).

    Args:
        text: Input text to chunk (plain text, JSON conversation, or JSONL)
        max_chars: Target maximum characters per chunk
        structured_chunk_size: Maximum characters for a single JSONL line or
            conversation turn to keep whole. Defaults to ``max_chars``.

    Returns:
        List of text chunks, roughly under max_chars
    """
    # If text is small enough, return as-is
    if len(text) <= max_chars:
        return [text]

    structured_limit = structured_chunk_size if structured_chunk_size is not None else max_chars

    # Try to parse as JSON conversation array
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None

    if isinstance(parsed, list) and all(isinstance(turn, dict) for turn in parsed):
        # This looks like a conversation - chunk at turn boundaries
        return _chunk_conversation(parsed, max_chars, structured_limit)

    if isinstance(parsed, dict):
        # A single JSON object — e.g. one JSONL line handed back to the extractor
        # after the producer already pre-chunked it. It is one structured unit:
        # keep it whole up to the structured limit, else split it as text within
        # the chunk budget. Without this, a lone object (one line, so _chunk_jsonl
        # declines) would fall through to plain-text splitting and re-split a chunk
        # the producer deliberately kept whole — breaking idempotency (issue #2301).
        if len(text) <= structured_limit:
            return [text]
        return _split_oversized_unit(text, max_chars)

    # Try to parse as JSONL (newline-delimited JSON objects, e.g. session logs)
    jsonl_chunks = _chunk_jsonl(text, max_chars, structured_limit)
    if jsonl_chunks is not None:
        return jsonl_chunks

    # Fall back to sentence-aware text splitting
    return _split_oversized_unit(text, max_chars)


def _chunk_conversation(turns: list[dict], max_chars: int, structured_limit: int) -> list[str]:
    """
    Chunk a conversation array at turn boundaries, preserving complete turns.

    Args:
        turns: List of conversation turn dicts (with 'role' and 'content' keys)
        max_chars: Maximum characters per chunk
        structured_limit: Maximum characters for a single turn to keep whole

    Returns:
        List of JSON-serialized chunks, each containing complete turns
    """

    chunks = []
    current_chunk = []
    current_size = 2  # Account for "[]"

    def _flush() -> None:
        nonlocal current_chunk, current_size
        if current_chunk:
            chunks.append(json.dumps(current_chunk, ensure_ascii=False))
            current_chunk = []
            current_size = 2  # Reset to "[]"

    for turn in turns:
        # Estimate size of this turn when serialized (with comma separator)
        turn_json = json.dumps(turn, ensure_ascii=False)
        turn_unit_size = len(turn_json)
        turn_size = turn_unit_size + 1  # +1 for comma

        # A turn too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if turn_unit_size > structured_limit:
            _flush()
            chunks.extend(_split_oversized_unit(turn_json, min(structured_limit, max_chars)))
            continue

        # If adding this turn would exceed limit and we have turns, save current chunk
        if current_size + turn_size > max_chars and current_chunk:
            _flush()

        # Add turn to current chunk
        current_chunk.append(turn)
        current_size += turn_size

    # Add final chunk if non-empty
    _flush()

    return chunks if chunks else [json.dumps(turns, ensure_ascii=False)]


def _chunk_jsonl(text: str, max_chars: int, structured_limit: int) -> list[str] | None:
    """Chunk newline-delimited JSON (JSONL) at line boundaries.

    Detects JSONL — two or more non-empty lines, each a complete JSON object —
    and packs whole lines into chunks so no line is split across chunks (multiple
    short lines may share a chunk). A line that overflows ``max_chars`` is kept
    whole only up to ``structured_limit``. Returns ``None`` if the input is not
    JSONL, so the caller falls back to plain-text splitting.

    Args:
        text: Input text to inspect/chunk.
        max_chars: Maximum characters per chunk.
        structured_limit: Maximum characters for a single JSONL line to
            keep whole.

    Returns:
        List of JSONL chunks (lines joined by newline), or ``None`` if not JSONL.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    for line in lines:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(obj, dict):
            return None

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    def _flush() -> None:
        nonlocal current_chunk, current_size
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

    for line in lines:
        line_unit_size = len(line)
        line_size = len(line) + 1  # +1 for the joining newline

        # A line too large to keep whole even alone: flush, then split it as
        # text. Fragment within min(structured_limit, max_chars) so no fragment
        # exceeds the chunk budget — otherwise a downstream re-chunk would split
        # it again and collide on chunk_id (issue #2301).
        if line_unit_size > structured_limit:
            _flush()
            chunks.extend(_split_oversized_unit(line, min(structured_limit, max_chars)))
            continue

        # If adding this line would exceed the limit and we have lines, flush.
        # A line up to structured_limit is kept whole (a bounded overflow).
        if current_size + line_size > max_chars and current_chunk:
            _flush()

        current_chunk.append(line)
        current_size += line_size

    _flush()

    return chunks


# =============================================================================
# FACT EXTRACTION PROMPTS
# =============================================================================

# Base prompt template (shared by concise and custom modes)
# Uses {extraction_guidelines} placeholder for mode-specific instructions
_BASE_FACT_EXTRACTION_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}{extraction_guidelines}

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. "what": Core fact - concise but complete (1-2 sentences max)
2. "when": Temporal info if mentioned. "N/A" if none. Use day name when known.
3. "where": Location if relevant. "N/A" if none.
4. "who": People involved with relationships. "N/A" if just general info.
5. "why": Context/significance ONLY if important. "N/A" if obvious.

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
- "world": Objective/external facts, including the user's preferences, rules, corrections, constraints, plans, traits, or context. These stay "world" even when the user states them during an assistant interaction (e.g., "User prefers browser_navigate over web_search", "User corrected the project deadline").
- "assistant": Actions, experiences, or observations the assistant/agent actually performed (e.g., "I changed X", "I discovered Y", "I debugged Z"). Use this for the assistant/agent doing, trying, learning, deciding, recommending, or responding — not merely for user facts mentioned in conversation.

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- CRITICAL: Convert ALL relative temporal expressions to absolute dates in the fact text itself.
  "yesterday" → write the resolved date (e.g. "on November 12, 2024"), NOT the word "yesterday"
  "last night", "this morning", "today", "tonight" → convert to the resolved absolute date
- For events: set occurred_start AND occurred_end (same for point events)
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

ALWAYS return "entities" as an array of plain strings — never objects, never null.
Correct: entities=["Alice", "Kubernetes", "CKA"]
Wrong:   entities as an array of objects with a "text" key ← never use this form
Use an empty array [] only when the fact truly names nothing.

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
EXAMPLES (shown in English for illustration; for non-English input, ALL output values MUST be in the input language)
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

Ask: "Would this be useful to recall in 6 months?" If no, skip it.

IMPORTANT: Sensory/emotional details and observations that provide meaningful context
about experiences ARE important to remember, even if they seem small (e.g., how food
tasted, how someone looked, how loud music was). Extract these if they characterize
an experience or person."""

# Assembled concise prompt
CONCISE_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_CONCISE_GUIDELINES,
    examples=_CONCISE_EXAMPLES,
)

# Custom prompt uses same base but without examples
CUSTOM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines="{custom_instructions}",
    examples="",  # No examples for custom mode
)

# Verbatim mode: preserve the original text exactly, but still extract metadata
_VERBATIM_GUIDELINES = """══════════════════════════════════════════════════════════════════════════
VERBATIM MODE — Extract metadata only
══════════════════════════════════════════════════════════════════════════

The original text will be stored as-is in code. Your ONLY job is to extract metadata.

RULES:
- Produce EXACTLY ONE entry per input chunk.
- DO NOT include a "what" field — it is not part of the output schema.
- Extract all entities (people, places, organizations, objects, concepts).
- Extract temporal information (occurred_start, occurred_end, fact_kind, when).
- Extract location (where) and people (who).
- fact_type: use "world" for user preferences, rules, corrections, constraints, traits, and other objective facts, even when stated during an assistant interaction. Use "assistant" only for actions or experiences the assistant/agent actually performed."""

VERBATIM_FACT_EXTRACTION_PROMPT = _BASE_FACT_EXTRACTION_PROMPT.format(
    retain_mission_section="{retain_mission_section}",
    extraction_guidelines=_VERBATIM_GUIDELINES,
    examples="",
)


# Verbose extraction prompt - detailed, comprehensive facts (legacy mode)
VERBOSE_FACT_EXTRACTION_PROMPT = """Extract facts from text into structured format with FIVE required dimensions - BE EXTREMELY DETAILED.

LANGUAGE: MANDATORY — Detect the language of the input text and produce ALL output in that EXACT same language. You are STRICTLY FORBIDDEN from translating or switching to any other language. Every single word of your output must be in the same language as the input. Do NOT output in a different language under any circumstance.

{retain_mission_section}══════════════════════════════════════════════════════════════════════════
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
- CRITICAL: If the content mentions an absolute date (e.g., "March 15, 2024", "2024-03-15"), you MUST extract it and set occurred_start in ISO format
- Always include the day name (Monday, Tuesday, etc.) in the 'when' field
- Set occurred_start AND occurred_end to WHEN IT HAPPENED (not when mentioned)
- For single-day/point events: set occurred_end = occurred_start (same timestamp)

For CONVERSATIONS (fact_kind="conversation"):
- General info, preferences, ongoing states → NO occurred dates
- Examples: "loves coffee", "works as engineer"

══════════════════════════════════════════════════════════════════════════
FACT TYPE
══════════════════════════════════════════════════════════════════════════

- **world**: User's life, preferences, rules, corrections, constraints, other people, and events (facts that would exist without this conversation)
- **assistant**: Actions or experiences the assistant/agent actually performed while helping the user (requests, recommendations, help)
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


def _append_map_fields_prompt(fields: dict[str, "MapField"], lines: list[str], indent: int = 4) -> None:
    """Recursively append map field descriptions to the prompt lines."""
    pad = " " * indent
    for field_name, map_field in fields.items():
        field_desc = f": {map_field.description}" if map_field.description else ""
        if map_field.type == "map" and map_field.fields:
            lines.append(f"{pad}• {field_name} (object){field_desc}")
            _append_map_fields_prompt(map_field.fields, lines, indent + 4)
        elif map_field.type == "multi-values":
            vals = ", ".join(v.value for v in map_field.values if v.value)
            type_hint = f"multi-values: {vals}" if vals else "multi-values"
            lines.append(f"{pad}• {field_name} ({type_hint}){field_desc}")
        elif map_field.type == "value" and map_field.values:
            vals = ", ".join(v.value for v in map_field.values if v.value)
            lines.append(f"{pad}• {field_name} (one of: {vals}){field_desc}")
        else:
            lines.append(f"{pad}• {field_name} (text){field_desc}")


def _build_labels_prompt_section(labels_cfg: EntityLabelsConfig | list | None, free_form_entities: bool = True) -> str:
    """Build the entity labels classification section for the extraction prompt."""
    if labels_cfg is None:
        return ""

    # Accept raw list for backwards compatibility
    if isinstance(labels_cfg, list):
        if not labels_cfg:
            return ""
        labels_cfg = parse_entity_labels(labels_cfg)
        if labels_cfg is None:
            return ""

    if not labels_cfg.attributes:
        return ""

    if free_form_entities:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Continue extracting regular named entities in the 'entities' field."
    else:
        entities_instruction = "Classify each fact using the structured 'labels' field below. Do NOT add regular named entities — labels-only mode."

    lines = [
        "\n\n══════════════════════════════════════════════════════════════════════════",
        "ENTITY LABELS - CLASSIFICATION ATTRIBUTES",
        "══════════════════════════════════════════════════════════════════════════",
        "",
        entities_instruction,
        "",
        "For each fact, fill the 'labels' object. Each field is a label group:",
        "",
    ]

    has_classification_attrs = False
    has_map_attrs = False

    for attr in labels_cfg.attributes:
        if attr.type == "map":
            has_map_attrs = True
            continue
        has_classification_attrs = True
        if attr.type == "text":
            # Free-text: no predefined values — LLM writes any relevant string or null
            lines.append(f"- {attr.key} (free text or null): {attr.description}")
        else:
            mode = "multi-value (list)" if attr.type == "multi-values" else "single value or null"
            lines.append(f"- {attr.key} ({mode}): {attr.description}")
            for v in attr.values:
                desc = f" — {v.description}" if v.description else ""
                lines.append(f'    • "{v.value}"{desc}')
        lines.append("")

    if has_classification_attrs:
        lines.append("Only assign labels when clearly applicable. Leave null/empty if the fact does not match.")
        lines.append("")

    # Add structured entity types (map groups)
    if has_map_attrs:
        lines.append("")
        lines.append("══════════════════════════════════════════════════════════════════════════")
        lines.append("STRUCTURED ENTITY TYPES")
        lines.append("══════════════════════════════════════════════════════════════════════════")
        lines.append("")
        lines.append("For each fact, extract structured entities into the corresponding list field in 'labels'.")
        lines.append("Each structured entity type has defined fields. Return a list of objects, one per entity found.")
        lines.append("")
        for attr in labels_cfg.attributes:
            if attr.type != "map" or not attr.fields:
                continue
            desc = f": {attr.description}" if attr.description else ""
            lines.append(f"- {attr.key}{desc}")
            _append_map_fields_prompt(attr.fields, lines, indent=4)
            lines.append("")
        lines.append(
            "Only extract structured entities when clearly present in the text. Leave the list empty if none found."
        )

    return "\n".join(lines)


def _build_extraction_prompt_and_schema(config) -> tuple[str, type]:
    """
    Build extraction prompt and response schema based on config.

    When a taxonomy is configured, dynamically builds a Pydantic model with a
    typed `taxonomy_entities` field using an Enum built from valid taxonomy values.
    This enables JSON schema enforcement for structured outputs.

    Returns:
        Tuple of (prompt, response_schema)
    """
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # The per-bank retain mission is NOT baked into this system prompt: it would
    # make the prompt bank-specific and force a separate Gemini context cache per
    # mission (one per bank). Instead the prompt is bank-agnostic so a single
    # CachedContent serves every bank, and the mission rides in the per-request
    # user message via _retain_mission_preamble(). The {retain_mission_section}
    # placeholder is kept (templates still reference it) but always empty here.
    from hindsight_api.engine.prompt_utils import escape_for_prompt

    retain_mission_section = ""

    # Select base prompt based on extraction mode
    if extraction_mode == "custom":
        if not config.retain_custom_instructions:
            base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
            )
        else:
            base_prompt = CUSTOM_FACT_EXTRACTION_PROMPT
            prompt = base_prompt.format(
                retain_mission_section=retain_mission_section,
                custom_instructions=escape_for_prompt(config.retain_custom_instructions),
            )
    elif extraction_mode == "verbose":
        prompt = VERBOSE_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    elif extraction_mode == "verbatim":
        prompt = VERBATIM_FACT_EXTRACTION_PROMPT.format(
            retain_mission_section=retain_mission_section,
        )
    else:
        base_prompt = CONCISE_FACT_EXTRACTION_PROMPT
        prompt = base_prompt.format(
            retain_mission_section=retain_mission_section,
        )

    # Add causal relationships section if enabled
    # Verbatim mode never uses causal relations (no fact text to relate causally)
    if extraction_mode == "verbatim":
        base_fact_class = VerbatimExtractedFact
        base_response_class = VerbatimFactExtractionResponse
    elif extract_causal_links:
        prompt = prompt + CAUSAL_RELATIONSHIPS_SECTION
        base_fact_class = ExtractedFactVerbose if extraction_mode == "verbose" else ExtractedFact
        base_response_class = FactExtractionResponseVerbose if extraction_mode == "verbose" else FactExtractionResponse
    else:
        base_fact_class = ExtractedFactNoCausal
        base_response_class = FactExtractionResponseNoCausal

    # Add entity labels section if configured and build dynamic schema
    entity_labels_raw = getattr(config, "entity_labels", None)
    labels_cfg = parse_entity_labels(entity_labels_raw)
    free_form_entities = getattr(config, "entities_allow_free_form", True)
    labels_section = _build_labels_prompt_section(labels_cfg, free_form_entities)
    if labels_section:
        prompt = prompt + labels_section

    # Force the LLM to emit fact text in the configured language, regardless of
    # the source content's language. Same directive is applied to consolidation
    # and reflect so HINDSIGHT_API_LLM_OUTPUT_LANGUAGE has a uniform effect
    # across the pipeline. This is independent of the BM25 indexing language
    # (HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE) by design — search
    # tokenization and LLM output language are separate concerns.
    from ..prompt_utils import output_language_directive

    prompt = prompt + output_language_directive(getattr(config, "llm_output_language", None))

    response_schema = base_response_class

    if labels_cfg and labels_cfg.attributes:
        LabelsModel = build_labels_model(labels_cfg)
        if LabelsModel is not None:
            dynamic_fields: dict = {
                "labels": (
                    LabelsModel,
                    Field(
                        description="Classification labels for this fact. Fill each applicable field; leave others null/empty."
                    ),
                )
            }
            if not free_form_entities:
                dynamic_fields["entities"] = (
                    list[str],
                    Field(default_factory=list, description="Leave empty — labels-only mode"),
                )
            # Inherit parent's required fields and add 'labels' so it appears in the JSON schema
            # required array (the base class json_schema_extra overrides required entirely)
            base_extra = base_fact_class.model_config.get("json_schema_extra")
            base_required = cast(dict, base_extra).get("required", []) if isinstance(base_extra, dict) else []
            DynamicFact = create_model(
                "LabelsFact",
                __base__=base_fact_class,
                __config__=ConfigDict(
                    json_schema_mode="validation",
                    json_schema_extra={"required": [*base_required, "labels"]},
                ),
                **dynamic_fields,
            )
            DynamicResponse = create_model("LabelsResponse", facts=(list[DynamicFact], ...))  # type: ignore[valid-type]
            response_schema = DynamicResponse

    return prompt, response_schema


def _retain_mission_preamble(config) -> str:
    """The bank's retain mission, formatted for the per-request user message.

    Kept OUT of the cached system prompt (which must stay bank-agnostic so one
    CachedContent serves every bank — otherwise each distinct mission spawns its
    own cache) and prepended to the user message instead. Returns "" when unset.
    No brace-escaping needed: unlike the system template, the user message is
    used verbatim, not passed through str.format().
    """
    retain_mission = getattr(config, "retain_mission", None)
    if not retain_mission:
        return ""
    return (
        "══════════════════════════════════════════════════════════════════════════\n"
        "FOCUS — What to retain for this bank (takes priority over the general guidelines)\n"
        "══════════════════════════════════════════════════════════════════════════\n\n"
        f"{retain_mission}\n\n"
    )


def _build_user_message(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    metadata: dict[str, str] | None = None,
    agent_name: str | None = None,
    mission_preamble: str = "",
) -> str:
    """Build user message for fact extraction.

    ``mission_preamble`` (the bank's retain mission, possibly empty) is prepended
    so the bank-specific focus lives in the variable user turn rather than the
    cached, bank-agnostic system prompt.
    """
    from .orchestrator import parse_datetime_flexible

    sanitized_chunk = _sanitize_text(chunk)
    sanitized_context = _sanitize_text(context) if context else "none"

    if event_date is not None:
        event_date = parse_datetime_flexible(event_date)
        event_date_str = f"{event_date.strftime('%A, %B %d, %Y')} ({event_date.isoformat()})"
    else:
        event_date_str = "Unknown"

    metadata_section = ""
    if metadata:
        metadata_lines = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
        metadata_section = f"\nMetadata:\n{metadata_lines}"

    narrator_section = ""
    if agent_name:
        narrator_section = (
            f"\nNarrator: {agent_name} (the AI agent whose memory this is). By default, "
            f'first-person statements like "I did X" are {agent_name}\'s own actions → classify as '
            f'"assistant".'
        )
        # Only defer to the Context when one was actually provided — otherwise this
        # clause points at a "Context: none" line and just adds noise.
        if context:
            narrator_section += (
                " BUT the Context above takes precedence: if it identifies a different "
                "first-person speaker (e.g. a user or customer in a transcript), attribute those "
                'statements to that speaker and classify them as "world", not "assistant".'
            )

    return f"""{mission_preamble}Extract facts from the following text chunk.

Chunk: {chunk_index + 1}/{total_chunks}
Event Date: {event_date_str}
Context: {sanitized_context}{metadata_section}{narrator_section}

Text:
{sanitized_chunk}"""


def _build_request_body(llm_config, config, prompt: str, user_message: str, response_schema: type) -> dict:
    """Build request body for LLM API call."""
    request_body = {
        "model": llm_config.model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
    }

    # Honour the configured retain temperature. ``None`` omits the parameter
    # entirely (for models like Azure GPT-5.5 that reject explicit temperatures),
    # mirroring LLMProvider.call, which drops temperature when it is None. The
    # batch path builds the request body directly instead of going through
    # LLMProvider.call (#2469 only de-hardcoded the streaming path), so it must
    # apply the same rule here.
    if config.llm_temperature_retain is not None:
        request_body["temperature"] = config.llm_temperature_retain

    # Add max_completion_tokens if configured
    if config.retain_max_completion_tokens:
        request_body["max_completion_tokens"] = config.retain_max_completion_tokens

    # Add service_tier for OpenAI Flex Processing
    if llm_config.provider == "openai" and llm_config._provider_impl.openai_service_tier:
        request_body["service_tier"] = llm_config._provider_impl.openai_service_tier

    # Add response_format (JSON schema). The batch path builds the request body
    # directly instead of going through LLMProvider.call(), so honour
    # HINDSIGHT_API_LLM_STRICT_SCHEMA here too: strict=True grammar-enforces the
    # output on capable backends rather than relying on the model to emit clean JSON.
    if hasattr(response_schema, "model_json_schema"):
        schema = (
            strict_json_schema(response_schema) if config.llm_strict_schema else response_schema.model_json_schema()
        )
        request_body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "facts", "schema": schema, "strict": config.llm_strict_schema},
        }

    return request_body


def _coerce_fact_response(response: Any) -> dict[str, Any] | None:
    """Accept the schema wrapper, or a recoverable top-level facts array."""
    if isinstance(response, dict):
        return response
    if isinstance(response, list) and all(isinstance(item, dict) for item in response):
        return {"facts": response}
    return None


async def _extract_facts_from_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: "LLMConfig",
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], TokenUsage]:
    """
    Extract facts from a single chunk (internal helper for parallel processing).

    Note: event_date parameter is kept for backward compatibility but not used in prompt.
    The LLM extracts temporal information from the context string instead.
    """
    import logging

    from openai import BadRequestError

    logger = logging.getLogger(__name__)

    # Build prompt and schema using helper function
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    # Check config for extraction mode and causal link extraction
    extraction_mode = config.retain_extraction_mode
    extract_causal_links = config.retain_extract_causal_links

    # Build user message — the bank mission rides here (not in the cached prefix).
    user_message = _build_user_message(
        chunk,
        chunk_index,
        total_chunks,
        event_date,
        context,
        metadata,
        agent_name,
        mission_preamble=_retain_mission_preamble(config),
    )

    # Opt into context caching when the provider supports it. The prompt and
    # response_schema are bank-agnostic (the mission lives in the user message),
    # so one cached prefix serves every bank; reusing it across many small-payload
    # retain calls dramatically lowers per-call input
    # cost. ``get_or_create_cached_prefix`` returns None when caching is
    # disabled, unsupported, or the prefix is too small; the LLM call
    # transparently falls back to the uncached path in that case.
    cached_prefix_name: str | None = None
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is not None and provider_impl.supports_prompt_caching():
        try:
            cached_prefix_name = await provider_impl.get_or_create_cached_prefix(
                system_instruction=prompt,
                response_schema=response_schema,
            )
        except Exception:
            # Caching is a soft optimisation — never let a cache-side
            # error block a retain operation.
            logger.exception("Cache prefix lookup failed; falling back to uncached call")
            cached_prefix_name = None

    # Retry logic for JSON validation errors
    # Use retain-specific overrides if set, otherwise fall back to global LLM config
    llm_max_retries = (
        config.retain_llm_max_retries if config.retain_llm_max_retries is not None else config.llm_max_retries
    )
    last_error: Exception | None = None

    usage = TokenUsage()  # Track cumulative usage across retries
    for attempt in range(llm_max_retries):
        try:
            initial_backoff = (
                config.retain_llm_initial_backoff
                if config.retain_llm_initial_backoff is not None
                else config.llm_initial_backoff
            )
            max_backoff = (
                config.retain_llm_max_backoff if config.retain_llm_max_backoff is not None else config.llm_max_backoff
            )

            call_kwargs: dict[str, Any] = dict(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_message}],
                response_format=response_schema,
                scope="retain_extract_facts",
                temperature=config.llm_temperature_retain,
                max_completion_tokens=config.retain_max_completion_tokens,
                max_retries=llm_max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=True,  # Get raw JSON, we'll validate leniently
                return_usage=True,
            )
            if cached_prefix_name is not None:
                call_kwargs["cached_prefix"] = cached_prefix_name

            extraction_response_json, call_usage = await llm_config.call(**call_kwargs)
            usage = usage + call_usage  # Aggregate usage across retries

            # Lenient parsing of facts from raw JSON
            chunk_facts = []
            has_malformed_facts = False

            # Handle malformed LLM responses
            coerced_response_json = _coerce_fact_response(extraction_response_json)
            if coerced_response_json is None:
                if attempt < llm_max_retries - 1:
                    logger.warning(
                        f"LLM returned non-dict JSON on attempt {attempt + 1}/{llm_max_retries}: {type(extraction_response_json).__name__}. Retrying..."
                    )
                    continue
                else:
                    # A non-dict response is malformed (the schema is {"facts": [...]}).
                    # Raise instead of returning [] so the failure propagates to the
                    # worker's retry machinery and ultimately fails loudly — never
                    # silently commit the document with 0 facts. See issue #1833.
                    raise RuntimeError(
                        f"Fact extraction failed: LLM returned non-dict JSON after {llm_max_retries} attempts "
                        f"({type(extraction_response_json).__name__}). Raw: {str(extraction_response_json)[:500]}"
                    )
            extraction_response_json = coerced_response_json

            raw_facts = extraction_response_json.get("facts", [])

            if not raw_facts:
                logger.debug(
                    f"LLM response missing 'facts' field or returned empty list. "
                    f"Response: {extraction_response_json}. "
                    f"Input: "
                    f"date: {event_date.isoformat() if event_date else 'unset'}, "
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
                    # In verbatim mode, 'what' is intentionally absent — text is backfilled from chunk
                    if extraction_mode != "verbatim":
                        logger.warning(f"Skipping fact {i}: missing 'what' field")
                        continue

                # Critical field: fact_type — "assistant" maps to "experience", everything else is "world".
                # If fact_type is unexpected, fall back to fact_kind before defaulting to "world".
                raw_fact_type = llm_fact.get("fact_type")
                if raw_fact_type == "assistant":
                    fact_type = "experience"
                elif raw_fact_type == "world":
                    fact_type = "world"
                else:
                    raw_fact_kind = llm_fact.get("fact_kind")
                    fact_type = "experience" if raw_fact_kind == "assistant" else "world"

                # Get fact_kind for temporal handling (but don't store it)
                fact_kind = llm_fact.get("fact_kind", "conversation")
                if fact_kind not in ["conversation", "event", "other"]:
                    fact_kind = "conversation"

                # Build combined fact text from the 4 dimensions: what | when | who | why
                # In verbatim mode, leave combined_text empty — _collapse_to_verbatim backfills it
                fact_data = {}
                if extraction_mode == "verbatim":
                    combined_text = ""
                else:
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

                # Entities are plain strings. Older prompts taught a {"text": ...}
                # object form, so keep unwrapping it for models that still emit it.
                validated_entities = _coerce_entity_strings(get_value("entities"))

                # Post-process label entities from structured labels object
                entity_labels_raw = getattr(config, "entity_labels", None)
                labels_cfg = parse_entity_labels(entity_labels_raw)
                free_form_entities = getattr(config, "entities_allow_free_form", True)
                if labels_cfg and labels_cfg.attributes:
                    labels_lookup = build_labels_lookup(labels_cfg)
                    labels_data = llm_fact.get("labels") or {}
                    if isinstance(labels_data, dict):
                        existing_texts_lower = {e.lower() for e in validated_entities}
                        for group in labels_cfg.attributes:
                            value = labels_data.get(group.key)
                            if not value:
                                continue
                            # Map-type groups: recursively extract key:field:value strings
                            if group.type == "map" and group.fields:
                                entities_list = value if isinstance(value, list) else [value]
                                for entity_obj in entities_list:
                                    if isinstance(entity_obj, dict):
                                        _extract_map_entities(
                                            entity_obj,
                                            group.fields,
                                            f"{group.key}:",
                                            validated_entities,
                                            existing_texts_lower,
                                        )
                                continue
                            values_list = value if isinstance(value, list) else [value]
                            for v in values_list:
                                if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                                    continue
                                label_str = f"{group.key}:{v.strip()}"
                                if group.type == "text":
                                    if label_str.lower() not in existing_texts_lower:
                                        validated_entities.append(label_str)
                                        existing_texts_lower.add(label_str.lower())
                                elif (
                                    label_str.lower() in labels_lookup and label_str.lower() not in existing_texts_lower
                                ):
                                    validated_entities.append(label_str)
                                    existing_texts_lower.add(label_str.lower())
                                else:
                                    logger.warning(f"Label '{label_str}' not in valid label values, skipping")

                    # In labels-only mode, keep only label entities
                    if not free_form_entities:
                        validated_entities = [
                            e for e in validated_entities if is_label_entity(e, labels_cfg, labels_lookup)
                        ]
                elif not free_form_entities:
                    # No labels but free_form disabled: clear all entities
                    validated_entities = []

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
                                    )
                                )
                            except Exception as e:
                                logger.debug(f"Invalid causal relation {rel}: {e}")

                    if validated_relations:
                        fact_data["causal_relations"] = validated_relations

                # Set mentioned_at to the event_date (when the conversation/document occurred),
                # or None when the caller opted into no timestamp.
                fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

                # Build Fact model instance
                try:
                    fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                    chunk_facts.append(fact)
                except Exception as e:
                    logger.error(f"Failed to create Fact model for fact {i}: {e}")
                    has_malformed_facts = True
                    continue

            # If we got malformed facts and haven't exhausted retries, try again
            if has_malformed_facts and len(chunk_facts) < len(raw_facts) * 0.8 and attempt < llm_max_retries - 1:
                logger.warning(
                    f"Got {len(raw_facts) - len(chunk_facts)} malformed facts out of {len(raw_facts)} on attempt {attempt + 1}/{llm_max_retries}. Retrying..."
                )
                continue

            return chunk_facts, usage

        except BadRequestError as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is related to max_tokens/completion_tokens not being supported
            if any(
                keyword in error_str
                for keyword in [
                    "max_tokens",
                    "max_completion_tokens",
                    "maximum context",
                    "token limit",
                    "context length",
                ]
            ):
                # Provide helpful error message with configuration suggestions
                raise ValueError(
                    f"Model does not support the required output token limit.\n\n"
                    f"The model '{llm_config.model}' (provider: {llm_config.provider}) failed with: {e}\n\n"
                    f"You have two options to fix this:\n"
                    f"  1. Use a different model that supports at least {config.retain_max_completion_tokens} output tokens\n"
                    f"  2. Decrease HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS to a value your model supports\n"
                    f"     (current value: {config.retain_max_completion_tokens}, must be > RETAIN_CHUNK_SIZE={config.retain_chunk_size})"
                ) from e

            # Don't retry json_validate_failed here — the inner provider
            # loop already retried the 400 error.  Re-entering the LLM call
            # with the same input just multiplies wasted calls.
            raise

    # If we exhausted all retries, raise the last error or a descriptive fallback
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Fact extraction failed after {llm_max_retries} attempts: LLM did not return valid JSON")


async def _extract_facts_with_auto_split(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    event_date: datetime | None,
    context: str,
    llm_config: LLMConfig,
    config,
    agent_name: str = None,
    metadata: dict[str, str] | None = None,
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
        config: Resolved HindsightConfig for this bank
        agent_name: Optional agent name (memory owner)
        metadata: Optional document metadata key-value pairs

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
            config=config,
            agent_name=agent_name,
            metadata=metadata,
        )
    except OutputTooLongError:
        # Output exceeded token limits - split the chunk and retry. Conversation
        # chunks are JSON arrays, so preserve array/turn boundaries when possible.
        logger.warning(
            f"Output too long for chunk {chunk_index + 1}/{total_chunks} "
            f"({len(chunk)} chars). Splitting and retrying..."
        )

        split_chunks = _split_chunk_for_output_retry(chunk)
        if split_chunks is None:
            logger.warning(
                f"Cannot make progress splitting chunk {chunk_index + 1}/{total_chunks} "
                f"({len(chunk)} chars); dropping this sub-chunk."
            )
            return [], TokenUsage()

        first_half, second_half = split_chunks

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
                config=config,
                agent_name=agent_name,
                metadata=metadata,
            ),
            _extract_facts_with_auto_split(
                chunk=second_half,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                event_date=event_date,
                context=context,
                llm_config=llm_config,
                config=config,
                agent_name=agent_name,
                metadata=metadata,
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
    event_date: datetime | None,
    llm_config: LLMConfig,
    agent_name: str,
    config,
    context: str = "",
    metadata: dict[str, str] | None = None,
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
        llm_config: LLM configuration to use
        agent_name: Agent name (memory owner)
        config: Resolved HindsightConfig for this bank
        context: Context about the conversation/document
        metadata: Optional document metadata key-value pairs

    Returns:
        Tuple of (facts, chunks, usage) where:
        - facts: List of Fact model instances
        - chunks: List of tuples (chunk_text, fact_count) for each chunk
        - usage: Aggregated token usage across all LLM calls
    """
    chunks = chunk_text(
        text,
        max_chars=config.retain_chunk_size,
        structured_chunk_size=config.retain_structured_chunk_size,
    )

    # Log chunk count before starting LLM requests
    total_chars = sum(len(c) for c in chunks)
    if len(chunks) > 1:
        logger.debug(
            f"[FACT_EXTRACTION] Text chunked into {len(chunks)} chunks ({total_chars:,} chars total, "
            f"chunk_size={config.retain_chunk_size:,}) - starting parallel LLM extraction"
        )

    # Transient LLM failures (timeouts, rate limits) are already retried inside
    # the provider's inner loop.  Content-quality retries (malformed facts) are
    # handled by the middle loop in _extract_facts_from_chunk.  Adding a third
    # retry layer here would multiply wasted calls on deterministic failures
    # (see https://github.com/vectorize-io/hindsight/issues/1412).
    tasks = [
        _extract_facts_with_auto_split(
            chunk=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            event_date=event_date,
            context=context,
            llm_config=llm_config,
            config=config,
            agent_name=agent_name,
            metadata=metadata,
        )
        for i, chunk in enumerate(chunks)
    ]

    # return_exceptions=True so we can collect all results even if some chunks
    # exhausted their retries. We check for failures below and fail the retain
    # if ANY chunk could not be extracted — partial extraction is not acceptable.
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_facts = []
    chunk_metadata = []  # [(chunk_text, fact_count), ...]
    total_usage = TokenUsage()
    failed_chunks = []
    for i, (chunk, result) in enumerate(zip(chunks, chunk_results)):
        if isinstance(result, Exception):
            failed_chunks.append((i, result))
            continue
        chunk_facts, chunk_usage = result
        all_facts.extend(chunk_facts)
        chunk_metadata.append((chunk, len(chunk_facts)))
        total_usage = total_usage + chunk_usage

    if failed_chunks:
        # Include the exception message — not just the type — so operators
        # can tell a structured-JSON parse failure apart from a rate limit
        # apart from a network 5xx, all of which can surface as the same
        # exception types. The error_message we propagate to the
        # async_operations row is the only inspection surface a worker-side
        # failure leaves behind, and a bare "chunk 0: RuntimeError" is not
        # actionable.
        failed_summary = ", ".join(f"chunk {idx}: {type(err).__name__}: {err}" for idx, err in failed_chunks[:5])
        quota_errors = [err for _, err in failed_chunks if isinstance(err, ProviderRateLimitResetError)]
        if quota_errors and len(quota_errors) == len(failed_chunks):
            retry_at = max(err.retry_at for err in quota_errors)
            raise ProviderRateLimitResetError(
                retry_at=retry_at,
                message=(
                    f"Fact extraction deferred by provider quota: {len(failed_chunks)}/{len(chunks)} chunks failed. "
                    f"First failures: {failed_summary}. Provider detail: {quota_errors[0]}"
                ),
            ) from quota_errors[0]

        # Fail the entire retain — partial extraction is not acceptable.
        # All successfully extracted facts are discarded because the transaction
        # hasn't committed yet. The worker poller will retry the entire task.
        raise RuntimeError(
            f"Fact extraction failed: {len(failed_chunks)}/{len(chunks)} chunks failed. "
            f"First failures: {failed_summary}"
        )

    return all_facts, chunk_metadata, total_usage


# ============================================================================
# ORCHESTRATION LAYER
# ============================================================================

# Import types for the orchestration layer (note: ExtractedFact here is different from the Pydantic model above)

from .types import CausalRelation as CausalRelationType
from .types import ChunkMetadata, RetainContent
from .types import ExtractedFact as ExtractedFactType

logger = logging.getLogger(__name__)

# Each fact gets 10ms offset to preserve ordering within a document
SECONDS_PER_FACT = 0.01


async def _write_batch_extraction_errors(
    pool: Any,
    operation_id: str | None,
    schema: str | None,
    errors: RetainExtractionErrors,
) -> None:
    """Persist non-fatal Batch API extraction errors into operation result_metadata."""
    if not pool or not operation_id or errors.count == 0:
        return

    from ..db_utils import acquire_with_retry
    from ..task_backend import fq_table

    # `errors` is the complete set for this extraction run, so overwrite the
    # extraction_errors_* keys rather than folding in what's already stored. On
    # batch crash recovery the resumed batch reprocesses every result and
    # recomputes `errors` from scratch; reading + merging the prior run's
    # counters here would double-count them. The SQL `||` merge still preserves
    # unrelated keys (e.g. batch_id) already on result_metadata.
    table = fq_table("async_operations", schema)
    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {table}
            SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $2::jsonb,
                updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
            json.dumps(errors.to_dict()),
        )


async def extract_facts_from_contents_batch_api(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts using LLM Batch API (OpenAI/Groq).

    Submits all chunks as a single batch, polls until complete, then processes results.
    Only called when config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration with batch API support
        agent_name: Name of the agent
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (for storing batch state)
        operation_id: Async operation ID (for crash recovery)
        schema: Database schema (for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)
    """
    if not contents:
        return [], [], TokenUsage()

    logger.info(f"Using Batch API for fact extraction ({len(contents)} contents)")

    # Check config for causal link extraction (used throughout)
    extract_causal_links = config.retain_extract_causal_links

    # Check if provider supports batch API
    if not await llm_config._provider_impl.supports_batch_api():
        raise RuntimeError(
            f"retain_batch_enabled=True but provider '{llm_config.provider}' does not "
            f"support the batch API. This should have been caught at startup — check "
            f"HINDSIGHT_API_RETAIN_BATCH_ENABLED and your LLM provider configuration."
        )

    # Check if we're resuming an existing batch (crash recovery)
    batch_id = None
    if operation_id and pool:
        from ..db_utils import acquire_with_retry
        from ..task_backend import fq_table

        table = fq_table("async_operations", schema)
        async with acquire_with_retry(pool) as conn:
            row = await conn.fetchrow(
                f"SELECT result_metadata FROM {table} WHERE operation_id = $1",
                operation_id,
            )

        if row and row["result_metadata"]:
            metadata = row["result_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            batch_id = metadata.get("batch_id")

            if batch_id:
                logger.info(f"Resuming existing batch: batch_id={batch_id} (crash recovery)")

    # Step 1: Chunk all contents and build batch requests (skip if resuming)
    all_chunks_info = []  # List of (chunk_text, content_index, chunk_index_in_content, event_date, context)
    batch_requests = []

    # Build prompt and schema once (same for all chunks)
    prompt, response_schema = _build_extraction_prompt_and_schema(config)

    for content_index, item in enumerate(contents):
        chunks = chunk_text(
            item.content,
            max_chars=config.retain_chunk_size,
            structured_chunk_size=config.retain_structured_chunk_size,
        )

        for chunk_index_in_content, chunk in enumerate(chunks):
            all_chunks_info.append((chunk, content_index, chunk_index_in_content, item.event_date, item.context))

            # Build batch request for this chunk
            custom_id = f"chunk_{len(all_chunks_info) - 1}"  # Global chunk index

            # Build user message using helper function
            user_message = _build_user_message(
                chunk,
                chunk_index_in_content,
                len(chunks),
                item.event_date,
                item.context,
                item.metadata or None,
                agent_name,
                mission_preamble=_retain_mission_preamble(config),
            )

            # Build request body using helper function
            request_body = _build_request_body(llm_config, config, prompt, user_message, response_schema)

            batch_requests.append(
                {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": request_body}
            )

    if not batch_requests and not batch_id:  # No requests and not resuming
        return [], [], TokenUsage()

    # Step 2: Submit batch (skip if resuming)
    if not batch_id:
        logger.info(f"Submitting batch with {len(batch_requests)} chunk requests")

        batch_metadata = await llm_config._provider_impl.submit_batch(batch_requests)
        batch_id = batch_metadata["batch_id"]

        logger.info(f"Batch submitted: {batch_id}, polling every {config.retain_batch_poll_interval_seconds}s")

        # CRITICAL: Store minimal batch state in operation metadata for crash recovery
        # This allows resuming polling if worker restarts
        if operation_id and pool:
            batch_state = {
                "batch_id": batch_id,
                "batch_provider": llm_config.provider,
                "chunk_count": len(batch_requests),
            }

            # Update operation result_metadata
            from ..db_utils import acquire_with_retry
            from ..task_backend import fq_table

            table = fq_table("async_operations", schema)
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET result_metadata = result_metadata || $1::jsonb, updated_at = now()
                    WHERE operation_id = $2
                    """,
                    json.dumps(batch_state),
                    operation_id,
                )
            logger.info(f"Stored batch state for operation {operation_id} (crash recovery enabled)")
    else:
        logger.info(f"Resuming polling for existing batch: {batch_id}")

    # Step 3: Poll until complete
    import time

    start_time = time.time()
    while True:
        status_info = await llm_config._provider_impl.get_batch_status(batch_id)
        status = status_info["status"]

        elapsed = time.time() - start_time
        logger.info(
            f"Batch {batch_id}: status={status}, "
            f"completed={status_info['request_counts']['completed']}/{status_info['request_counts']['total']}, "
            f"elapsed={elapsed:.0f}s"
        )

        if status == "completed":
            break
        elif status in ("failed", "expired", "cancelled"):
            error_msg = status_info.get("errors", "Unknown error")
            raise RuntimeError(f"Batch {batch_id} failed with status {status}: {error_msg}")

        # Wait before polling again
        await asyncio.sleep(config.retain_batch_poll_interval_seconds)

    logger.info(f"Batch {batch_id} completed in {elapsed:.0f}s, retrieving results")

    # Step 4: Retrieve results
    batch_results = await llm_config._provider_impl.retrieve_batch_results(batch_id)

    # Map results by custom_id
    results_by_id = {result["custom_id"]: result for result in batch_results}

    # Step 5: Parse results into facts (same as sync mode)
    all_facts_from_llm = []
    chunks_metadata = []
    total_usage = TokenUsage()
    extraction_errors = RetainExtractionErrors()

    for chunk_idx, (chunk_content, content_index, chunk_index_in_content, event_date, context) in enumerate(
        all_chunks_info
    ):
        custom_id = f"chunk_{chunk_idx}"
        result = results_by_id.get(custom_id)

        if not result:
            message = f"{custom_id}: missing batch result"
            logger.warning(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Check for errors
        if result.get("error"):
            message = f"{custom_id}: {result['error']}"
            logger.error(f"Error in {message}")
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Extract response
        response_body = result.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])

        if not choices:
            message = f"{custom_id}: no choices in response"
            logger.warning(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse JSON content
        message = choices[0].get("message", {})
        content_str = message.get("content", "{}")

        try:
            # #2701: use the lenient parser (strips markdown fences, scrubs
            # embedded control chars) so recoverable batch responses — e.g.
            # transient Gemini quirks — aren't dropped along with all their facts.
            extraction_response_json = parse_llm_json(content_str)
        except json.JSONDecodeError as e:
            message = f"{custom_id}: failed to parse JSON: {e}"
            logger.error(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        response_type_name = type(extraction_response_json).__name__
        extraction_response_json = _coerce_fact_response(extraction_response_json)
        if extraction_response_json is None:
            message = f"{custom_id}: LLM returned non-dict JSON ({response_type_name})"
            logger.error(message)
            extraction_errors.add(message)
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk_content, fact_count=0, content_index=content_index, chunk_index=chunk_idx
                )
            )
            continue

        # Parse facts (reuse existing logic from _extract_facts_from_chunk)
        raw_facts = extraction_response_json.get("facts", [])
        chunk_facts = []

        for i, llm_fact in enumerate(raw_facts):
            if not isinstance(llm_fact, dict):
                continue

            def get_value(field_name):
                value = llm_fact.get(field_name)
                if value and value != "" and value != [] and value != {} and str(value).upper() != "N/A":
                    return value
                return None

            what = get_value("what")
            if not what:
                what = get_value("factual_core")
            if not what:
                continue

            when = get_value("when")
            who = get_value("who")
            why = get_value("why")

            # Critical field: fact_type — only "assistant" maps to "experience", everything else is "world"
            # Critical field: fact_type — "assistant" maps to "experience", everything else is "world".
            # If fact_type is unexpected, fall back to fact_kind before defaulting to "world".
            raw_fact_type = llm_fact.get("fact_type")
            if raw_fact_type == "assistant":
                fact_type = "experience"
            elif raw_fact_type == "world":
                fact_type = "world"
            else:
                raw_fact_kind = llm_fact.get("fact_kind")
                fact_type = "experience" if raw_fact_kind == "assistant" else "world"

            # Build combined fact text
            combined_parts = [what]
            if when:
                combined_parts.append(f"When: {when}")
            if who:
                combined_parts.append(f"Involving: {who}")
            if why:
                combined_parts.append(why)
            combined_text = " | ".join(combined_parts)

            # Temporal fields
            fact_data = {}
            fact_kind = llm_fact.get("fact_kind", "conversation")
            if fact_kind not in ["conversation", "event", "other"]:
                fact_kind = "conversation"

            if fact_kind == "event":
                occurred_start = get_value("occurred_start")
                occurred_end = get_value("occurred_end")

                if not occurred_start:
                    fact_data["occurred_start"] = _infer_temporal_date(combined_text, event_date)
                else:
                    fact_data["occurred_start"] = occurred_start

                if occurred_end:
                    fact_data["occurred_end"] = occurred_end
                elif fact_data.get("occurred_start"):
                    fact_data["occurred_end"] = fact_data["occurred_start"]

            # Entities are plain strings. Older prompts taught a {"text": ...}
            # object form, so keep unwrapping it for models that still emit it.
            validated_entities = _coerce_entity_strings(get_value("entities"))

            # Post-process label entities from structured labels object
            entity_labels_raw = getattr(config, "entity_labels", None)
            labels_cfg_batch = parse_entity_labels(entity_labels_raw)
            free_form_entities_batch = getattr(config, "entities_allow_free_form", True)
            if labels_cfg_batch and labels_cfg_batch.attributes:
                labels_lookup_batch = build_labels_lookup(labels_cfg_batch)
                labels_data = llm_fact.get("labels") or {}
                if isinstance(labels_data, dict):
                    existing_texts_lower = {e.lower() for e in validated_entities}
                    for group in labels_cfg_batch.attributes:
                        value = labels_data.get(group.key)
                        if not value:
                            continue
                        # Map-type groups: recursively extract key:field:value strings
                        if group.type == "map" and group.fields:
                            entities_list = value if isinstance(value, list) else [value]
                            for entity_obj in entities_list:
                                if isinstance(entity_obj, dict):
                                    _extract_map_entities(
                                        entity_obj,
                                        group.fields,
                                        f"{group.key}:",
                                        validated_entities,
                                        existing_texts_lower,
                                    )
                            continue
                        values_list = value if isinstance(value, list) else [value]
                        for v in values_list:
                            if not isinstance(v, str) or not v.strip() or v.lower() in ("none", "null", "n/a"):
                                continue
                            label_str = f"{group.key}:{v.strip()}"
                            if group.type == "text":
                                if label_str.lower() not in existing_texts_lower:
                                    validated_entities.append(label_str)
                                    existing_texts_lower.add(label_str.lower())
                            elif (
                                label_str.lower() in labels_lookup_batch
                                and label_str.lower() not in existing_texts_lower
                            ):
                                validated_entities.append(label_str)
                                existing_texts_lower.add(label_str.lower())

                if not free_form_entities_batch:
                    validated_entities = [
                        e for e in validated_entities if is_label_entity(e, labels_cfg_batch, labels_lookup_batch)
                    ]
            elif not free_form_entities_batch:
                validated_entities = []

            if validated_entities:
                fact_data["entities"] = validated_entities

            # Causal relations
            if extract_causal_links:
                validated_relations = []
                causal_relations_raw = get_value("causal_relations")
                if causal_relations_raw:
                    for rel in causal_relations_raw:
                        if not isinstance(rel, dict):
                            continue
                        target_idx = rel.get("target_index")
                        relation_type = rel.get("relation_type")

                        if target_idx is None or relation_type is None:
                            continue
                        if target_idx < 0 or target_idx >= i:
                            continue

                        try:
                            validated_relations.append(
                                CausalRelation(target_fact_index=target_idx, relation_type=relation_type)
                            )
                        except Exception:
                            pass

                if validated_relations:
                    fact_data["causal_relations"] = validated_relations

            # Set mentioned_at to the event_date (when the conversation/document occurred),
            # or None when the caller opted into no timestamp.
            fact_data["mentioned_at"] = event_date.isoformat() if event_date is not None else None

            try:
                fact = Fact(fact=combined_text, fact_type=fact_type, **fact_data)
                chunk_facts.append(fact)
            except Exception as e:
                message = f"{custom_id}: failed to create Fact model for fact {i}: {e}"
                logger.error(message)
                extraction_errors.add(message)
                continue

        all_facts_from_llm.extend(chunk_facts)
        chunks_metadata.append(
            ChunkMetadata(
                chunk_text=chunk_content,
                fact_count=len(chunk_facts),
                content_index=content_index,
                chunk_index=chunk_idx,
            )
        )

        # Track token usage
        usage_data = response_body.get("usage", {})
        if usage_data:
            total_usage = total_usage + TokenUsage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

    # Step 6: Convert to ExtractedFact objects with proper chunk mapping
    # Group facts by chunk
    facts_by_chunk = []  # List of (chunk_metadata, [facts])
    fact_start_idx = 0

    for chunk_meta in chunks_metadata:
        chunk_facts = all_facts_from_llm[fact_start_idx : fact_start_idx + chunk_meta.fact_count]
        facts_by_chunk.append((chunk_meta, chunk_facts))
        fact_start_idx += chunk_meta.fact_count

    # Now convert to ExtractedFactType
    extracted_facts = []
    global_fact_idx = 0

    for chunk_meta, chunk_facts in facts_by_chunk:
        content = contents[chunk_meta.content_index]

        for fact_from_llm in chunk_facts:
            extracted_fact = ExtractedFactType(
                fact_text=fact_from_llm.fact,
                fact_type=fact_from_llm.fact_type,
                entities=list(fact_from_llm.entities or []),
                occurred_start=_parse_datetime(fact_from_llm.occurred_start) if fact_from_llm.occurred_start else None,
                occurred_end=_parse_datetime(fact_from_llm.occurred_end) if fact_from_llm.occurred_end else None,
                causal_relations=_convert_causal_relations(fact_from_llm.causal_relations or [], global_fact_idx),
                content_index=chunk_meta.content_index,
                chunk_index=chunk_meta.chunk_index,
                context=content.context,
                mentioned_at=content.event_date,
                metadata=content.metadata,
                tags=content.tags,
                observation_scopes=content.observation_scopes,
            )

            extracted_facts.append(extracted_fact)
            global_fact_idx += 1

    # Step 7: Add temporal offsets
    _add_temporal_offsets(extracted_facts, contents)

    # Step 8: Auto-tag facts from label groups with tag=True
    _inject_label_tags(extracted_facts, config)

    await _write_batch_extraction_errors(pool, operation_id, schema, extraction_errors)

    logger.info(f"Batch API extracted {len(extracted_facts)} facts from {len(all_chunks_info)} chunks")

    return extracted_facts, chunks_metadata, total_usage


def _extract_facts_chunks(
    contents: list[RetainContent],
    config,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    chunks mode: no LLM call, no entity extraction.

    Each chunk becomes one memory unit with the raw text as fact_text.
    User-provided entities from RetainContent.entities are picked up downstream
    by entity_processing.py — they are the sole source of entity data in this mode.
    """
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    global_chunk_idx = 0

    for content_index, content in enumerate(contents):
        chunks = chunk_text(
            content.content,
            config.retain_chunk_size,
            structured_chunk_size=config.retain_structured_chunk_size,
        )
        for chunk in chunks:
            chunks_metadata.append(
                ChunkMetadata(
                    chunk_text=chunk,
                    fact_count=1,
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                )
            )
            extracted_facts.append(
                ExtractedFactType(
                    fact_text=chunk,
                    fact_type="world",
                    entities=[],
                    content_index=content_index,
                    chunk_index=global_chunk_idx,
                    context=content.context,
                    mentioned_at=content.event_date,
                    metadata=content.metadata,
                    tags=content.tags,
                    observation_scopes=content.observation_scopes,
                )
            )
            global_chunk_idx += 1

    _add_temporal_offsets(extracted_facts, contents)
    return extracted_facts, chunks_metadata, TokenUsage()


async def extract_facts_from_contents(
    contents: list[RetainContent],
    llm_config,
    agent_name: str,
    config,
    pool=None,
    operation_id: str | None = None,
    schema: str | None = None,
) -> tuple[list[ExtractedFactType], list[ChunkMetadata], TokenUsage]:
    """
    Extract facts from multiple content items in parallel.

    This function:
    1. Extracts facts from all contents in parallel using the LLM
    2. Tracks which facts came from which chunks
    3. Adds time offsets to preserve fact ordering within each content
    4. Returns typed ExtractedFact and ChunkMetadata objects

    Routes to batch API mode if config.retain_batch_enabled=True.

    Args:
        contents: List of RetainContent objects to process
        llm_config: LLM configuration for fact extraction
        agent_name: Name of the agent (for agent-related fact detection)
        config: Resolved HindsightConfig for this bank
        pool: Database connection pool (passed to batch API for state storage)
        operation_id: Async operation ID (passed to batch API for crash recovery)
        schema: Database schema (passed to batch API for multi-tenant support)

    Returns:
        Tuple of (extracted_facts, chunks_metadata, usage)
    """
    if not contents:
        return [], [], TokenUsage()

    # chunks mode: skip LLM entirely, store each chunk as-is
    # Must come before the batch-API check so no LLM queue/locks are acquired
    if config.retain_extraction_mode == "chunks":
        return _extract_facts_chunks(contents, config)

    # Route to batch API if enabled
    if config.retain_batch_enabled:
        return await extract_facts_from_contents_batch_api(
            contents, llm_config, agent_name, config, pool, operation_id, schema
        )

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
            config=config,
            metadata=item.metadata or None,
        )
        fact_extraction_tasks.append(task)

    # Step 2: Wait for all fact extractions to complete.
    # return_exceptions=True so a failing item doesn't cancel its still-running
    # siblings (which would leave orphaned LLM calls / partial work); we await
    # them all, then propagate.
    all_fact_results = await asyncio.gather(*fact_extraction_tasks, return_exceptions=True)

    # Step 3: Flatten and convert to typed objects
    extracted_facts: list[ExtractedFactType] = []
    chunks_metadata: list[ChunkMetadata] = []
    total_usage = TokenUsage()

    global_chunk_idx = 0
    global_fact_idx = 0

    # Never silently drop a document's memory. Any extraction failure (provider
    # rate-limit / timeout / 5xx, malformed response, token-limit, etc.)
    # propagates so the streaming producer surfaces it and the worker's
    # RetryTaskAt machinery retries the task — and ultimately fails it *loudly*
    # if the problem persists. Swallowing the error and substituting an empty
    # result here used to commit the document with 0 facts and mark the
    # operation `completed`, losing the memory with no signal. See issue #1833.
    valid_results = []
    for content, result in zip(contents, all_fact_results):
        if isinstance(result, Exception):
            raise result
        valid_results.append((content, result))

    for content_index, (content, (facts_from_llm, chunks_from_llm, content_usage)) in enumerate(valid_results):
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
                        entities=list(fact_from_llm.entities or []),
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
                        observation_scopes=content.observation_scopes,
                    )

                    extracted_facts.append(extracted_fact)
                    global_fact_idx += 1
                    fact_idx_in_content += 1

    # Step 4: For verbatim mode, collapse to one fact per chunk with original text
    if config.retain_extraction_mode == "verbatim":
        extracted_facts = _collapse_to_verbatim(extracted_facts, chunks_metadata)

    # Step 5: Add time offsets to preserve ordering within each content
    _add_temporal_offsets(extracted_facts, contents)

    # Step 6: Auto-tag facts from label groups with tag=True
    _inject_label_tags(extracted_facts, config)

    return extracted_facts, chunks_metadata, total_usage


def _collapse_to_verbatim(facts: list[ExtractedFactType], chunks: list[ChunkMetadata]) -> list[ExtractedFactType]:
    """
    For verbatim mode: ensure one fact per chunk with the original chunk text preserved.

    The LLM prompt asks for exactly one fact per chunk, but if it returns more,
    this collapses them: keeps the first fact as representative, overrides its
    fact_text with the raw chunk text, and merges entities from any extra facts.
    """
    chunk_text_map = {c.chunk_index: c.chunk_text for c in chunks}
    seen: dict[int, ExtractedFactType] = {}
    result: list[ExtractedFactType] = []

    for fact in facts:
        if fact.chunk_index not in seen:
            fact.fact_text = chunk_text_map.get(fact.chunk_index, fact.fact_text)
            seen[fact.chunk_index] = fact
            result.append(fact)
        else:
            # Merge entities from extra facts into the representative
            representative = seen[fact.chunk_index]
            for entity in fact.entities:
                if entity not in representative.entities:
                    representative.entities.append(entity)

    return result


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
        )
        causal_relations.append(causal_relation)
    return causal_relations


def _add_temporal_offsets(facts: list[ExtractedFactType], contents: list[RetainContent]) -> None:
    """
    Add time offsets to preserve fact ordering across all contents.

    This allows retrieval to distinguish between facts from different documents/conversations
    even when they have the same base event_date, and also between facts within the same
    conversation.

    Uses absolute position across all facts to ensure unique timestamps.

    Modifies facts in place.
    """
    from .orchestrator import parse_datetime_flexible

    for i, fact in enumerate(facts):
        # Use absolute position across all facts to ensure uniqueness across different contents
        offset = timedelta(seconds=i * SECONDS_PER_FACT)

        # Apply offset to all temporal fields (handle both datetime objects and ISO strings)
        if fact.occurred_start:
            fact.occurred_start = parse_datetime_flexible(fact.occurred_start) + offset
        if fact.occurred_end:
            fact.occurred_end = parse_datetime_flexible(fact.occurred_end) + offset
        if fact.mentioned_at:
            fact.mentioned_at = parse_datetime_flexible(fact.mentioned_at) + offset


def _inject_label_tags(facts: list[ExtractedFactType], config) -> None:
    """
    For label groups with tag=True, add extracted key:value label entities
    to each fact's tags list. Modifies facts in place.

    This lets entity labels double as tags, enabling filtering via the
    existing tags API without any extra query infrastructure.
    """
    labels_cfg = parse_entity_labels(getattr(config, "entity_labels", None))
    if not labels_cfg:
        return
    tag_group_keys = {g.key.lower() for g in labels_cfg.attributes if g.tag}
    if not tag_group_keys:
        return
    for fact in facts:
        label_tags = [e for e in fact.entities if ":" in e and e.split(":", 1)[0].lower() in tag_group_keys]
        if label_tags:
            existing = set(fact.tags)
            fact.tags = fact.tags + [t for t in label_tags if t not in existing]
