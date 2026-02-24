"""Prompts for the consolidation engine."""

# Default mission when no bank-specific mission is set
_DEFAULT_MISSION = "Track every detail: names, numbers, dates, places, and relationships. Prefer specifics over abstractions, never generalise."

# Processing rules — always present regardless of mission
_PROCESSING_RULES = """Processing rules (always apply):
- REDUNDANT: same info worded differently → UPDATE the existing observation.
- CONTRADICTION/UPDATE: capture both states with temporal markers ("used to X, now Y").
- RESOLVE REFERENCES: when a new fact provides a concrete value resolving a vague placeholder in an existing observation (e.g. "home country", "hometown", "birthplace", "native language", "her ex", "that city"), UPDATE the observation to embed the resolved value explicitly. Example: new fact says "grandma in Sweden" + existing observation says "moved from her home country" → update to "home country is Sweden".
- NEVER merge observations about different people or unrelated topics."""

# Data section — format placeholders {facts_text} and {observations_text} are substituted at call time
_BATCH_DATA_SECTION = """
NEW FACTS:
{facts_text}

EXISTING OBSERVATIONS (JSON array, pooled from recalls across all facts above):
{observations_text}

Each observation includes:
- id: unique identifier for updating
- text: the observation content
- proof_count: number of supporting memories
- occurred_start/occurred_end: temporal range of source facts
- source_memories: array of supporting facts with their text and dates

Compare the facts against existing observations:
- Same topic as an existing observation → UPDATE it (observation_id + source_fact_ids)
- New topic with durable knowledge → CREATE a new observation (source_fact_ids)
- Cross-reference facts within the batch: a later fact may resolve a vague reference in an earlier one
- Purely ephemeral facts → omit them (no create/update needed)"""

# Output format — JSON braces escaped as {{ }} so .format() leaves them literal
_BATCH_OUTPUT_FORMAT = """
Output a JSON object with three arrays.

Example (showing the required UUID format for all IDs):
{{"creates": [{{"text": "Alice lives in Berlin", "source_fact_ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "b2c3d4e5-f6a7-8901-bcde-f12345678901"]}}],
  "updates": [{{"text": "Alice works at Acme Corp as a senior engineer", "observation_id": "c3d4e5f6-a7b8-9012-cdef-123456789012", "source_fact_ids": ["d4e5f6a7-b8c9-0123-defa-234567890123"]}}],
  "deletes": [{{"observation_id": "e5f6a7b8-c9d0-1234-efab-345678901234"}}]}}

Rules:
- "source_fact_ids": copy the EXACT UUID strings shown in brackets [uuid] from NEW FACTS — never use integers or positions.
- "observation_id": copy the EXACT "id" UUID string from EXISTING OBSERVATIONS.
- One create/update may reference multiple facts when they jointly support the observation.
- "deletes": only when an observation is directly superseded or contradicted by new facts.
- Do NOT include "tags" — handled automatically.
- Return {{"creates": [], "updates": [], "deletes": []}} if nothing durable is found."""


def build_batch_consolidation_prompt(observations_mission: str | None = None) -> str:
    """
    Build the consolidation prompt for batch mode (multiple facts per LLM call).

    The mission defines *what* to track (customisable per bank).
    Processing rules and output format are always present regardless of mission.
    """
    mission = observations_mission or _DEFAULT_MISSION

    return (
        "You are a memory consolidation system. Synthesize facts into observations "
        "and merge with existing observations when appropriate.\n\n"
        f"## MISSION\n{mission}\n\n"
        f"{_PROCESSING_RULES}"
        + _BATCH_DATA_SECTION
        + _BATCH_OUTPUT_FORMAT
    )
