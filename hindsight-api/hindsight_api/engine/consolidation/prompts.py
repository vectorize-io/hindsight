"""Prompts for the consolidation engine."""

# Output format instructions
_OUTPUT_FORMAT = """
Output a JSON object with an "actions" array:
{{"actions": [
  {{"action": "update", "learning_id": "uuid-from-observations", "text": "...", "reason": "..."}},
  {{"action": "create", "text": "...", "reason": "..."}}
]}}

Return {{"actions": []}} if the fact contains no durable knowledge.
Do NOT include "tags" in output — tags are handled automatically."""

# Data section - holds the dynamic per-call data
_DATA_SECTION = """
NEW FACT: {fact_text}

EXISTING OBSERVATIONS (JSON array with source memories and dates):
{observations_text}

Each observation includes:
- id: unique identifier for updating
- text: the observation content
- proof_count: number of supporting memories
- occurred_start/occurred_end: temporal range of source facts
- source_memories: array of supporting facts with their text and dates

Compare the new fact against existing observations:
- Same topic → UPDATE with learning_id
- New topic → CREATE new observation
- Purely ephemeral → return empty actions list"""

# Default rules used when no observations_spec is set
_DEFAULT_RULES = """Extract DURABLE KNOWLEDGE from facts — the stable truth implied by an event, not transient state.

Example: "User moved to Room 203" → observe "Room 203 exists", not "User is in Room 203".

Rules:
- Keep specifics: names, numbers, locations. Never abstract into general principles.
- NEVER merge observations about different people or unrelated topics.
- REDUNDANT: same info worded differently → update existing.
- CONTRADICTION/UPDATE: capture both states with temporal markers ("used to X, now Y")."""


def build_consolidation_prompt(observations_spec: str | None = None) -> str:
    """
    Build the consolidation prompt.

    If observations_spec is provided, it defines what observations are for this bank,
    replacing the default durable-knowledge rules. Otherwise the default rules apply.
    """
    rules_section = f"## OBSERVATIONS DEFINITION\n{observations_spec}" if observations_spec else _DEFAULT_RULES

    return (
        "You are a memory consolidation system. Synthesize facts into observations "
        "and merge with existing observations when appropriate.\n\n" + rules_section + _DATA_SECTION + _OUTPUT_FORMAT
    )
