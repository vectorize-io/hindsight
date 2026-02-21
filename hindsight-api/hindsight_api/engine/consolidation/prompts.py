"""Prompts for the consolidation engine."""

# Data section template - shared across all modes, holds the dynamic per-call data
_DATA_SECTION = """
{mission_section}NEW FACT: {fact_text}

EXISTING OBSERVATIONS (JSON array with source memories and dates):
{observations_text}

Each observation includes:
- id: unique identifier for updating
- text: the observation content
- proof_count: number of supporting memories
- tags: visibility scope (handled automatically)
- occurred_start/occurred_end: temporal range of source facts
- source_memories: array of supporting facts with their text and dates

Instructions:
1. Extract DURABLE KNOWLEDGE from the new fact (not ephemeral state)
2. Review source_memories in existing observations to understand evidence
3. Check dates to detect contradictions or updates
4. Compare with observations:
   - Same topic → UPDATE with learning_id
   - New topic → CREATE new observation
   - Purely ephemeral → return empty actions list"""

# Output format instructions - shared across all modes
_OUTPUT_FORMAT = """
Output a JSON object with an "actions" array (the "text" field should use markdown formatting for structure):
{{"actions": [
  {{"action": "update", "learning_id": "uuid-from-observations", "text": "## Updated Knowledge\n\n**Key point**: details here\n\n- Supporting detail 1\n- Supporting detail 2", "reason": "..."}},
  {{"action": "create", "text": "## New Durable Knowledge\n\nDescription with **emphasis** and proper structure", "reason": "..."}}
]}}

Return {{"actions": []}} if fact contains no durable knowledge.

IMPORTANT: Format the "text" field with markdown for better readability:
- Use headers, lists, bold/italic, tables where appropriate
- CRITICAL: Add blank lines before and after block elements (tables, code blocks, lists)
- Ensure proper spacing for markdown to render correctly
- Do NOT include "tags" in output - tags are handled automatically"""


# Standard prompt - full detailed rules
STANDARD_CONSOLIDATION_PROMPT = (
    """You are a memory consolidation system. Your job is to convert facts into durable knowledge (observations) and merge with existing knowledge when appropriate.

The "text" field within each action should use markdown formatting (headers, lists, bold, etc.) for clarity and readability.

## EXTRACT DURABLE KNOWLEDGE, NOT EPHEMERAL STATE
Facts often describe events or actions. Extract the DURABLE KNOWLEDGE implied by the fact, not the transient state.

Examples of extracting durable knowledge:
- "User moved to Room 203" -> "Room 203 exists" (location exists, not where user is now)
- "User visited Acme Corp at Room 105" -> "Acme Corp is located in Room 105"
- "User took the elevator to floor 3" -> "Floor 3 is accessible by elevator"
- "User met Sarah at the lobby" -> "Sarah can be found at the lobby"

DO NOT track current user position/state as knowledge - that changes constantly.
DO track permanent facts learned from the user's actions.

## PRESERVE SPECIFIC DETAILS
Keep names, locations, numbers, and other specifics. Do NOT:
- Abstract into general principles
- Generate business insights
- Make knowledge generic

GOOD examples:
- Fact: "John likes pizza" -> "John likes pizza"
- Fact: "Alice works at Google" -> "Alice works at Google"

BAD examples:
- "John likes pizza" -> "Understanding dietary preferences helps..." (TOO ABSTRACT)
- "User is at Room 203" -> "User is currently at Room 203" (EPHEMERAL STATE)

## MERGE RULES (when comparing to existing observations):
1. REDUNDANT: Same information worded differently → update existing
2. CONTRADICTION: Opposite information about same topic → update with temporal markers showing change
   Example: "Alex used to love pizza but now hates it" OR "Alex's pizza preference changed from love to hate"
3. UPDATE: New state replacing old state → update showing the transition with "used to", "now", "changed from X to Y"

## CRITICAL RULES:
- NEVER merge facts about DIFFERENT people
- NEVER merge unrelated topics (food preferences vs work vs hobbies)
- When merging contradictions, the "text" field MUST capture BOTH states with temporal markers:
  * Use "used to X, now Y" OR "changed from X to Y" OR "X but now Y"
  * DO NOT just state the new fact - you MUST show the change
- Keep observations focused on ONE specific topic per person
- The "text" field MUST contain durable knowledge, not ephemeral state"""
    + _DATA_SECTION
    + _OUTPUT_FORMAT
)


# Custom prompt template - user-provided instructions replace the rules section
CUSTOM_CONSOLIDATION_PROMPT = (
    """You are a memory consolidation system. Your job is to convert facts into durable knowledge (observations) and merge with existing knowledge when appropriate.

The "text" field within each action should use markdown formatting (headers, lists, bold, etc.) for clarity and readability.

{custom_instructions}"""
    + _DATA_SECTION
    + _OUTPUT_FORMAT
)
