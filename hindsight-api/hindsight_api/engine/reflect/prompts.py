"""
System prompts for the reflect agent.
"""

# Tool descriptions
_TOOL_LOOKUP = """### lookup(model_id?)
Get mental models - your synthesized knowledge about entities, concepts, and events.
- Without model_id: Lists all mental models (name + description only)
- With model_id: Gets full details including summary"""

_TOOL_RECALL = """### recall(query)
Search facts using semantic + temporal retrieval. Returns relevant memories from experience and world knowledge."""

_TOOL_LEARN = """### learn(name, type, description, summary, triggers, entity_id?)
Store a new insight as a mental model. Use when you've synthesized understanding worth remembering.
- name: Human-readable name
- type: "entity", "concept", or "event"
- description: One-liner for quick reference
- summary: Full synthesized understanding
- triggers: Keywords for retrieval
- entity_id: Optional link to existing entity"""

_TOOL_EXPAND = """### expand(memory_id, depth)
Get more context for a memory.
- depth="chunk": Returns the surrounding text chunk
- depth="document": Returns the full source document"""

_TOOL_DONE = """### done(answer)
Signal completion with your final answer."""


def build_system_prompt(enable_learn: bool = True) -> str:
    """Build the system prompt with optional learn tool."""
    tools = [_TOOL_LOOKUP, _TOOL_RECALL]
    if enable_learn:
        tools.append(_TOOL_LEARN)
    tools.extend([_TOOL_EXPAND, _TOOL_DONE])

    learn_goal = "- When you learn something new worth remembering, use the learn tool\n" if enable_learn else ""

    return f"""You are a reflection agent that answers questions ONLY using information retrieved from tools.

## CRITICAL RULES
- You must NEVER make up or hallucinate information
- Your answers must be based ONLY on data returned by the tools
- You MUST call recall() before saying you don't have information - mental models alone are not enough
- Only say "I don't have information about this" AFTER trying both lookup AND recall with no relevant results

## Available Tools

{chr(10).join(tools)}

## Workflow

1. Check if a mental model in the context matches the question exactly - if so, use lookup(model_id) to get its summary
2. If no mental model matches OR you need more details: ALWAYS call recall(query) to search for facts
3. Try multiple recall queries with different phrasings if the first doesn't find what you need
4. Use expand(memory_id, depth) if you need more context on a specific fact
{learn_goal}5. When ready, use done(answer) with your response based ONLY on retrieved data

IMPORTANT: Even if no mental model matches, you MUST try recall() before concluding you don't have information.

## Response Format

Respond with JSON containing an "actions" array:

{{
  "actions": [
    {{"tool": "lookup", "reasoning": "Check for matching mental model"}},
    {{"tool": "recall", "query": "specific search query", "reasoning": "Search for facts"}}
  ]
}}

When done:
{{
  "actions": [
    {{"tool": "done", "answer": "Based on the retrieved information: ..."}}
  ]
}}
"""


# Default system prompt with learn enabled (for backward compatibility)
REFLECT_AGENT_SYSTEM_PROMPT = build_system_prompt(enable_learn=True)


def build_agent_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
) -> str:
    """Build the user prompt for the reflect agent."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    background = bank_profile.get("background", "")
    goal = bank_profile.get("goal", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if background:
        parts.append(f"Background: {background}")
    if goal:
        parts.append(f"Goal: {goal}")

    # Disposition traits if present
    disposition = bank_profile.get("disposition", {})
    if disposition:
        traits = []
        if "skepticism" in disposition:
            traits.append(f"skepticism={disposition['skepticism']}")
        if "literalism" in disposition:
            traits.append(f"literalism={disposition['literalism']}")
        if "empathy" in disposition:
            traits.append(f"empathy={disposition['empathy']}")
        if traits:
            parts.append(f"Disposition: {', '.join(traits)}")

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history
    if context_history:
        parts.append("\n## Tool Results (use ONLY this data for your answer)")
        for i, entry in enumerate(context_history, 1):
            tool = entry["tool"]
            output = entry["output"]
            # Truncate long outputs
            output_str = str(output)
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "... (truncated)"
            parts.append(f"\n### Call {i}: {tool}\n```json\n{output_str}\n```")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Instructions
    if context_history:
        parts.append(
            "\n## Instructions\n"
            "Based on the tool results above, either call more tools or provide your final answer. "
            "Your answer must be grounded ONLY in the data shown above - do not add information you don't have."
        )
    else:
        parts.append(
            "\n## Instructions\n"
            "Check if any mental model above matches your question. If so, use lookup(model_id) to get details. "
            "If no mental model matches, you MUST call recall(query) to search for facts. "
            "Never answer without first trying recall() - it searches all stored memories."
        )

    return "\n".join(parts)


def build_final_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
) -> str:
    """Build the final prompt when forcing a text response (no tools)."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    background = bank_profile.get("background", "")
    goal = bank_profile.get("goal", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if background:
        parts.append(f"Background: {background}")
    if goal:
        parts.append(f"Goal: {goal}")

    # Disposition traits if present
    disposition = bank_profile.get("disposition", {})
    if disposition:
        traits = []
        if "skepticism" in disposition:
            traits.append(f"skepticism={disposition['skepticism']}")
        if "literalism" in disposition:
            traits.append(f"literalism={disposition['literalism']}")
        if "empathy" in disposition:
            traits.append(f"empathy={disposition['empathy']}")
        if traits:
            parts.append(f"Disposition: {', '.join(traits)}")

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history
    if context_history:
        parts.append("\n## Retrieved Data (use ONLY this for your answer)")
        for entry in context_history:
            tool = entry["tool"]
            output = entry["output"]
            output_str = str(output)
            if len(output_str) > 2000:
                output_str = output_str[:2000] + "... (truncated)"
            parts.append(f"\n### From {tool}:\n{output_str}")
    else:
        parts.append("\n## Retrieved Data\nNo data was retrieved.")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Final instructions
    parts.append(
        "\n## Instructions\n"
        "Provide your answer based ONLY on the retrieved data above. "
        "Do not make up or hallucinate any information. "
        "If no relevant data was found, say 'I don't have information about this topic.'"
    )

    return "\n".join(parts)


FINAL_SYSTEM_PROMPT = """You are a grounded assistant that answers ONLY based on retrieved data.
CRITICAL: Never hallucinate or make up information. Only use the data shown in the prompt.
If no relevant data exists, clearly state that you don't have information about this topic."""
