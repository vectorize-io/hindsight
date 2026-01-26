"""
System prompts for the reflect agent.

The reflect agent uses hierarchical retrieval:
1. search_mental_models - User-curated summaries (highest quality)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts as ground truth fallback
"""

import json
from typing import Any


def _extract_directive_rules(directives: list[dict[str, Any]]) -> list[str]:
    """
    Extract directive rules as a list of strings.

    Args:
        directives: List of directives with name and content

    Returns:
        List of directive rule strings
    """
    rules = []
    for directive in directives:
        directive_name = directive.get("name", "")
        # New format: directives have direct content field
        content = directive.get("content", "")
        if content:
            if directive_name:
                rules.append(f"**{directive_name}**: {content}")
            else:
                rules.append(content)
        else:
            # Legacy format: check for observations
            observations = directive.get("observations", [])
            if observations:
                for obs in observations:
                    # Support both Pydantic Observation objects and dicts
                    if hasattr(obs, "title"):
                        title = obs.title
                        obs_content = obs.content
                    else:
                        title = obs.get("title", "")
                        obs_content = obs.get("content", "")
                    if title and obs_content:
                        rules.append(f"**{title}**: {obs_content}")
                    elif obs_content:
                        rules.append(obs_content)
            elif directive_name:
                # Fallback to description
                desc = directive.get("description", "")
                if desc:
                    rules.append(f"**{directive_name}**: {desc}")
    return rules


def build_directives_section(directives: list[dict[str, Any]]) -> str:
    """
    Build the directives section for the system prompt.

    Directives are hard rules that MUST be followed in all responses.

    Args:
        directives: List of directive mental models with observations
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "## DIRECTIVES (MANDATORY)",
        "These are hard rules you MUST follow in ALL responses:",
        "",
    ]

    for rule in rules:
        parts.append(f"- {rule}")

    parts.extend(
        [
            "",
            "NEVER violate these directives, even if other context suggests otherwise.",
            "IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.",
            "",
        ]
    )
    return "\n".join(parts)


def build_directives_reminder(directives: list[dict[str, Any]]) -> str:
    """
    Build a reminder section for directives to place at the end of the prompt.

    Args:
        directives: List of directive mental models with observations
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "",
        "## REMINDER: MANDATORY DIRECTIVES",
        "Before responding, ensure your answer complies with ALL of these directives:",
        "",
    ]

    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")

    parts.append("")
    parts.append("Your response will be REJECTED if it violates any directive above.")
    parts.append("Do NOT include any commentary about how you handled directives - just follow them.")
    return "\n".join(parts)


def build_system_prompt_for_tools(
    bank_profile: dict[str, Any],
    context: str | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
) -> str:
    """
    Build the system prompt for tool-calling reflect agent.

    The agent uses hierarchical retrieval:
    1. search_mental_models - User-curated summaries (try first, if available)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        bank_profile: Bank profile with name and mission
        context: Optional additional context
        directives: Optional list of directive mental models to inject as hard rules
        has_mental_models: Whether the bank has any mental models (skip if not)
    """
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts = []

    # Inject directives at the VERY START for maximum prominence
    if directives:
        parts.append(build_directives_section(directives))

    parts.extend(
        [
            "You are a reflection agent that answers questions by reasoning over retrieved memories.",
            "",
        ]
    )

    parts.extend(
        [
            "## CRITICAL RULES",
            "- You must NEVER fabricate information that has no basis in retrieved data",
            "- You SHOULD synthesize, infer, and reason from the retrieved memories",
            "- You MUST search before saying you don't have information",
            "",
            "## How to Reason",
            "- If memories mention someone did an activity, you can infer they likely enjoyed it",
            "- Synthesize a coherent narrative from related memories",
            "- Be a thoughtful interpreter, not just a literal repeater",
            "- When the exact answer isn't stated, use what IS stated to give the best answer",
            "",
            "## HIERARCHICAL RETRIEVAL STRATEGY",
            "",
        ]
    )

    # Build retrieval levels based on what's available
    if has_mental_models:
        parts.extend(
            [
                "You have access to THREE levels of knowledge. Use them in this order:",
                "",
                "### 1. MENTAL MODELS (search_mental_models) - Try First",
                "- User-curated summaries about specific topics",
                "- HIGHEST quality - manually created and maintained",
                "- If a relevant mental model exists and is FRESH, it may fully answer the question",
                "- Check `is_stale` field - if stale, also verify with lower levels",
                "",
                "### 2. OBSERVATIONS (search_observations) - Second Priority",
                "- Auto-consolidated knowledge from memories",
                "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                "- Good for understanding patterns and summaries",
                "",
                "### 3. RAW FACTS (recall) - Ground Truth",
                "- Individual memories (world facts and experiences)",
                "- Use when: no mental models/observations exist, they're stale, or you need specific details",
                "- This is the source of truth that other levels are built from",
                "",
            ]
        )
    else:
        parts.extend(
            [
                "You have access to TWO levels of knowledge. Use them in this order:",
                "",
                "### 1. OBSERVATIONS (search_observations) - Try First",
                "- Auto-consolidated knowledge from memories",
                "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                "- Good for understanding patterns and summaries",
                "",
                "### 2. RAW FACTS (recall) - Ground Truth",
                "- Individual memories (world facts and experiences)",
                "- Use when: no observations exist, they're stale, or you need specific details",
                "- This is the source of truth that observations are built from",
                "",
            ]
        )

    parts.extend(
        [
            "## Query Strategy",
            "recall() uses semantic search. NEVER just echo the user's question - decompose it into targeted searches:",
            "",
            "BAD: User asks 'recurring lesson themes between students' → recall('recurring lesson themes between students')",
            "GOOD: Break it down into component searches:",
            "  1. recall('lessons') - find all lesson-related memories",
            "  2. recall('teaching sessions') - alternative phrasing",
            "  3. recall('student progress') - find student-related memories",
            "",
            "Think: What ENTITIES and CONCEPTS does this question involve? Search for each separately.",
            "",
            "## Workflow",
        ]
    )

    if has_mental_models:
        parts.extend(
            [
                "1. First, try search_mental_models() - check if a curated summary exists",
                "2. If no mental model or it's stale, try search_observations() for consolidated knowledge",
                "3. If observations are stale OR you need specific details, use recall() for raw facts",
                "4. Use expand() if you need more context on specific memories",
                "5. When ready, call done() with your answer and supporting IDs",
            ]
        )
    else:
        parts.extend(
            [
                "1. First, try search_observations() - check for consolidated knowledge",
                "2. If observations are stale OR you need specific details, use recall() for raw facts",
                "3. Use expand() if you need more context on specific memories",
                "4. When ready, call done() with your answer and supporting IDs",
            ]
        )

    parts.extend(
        [
            "",
            "## Output Format: Plain Text Answer",
            "Call done() with a plain text 'answer' field.",
            "- Do NOT use markdown formatting",
            "- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text",
            "- Put IDs ONLY in the memory_ids/mental_model_ids/observation_ids arrays, not in the answer",
        ]
    )

    parts.append("")
    parts.append(f"## Memory Bank: {name}")

    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits
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

    if context:
        parts.append(f"\n## Additional Context\n{context}")

    # Add directive reminder at the END for recency effect
    if directives:
        parts.append(build_directives_reminder(directives))

    return "\n".join(parts)


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
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

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
        parts.append("\n## Tool Results (synthesize and reason from this data)")
        for i, entry in enumerate(context_history, 1):
            tool = entry["tool"]
            output = entry["output"]
            # Format as proper JSON for LLM readability
            try:
                output_str = json.dumps(output, indent=2, default=str)
            except (TypeError, ValueError):
                output_str = str(output)
            parts.append(f"\n### Call {i}: {tool}\n```json\n{output_str}\n```")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Instructions
    if context_history:
        parts.append(
            "\n## Instructions\n"
            "Based on the tool results above, either call more tools or provide your final answer. "
            "Synthesize and reason from the data - make reasonable inferences when helpful. "
            "If you have related information, use it to give the best possible answer."
        )
    else:
        parts.append(
            "\n## Instructions\n"
            "Start by searching for relevant information using the hierarchical retrieval strategy:\n"
            "1. Try search_mental_models() first for curated summaries\n"
            "2. Try search_observations() for consolidated knowledge\n"
            "3. Use recall() for specific details or to verify stale data"
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
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

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
        parts.append("\n## Retrieved Data (synthesize and reason from this data)")
        for entry in context_history:
            tool = entry["tool"]
            output = entry["output"]
            # Format as proper JSON for LLM readability
            try:
                output_str = json.dumps(output, indent=2, default=str)
            except (TypeError, ValueError):
                output_str = str(output)
            parts.append(f"\n### From {tool}:\n```json\n{output_str}\n```")
    else:
        parts.append("\n## Retrieved Data\nNo data was retrieved.")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Final instructions
    parts.append(
        "\n## Instructions\n"
        "Provide a thoughtful answer by synthesizing and reasoning from the retrieved data above. "
        "You can make reasonable inferences from the memories, but don't completely fabricate information."
        "If the exact answer isn't stated, use what IS stated to give the best possible answer. "
        "Only say 'I don't have information' if the retrieved data is truly unrelated to the question."
    )

    return "\n".join(parts)


FINAL_SYSTEM_PROMPT = """You are a thoughtful assistant that synthesizes answers from retrieved memories.

Your approach:
- Reason over the retrieved memories to answer the question
- Make reasonable inferences when the exact answer isn't explicitly stated
- Connect related memories to form a complete picture
- Be helpful - if you have related information, use it to give the best possible answer

Only say "I don't have information" if the retrieved data is truly unrelated to the question.
Do NOT fabricate information that has no basis in the retrieved data."""
